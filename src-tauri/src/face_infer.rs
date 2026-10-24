//! Face inference engine: a port of node/face/image-ops.ts,
//! ort-provider.ts, scrfd.ts, arcface.ts, lvface.ts, genderage.ts,
//! face-scan.ts and the worker's attribute job (design doc Phase C).
//!
//! Parity: every resampling step mirrors the TypeScript implementation
//! (which mirrors OpenCV semantics), and the whole pipeline was checked
//! against the spike golden data before this port was written —
//! embedding cosine vs the Node implementation was 0.9996+ on all
//! golden faces (docs/TAURI_FACE_PORT_DESIGN.md, decision T6).
//!
//! ONNX Runtime is loaded dynamically (`ort` crate, load-dynamic): the
//! library is located via ORT_DYLIB_PATH, next to the executable, or the
//! system loader, in that order. GPU execution providers are only
//! available when the crate is built with the matching ort features;
//! without them the provider ladder records the attempt as failed and
//! lands on CPU, exactly like a machine without the CUDA libraries.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{Value, json};
use wait_timeout::ChildExt;

use crate::face::{FaceStore, ScannedFace};
use crate::face_cluster;

// ---------------------------------------------------------------------------
// Raw images and image operations (image-ops.ts)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RawImage {
  pub data: Vec<u8>, // RGB, row-major, width*height*3
  pub width: usize,
  pub height: usize,
}

/// Decode a JPEG/PNG file into raw RGB (sharp's role in the TS code).
pub fn decode_image(path: &Path) -> Result<RawImage, String> {
  let img = image::open(path)
    .map_err(|error| format!("Cannot decode {}: {error}", path.display()))?;
  let rgb = img.to_rgb8();
  let (width, height) = rgb.dimensions();
  Ok(RawImage {
    data: rgb.into_raw(),
    width: width as usize,
    height: height as usize,
  })
}

/// Encode raw RGB as a JPEG file (sharp's role in the TS code).
pub fn encode_jpeg(img: &RawImage, out_path: &Path, quality: u8) -> Result<(), String> {
  let mut bytes: Vec<u8> = Vec::new();
  {
    let mut encoder =
      image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality);
    encoder
      .encode(
        &img.data,
        img.width as u32,
        img.height as u32,
        image::ExtendedColorType::Rgb8,
      )
      .map_err(|error| format!("Cannot encode JPEG: {error}"))?;
  }
  std::fs::write(out_path, bytes)
    .map_err(|error| format!("Cannot write {}: {error}", out_path.display()))
}

fn reflect101(mut i: i64, n: usize) -> usize {
  if n == 1 {
    return 0;
  }
  let n = n as i64;
  while i < 0 || i >= n {
    if i < 0 {
      i = -i;
    }
    if i >= n {
      i = 2 * n - 2 - i;
    }
  }
  i as usize
}

/// Bilinear resize with cv2.resize INTER_LINEAR coordinate mapping.
pub fn resize_bilinear(src: &RawImage, new_width: usize, new_height: usize) -> RawImage {
  let (sw, sh) = (src.width, src.height);
  let mut out = vec![0u8; new_width * new_height * 3];
  let scale_x = sw as f64 / new_width as f64;
  let scale_y = sh as f64 / new_height as f64;
  for y in 0..new_height {
    let fy = (y as f64 + 0.5) * scale_y - 0.5;
    let y0 = fy.floor() as i64;
    let wy = fy - y0 as f64;
    let (ya, yb) = (reflect101(y0, sh), reflect101(y0 + 1, sh));
    for x in 0..new_width {
      let fx = (x as f64 + 0.5) * scale_x - 0.5;
      let x0 = fx.floor() as i64;
      let wx = fx - x0 as f64;
      let (xa, xb) = (reflect101(x0, sw), reflect101(x0 + 1, sw));
      let o = (y * new_width + x) * 3;
      let i00 = (ya * sw + xa) * 3;
      let i01 = (ya * sw + xb) * 3;
      let i10 = (yb * sw + xa) * 3;
      let i11 = (yb * sw + xb) * 3;
      for c in 0..3 {
        let v = src.data[i00 + c] as f64 * (1.0 - wx) * (1.0 - wy)
          + src.data[i01 + c] as f64 * wx * (1.0 - wy)
          + src.data[i10 + c] as f64 * (1.0 - wx) * wy
          + src.data[i11 + c] as f64 * wx * wy;
        out[o + c] = v.round() as u8;
      }
    }
  }
  RawImage {
    data: out,
    width: new_width,
    height: new_height,
  }
}

/// Affine warp with cv2.warpAffine semantics (M maps source ->
/// destination; bilinear sampling; out-of-source samples are 0).
pub fn warp_affine(src: &RawImage, m: [f64; 6], out_width: usize, out_height: usize) -> RawImage {
  let [a, b, tx, c, d, ty] = m;
  let det = a * d - b * c;
  assert!(det != 0.0, "singular affine matrix");
  let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
  let itx = -(ia * tx + ib * ty);
  let ity = -(ic * tx + id * ty);
  let (sw, sh) = (src.width as i64, src.height as i64);
  let sample = |x: i64, y: i64, ch: usize| -> f64 {
    if x < 0 || y < 0 || x >= sw || y >= sh {
      0.0
    } else {
      src.data[((y * sw + x) * 3) as usize + ch] as f64
    }
  };
  let mut out = vec![0u8; out_width * out_height * 3];
  for y in 0..out_height {
    for x in 0..out_width {
      let sx = ia * x as f64 + ib * y as f64 + itx;
      let sy = ic * x as f64 + id * y as f64 + ity;
      let x0 = sx.floor() as i64;
      let y0 = sy.floor() as i64;
      let wx = sx - x0 as f64;
      let wy = sy - y0 as f64;
      let o = (y * out_width + x) * 3;
      for ch in 0..3 {
        let v = sample(x0, y0, ch) * (1.0 - wx) * (1.0 - wy)
          + sample(x0 + 1, y0, ch) * wx * (1.0 - wy)
          + sample(x0, y0 + 1, ch) * (1.0 - wx) * wy
          + sample(x0 + 1, y0 + 1, ch) * wx * wy;
        out[o + ch] = v.round() as u8;
      }
    }
  }
  RawImage {
    data: out,
    width: out_width,
    height: out_height,
  }
}

/// Place src at the top-left of a zero-filled canvas.
pub fn pad_to_canvas(src: &RawImage, width: usize, height: usize) -> RawImage {
  let mut out = vec![0u8; width * height * 3];
  for y in 0..src.height {
    let offset = y * width * 3;
    out[offset..offset + src.width * 3]
      .copy_from_slice(&src.data[y * src.width * 3..(y + 1) * src.width * 3]);
  }
  RawImage {
    data: out,
    width,
    height,
  }
}

/// NCHW float blob: (v - mean) / std per channel, RGB order preserved.
pub fn to_blob(img: &RawImage, mean: f32, std: f32) -> Vec<f32> {
  let n = img.width * img.height;
  let mut blob = vec![0.0f32; 3 * n];
  for i in 0..n {
    for c in 0..3 {
      blob[c * n + i] = (img.data[i * 3 + c] as f32 - mean) / std;
    }
  }
  blob
}

// ---------------------------------------------------------------------------
// ffmpeg frame extraction (face-scan.ts)
// ---------------------------------------------------------------------------

fn ffmpeg_path() -> PathBuf {
  std::env::var_os("VHA_FFMPEG_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("ffmpeg"))
}

/// The generateFaceFrameArgs arguments from main-extract.ts.
pub fn face_frame_args(video_path: &Path, t: f64, max_width: u32, out_path: &Path) -> Vec<String> {
  vec![
    "-ss".to_string(),
    format_time(t),
    "-i".to_string(),
    video_path.to_string_lossy().into_owned(),
    "-frames:v".to_string(),
    "1".to_string(),
    "-vf".to_string(),
    format!("scale=w='min({max_width},iw)':h=-2"),
    "-q:v".to_string(),
    "2".to_string(),
    out_path.to_string_lossy().into_owned(),
  ]
}

/// JS Number -> string for the values sampleTimes produces (integers
/// print without a fraction, everything else in shortest round-trip
/// form; Rust's f64 Display matches that for these magnitudes).
fn format_time(t: f64) -> String {
  format!("{t}")
}

/// Run ffmpeg with the TS runner's 60-second timeout; the error carries
/// the last 300 characters of stderr.
pub fn run_ffmpeg(args: &[String]) -> Result<(), String> {
  let mut child = Command::new(ffmpeg_path())
    .args(args)
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .map_err(|error| format!("Cannot start ffmpeg: {error}"))?;
  match child.wait_timeout(Duration::from_secs(60)) {
    Ok(Some(status)) => {
      if status.success() {
        Ok(())
      } else {
        let mut stderr = String::new();
        if let Some(mut pipe) = child.stderr.take() {
          use std::io::Read;
          let _ = pipe.read_to_string(&mut stderr);
        }
        let tail: String = stderr.chars().rev().take(300).collect::<String>()
          .chars().rev().collect();
        Err(format!("ffmpeg exited {status}: {tail}"))
      }
    }
    Ok(None) => {
      let _ = child.kill();
      Err("ffmpeg timed out".to_string())
    }
    Err(error) => Err(format!("ffmpeg wait failed: {error}")),
  }
}

/// Extract and decode the frame at `t` seconds (grabFrameImage).
pub fn grab_frame_image(video_path: &Path, t: f64, out_path: &Path) -> Option<RawImage> {
  let _ = std::fs::remove_file(out_path);
  if run_ffmpeg(&face_frame_args(video_path, t, 1280, out_path)).is_err() {
    return None;
  }
  if !out_path.exists() {
    return None;
  }
  decode_image(out_path).ok()
}

// ---------------------------------------------------------------------------
// Provider ladder (ort-provider.ts, decision D9)
// ---------------------------------------------------------------------------

/// Ordered candidates for a normalized setting on this platform.
pub fn provider_candidates(setting: &str) -> Vec<&'static str> {
  match setting {
    "cpu" => vec!["cpu"],
    "cuda" => vec!["cuda", "cpu"],
    "directml" => vec!["directml", "cpu"],
    "coreml" => vec!["coreml", "cpu"],
    _ => {
      // auto
      if cfg!(target_os = "windows") {
        vec!["cuda", "directml", "cpu"]
      } else if cfg!(target_os = "macos") {
        vec!["coreml", "cpu"]
      } else {
        vec!["cuda", "cpu"]
      }
    }
  }
}

fn first_line(error: &str) -> String {
  error.lines().next().unwrap_or_default().chars().take(300).collect()
}

/// Build the execution-provider dispatch for a ladder candidate. The
/// registrations are compiled per platform, matching the target-gated
/// ort features in Cargo.toml (Phase D): Windows gets CUDA + DirectML,
/// macOS CoreML, Linux CUDA. A provider this platform cannot offer —
/// or whose runtime libraries the loaded ONNX Runtime lacks — fails
/// here or at session creation and is recorded by the ladder.
#[cfg(target_os = "windows")]
fn execution_provider(name: &str) -> Result<ort::ep::ExecutionProviderDispatch, String> {
  use ort::ep::ExecutionProvider;
  match name {
    "cuda" => Ok(ort::ep::CUDA::default().build()),
    "directml" => Ok(ort::ep::DirectML::default().build()),
    other => Err(format!(
      "execution provider '{other}' is not compiled into this build"
    )),
  }
}

#[cfg(target_os = "macos")]
fn execution_provider(name: &str) -> Result<ort::ep::ExecutionProviderDispatch, String> {
  use ort::ep::ExecutionProvider;
  match name {
    "coreml" => Ok(ort::ep::CoreML::default().build()),
    other => Err(format!(
      "execution provider '{other}' is not compiled into this build"
    )),
  }
}

#[cfg(target_os = "linux")]
fn execution_provider(name: &str) -> Result<ort::ep::ExecutionProviderDispatch, String> {
  use ort::ep::ExecutionProvider;
  match name {
    "cuda" => Ok(ort::ep::CUDA::default().build()),
    other => Err(format!(
      "execution provider '{other}' is not compiled into this build"
    )),
  }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn execution_provider(name: &str) -> Result<ort::ep::ExecutionProviderDispatch, String> {
  Err(format!(
    "execution provider '{name}' is not compiled into this build"
  ))
}

static RESOURCE_DIR_HINT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Tell the loader where Tauri placed the app's bundled resources
/// (called once from the app setup; the ONNX Runtime library ships as
/// a bundle resource, see src-tauri/fetch-ort.sh).
pub fn set_resource_dir_hint(dir: PathBuf) {
  let _ = RESOURCE_DIR_HINT.set(dir);
}

/// Locate the ONNX Runtime shared library: ORT_DYLIB_PATH, then next
/// to the executable, then the Tauri resource dir (the bundle
/// placement), else None (system loader search via ort::init).
pub fn onnxruntime_dylib() -> Option<PathBuf> {
  if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
    return Some(PathBuf::from(path));
  }
  let file_name = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
  } else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
  } else {
    "libonnxruntime.so"
  };
  let mut search_dirs: Vec<PathBuf> = Vec::new();
  if let Ok(exe) = std::env::current_exe() {
    if let Some(dir) = exe.parent() {
      search_dirs.push(dir.to_path_buf());
      search_dirs.push(dir.join("lib"));
    }
  }
  if let Some(dir) = RESOURCE_DIR_HINT.get() {
    search_dirs.push(dir.clone());
    search_dirs.push(dir.join("lib"));
    search_dirs.push(dir.join("resources"));
  }
  for dir in search_dirs {
    let candidate = dir.join(file_name);
    if candidate.exists() {
      return Some(candidate);
    }
  }
  None
}

/// Session factory shared by all models of one engine, mirroring
/// OrtSessionFactory: the first session walks the ladder, later
/// sessions go straight to the winner, and every attempt is recorded
/// once for the provider resolution report.
pub struct OrtSessionFactory {
  pub requested: String,
  candidates: Vec<&'static str>,
  environment: Option<Arc<Environment>>,
  resolved: Option<String>,
  failed: HashMap<String, String>,
  pub attempts: Vec<Value>,
  pub dylib_path: Option<PathBuf>,
}

impl OrtSessionFactory {
  pub fn new(requested: &str) -> OrtSessionFactory {
    OrtSessionFactory {
      requested: requested.to_string(),
      candidates: provider_candidates(requested),
      environment: None,
      resolved: None,
      failed: HashMap::new(),
      attempts: Vec::new(),
      dylib_path: onnxruntime_dylib(),
    }
  }

  fn environment(&mut self) -> Result<Arc<Environment>, String> {
    if let Some(env) = &self.environment {
      return Ok(env.clone());
    }
    // One ORT environment per process (the Electron worker has one
    // too); ort::init_from may only run once, so the first factory's
    // library discovery wins for everyone.
    static SHARED: std::sync::OnceLock<Result<Arc<Environment>, String>> =
      std::sync::OnceLock::new();
    let dylib = self.dylib_path.clone();
    let shared = SHARED.get_or_init(move || {
      let builder = match &dylib {
        Some(path) => ort::init_from(path).map_err(|error| {
          format!("Cannot load ONNX Runtime from {}: {error}", path.display())
        })?,
        None => ort::init(),
      };
      builder
        .build()
        .map(|env| Arc::new(env))
        .map_err(|error| format!("Cannot initialize ONNX Runtime: {error}"))
    });
    let env = shared.clone()?;
    self.environment = Some(env.clone());
    Ok(env)
  }

  /// The directory holding the ONNX Runtime binary ('' when unknown).
  pub fn ort_bin_dir(&self) -> String {
    self
      .dylib_path
      .as_ref()
      .and_then(|path| path.parent())
      .map(|dir| dir.to_string_lossy().into_owned())
      .unwrap_or_default()
  }

  /// True when both CUDA provider libraries sit next to the ORT binary.
  pub fn cuda_libs_present(&self) -> bool {
    let dir = self.ort_bin_dir();
    if dir.is_empty() {
      return false;
    }
    let names: [&str; 2] = if cfg!(target_os = "windows") {
      ["onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"]
    } else {
      ["libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so"]
    };
    names.iter().all(|name| Path::new(&dir).join(name).exists())
  }

  /// FaceProviderResolution plus the bin-dir fields the settings UI
  /// shows (assembled in getFaceModelStatus on the Electron side).
  pub fn provider_info(&self) -> Value {
    json!({
      "requested": self.requested,
      "effective": self.resolved,
      "attempts": self.attempts,
      "cudaLibsPresent": self.cuda_libs_present(),
      "ortBinDir": self.ort_bin_dir()
    })
  }

  fn create_with_provider(
    &mut self,
    model_path: &Path,
    provider: &str,
  ) -> Result<Session, String> {
    let env = self.environment()?;
    let mut builder = Session::builder(&env)
      .map_err(|error| format!("Cannot create session builder: {error}"))?;
    if provider != "cpu" {
      let dispatch = execution_provider(provider)?;
      builder = builder
        .with_execution_providers([dispatch])
        .map_err(|error| format!("Cannot register provider '{provider}': {error}"))?;
    }
    builder
      .commit_from_file(model_path)
      .map_err(|error| format!("Cannot load {}: {error}", model_path.display()))
  }

  /// Mirror of OrtSessionFactory.createSession.
  pub fn create_session(&mut self, model_path: &Path) -> Result<Session, String> {
    let mut order: Vec<&'static str> = Vec::new();
    if let Some(resolved) = self.resolved.as_deref() {
      if let Some(candidate) = self.candidates.iter().find(|c| **c == resolved) {
        order.push(candidate);
      }
    }
    for candidate in &self.candidates {
      if !order.contains(candidate) && !self.failed.contains_key(*candidate) {
        order.push(candidate);
      }
    }
    if order.is_empty() {
      order = self.candidates.clone();
    }
    let mut last_error = String::new();
    for provider in order {
      match self.create_with_provider(model_path, provider) {
        Ok(session) => {
          if self.resolved.as_deref() != Some(provider) {
            self.resolved = Some(provider.to_string());
            self.attempts.push(json!({ "provider": provider, "ok": true }));
          }
          return Ok(session);
        }
        Err(error) => {
          let message = first_line(&error);
          if !self.failed.contains_key(provider) {
            self.failed.insert(provider.to_string(), message.clone());
            self.attempts.push(json!({
              "provider": provider, "ok": false, "error": message
            }));
          }
          if self.resolved.as_deref() == Some(provider) {
            self.resolved = None;
          }
          last_error = error;
        }
      }
    }
    Err(last_error)
  }
}

// ---------------------------------------------------------------------------
// Models (scrfd.ts / arcface.ts / lvface.ts / genderage.ts)
// ---------------------------------------------------------------------------

fn run_session(
  session: &mut Session,
  input_name: &str,
  blob: Vec<f32>,
  dims: [i64; 4],
) -> Result<HashMap<String, Vec<f32>>, String> {
  let tensor = Tensor::from_array((vec![dims[0], dims[1], dims[2], dims[3]], blob))
    .map_err(|error| format!("Cannot create input tensor: {error}"))?;
  let outputs = session
    .run(ort::inputs![input_name => tensor])
    .map_err(|error| format!("Inference failed: {error}"))?;
  let mut result = HashMap::new();
  for (name, value) in outputs.iter() {
    let array = value
      .try_extract_array::<f32>()
      .map_err(|error| format!("Cannot read output {name}: {error}"))?;
    result.insert(name.to_string(), array.iter().copied().collect());
  }
  Ok(result)
}

#[derive(Clone, Debug)]
pub struct DetectedFace {
  pub bbox: [f64; 4], // x1, y1, x2, y2 in original image px
  pub score: f64,
  pub kps: [[f64; 2]; 5],
}

const SCRFD_STRIDES: [usize; 3] = [8, 16, 32];
const SCRFD_NUM_ANCHORS: usize = 2;
const SCRFD_NMS_THRESH: f64 = 0.4;

pub struct Scrfd {
  session: Session,
  input_name: String,
  output_names: Vec<String>,
}

impl Scrfd {
  pub fn create(model_path: &Path, factory: &mut OrtSessionFactory) -> Result<Scrfd, String> {
    let session = factory.create_session(model_path)?;
    let input_name = session
      .inputs()
      .first()
      .map(|outlet| outlet.name().to_string())
      .unwrap_or_default();
    let output_names: Vec<String> = session
      .outputs()
      .iter()
      .map(|outlet| outlet.name().to_string())
      .collect();
    if output_names.len() != 9 {
      return Err(format!(
        "expected 9 SCRFD outputs, got {}",
        output_names.len()
      ));
    }
    Ok(Scrfd {
      session,
      input_name,
      output_names,
    })
  }

  /// Detect faces; `input_size` is [width, height] of the model canvas.
  pub fn detect(
    &mut self,
    img: &RawImage,
    input_size: [usize; 2],
    threshold: f64,
  ) -> Result<Vec<DetectedFace>, String> {
    let [in_w, in_h] = input_size;
    let im_ratio = img.height as f64 / img.width as f64;
    let model_ratio = in_h as f64 / in_w as f64;
    let (new_width, new_height) = if im_ratio > model_ratio {
      ((in_h as f64 / im_ratio).trunc() as usize, in_h)
    } else {
      (in_w, (in_w as f64 * im_ratio).trunc() as usize)
    };
    let det_scale = new_height as f64 / img.height as f64;
    let resized = resize_bilinear(img, new_width, new_height);
    let canvas = pad_to_canvas(&resized, in_w, in_h);
    let blob = to_blob(&canvas, 127.5, 128.0);
    let outs = run_session(
      &mut self.session,
      &self.input_name,
      blob,
      [1, 3, in_h as i64, in_w as i64],
    )?;

    let mut cands: Vec<DetectedFace> = Vec::new();
    for (idx, stride) in SCRFD_STRIDES.iter().enumerate() {
      let stride = *stride as f64;
      let scores = &outs[&self.output_names[idx]];
      let bbox_pred = &outs[&self.output_names[idx + 3]];
      let kps_pred = &outs[&self.output_names[idx + 6]];
      let h = in_h / SCRFD_STRIDES[idx];
      let w = in_w / SCRFD_STRIDES[idx];
      let total = h * w * SCRFD_NUM_ANCHORS;
      for i in 0..total {
        let score = scores[i] as f64;
        if score < threshold {
          continue;
        }
        // Anchor centre: anchors are laid out (y, x, anchor) row-major.
        let cell = i / SCRFD_NUM_ANCHORS;
        let cx = (cell % w) as f64 * stride;
        let cy = (cell / w) as f64 * stride;
        let bo = i * 4;
        let bbox = [
          (cx - bbox_pred[bo] as f64 * stride) / det_scale,
          (cy - bbox_pred[bo + 1] as f64 * stride) / det_scale,
          (cx + bbox_pred[bo + 2] as f64 * stride) / det_scale,
          (cy + bbox_pred[bo + 3] as f64 * stride) / det_scale,
        ];
        let ko = i * 10;
        let mut kps = [[0.0; 2]; 5];
        for (k, point) in kps.iter_mut().enumerate() {
          point[0] = (cx + kps_pred[ko + k * 2] as f64 * stride) / det_scale;
          point[1] = (cy + kps_pred[ko + k * 2 + 1] as f64 * stride) / det_scale;
        }
        cands.push(DetectedFace {
          bbox,
          score,
          kps,
        });
      }
    }
    // Stable descending sort + NMS (the "+1 pixel" area convention).
    cands.sort_by(|p, q| q.score.total_cmp(&p.score));
    let mut keep: Vec<DetectedFace> = Vec::new();
    let mut suppressed = vec![false; cands.len()];
    for i in 0..cands.len() {
      if suppressed[i] {
        continue;
      }
      keep.push(cands[i].clone());
      let [ax1, ay1, ax2, ay2] = cands[i].bbox;
      let area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0);
      for j in (i + 1)..cands.len() {
        if suppressed[j] {
          continue;
        }
        let [bx1, by1, bx2, by2] = cands[j].bbox;
        let wgt = (ax2.min(bx2) - ax1.max(bx1) + 1.0).max(0.0);
        let hgt = (ay2.min(by2) - ay1.max(by1) + 1.0).max(0.0);
        let inter = wgt * hgt;
        let area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0);
        if inter / (area_a + area_b - inter) > SCRFD_NMS_THRESH {
          suppressed[j] = true;
        }
      }
    }
    Ok(keep)
  }
}

/// ArcFace alignment template for 112x112.
pub const ARCFACE_DST: [[f64; 2]; 5] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/// Least-squares similarity transform mapping src points onto dst
/// (estimateNorm in arcface.ts; returns [a, b, tx, c, d, ty]).
pub fn estimate_norm(lmk: &[[f64; 2]; 5], image_size: f64) -> [f64; 6] {
  let ratio = image_size / 112.0;
  let dst: Vec<[f64; 2]> = ARCFACE_DST
    .iter()
    .map(|p| [p[0] * ratio, p[1] * ratio])
    .collect();
  let n = lmk.len() as f64;
  let (mut msx, mut msy, mut mdx, mut mdy) = (0.0, 0.0, 0.0, 0.0);
  for i in 0..lmk.len() {
    msx += lmk[i][0];
    msy += lmk[i][1];
    mdx += dst[i][0];
    mdy += dst[i][1];
  }
  msx /= n;
  msy /= n;
  mdx /= n;
  mdy /= n;
  let (mut num1, mut num2, mut den) = (0.0, 0.0, 0.0);
  for i in 0..lmk.len() {
    let sx = lmk[i][0] - msx;
    let sy = lmk[i][1] - msy;
    let dx = dst[i][0] - mdx;
    let dy = dst[i][1] - mdy;
    num1 += sx * dx + sy * dy;
    num2 += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  let a = num1 / den;
  let b = num2 / den;
  let tx = mdx - (a * msx - b * msy);
  let ty = mdy - (b * msx + a * msy);
  [a, -b, tx, b, a, ty]
}

/// Warp the image so the landmarks land on the template (normCrop).
pub fn norm_crop(img: &RawImage, face: &DetectedFace, image_size: usize) -> RawImage {
  let m = estimate_norm(&face.kps, image_size as f64);
  warp_affine(img, m, image_size, image_size)
}

fn normalized_embedding(session: &mut Session, input_name: &str, aligned: &RawImage, validate: bool) -> Result<Vec<f32>, String> {
  let blob = to_blob(aligned, 127.5, 127.5);
  let outs = run_session(session, input_name, blob, [1, 3, 112, 112])?;
  let raw = outs
    .values()
    .next()
    .ok_or_else(|| "recognition model produced no output".to_string())?;
  let mut emb: Vec<f32> = raw.clone();
  let mut norm = 0.0f64;
  for value in &emb {
    norm += *value as f64 * *value as f64;
  }
  norm = norm.sqrt();
  if validate && (!norm.is_finite() || norm == 0.0) {
    return Err(format!("invalid LVFace embedding norm: {norm}"));
  }
  for value in &mut emb {
    *value = (*value as f64 / norm) as f32;
  }
  Ok(emb)
}

/// Recognition model: ArcFace or LVFace (identical input path; LVFace
/// validates the embedding norm like the TS port).
pub enum Recognizer {
  ArcFace { session: Session, input_name: String },
  LVFace { session: Session, input_name: String },
}

impl Recognizer {
  pub fn create(
    model_path: &Path,
    kind: &str,
    factory: &mut OrtSessionFactory,
  ) -> Result<Recognizer, String> {
    let session = factory.create_session(model_path)?;
    let input_name = session
      .inputs()
      .first()
      .map(|outlet| outlet.name().to_string())
      .unwrap_or_default();
    Ok(if kind == "lvface" {
      Recognizer::LVFace {
        session,
        input_name,
      }
    } else {
      Recognizer::ArcFace {
        session,
        input_name,
      }
    })
  }

  /// The L2-normalized 512-d embedding of an aligned 112x112 face.
  pub fn embedding(&mut self, aligned: &RawImage) -> Result<Vec<f32>, String> {
    match self {
      Recognizer::ArcFace {
        session,
        input_name,
      } => normalized_embedding(session, input_name, aligned, false),
      Recognizer::LVFace {
        session,
        input_name,
      } => normalized_embedding(session, input_name, aligned, true),
    }
  }
}

pub struct GenderAge {
  session: Session,
  input_name: String,
}

pub struct AttributeSample {
  pub sex: &'static str, // "M" | "F"
  pub age: i64,
}

impl GenderAge {
  pub const INPUT_SIZE: usize = 96;

  pub fn create(model_path: &Path, factory: &mut OrtSessionFactory) -> Result<GenderAge, String> {
    let session = factory.create_session(model_path)?;
    let input_name = session
      .inputs()
      .first()
      .map(|outlet| outlet.name().to_string())
      .unwrap_or_default();
    Ok(GenderAge {
      session,
      input_name,
    })
  }

  /// Port of Attribute.get: the crop centres a similarity transform on
  /// the bbox centre with scale = 96 / (max(w, h) * 1.5), rotation 0,
  /// and the model consumes raw pixels (its graph normalizes itself).
  pub fn get(&mut self, img: &RawImage, bbox: [f64; 4]) -> Result<AttributeSample, String> {
    let size = Self::INPUT_SIZE as f64;
    let [x1, y1, x2, y2] = bbox;
    let w = x2 - x1;
    let h = y2 - y1;
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let s = size / (w.max(h) * 1.5);
    let m = [s, 0.0, size / 2.0 - s * cx, 0.0, s, size / 2.0 - s * cy];
    let crop = warp_affine(img, m, Self::INPUT_SIZE, Self::INPUT_SIZE);
    let blob = to_blob(&crop, 0.0, 1.0);
    let outs = run_session(
      &mut self.session,
      &self.input_name,
      blob,
      [1, 3, Self::INPUT_SIZE as i64, Self::INPUT_SIZE as i64],
    )?;
    let pred = outs
      .values()
      .next()
      .ok_or_else(|| "genderage model produced no output".to_string())?;
    let gender_m = pred[1] > pred[0];
    Ok(AttributeSample {
      sex: if gender_m { "M" } else { "F" },
      age: (pred[2] * 100.0).round() as i64,
    })
  }
}

/// Aggregate per-face samples into the person-level record
/// (aggregateAttributes in face-types.ts).
pub fn aggregate_attributes(samples: &[AttributeSample]) -> Option<Value> {
  if samples.is_empty() {
    return None;
  }
  let mut ages: Vec<i64> = samples.iter().map(|s| s.age).collect();
  ages.sort();
  let mid = ages.len() / 2;
  let age_median = if ages.len() % 2 == 1 {
    ages[mid]
  } else {
    ((ages[mid - 1] + ages[mid]) as f64 / 2.0).round() as i64
  };
  let f_count = samples.iter().filter(|s| s.sex == "F").count();
  let sex = if f_count * 2 >= samples.len() { "F" } else { "M" };
  let majority = f_count.max(samples.len() - f_count);
  let agreement = ((majority as f64 / samples.len() as f64) * 100.0).round() / 100.0;
  Some(json!({
    "sex": sex,
    "sexAgreement": agreement,
    "ageMedian": age_median,
    "sampleCount": samples.len()
  }))
}

// ---------------------------------------------------------------------------
// Scan pipeline (face-scan.ts) and orchestration passes (main-face.ts)
// ---------------------------------------------------------------------------

pub struct FaceScanJob {
  pub video_path: PathBuf,
  pub duration: f64,
  pub hash: String,
  pub faces_dir: PathBuf,
  pub max_frames_per_video: i64,
  pub sample_interval_sec: f64,
  pub min_face_size_px: f64,
}

pub struct FaceScanResult {
  pub status: &'static str, // "done" | "none" | "error"
  pub frame_count: i64,
  pub faces: Vec<ScannedFace>,
  pub error: Option<String>,
}

fn sample_times(duration: f64, interval_sec: f64, max_frames: i64) -> Vec<f64> {
  let count = 10.max(max_frames.min((duration / interval_sec).ceil() as i64));
  (0..count)
    .map(|i| (duration - 0.1).min(((i as f64 + 0.5) * duration) / count as f64))
    .collect()
}

fn round_to(value: f64, digits: f64) -> f64 {
  (value * digits).round() / digits
}

/// Display crop (§5.3): square around the face centre, side =
/// max(w, h) * 1.35, clamped to the frame, resized to 256x256. All
/// geometry stays integer (see the Phase 2 note in the TS source).
fn display_crop(img: &RawImage, face: &DetectedFace) -> RawImage {
  let [x1, y1, x2, y2] = face.bbox;
  let side = 1.max(((x2 - x1).max(y2 - y1) * 1.35).round() as i64);
  let cx = (x1 + x2) / 2.0;
  let cy = (y1 + y2) / 2.0;
  let sx = 0.max((img.width as i64 - side).min((cx - side as f64 / 2.0).round() as i64));
  let sy = 0.max((img.height as i64 - side).min((cy - side as f64 / 2.0).round() as i64));
  let sw = side.min(img.width as i64 - sx);
  let sh = side.min(img.height as i64 - sy);
  let mut data = vec![0u8; (sw * sh * 3) as usize];
  for y in 0..sh {
    for x in 0..sw {
      let si = (((sy + y) * img.width as i64 + (sx + x)) * 3) as usize;
      let di = ((y * sw + x) * 3) as usize;
      data[di] = img.data[si];
      data[di + 1] = img.data[si + 1];
      data[di + 2] = img.data[si + 2];
    }
  }
  resize_bilinear(
    &RawImage {
      data,
      width: sw as usize,
      height: sh as usize,
    },
    256,
    256,
  )
}

struct KeptFace {
  scanned: ScannedFace,
  frame_path: PathBuf,
  det: DetectedFace,
}

pub struct FaceEngine {
  detector: Scrfd,
  recognizer: Recognizer,
  factory: OrtSessionFactory,
}

impl FaceEngine {
  pub fn create(
    det_model_path: &Path,
    rec_model_path: &Path,
    rec_kind: &str,
    provider_setting: &str,
  ) -> Result<FaceEngine, String> {
    let mut factory = OrtSessionFactory::new(provider_setting);
    let detector = Scrfd::create(det_model_path, &mut factory)?;
    let recognizer = Recognizer::create(rec_model_path, rec_kind, &mut factory)?;
    Ok(FaceEngine {
      detector,
      recognizer,
      factory,
    })
  }

  /// Which execution provider the sessions actually run on (D9).
  pub fn provider_info(&self) -> Value {
    self.factory.provider_info()
  }

  /// Scan one video's faces. Mirrors FaceEngine.scanVideo: per-video
  /// "soft" failures (no duration, no frames) come back as an error
  /// result, while decode / inference exceptions propagate as Err (the
  /// worker turned them into an error result with frameCount 0).
  pub fn scan_video(&mut self, job: &FaceScanJob) -> Result<FaceScanResult, String> {
    if job.duration <= 0.0 {
      return Ok(FaceScanResult {
        status: "error",
        frame_count: 0,
        faces: Vec::new(),
        error: Some("video duration unknown".to_string()),
      });
    }
    let times = sample_times(job.duration, job.sample_interval_sec, job.max_frames_per_video);
    let tmp_dir = job.faces_dir.join("tmp").join(&job.hash);
    std::fs::create_dir_all(&tmp_dir)
      .map_err(|e| format!("Cannot create {}: {e}", tmp_dir.display()))?;
    std::fs::create_dir_all(job.faces_dir.join("aligned"))
      .map_err(|e| format!("Cannot create aligned dir: {e}"))?;
    std::fs::create_dir_all(job.faces_dir.join("crops"))
      .map_err(|e| format!("Cannot create crops dir: {e}"))?;

    let mut kept: Vec<KeptFace> = Vec::new();
    let mut frames_read = 0i64;
    for (fi, t) in times.iter().enumerate() {
      let frame_path = tmp_dir.join(format!("frame-{fi:03}.jpg"));
      if run_ffmpeg(&face_frame_args(&job.video_path, *t, 1280, &frame_path)).is_err() {
        continue; // a single un-grabbable frame must not fail the video
      }
      if !frame_path.exists() {
        continue;
      }
      frames_read += 1;
      let img = decode_image(&frame_path)?;
      let mut faces = self.detector.detect(&img, [640, 640], 0.5)?;
      if faces.is_empty() {
        // §5.2: SCRFD at 640 can miss faces that fill most of the frame.
        faces = self.detector.detect(&img, [320, 320], 0.5)?;
      }
      for face in &faces {
        let w = face.bbox[2] - face.bbox[0];
        let h = face.bbox[3] - face.bbox[1];
        if w.min(h) < job.min_face_size_px {
          continue;
        }
        let aligned = norm_crop(&img, face, 112);
        let embedding = self.recognizer.embedding(&aligned)?;
        let face_area_ratio = (w * h) / (img.width * img.height) as f64;
        let face_id = format!("{}-{:04}", job.hash, kept.len() + 1);
        kept.push(KeptFace {
          scanned: ScannedFace {
            face_id: face_id.clone(),
            t: round_to(*t, 10.0),
            bbox: [
              round_to(face.bbox[0] / img.width as f64, 10000.0),
              round_to(face.bbox[1] / img.height as f64, 10000.0),
              round_to(w / img.width as f64, 10000.0),
              round_to(h / img.height as f64, 10000.0),
            ],
            score: round_to(face.score, 1000.0),
            quality: round_to(face.score * face_area_ratio.sqrt(), 10000.0),
            embedding,
            group_id: -1,
            representative: false,
          },
          frame_path: frame_path.clone(),
          det: face.clone(),
        });
        encode_jpeg(
          &aligned,
          &job.faces_dir.join("aligned").join(format!("{face_id}.jpg")),
          90,
        )?;
      }
    }

    if frames_read == 0 {
      let _ = std::fs::remove_dir_all(&tmp_dir);
      return Ok(FaceScanResult {
        status: "error",
        frame_count: times.len() as i64,
        faces: Vec::new(),
        error: Some("frame extraction failed for all sampled times".to_string()),
      });
    }

    // §5.5 within-video grouping; groups ordered by representative quality.
    let embeddings: Vec<Vec<f32>> = kept.iter().map(|k| k.scanned.embedding.clone()).collect();
    let qualities: Vec<f64> = kept.iter().map(|k| k.scanned.quality).collect();
    let mut groups = face_cluster::group_within_video(&embeddings, &qualities, 0.5);
    groups.sort_by(|a, b| qualities[b.representative].total_cmp(&qualities[a.representative]));
    for (gi, group) in groups.iter().enumerate() {
      for member in &group.members {
        kept[*member].scanned.group_id = gi as i64;
        kept[*member].scanned.representative = *member == group.representative;
      }
      let rep = &kept[group.representative];
      let frame_img = decode_image(&rep.frame_path)?;
      let crop = display_crop(&frame_img, &rep.det);
      encode_jpeg(
        &crop,
        &job
          .faces_dir
          .join("crops")
          .join(format!("{}.jpg", rep.scanned.face_id)),
        90,
      )?;
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(FaceScanResult {
      status: if kept.is_empty() { "none" } else { "done" },
      frame_count: times.len() as i64,
      faces: kept.into_iter().map(|k| k.scanned).collect(),
      error: None,
    })
  }
}

/// Resolve an element's video file path through the hub's input dirs
/// (the same rule media.rs uses for thumbnails).
pub fn video_path_for(input_dirs: &Value, element: &Value) -> PathBuf {
  let input_source = match element.get("inputSource") {
    Some(Value::Number(number)) => number.as_u64().unwrap_or(0),
    Some(Value::String(text)) => text.parse::<u64>().unwrap_or(0),
    _ => 0,
  };
  let source_path = input_dirs
    .get(input_source.to_string())
    .and_then(|source| source.get("path"))
    .and_then(Value::as_str)
    .unwrap_or_default();
  crate::hub::video_path(
    Path::new(source_path),
    element.get("partialPath").and_then(Value::as_str).unwrap_or(""),
    element.get("fileName").and_then(Value::as_str).unwrap_or(""),
  )
}

/// hash -> video path for the attribute job's frame re-extraction
/// (the videoPathRegistry of main-face.ts, rebuilt per request).
pub fn registry_for(input_dirs: &Value, elements: &[Value]) -> HashMap<String, PathBuf> {
  let mut registry = HashMap::new();
  for element in elements {
    if let Some(hash) = element.get("hash").and_then(Value::as_str) {
      registry.insert(hash.to_string(), video_path_for(input_dirs, element));
    }
  }
  registry
}

fn setting_bool(settings: &Value, key: &str) -> bool {
  settings.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn setting_f64(settings: &Value, key: &str, fallback: f64) -> f64 {
  settings.get(key).and_then(Value::as_f64).unwrap_or(fallback)
}

/// Scan every not-yet-scanned element (scanVideoFaces in main-face.ts,
/// with the engine shared across the pass). Returns the engine's
/// provider info when an engine was created.
pub fn scan_missing_pass(
  settings: &Value,
  asset_folder: &Path,
  input_dirs: &Value,
  elements: &[Value],
) -> Option<Value> {
  if !setting_bool(settings, "enabled") {
    return None;
  }
  let status = crate::face::model_status(settings);
  if status["ready"] != Value::Bool(true) {
    eprintln!(
      "face scan skipped (models not placed): {}",
      status["modelsDir"].as_str().unwrap_or_default()
    );
    return None;
  }
  let pack = settings
    .get("modelPack")
    .and_then(Value::as_str)
    .unwrap_or("quality")
    .to_string();
  let info = crate::face::pack_info(&pack);
  let models_dir = crate::face::face_models_root().join(&pack);
  let threshold = setting_f64(settings, "clusterThreshold", 0.5);
  let faces_dir = asset_folder.join("faces");
  let mut store = FaceStore::load(&faces_dir, info.model_id, threshold);
  let mut engine: Option<FaceEngine> = None;

  for element in elements {
    let Some(hash) = element.get("hash").and_then(Value::as_str) else {
      continue;
    };
    if store.model_mismatch {
      eprintln!("face index model mismatch; resetting face data for this hub");
      store = FaceStore::reset(&faces_dir, info.model_id, threshold);
    }
    if store.is_scanned(hash) {
      continue;
    }
    if engine.is_none() {
      match FaceEngine::create(
        &models_dir.join(info.det_file),
        &models_dir.join(info.rec_file),
        if pack == "lvface" { "lvface" } else { "arcface" },
        &crate::face::normalize_provider(settings),
      ) {
        Ok(created) => engine = Some(created),
        Err(error) => {
          eprintln!("face engine unavailable: {error}");
          return None;
        }
      }
    }
    let engine = engine.as_mut().expect("engine");
    let job = FaceScanJob {
      video_path: video_path_for(input_dirs, element),
      duration: element.get("duration").and_then(Value::as_f64).unwrap_or(0.0),
      hash: hash.to_string(),
      faces_dir: faces_dir.clone(),
      max_frames_per_video: settings
        .get("maxFramesPerVideo")
        .and_then(Value::as_i64)
        .unwrap_or(60),
      sample_interval_sec: setting_f64(settings, "sampleIntervalSec", 30.0),
      min_face_size_px: setting_f64(settings, "minFaceSizePx", 40.0),
    };
    match engine.scan_video(&job) {
      Ok(result) => {
        let stored = if result.status == "error" {
          store.set_video_error(
            hash,
            result.frame_count,
            result.error.as_deref().unwrap_or("unknown error"),
          )
        } else {
          store.set_video(hash, result.status, result.frame_count, &result.faces, None)
        };
        if let Err(error) = stored {
          eprintln!("face scan result could not be stored for {hash}: {error}");
        }
      }
      Err(error) => {
        eprintln!("face scan failed for {hash}: {error}");
        if let Err(store_error) = store.set_video_error(hash, 0, &error) {
          eprintln!("could not record face scan error: {store_error}");
        }
      }
    }
  }
  engine.map(|engine| engine.provider_info())
}

/// Re-run the global clustering (reclusterFaces in main-face.ts).
pub fn recluster_pass(settings: &Value, store: &mut FaceStore) {
  if !setting_bool(settings, "enabled") {
    return;
  }
  if store.model_mismatch || !store.store_has_data() {
    return;
  }
  let threshold = setting_f64(settings, "clusterThreshold", 0.5);
  match store.recluster(threshold) {
    Ok(()) => {
      if !setting_bool(settings, "attributesEnabled") {
        let _ = store.clear_persons_attributes();
      }
    }
    Err(error) => eprintln!("face re-clustering failed: {error}"),
  }
}

/// Compute the person-level attribute aggregates (computeFaceAttributes
/// + the worker's attribute job). Returns the genderage factory's
/// provider info when the job ran.
pub fn attributes_pass(
  settings: &Value,
  store: &mut FaceStore,
  registry: &HashMap<String, PathBuf>,
) -> Option<Value> {
  if !setting_bool(settings, "enabled") || !setting_bool(settings, "attributesEnabled") {
    return None;
  }
  let genderage_path = crate::face::genderage_model_path(settings)?;
  if store.model_mismatch {
    return None;
  }
  let index = store.payload()["index"].clone();
  let persons = index.get("persons").and_then(Value::as_array)?.clone();
  if persons.is_empty() {
    return None;
  }
  let samples_per_group = settings
    .get("attributeSamplesPerGroup")
    .and_then(Value::as_i64)
    .unwrap_or(5);

  // Sample faces per person: per (video, group), the top-N by quality.
  struct SampleRequest {
    face_id: String,
    video_path: PathBuf,
    t: f64,
    bbox: [f64; 4],
  }
  let mut job_persons: Vec<(String, Vec<SampleRequest>)> = Vec::new();
  for person in &persons {
    let person_id = person
      .get("personId")
      .and_then(Value::as_str)
      .unwrap_or_default()
      .to_string();
    let mut faces: Vec<SampleRequest> = Vec::new();
    if let Some(hashes) = person.get("videoHashes").and_then(Value::as_array) {
      for hash_value in hashes {
        let Some(hash) = hash_value.as_str() else { continue };
        let (Some(video_path), Some(entry)) =
          (registry.get(hash), index.get("videos").and_then(|v| v.get(hash)))
        else {
          continue;
        };
        let Some(entry_faces) = entry.get("faces").and_then(Value::as_array) else {
          continue;
        };
        let mut group_order: Vec<i64> = Vec::new();
        let mut groups: HashMap<i64, Vec<&Value>> = HashMap::new();
        for face in entry_faces {
          let gid = face.get("groupId").and_then(Value::as_i64).unwrap_or(-1);
          if face.get("personId").and_then(Value::as_str) == Some(person_id.as_str()) && gid >= 0
          {
            if !groups.contains_key(&gid) {
              group_order.push(gid);
              groups.insert(gid, Vec::new());
            }
            groups.get_mut(&gid).expect("group").push(face);
          }
        }
        for gid in group_order {
          let mut members = groups.remove(&gid).expect("members");
          members.sort_by(|a, b| {
            b.get("quality")
              .and_then(Value::as_f64)
              .unwrap_or(0.0)
              .total_cmp(&a.get("quality").and_then(Value::as_f64).unwrap_or(0.0))
          });
          for face in members.iter().take(samples_per_group.max(0) as usize) {
            let bbox = face.get("bbox").and_then(Value::as_array);
            let component = |i: usize| {
              bbox
                .and_then(|b| b.get(i))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
            };
            faces.push(SampleRequest {
              face_id: face
                .get("faceId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
              video_path: video_path.clone(),
              t: face.get("t").and_then(Value::as_f64).unwrap_or(0.0),
              bbox: [component(0), component(1), component(2), component(3)],
            });
          }
        }
      }
    }
    if !faces.is_empty() {
      job_persons.push((person_id, faces));
    }
  }
  if job_persons.is_empty() {
    return None;
  }

  let mut factory = OrtSessionFactory::new(&crate::face::normalize_provider(settings));
  let mut gender_age = match GenderAge::create(&genderage_path, &mut factory) {
    Ok(created) => created,
    Err(error) => {
      eprintln!("genderage unavailable: {error}");
      return None;
    }
  };
  let faces_dir = store.faces_dir_path().to_path_buf();
  let tmp_dir = faces_dir.join("tmp").join("attributes");
  let _ = std::fs::create_dir_all(&tmp_dir);
  let mut frame_cache: HashMap<String, Option<RawImage>> = HashMap::new();
  let mut results: Vec<(String, Value)> = Vec::new();
  for (person_id, faces) in &job_persons {
    let mut samples: Vec<AttributeSample> = Vec::new();
    for sample in faces {
      let cache_key = format!("{}@{}", sample.video_path.display(), sample.t);
      if !frame_cache.contains_key(&cache_key) {
        let frame_path = tmp_dir.join(format!("frame-{}.jpg", frame_cache.len()));
        let image = grab_frame_image(&sample.video_path, sample.t, &frame_path);
        frame_cache.insert(cache_key.clone(), image);
      }
      let Some(Some(img)) = frame_cache.get(&cache_key) else {
        continue;
      };
      let bbox_abs = [
        sample.bbox[0] * img.width as f64,
        sample.bbox[1] * img.height as f64,
        (sample.bbox[0] + sample.bbox[2]) * img.width as f64,
        (sample.bbox[1] + sample.bbox[3]) * img.height as f64,
      ];
      match gender_age.get(img, bbox_abs) {
        Ok(attribute) => samples.push(attribute),
        Err(error) => eprintln!("genderage failed for face {}: {error}", sample.face_id),
      }
    }
    if let Some(attributes) = aggregate_attributes(&samples) {
      results.push((person_id.clone(), attributes));
    }
  }
  let _ = std::fs::remove_dir_all(&tmp_dir);
  if !results.is_empty() {
    if let Err(error) = store.set_persons_attributes(&results) {
      eprintln!("face attributes could not be stored: {error}");
    }
  }
  Some(factory.provider_info())
}

// ---------------------------------------------------------------------------
// Background orchestration (the Electron face queue's drain behaviour)
// ---------------------------------------------------------------------------

/// Run the full post-scan sequence over a fresh store: re-clustering,
/// then attributes, remembering the latest provider resolution in the
/// app state and notifying the renderer (reclusterAfterScan +
/// notifyFaceIndexUpdated in main-face.ts).
pub fn finish_face_pass(
  app: &tauri::AppHandle,
  settings: &Value,
  asset_folder: &Path,
  input_dirs: &Value,
  elements: &[Value],
  scan_provider_info: Option<Value>,
) {
  use tauri::{Emitter, Manager};
  let mut provider_info = scan_provider_info;
  let mut store = crate::face::hub_store(asset_folder, settings);
  recluster_pass(settings, &mut store);
  let registry = registry_for(input_dirs, elements);
  if let Some(info) = attributes_pass(settings, &mut store, &registry) {
    provider_info = Some(info);
  }
  let payload = store.payload();
  if let Some(info) = provider_info {
    let _ = app
      .state::<crate::state::VhaState>()
      .set_face_provider_info(info);
  }
  let _ = app.emit("face-index-updated", Value::Array(vec![payload]));
}

/// Auto-scan hook for the media pipeline: after an import / thumbnail
/// job finishes, scan any hub videos that have no face data yet when
/// the user enabled face recognition with autoScanNew (the Electron
/// app feeds its face queue from the screenshot queue the same way).
/// Fire-and-forget: failures are logged, never propagated.
pub fn auto_scan_hook(app: &tauri::AppHandle) {
  use tauri::Manager;
  let state = app.state::<crate::state::VhaState>();
  let Ok(settings) = state.face_settings() else {
    return;
  };
  if !setting_bool(&settings, "enabled") || !setting_bool(&settings, "autoScanNew") {
    return;
  }
  let Ok(Some(current)) = state.current_hub() else {
    return;
  };
  let Ok(Some(vha_file)) = state.current_vha_file() else {
    return;
  };
  let app = app.clone();
  std::thread::spawn(move || {
    let Ok(loaded) = crate::hub::load(&vha_file) else {
      return;
    };
    let elements: Vec<Value> = loaded
      .final_object
      .get("images")
      .and_then(Value::as_array)
      .cloned()
      .unwrap_or_default();
    if elements.is_empty() {
      return;
    }
    let provider_info = scan_missing_pass(
      &settings,
      &current.asset_folder,
      &current.input_dirs,
      &elements,
    );
    finish_face_pass(
      &app,
      &settings,
      &current.asset_folder,
      &current.input_dirs,
      &elements,
      provider_info,
    );
  });
}

// ---------------------------------------------------------------------------
// Verification tests. Both are environment-gated (they need the spike
// models, the golden data, an ONNX Runtime library, and ffmpeg), so a
// plain `cargo test` skips them:
//   FACE_SPIKE_DIR=<repo>/tools/face-spike \
//   ORT_DYLIB_PATH=<...>/libonnxruntime.so.1.31.0 \
//   cargo test face_infer -- --nocapture
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn spike_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("FACE_SPIKE_DIR").ok()?);
    if dir.join("golden").join("golden_buffalo_l.json").exists() {
      Some(dir)
    } else {
      None
    }
  }

  /// The T6 numeric-parity gate, as a test: detection, alignment,
  /// embedding, and attributes from this Rust port against the golden
  /// data recorded from the TypeScript implementation.
  #[test]
  fn golden_parity_gate() {
    let Some(spike) = spike_dir() else {
      return;
    };
    let golden: Value = serde_json::from_str(
      &std::fs::read_to_string(spike.join("golden").join("golden_buffalo_l.json"))
        .expect("golden file"),
    )
    .expect("golden json");
    let mut factory = OrtSessionFactory::new("cpu");
    let models = spike.join("models").join("buffalo_l");
    let mut detector = Scrfd::create(&models.join("det_10g.onnx"), &mut factory)
      .expect("detector");
    let mut recognizer =
      Recognizer::create(&models.join("w600k_r50.onnx"), "arcface", &mut factory)
        .expect("recognizer");
    let mut gender_age = GenderAge::create(&models.join("genderage.onnx"), &mut factory)
      .expect("genderage");

    for image in golden["images"].as_array().expect("images") {
      let file = image["file"].as_str().expect("file");
      let img = decode_image(&spike.join("samples").join(file)).expect("decode");
      let det_size = image["detSize"][0].as_u64().expect("detSize") as usize;
      let faces = detector
        .detect(&img, [det_size, det_size], 0.5)
        .expect("detect");
      let golden_faces = image["faces"].as_array().expect("faces");
      assert_eq!(faces.len(), golden_faces.len(), "{file}: face count");
      for golden_face in golden_faces {
        let gb: Vec<f64> = golden_face["bbox"]
          .as_array()
          .expect("bbox")
          .iter()
          .map(|v| v.as_f64().expect("coord"))
          .collect();
        let rust_face = faces
          .iter()
          .min_by(|a, b| {
            let da: f64 = (0..4).map(|i| (a.bbox[i] - gb[i]).abs()).sum();
            let db: f64 = (0..4).map(|i| (b.bbox[i] - gb[i]).abs()).sum();
            da.total_cmp(&db)
          })
          .expect("a matching face");
        for i in 0..4 {
          assert!(
            (rust_face.bbox[i] - gb[i]).abs() < 1.0,
            "{file}: bbox[{i}] drifted: {} vs {}",
            rust_face.bbox[i],
            gb[i]
          );
        }
        let aligned = norm_crop(&img, rust_face, 112);
        let embedding = recognizer.embedding(&aligned).expect("embedding");
        let golden_embedding: Vec<f64> = golden_face["embedding"]
          .as_array()
          .expect("embedding")
          .iter()
          .map(|v| v.as_f64().expect("component"))
          .collect();
        let cosine: f64 = embedding
          .iter()
          .zip(&golden_embedding)
          .map(|(a, b)| *a as f64 * b)
          .sum();
        assert!(
          cosine >= 0.999,
          "{file}: embedding cosine {cosine} below the T6 gate (0.999)"
        );
        let attributes = gender_age.get(&img, rust_face.bbox).expect("attributes");
        assert_eq!(
          attributes.sex,
          golden_face["sex"].as_str().expect("sex"),
          "{file}: sex mismatch"
        );
        assert!(
          (attributes.age - golden_face["age"].as_i64().expect("age")).abs() <= 2,
          "{file}: age drifted: {} vs {}",
          attributes.age,
          golden_face["age"]
        );
      }
    }

    // The auto ladder must walk the compiled GPU provider first: on
    // this machine (no CUDA provider library beside the dylib) the
    // CUDA attempt is recorded as failed and CPU wins; where the
    // provider library exists, CUDA may win instead.
    let mut auto_factory = OrtSessionFactory::new("auto");
    let _ga = GenderAge::create(&models.join("genderage.onnx"), &mut auto_factory)
      .expect("genderage via the auto ladder");
    let info = auto_factory.provider_info();
    let attempts = info["attempts"].as_array().expect("attempts recorded");
    assert_eq!(attempts[0]["provider"], "cuda");
    let effective = info["effective"].as_str().expect("a provider resolved");
    assert!(
      effective == "cpu" || effective == "cuda",
      "effective: {effective}"
    );
  }

  /// End-to-end scan of a synthetic video (a looped golden still)
  /// through the whole pass pipeline: scan -> store -> recluster.
  #[test]
  fn scan_pipeline_end_to_end() {
    let Some(spike) = spike_dir() else {
      return;
    };
    // A private config home so face_models_root() resolves to the
    // symlinked spike models.
    let root = PathBuf::from(format!(
      "{}/vha-face-e2e-{}",
      std::env::temp_dir().display(),
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let config_home = root.join("config");
    let pack_dir = config_home
      .join("Video Hub App 3")
      .join("face-models")
      .join("quality");
    std::fs::create_dir_all(&pack_dir).expect("pack dir");
    for file in ["det_10g.onnx", "w600k_r50.onnx", "genderage.onnx"] {
      std::os::unix::fs::symlink(spike.join("models").join("buffalo_l").join(file), pack_dir.join(file))
        .expect("model symlink");
    }
    // SAFETY: test-only; no other test in this binary resolves the
    // config home while this runs.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", &config_home) };

    // A 3-second video looping the t1 still (6 faces).
    let video_dir = root.join("videos");
    std::fs::create_dir_all(&video_dir).expect("video dir");
    let video_path = video_dir.join("clip.mp4");
    let status = Command::new("ffmpeg")
      .args([
        "-y", "-loglevel", "error", "-loop", "1", "-i",
        spike.join("samples").join("t1.jpg").to_str().expect("sample path"),
        "-t", "3", "-pix_fmt", "yuv420p",
        video_path.to_str().expect("video path"),
      ])
      .status()
      .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg could not build the test video");

    let asset_folder = root.join("vha-test");
    std::fs::create_dir_all(&asset_folder).expect("asset folder");
    let settings = json!({
      "enabled": true, "modelPack": "quality", "maxFramesPerVideo": 60,
      "sampleIntervalSec": 30, "minFaceSizePx": 40, "clusterThreshold": 0.5,
      "autoScanNew": true, "attributesEnabled": false,
      "attributeSamplesPerGroup": 5, "provider": "cpu"
    });
    let input_dirs = json!({ "0": { "path": video_dir.to_string_lossy(), "watch": false } });
    let elements = vec![json!({
      "hash": "e2e0000000000000000000001", "partialPath": "", "fileName": "clip.mp4",
      "duration": 3, "inputSource": 0
    })];

    let provider_info = scan_missing_pass(&settings, &asset_folder, &input_dirs, &elements);
    let provider_info = provider_info.expect("engine ran");
    assert_eq!(provider_info["effective"], json!("cpu"));

    let mut store = crate::face::hub_store(&asset_folder, &settings);
    recluster_pass(&settings, &mut store);
    let payload = store.payload();
    // 10 sampled frames x the same 6 faces; identical frames group
    // into 6 within-video groups and 6 persons.
    assert_eq!(payload["stats"]["scannedVideos"], 1);
    assert_eq!(payload["stats"]["totalFaces"], 60);
    assert_eq!(payload["stats"]["totalGroups"], 6);
    assert_eq!(payload["stats"]["totalPersons"], 6);
    let faces_dir = asset_folder.join("faces");
    let aligned = std::fs::read_dir(faces_dir.join("aligned")).expect("aligned dir").count();
    let crops = std::fs::read_dir(faces_dir.join("crops")).expect("crops dir").count();
    assert_eq!(aligned, 60, "one aligned crop per kept face");
    assert_eq!(crops, 6, "one display crop per group representative");
    let embeddings_len = std::fs::metadata(faces_dir.join("embeddings.bin"))
      .expect("embeddings.bin")
      .len();
    assert_eq!(embeddings_len, 60 * 2048);
    // SAFETY: see above.
    unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
    let _ = std::fs::remove_dir_all(&root);
  }
}
