//! One-click face model downloads: fetch the selected pack's files
//! from their official sources (InsightFace GitHub releases for the
//! buffalo packs, Hugging Face for LVFace-T) into the face-models
//! folder, with progress events and cancellation. The models are never
//! bundled with the app; this only automates the manual placement the
//! README describes, at the user's explicit click.

use std::{
  fs,
  io::{Read, Write},
  path::{Path, PathBuf},
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
  time::{Duration, Instant},
};

use serde_json::{Value, json};
use tauri::{AppHandle, Emitter};

pub const BUFFALO_L_URL: &str =
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip";
pub const BUFFALO_SC_URL: &str =
  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip";
pub const LVFACE_T_URL: &str =
  "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-T_Glint360K/LVFace-T_Glint360K.onnx";

const QUALITY_FILES: [&str; 3] = ["det_10g.onnx", "w600k_r50.onnx", "genderage.onnx"];
const LIGHT_FILES: [&str; 2] = ["det_500m.onnx", "w600k_mbf.onnx"];
const LVFACE_REC_FILE: &str = "LVFace-T_Glint360K.onnx";
const LVFACE_DET_FILE: &str = "det_10g.onnx";

pub fn is_downloadable_pack(pack: &str) -> bool {
  matches!(pack, "quality" | "light" | "lvface")
}

fn emit_progress(app: &AppHandle, payload: Value) {
  let _ = app.emit("face-download-progress", Value::Array(vec![payload]));
}

fn progress_payload(pack: &str, stage: &str, received: u64, total: u64, message: &str) -> Value {
  json!({
    "pack": pack,
    "stage": stage,
    "received": received,
    "total": total,
    "message": message,
  })
}

/// Download `url` to `dest` (via a `.part` sibling), emitting progress
/// at most every ~200 ms. Returns Err("cancelled") when the cancel
/// flag is set; the partial file is removed in that case and on error.
fn download_file(
  app: &AppHandle,
  pack: &str,
  url: &str,
  dest: &Path,
  cancel: &AtomicBool,
) -> Result<(), String> {
  let client = reqwest::blocking::Client::builder()
    .user_agent("VideoHubApp/3 (face model download)")
    .build()
    .map_err(|error| format!("Cannot create download client: {error}"))?;
  let mut response = client
    .get(url)
    .send()
    .map_err(|error| format!("Download failed for {url}: {error}"))?;
  if !response.status().is_success() {
    return Err(format!("Download of {url} returned {}", response.status()));
  }
  let total = response.content_length().unwrap_or(0);

  if let Some(parent) = dest.parent() {
    fs::create_dir_all(parent)
      .map_err(|error| format!("Cannot create {}: {error}", parent.display()))?;
  }
  let part_path = dest.with_extension("part");
  let mut file = fs::File::create(&part_path)
    .map_err(|error| format!("Cannot create {}: {error}", part_path.display()))?;

  let cleanup = |file: fs::File| {
    drop(file);
    let _ = fs::remove_file(&part_path);
  };

  let mut received: u64 = 0;
  let mut last_emit = Instant::now() - Duration::from_millis(500);
  let mut buffer = vec![0u8; 256 * 1024];
  loop {
    if cancel.load(Ordering::Relaxed) {
      cleanup(file);
      return Err("cancelled".to_string());
    }
    let read = match response.read(&mut buffer) {
      Ok(0) => break,
      Ok(n) => n,
      Err(error) => {
        cleanup(file);
        return Err(format!("Download failed for {url}: {error}"));
      }
    };
    if let Err(error) = file.write_all(&buffer[..read]) {
      cleanup(file);
      return Err(format!("Cannot write {}: {error}", part_path.display()));
    }
    received += read as u64;
    if last_emit.elapsed() >= Duration::from_millis(200) {
      last_emit = Instant::now();
      emit_progress(app, progress_payload(pack, "downloading", received, total, ""));
    }
  }
  drop(file);
  fs::rename(&part_path, dest)
    .map_err(|error| format!("Cannot move {} into place: {error}", part_path.display()))?;
  emit_progress(app, progress_payload(pack, "downloading", received, total, ""));
  Ok(())
}

/// Extract only the named entries (matched by file name) from a zip
/// archive into `dest_dir`.
fn extract_zip_entries(zip_path: &Path, dest_dir: &Path, wanted: &[&str]) -> Result<(), String> {
  let file = fs::File::open(zip_path)
    .map_err(|error| format!("Cannot open {}: {error}", zip_path.display()))?;
  let mut archive = zip::ZipArchive::new(file)
    .map_err(|error| format!("Cannot read {} as a zip archive: {error}", zip_path.display()))?;
  fs::create_dir_all(dest_dir)
    .map_err(|error| format!("Cannot create {}: {error}", dest_dir.display()))?;
  let mut extracted = 0;
  for index in 0..archive.len() {
    let mut entry = archive
      .by_index(index)
      .map_err(|error| format!("Cannot read zip entry {index}: {error}"))?;
    let Some(name) = entry
      .enclosed_name()
      .map(|path| path.file_name().map(|name| name.to_string_lossy().to_string()))
    else {
      continue;
    };
    let Some(name) = name else { continue };
    if !wanted.contains(&name.as_str()) {
      continue;
    }
    let out_path = dest_dir.join(&name);
    let mut out = fs::File::create(&out_path)
      .map_err(|error| format!("Cannot create {}: {error}", out_path.display()))?;
    std::io::copy(&mut entry, &mut out)
      .map_err(|error| format!("Cannot extract {name}: {error}"))?;
    extracted += 1;
  }
  if extracted == 0 {
    return Err(format!(
      "{} contains none of the expected model files",
      zip_path.display()
    ));
  }
  Ok(())
}

/// Download a buffalo pack zip and extract its model files.
fn download_buffalo_pack(
  app: &AppHandle,
  pack: &str,
  url: &str,
  files: &[&str],
  models_root: &Path,
  cancel: &AtomicBool,
) -> Result<(), String> {
  let temp_zip = models_root.join(format!(".download-{pack}.zip"));
  let result = download_file(app, pack, url, &temp_zip, cancel).and_then(|()| {
    emit_progress(app, progress_payload(pack, "extracting", 0, 0, ""));
    extract_zip_entries(&temp_zip, &models_root.join(pack), files)
  });
  let _ = fs::remove_file(&temp_zip);
  result
}

/// Make sure the lvface folder has det_10g.onnx: copy it from the
/// quality pack when present, otherwise download the quality pack
/// (which also leaves quality ready to use) and copy it over.
fn ensure_lvface_detector(
  app: &AppHandle,
  models_root: &Path,
  cancel: &AtomicBool,
) -> Result<(), String> {
  let lvface_det = models_root.join("lvface").join(LVFACE_DET_FILE);
  if lvface_det.is_file() {
    return Ok(());
  }
  let quality_det = models_root.join("quality").join(LVFACE_DET_FILE);
  if !quality_det.is_file() {
    download_buffalo_pack(app, "lvface", BUFFALO_L_URL, &QUALITY_FILES, models_root, cancel)?;
  }
  fs::create_dir_all(models_root.join("lvface"))
    .map_err(|error| format!("Cannot create lvface folder: {error}"))?;
  fs::copy(&quality_det, &lvface_det)
    .map_err(|error| format!("Cannot copy the detector into the lvface folder: {error}"))?;
  Ok(())
}

/// Run the full download for `pack` into `models_root`, emitting
/// `face-download-progress` events. Designed to run on a background
/// thread; pair with the VhaState begin/cancel/finish bookkeeping.
pub fn run_download(
  app: &AppHandle,
  pack: &str,
  models_root: &Path,
  cancel: &Arc<AtomicBool>,
) -> Result<(), String> {
  fs::create_dir_all(models_root)
    .map_err(|error| format!("Cannot create {}: {error}", models_root.display()))?;

  match pack {
    "quality" => download_buffalo_pack(app, pack, BUFFALO_L_URL, &QUALITY_FILES, models_root, cancel)?,
    "light" => download_buffalo_pack(app, pack, BUFFALO_SC_URL, &LIGHT_FILES, models_root, cancel)?,
    "lvface" => {
      download_file(
        app,
        pack,
        LVFACE_T_URL,
        &models_root.join("lvface").join(LVFACE_REC_FILE),
        cancel,
      )?;
      ensure_lvface_detector(app, models_root, cancel)?;
    }
    _ => return Err(format!("Unknown model pack: {pack}")),
  }

  // Verify the pack is actually usable afterwards (detector +
  // recognizer present), so a truncated or changed archive is
  // reported instead of silently leaving a broken pack behind.
  let info = crate::face::pack_info(pack);
  let pack_dir: PathBuf = models_root.join(pack);
  for required in [info.det_file, info.rec_file] {
    if !pack_dir.join(required).is_file() {
      return Err(format!(
        "Download finished but {required} is missing from {}",
        pack_dir.display()
      ));
    }
  }
  emit_progress(app, progress_payload(pack, "done", 0, 0, ""));
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::io::Write as _;

  #[test]
  fn only_known_packs_are_downloadable() {
    assert!(is_downloadable_pack("quality"));
    assert!(is_downloadable_pack("light"));
    assert!(is_downloadable_pack("lvface"));
    assert!(!is_downloadable_pack("buffalo_l"));
    assert!(!is_downloadable_pack(""));
  }

  #[test]
  fn zip_extraction_keeps_only_wanted_files() {
    let root = std::env::temp_dir().join(format!("vha-model-dl-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create test dir");
    let zip_path = root.join("pack.zip");
    {
      let file = fs::File::create(&zip_path).expect("create zip");
      let mut writer = zip::ZipWriter::new(file);
      let options = zip::write::SimpleFileOptions::default();
      for (name, bytes) in [
        ("det_500m.onnx", &b"det"[..]),
        ("w600k_mbf.onnx", &b"rec"[..]),
        ("unwanted.onnx", &b"skip"[..]),
      ] {
        writer.start_file(name, options).expect("start entry");
        writer.write_all(bytes).expect("write entry");
      }
      writer.finish().expect("finish zip");
    }
    let dest = root.join("light");
    extract_zip_entries(&zip_path, &dest, &LIGHT_FILES).expect("extract");
    assert!(dest.join("det_500m.onnx").is_file());
    assert!(dest.join("w600k_mbf.onnx").is_file());
    assert!(!dest.join("unwanted.onnx").exists());
    fs::remove_dir_all(root).expect("cleanup");
  }

  /// Real end-to-end download of the smallest pack (15 MB) from the
  /// official release. Ignored by default: needs network access.
  /// Run with: cargo test download_light_pack_end_to_end -- --ignored
  #[test]
  #[ignore]
  fn download_light_pack_end_to_end() {
    let root = std::env::temp_dir().join(format!("vha-model-dl-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create test dir");

    // A minimal AppHandle stand-in is not available in tests; the
    // download path is exercised through the same helpers the IPC
    // handler composes, minus event emission.
    let zip_path = root.join("buffalo_sc.zip");
    let client = reqwest::blocking::Client::builder()
      .user_agent("VideoHubApp/3 (face model download test)")
      .build()
      .expect("client");
    let bytes = client
      .get(BUFFALO_SC_URL)
      .send()
      .expect("download")
      .error_for_status()
      .expect("status")
      .bytes()
      .expect("body");
    assert!(bytes.len() > 10_000_000, "buffalo_sc.zip should be ~15 MB");
    fs::write(&zip_path, &bytes).expect("write zip");
    extract_zip_entries(&zip_path, &root.join("light"), &LIGHT_FILES).expect("extract");
    assert!(root.join("light").join("det_500m.onnx").is_file());
    assert!(root.join("light").join("w600k_mbf.onnx").is_file());
    fs::remove_dir_all(root).expect("cleanup");
  }
}
