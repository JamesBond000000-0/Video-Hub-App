use std::{
  collections::{BTreeMap, HashSet},
  ffi::OsString,
  fs::{self, File},
  io::{Read, Seek, SeekFrom},
  path::{Path, PathBuf},
  process::{Child, Command, ExitStatus, Stdio},
  sync::{Arc, Mutex},
  thread,
  time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};
use tauri::{AppHandle, Emitter, Manager};
use wait_timeout::ChildWaitTimeoutExt;
use walkdir::WalkDir;

use crate::{hub, state::VhaState, storage::IPC_MAP_KEY};

const ACCEPTABLE_EXTENSIONS: [&str; 32] = [
  "264", "265", "3g2", "3gp", "asf", "avi", "divx", "flv", "h264", "h265", "hevc",
  "m2ts", "m4a", "m4v", "mkv", "mov", "mp2", "mp4", "mpe", "mpeg", "mpg", "mts", "mxf",
  "ogg", "ogm", "ogv", "rm", "rmvb", "ts", "vob", "webm", "wmv",
];

#[derive(Clone)]
struct SourceFolder {
  path: PathBuf,
  #[allow(dead_code)]
  watch: bool,
}

#[derive(Clone)]
struct VideoFile {
  full_path: PathBuf,
  input_source: u64,
  file_name: String,
  partial_path: String,
}

struct ExtractionItem {
  element: Value,
  full_path: PathBuf,
}

#[derive(Clone)]
struct MediaTools {
  ffmpeg: PathBuf,
  ffprobe: PathBuf,
}

impl MediaTools {
  fn from_environment() -> Self {
    Self {
      ffmpeg: std::env::var_os("VHA_FFMPEG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg")),
      ffprobe: std::env::var_os("VHA_FFPROBE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffprobe")),
    }
  }

  fn verify(&self, app: &AppHandle, generation: u64) -> Result<(), String> {
    for (description, executable) in [
      ("FFprobe", &self.ffprobe),
      ("FFmpeg", &self.ffmpeg),
    ] {
      let result = run_process(
        executable,
        vec![OsString::from("-version")],
        Duration::from_secs(10),
        false,
        app,
        generation,
      )?;
      if !result.status.success() {
        return Err(format!("{description} -version exited with {}", result.status));
      }
    }
    Ok(())
  }
}

struct ProcessResult {
  status: ExitStatus,
  stdout: Vec<u8>,
}

fn is_current(app: &AppHandle, generation: u64) -> bool {
  app.state::<VhaState>().background_work_is_current(generation)
}

fn emit(app: &AppHandle, channel: &str, args: Vec<Value>) {
  if let Err(error) = app.emit(channel, Value::Array(args)) {
    eprintln!("Could not emit {channel}: {error}");
  }
}

fn emit_job(app: &AppHandle, generation: u64, channel: &str, args: Vec<Value>) -> bool {
  if !is_current(app, generation) {
    return false;
  }
  emit(app, channel, args);
  true
}

fn report_job_error(app: &AppHandle, generation: u64, content: &str, details: &str) {
  let title = app
    .state::<VhaState>()
    .system_message("error", "Error");
  emit_job(
    app,
    generation,
    "show-msg-dialog",
    vec![
      Value::String(title),
      Value::String(content.to_string()),
      Value::String(details.to_string()),
    ],
  );
}

fn finish_job(app: &AppHandle, generation: u64) {
  emit_job(
    app,
    generation,
    "import-progress-update",
    vec![Value::from(1), Value::from(1), Value::String("done".to_string())],
  );
}

fn kill_and_wait(child: &Arc<Mutex<Child>>) {
  if let Ok(mut child) = child.lock() {
    let _ = child.kill();
    let _ = child.wait();
  }
}

/// Run a child process without allowing corrupt media or cancellation to block a job forever.
fn run_process(
  executable: &Path,
  args: Vec<OsString>,
  timeout: Duration,
  capture_stdout: bool,
  app: &AppHandle,
  generation: u64,
) -> Result<ProcessResult, String> {
  if !is_current(app, generation) {
    return Err("Media job was cancelled".to_string());
  }

  let mut command = Command::new(executable);
  command
    .args(args)
    .stdin(Stdio::null())
    .stderr(Stdio::null());
  if capture_stdout {
    command.stdout(Stdio::piped());
  } else {
    command.stdout(Stdio::null());
  }

  let mut child = command.spawn().map_err(|error| {
    format!(
      "Could not start {}: {error}. Install it on PATH or set VHA_FFMPEG_PATH / VHA_FFPROBE_PATH.",
      executable.display()
    )
  })?;

  let stdout_reader = if capture_stdout {
    child.stdout.take().map(|mut stdout| {
      thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
      })
    })
  } else {
    None
  };

  let child = Arc::new(Mutex::new(child));
  if !is_current(app, generation) {
    kill_and_wait(&child);
    return Err("Media job was cancelled".to_string());
  }
  let child_id = match app.state::<VhaState>().register_child(child.clone()) {
    Ok(child_id) => child_id,
    Err(error) => {
      kill_and_wait(&child);
      return Err(error);
    }
  };

  let outcome = (|| -> Result<ProcessResult, String> {
    let started = Instant::now();
    let result = loop {
      if !is_current(app, generation) {
        kill_and_wait(&child);
        break Err("Media job was cancelled".to_string());
      }

      let wait_result = child
        .lock()
        .map_err(|_| "Child process state is unavailable".to_string())
        .and_then(|mut child| {
          child
            .wait_timeout(Duration::from_millis(100))
            .map_err(|error| error.to_string())
        });
      match wait_result {
        Ok(Some(status)) => break Ok(status),
        Ok(None) => {}
        Err(error) => {
          kill_and_wait(&child);
          break Err(format!("Could not wait for {}: {error}", executable.display()));
        }
      }

      if started.elapsed() >= timeout {
        kill_and_wait(&child);
        break Err(format!("{} timed out after {timeout:?}", executable.display()));
      }
    };

    let stdout = match stdout_reader {
      Some(reader) => reader
        .join()
        .map_err(|_| "Could not join the process output reader".to_string())?
        .map_err(|error| format!("Could not read process output: {error}"))?,
      None => Vec::new(),
    };

    Ok(ProcessResult {
      status: result?,
      stdout,
    })
  })();

  app.state::<VhaState>().unregister_child(child_id);
  outcome
}

fn timestamp_ms(time: std::io::Result<SystemTime>) -> u64 {
  time
    .ok()
    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
    .map(|duration| duration.as_millis() as u64)
    .unwrap_or(0)
}

/// Match VHA's existing hash so hubs can reuse thumbnails generated by Electron.
fn hash_file(path: &Path, file_size: u64) -> Result<String, String> {
  const SAMPLE_SIZE: usize = 16 * 1024;
  const SAMPLE_THRESHOLD: u64 = 128 * 1024;

  let mut file = File::open(path)
    .map_err(|error| format!("Cannot open {} for hashing: {error}", path.display()))?;
  let mut data = if file_size < SAMPLE_THRESHOLD {
    let mut bytes = Vec::new();
    file
      .read_to_end(&mut bytes)
      .map_err(|error| format!("Cannot hash {}: {error}", path.display()))?;
    bytes
  } else {
    let mut bytes = vec![0; SAMPLE_SIZE * 3];
    file
      .read_exact(&mut bytes[..SAMPLE_SIZE])
      .map_err(|error| format!("Cannot hash {}: {error}", path.display()))?;
    file
      .seek(SeekFrom::Start(file_size / 2))
      .and_then(|_| file.read_exact(&mut bytes[SAMPLE_SIZE..SAMPLE_SIZE * 2]))
      .map_err(|error| format!("Cannot hash {}: {error}", path.display()))?;
    file
      .seek(SeekFrom::Start(file_size - SAMPLE_SIZE as u64))
      .and_then(|_| file.read_exact(&mut bytes[SAMPLE_SIZE * 2..]))
      .map_err(|error| format!("Cannot hash {}: {error}", path.display()))?;
    bytes
  };
  data.extend_from_slice(file_size.to_string().as_bytes());

  // Electron hashes the hexadecimal representation, not the raw bytes.
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut hexadecimal = String::with_capacity(data.len() * 2);
  for byte in data {
    hexadecimal.push(HEX[(byte >> 4) as usize] as char);
    hexadecimal.push(HEX[(byte & 0x0f) as usize] as char);
  }
  Ok(format!("{:x}", md5::compute(hexadecimal.as_bytes())))
}

fn clean_file_name(file_name: &str) -> String {
  let without_extension = file_name
    .rsplit_once('.')
    .map(|(stem, _)| stem)
    .unwrap_or("");
  without_extension
    .replace('_', " ")
    .replace('.', " ")
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
}

fn number_from_probe(value: Option<&Value>) -> f64 {
  value
    .and_then(|value| match value {
      Value::String(number) => number.parse::<f64>().ok(),
      Value::Number(number) => number.as_f64(),
      _ => None,
    })
    .filter(|value| value.is_finite())
    .unwrap_or(0.0)
}

fn extract_metadata(
  video: &VideoFile,
  screenshot_settings: &Value,
  tools: &MediaTools,
  app: &AppHandle,
  generation: u64,
) -> Result<Value, String> {
  let probe = run_process(
    &tools.ffprobe,
    vec![
      OsString::from("-v"),
      OsString::from("error"),
      OsString::from("-of"),
      OsString::from("json"),
      OsString::from("-show_streams"),
      OsString::from("-show_format"),
      OsString::from("-select_streams"),
      OsString::from("V"),
      video.full_path.clone().into_os_string(),
    ],
    Duration::from_secs(30),
    true,
    app,
    generation,
  )?;
  if !probe.status.success() {
    return Err(format!("FFprobe exited with {}", probe.status));
  }

  let metadata: Value = serde_json::from_slice(&probe.stdout)
    .map_err(|error| format!("Could not parse FFprobe output: {error}"))?;
  let streams = metadata
    .get("streams")
    .and_then(Value::as_array)
    .ok_or_else(|| "FFprobe returned no video streams".to_string())?;
  let best_stream = streams
    .iter()
    .max_by(|left, right| {
      number_from_probe(left.get("width"))
        .partial_cmp(&number_from_probe(right.get("width")))
        .unwrap_or(std::cmp::Ordering::Equal)
    })
    .ok_or_else(|| "FFprobe returned no video streams".to_string())?;
  let first_stream = &streams[0];

  let stream_duration = first_stream.get("duration");
  let duration = number_from_probe(stream_duration.or_else(|| {
    metadata.get("format").and_then(|format| format.get("duration"))
  }))
  .trunc();
  let fps = first_stream
    .get("r_frame_rate")
    .and_then(Value::as_str)
    .and_then(|rate| {
      let (numerator, denominator) = rate.split_once('/')?;
      let numerator = numerator.parse::<f64>().ok()?;
      let denominator = denominator.parse::<f64>().ok()?;
      let fps = (numerator / denominator).round();
      if denominator == 0.0 || !fps.is_finite() {
        None
      } else {
        Some(fps)
      }
    })
    .unwrap_or(0.0);

  let file_metadata = fs::metadata(&video.full_path)
    .map_err(|error| format!("Cannot stat {}: {error}", video.full_path.display()))?;
  let file_size = file_metadata.len();
  let hash = hash_file(&video.full_path, file_size)?;
  let screens = hub::compute_number_of_screenshots(screenshot_settings, duration);

  Ok(json!({
    "birthtime": timestamp_ms(file_metadata.created()),
    "bitrate": 0,
    "cleanName": clean_file_name(&video.file_name),
    "duration": duration,
    "durationDisplay": "",
    "fileName": video.file_name.clone(),
    "fileSize": file_size,
    "fileSizeDisplay": "",
    "fps": fps,
    "hash": hash,
    "height": number_from_probe(best_stream.get("height")),
    "index": 0,
    "inputSource": video.input_source,
    "lastPlayed": 0,
    "mtime": timestamp_ms(file_metadata.modified()),
    "partialPath": video.partial_path.clone(),
    "resBucket": 0,
    "resolution": "",
    "screens": screens,
    "stars": 0.5,
    "timesPlayed": 0,
    "uuid": "",
    "width": number_from_probe(best_stream.get("width")),
  }))
}

fn source_map(hub: &crate::state::CurrentHub) -> BTreeMap<u64, SourceFolder> {
  let Some(sources) = hub.input_dirs.as_object() else {
    return BTreeMap::new();
  };
  sources
    .iter()
    .filter_map(|(key, source)| {
      Some((
        key.parse().ok()?,
        SourceFolder {
          path: PathBuf::from(source.get("path")?.as_str()?),
          watch: source.get("watch").and_then(Value::as_bool).unwrap_or(false),
        },
      ))
    })
    .collect()
}

fn acceptable_extensions(app: &AppHandle) -> HashSet<String> {
  let mut extensions: HashSet<String> = ACCEPTABLE_EXTENSIONS
    .iter()
    .map(|extension| extension.to_string())
    .collect();
  if let Ok(additional) = app.state::<VhaState>().additional_extensions() {
    extensions.extend(additional);
  }
  extensions
}

fn tagged_path_map(paths: &HashSet<PathBuf>) -> Value {
  let mut entries: Vec<Value> = paths
    .iter()
    .map(|path| json!([path.to_string_lossy(), 1]))
    .collect();
  entries.sort_by_key(|entry| entry.to_string());
  let mut map = serde_json::Map::new();
  map.insert(IPC_MAP_KEY.to_string(), Value::Array(entries));
  Value::Object(map)
}

fn scan_folder(
  source_index: u64,
  source: &SourceFolder,
  extensions: &HashSet<String>,
) -> Result<(HashSet<PathBuf>, Vec<VideoFile>), String> {
  if !source.path.is_dir() {
    return Err(format!("{} is not a readable directory", source.path.display()));
  }

  let mut all_paths = HashSet::new();
  let mut videos = Vec::new();
  let mut scan_failed = false;
  let walker = WalkDir::new(&source.path)
    .follow_links(false)
    .into_iter()
    .filter_entry(|entry| {
      entry.depth() == 0
        || !(entry.file_type().is_dir()
          && entry.file_name().to_string_lossy().starts_with("vha-"))
    });

  for entry in walker {
    let entry = match entry {
      Ok(entry) => entry,
      Err(error) => {
        scan_failed = true;
        eprintln!("Skipping folder scan error: {error}");
        continue;
      }
    };
    if !entry.file_type().is_file() {
      continue;
    }

    let extension = entry
      .path()
      .extension()
      .map(|extension| extension.to_string_lossy().to_lowercase())
      .unwrap_or_default();
    if !extensions.contains(&extension) {
      continue;
    }

    let full_path = entry.path().to_path_buf();
    let relative = full_path
      .strip_prefix(&source.path)
      .map_err(|error| format!("Cannot make a relative video path: {error}"))?;
    let relative_parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let partial_path = if relative_parent.as_os_str().is_empty() {
      "/".to_string()
    } else {
      let folders: Vec<String> = relative_parent
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect();
      format!("/{}", folders.join("/"))
    };
    let file_name = entry.file_name().to_string_lossy().to_string();

    all_paths.insert(full_path.clone());
    videos.push(VideoFile {
      full_path,
      input_source: source_index,
      file_name,
      partial_path,
    });
  }

  if scan_failed {
    return Err(format!(
      "Some directories under {} could not be read; the incomplete scan was discarded",
      source.path.display()
    ));
  }

  videos.sort_by(|left, right| left.full_path.cmp(&right.full_path));
  Ok((all_paths, videos))
}

fn scale_and_pad(width: f64, height: f64) -> String {
  format!(
    "scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
  )
}

fn ffmpeg_args(args: Vec<OsString>) -> Vec<OsString> {
  let mut result = vec![
    OsString::from("-nostdin"),
    OsString::from("-y"),
    OsString::from("-loglevel"),
    OsString::from("error"),
  ];
  result.extend(args);
  result
}

fn run_ffmpeg(
  tools: &MediaTools,
  args: Vec<OsString>,
  timeout: Duration,
  app: &AppHandle,
  generation: u64,
) -> Result<(), String> {
  let result = run_process(
    &tools.ffmpeg,
    ffmpeg_args(args),
    timeout,
    false,
    app,
    generation,
  )?;
  if result.status.success() {
    Ok(())
  } else {
    Err(format!("FFmpeg exited with {}", result.status))
  }
}

fn extraction_timeouts(element: &Value, settings: &Value) -> (Duration, Duration, Duration, Duration) {
  let number = |key: &str| element.get(key).and_then(Value::as_f64).unwrap_or(0.0);
  let setting = |key: &str| settings.get(key).and_then(Value::as_f64).unwrap_or(0.0);
  let source_height = number("height");
  let screens = number("screens");
  let screenshot_height = setting("height");
  let clip_snippets = setting("clipSnippets");
  let snippet_length = setting("clipSnippetLength");
  let clip_height = setting("clipHeight");

  let thumb_ratio = screenshot_height / 144.0;
  let thumb_factor = 1.0 + thumb_ratio * thumb_ratio / 4.0;
  let clip_ratio = clip_height / 144.0;
  let clip_factor = 1.0 + clip_ratio * clip_ratio / 4.0;
  let source_ratio = if source_height == 0.0 { 1.0 } else { source_height / 720.0 };
  let source_factor = 1.0 + source_ratio * source_ratio / 3.0;
  let duration = |milliseconds: f64| Duration::from_millis(milliseconds.max(1.0) as u64);

  (
    duration(500.0 * source_factor * thumb_factor),
    duration(350.0 * source_factor * thumb_factor * screens),
    duration(350.0 * source_factor * clip_factor * clip_snippets * snippet_length),
    duration(400.0 * clip_ratio),
  )
}

fn has_all_thumbnails(element: &Value, asset_folder: &Path, settings: &Value) -> bool {
  let Some(hash) = element.get("hash").and_then(Value::as_str) else {
    return false;
  };
  if hash.is_empty() || !hash.chars().all(|character| character.is_ascii_alphanumeric()) {
    return false;
  }
  let clips = settings
    .get("clipSnippets")
    .and_then(Value::as_f64)
    .unwrap_or(0.0)
    > 0.0;
  asset_folder.join("thumbnails").join(format!("{hash}.jpg")).is_file()
    && asset_folder.join("filmstrips").join(format!("{hash}.jpg")).is_file()
    && (!clips
      || (asset_folder.join("clips").join(format!("{hash}.mp4")).is_file()
        && asset_folder.join("clips").join(format!("{hash}.jpg")).is_file()))
}

fn extract_all(
  item: &ExtractionItem,
  asset_folder: &Path,
  settings: &Value,
  tools: &MediaTools,
  app: &AppHandle,
  generation: u64,
) -> Result<(), String> {
  let element = &item.element;
  let hash = element
    .get("hash")
    .and_then(Value::as_str)
    .filter(|hash| {
      !hash.is_empty() && hash.chars().all(|character| character.is_ascii_alphanumeric())
    })
    .ok_or_else(|| "Cannot extract media with an invalid file hash".to_string())?;
  if !item.full_path.is_file() {
    return Err(format!("{} is no longer available", item.full_path.display()));
  }

  let duration = element.get("duration").and_then(Value::as_f64).unwrap_or(0.0);
  let screens = element.get("screens").and_then(Value::as_i64).unwrap_or(0).max(1);
  let height = settings.get("height").and_then(Value::as_f64).unwrap_or(288.0);
  let clip_height = settings.get("clipHeight").and_then(Value::as_f64).unwrap_or(144.0);
  let clip_snippets = settings.get("clipSnippets").and_then(Value::as_i64).unwrap_or(0);
  let snippet_length = settings
    .get("clipSnippetLength")
    .and_then(Value::as_f64)
    .unwrap_or(1.0);
  let width = height * (16.0 / 9.0);
  let (thumb_timeout, filmstrip_timeout, clip_timeout, clip_thumb_timeout) =
    extraction_timeouts(element, settings);

  let thumbnail = asset_folder.join("thumbnails").join(format!("{hash}.jpg"));
  let filmstrip = asset_folder.join("filmstrips").join(format!("{hash}.jpg"));
  let clip = asset_folder.join("clips").join(format!("{hash}.mp4"));
  let clip_thumbnail = asset_folder.join("clips").join(format!("{hash}.jpg"));
  for folder in [
    asset_folder.join("thumbnails"),
    asset_folder.join("filmstrips"),
    asset_folder.join("clips"),
  ] {
    fs::create_dir_all(&folder)
      .map_err(|error| format!("Cannot create {}: {error}", folder.display()))?;
  }

  if !thumbnail.is_file() {
    run_ffmpeg(
      tools,
      vec![
        OsString::from("-ss"),
        OsString::from((duration / 10.0).to_string()),
        OsString::from("-i"),
        item.full_path.clone().into_os_string(),
        OsString::from("-frames"),
        OsString::from("1"),
        OsString::from("-q:v"),
        OsString::from("2"),
        OsString::from("-vf"),
        OsString::from(scale_and_pad(width, height)),
        thumbnail.clone().into_os_string(),
      ],
      thumb_timeout,
      app,
      generation,
    )?;
    if !thumbnail.is_file() {
      return Err("FFmpeg did not create a thumbnail".to_string());
    }
  }

  if !filmstrip.is_file() {
    let step = duration / (screens + 1) as f64;
    let mut args = Vec::new();
    let mut filters = String::new();
    let mut outputs = String::new();
    for current in 0..screens {
      args.extend([
        OsString::from("-ss"),
        OsString::from(((current + 1) as f64 * step).to_string()),
        OsString::from("-i"),
        item.full_path.clone().into_os_string(),
      ]);
      filters.push_str(&format!(
        "[{current}:V]{}[{current}];",
        scale_and_pad(width, height)
      ));
      outputs.push_str(&format!("[{current}]"));
    }
    args.extend([
      OsString::from("-frames"),
      OsString::from("1"),
      OsString::from("-filter_complex"),
      OsString::from(format!("{filters}{outputs}hstack=inputs={screens}")),
      filmstrip.clone().into_os_string(),
    ]);
    run_ffmpeg(tools, args, filmstrip_timeout, app, generation)?;
    if !filmstrip.is_file() {
      return Err("FFmpeg did not create a filmstrip".to_string());
    }
  }

  if clip_snippets > 0 {
    if !clip.is_file() {
      let step = duration / (clip_snippets + 1) as f64;
      let mut args = Vec::new();
      let mut concat = String::new();
      for current in 1..=clip_snippets {
        args.extend([
          OsString::from("-ss"),
          OsString::from((current as f64 * step).to_string()),
          OsString::from("-t"),
          OsString::from(snippet_length.to_string()),
          OsString::from("-i"),
          item.full_path.clone().into_os_string(),
        ]);
        let index = current - 1;
        concat.push_str(&format!("[{index}:V][{index}:a]"));
      }
      concat.push_str(&format!(
        "concat=n={clip_snippets}:v=1:a=1[v][a];[v]scale=-2:{clip_height}[v2]"
      ));
      args.extend([
        OsString::from("-filter_complex"),
        OsString::from(concat),
        OsString::from("-map"),
        OsString::from("[v2]"),
        OsString::from("-map"),
        OsString::from("[a]"),
        clip.clone().into_os_string(),
      ]);
      run_ffmpeg(tools, args, clip_timeout, app, generation)?;
      if !clip.is_file() {
        return Err("FFmpeg did not create a preview clip".to_string());
      }
    }

    if !clip_thumbnail.is_file() {
      run_ffmpeg(
        tools,
        vec![
          OsString::from("-ss"),
          OsString::from("0"),
          OsString::from("-i"),
          clip.clone().into_os_string(),
          OsString::from("-frames"),
          OsString::from("1"),
          OsString::from("-f"),
          OsString::from("image2"),
          clip_thumbnail.clone().into_os_string(),
        ],
        clip_thumb_timeout,
        app,
        generation,
      )?;
      if !clip_thumbnail.is_file() {
        return Err("FFmpeg did not create a clip thumbnail".to_string());
      }
    }
  }

  Ok(())
}

fn extract_thumbnails(
  app: &AppHandle,
  generation: u64,
  items: Vec<ExtractionItem>,
  tools: &MediaTools,
  asset_folder: &Path,
  screenshot_settings: &Value,
) -> Result<(), String> {
  let total = items.len();
  for (index, item) in items.into_iter().enumerate() {
    if !is_current(app, generation) {
      return Ok(());
    }
    if has_all_thumbnails(&item.element, asset_folder, screenshot_settings) {
      continue;
    }
    emit_job(
      app,
      generation,
      "import-progress-update",
      vec![
        Value::from(index),
        Value::from(total),
        Value::String("importingScreenshots".to_string()),
      ],
    );
    if let Err(error) = extract_all(
      &item,
      asset_folder,
      screenshot_settings,
      tools,
      app,
      generation,
    ) {
      if is_current(app, generation) {
        eprintln!("Could not extract {}: {error}", item.full_path.display());
      } else {
        return Ok(());
      }
    }
  }
  Ok(())
}

fn run_scan_pipeline(
  app: AppHandle,
  generation: u64,
  source_filter: Option<HashSet<u64>>,
) -> Result<(), String> {
  let current = app
    .state::<VhaState>()
    .current_hub()?
    .ok_or_else(|| "No hub is currently open".to_string())?;
  let sources = source_map(&current);
  let extensions = acceptable_extensions(&app);
  let mut pending = Vec::new();

  for (source_index, source) in sources {
    if source_filter
      .as_ref()
      .is_some_and(|filter| !filter.contains(&source_index))
    {
      continue;
    }
    if !is_current(&app, generation) {
      return Ok(());
    }
    emit_job(
      &app,
      generation,
      "started-watching-this-dir",
      vec![Value::from(source_index)],
    );
    let (all_paths, videos) = match scan_folder(source_index, &source, &extensions) {
      Ok(result) => result,
      Err(error) => {
        emit_job(
          &app,
          generation,
          "all-files-found-in-dir",
          vec![
            Value::from(source_index),
            tagged_path_map(&current.known_paths),
          ],
        );
        report_job_error(
          &app,
          generation,
          "A source folder could not be scanned; other sources will still be processed",
          &error,
        );
        continue;
      }
    };
    emit_job(
      &app,
      generation,
      "all-files-found-in-dir",
      vec![Value::from(source_index), tagged_path_map(&all_paths)],
    );
    pending.extend(
      videos
        .into_iter()
        .filter(|video| !current.known_paths.contains(&video.full_path)),
    );
  }

  if pending.is_empty() {
    return Ok(());
  }

  let tools = MediaTools::from_environment();
  tools.verify(&app, generation).map_err(|error| {
    format!("FFmpeg and FFprobe are required for media import. {error}")
  })?;

  let total = pending.len();
  let mut thumbnails = Vec::new();
  for (index, video) in pending.into_iter().enumerate() {
    if !is_current(&app, generation) {
      return Ok(());
    }
    emit_job(
      &app,
      generation,
      "import-progress-update",
      vec![
        Value::from(index),
        Value::from(total),
        Value::String("importingMeta".to_string()),
      ],
    );

    match extract_metadata(&video, &current.screenshot_settings, &tools, &app, generation) {
      Ok(mut element) => {
        hub::insert_temporary_fields_single(&mut element, index)?;
        if let Some(object) = element.as_object_mut() {
          object.insert(
            "uuid".to_string(),
            Value::String(format!("{:x}", md5::compute(video.full_path.to_string_lossy().as_bytes()))),
          );
        }
        if !emit_job(&app, generation, "new-video-meta", vec![element.clone()]) {
          return Ok(());
        }
        app.state::<VhaState>().add_known_path(video.full_path.clone())?;
        thumbnails.push(ExtractionItem {
          element,
          full_path: video.full_path,
        });
      }
      Err(error) => {
        if is_current(&app, generation) {
          eprintln!("Could not read metadata for {}: {error}", video.full_path.display());
        } else {
          return Ok(());
        }
      }
    }
  }

  extract_thumbnails(
    &app,
    generation,
    thumbnails,
    &tools,
    &current.asset_folder,
    &current.screenshot_settings,
  )
}

fn spawn_job<F>(app: AppHandle, generation: u64, job: F)
where
  F: FnOnce(AppHandle, u64) -> Result<(), String> + Send + 'static,
{
  thread::spawn(move || {
    let result = job(app.clone(), generation);
    if let Err(error) = result {
      if is_current(&app, generation) {
        report_job_error(&app, generation, "Media processing could not be completed", &error);
      }
    }
    finish_job(&app, generation);
  });
}

pub fn spawn_scan(app: AppHandle, generation: u64, sources: Option<Vec<u64>>) {
  let source_filter = sources.map(|sources| sources.into_iter().collect());
  spawn_job(app, generation, move |app, generation| {
    run_scan_pipeline(app, generation, source_filter)
  });
}

pub fn spawn_missing_thumbnails(app: AppHandle, generation: u64, images: Vec<Value>) {
  spawn_job(app, generation, move |app, generation| {
    let current = app
      .state::<VhaState>()
      .current_hub()?
      .ok_or_else(|| "No hub is currently open".to_string())?;
    let sources = source_map(&current);
    let mut items = Vec::new();

    for element in images {
      let input_source = element
        .get("inputSource")
        .and_then(|source| match source {
          Value::Number(number) => number.as_u64(),
          Value::String(source) => source.parse::<u64>().ok(),
          _ => None,
        })
        .ok_or_else(|| "A thumbnail item has no valid inputSource".to_string())?;
      let source = sources
        .get(&input_source)
        .ok_or_else(|| format!("Unknown input source {input_source}"))?;
      let full_path = hub::video_path(
        &source.path,
        element.get("partialPath").and_then(Value::as_str).unwrap_or(""),
        element.get("fileName").and_then(Value::as_str).unwrap_or(""),
      );
      items.push(ExtractionItem { element, full_path });
    }

    items.retain(|item| {
      !has_all_thumbnails(
        &item.element,
        &current.asset_folder,
        &current.screenshot_settings,
      )
    });
    if items.is_empty() {
      return Ok(());
    }
    let tools = MediaTools::from_environment();
    tools.verify(&app, generation)?;
    extract_thumbnails(
      &app,
      generation,
      items,
      &tools,
      &current.asset_folder,
      &current.screenshot_settings,
    )
  });
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn file_names_and_scaling_match_existing_import_behavior() {
    assert_eq!(clean_file_name("my_favorite.movie.mp4"), "my favorite movie");
    assert_eq!(scale_and_pad(512.0, 288.0),
      "scale=w=512:h=288:force_original_aspect_ratio=decrease,pad=512:288:(ow-iw)/2:(oh-ih)/2");
  }

  fn hash_fixture(label: &str, bytes: &[u8]) -> String {
    let path = std::env::temp_dir().join(format!(
      "vha-media-hash-{}-{label}",
      std::process::id()
    ));
    fs::write(&path, bytes).expect("write hash fixture");
    let result = hash_file(&path, bytes.len() as u64).expect("hash fixture");
    fs::remove_file(path).expect("remove hash fixture");
    result
  }

  #[test]
  fn hashes_match_the_electron_implementation() {
    // Expected values were generated with Node's crypto.createHash('md5')
    // after Buffer.concat([...samples, Buffer.from(fileSize.toString())])
    // and buf.toString('hex'), exactly as node/main-support.ts does.
    assert_eq!(
      hash_fixture("small", &[0, 1, 2, 250, 255]),
      "f8615f76a8eb4bbe85d13e42e8ee885f"
    );

    let bytes: Vec<u8> = (0..128 * 1024)
      .map(|index| ((index * 31 + 7) & 255) as u8)
      .collect();
    assert_eq!(
      hash_fixture("large", &bytes),
      "b6511447374a2435e4a99fed74fbc42c"
    );
  }

  #[test]
  fn recursive_scans_skip_generated_folders_and_keep_relative_paths() {
    let root = std::env::temp_dir().join(format!(
      "vha-media-scan-{}",
      std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("sub")).expect("create scan subdirectory");
    fs::create_dir_all(root.join("vha-generated")).expect("create generated directory");
    fs::write(root.join("root.mp4"), b"video").expect("write root video");
    fs::write(root.join("ignore.txt"), b"text").expect("write non-video");
    fs::write(root.join("sub").join("nested.MKV"), b"video").expect("write nested video");
    fs::write(root.join("vha-generated").join("skip.mp4"), b"video")
      .expect("write generated video");

    let extensions: HashSet<String> = ACCEPTABLE_EXTENSIONS
      .iter()
      .map(|extension| extension.to_string())
      .collect();
    let source = SourceFolder {
      path: root.clone(),
      watch: false,
    };
    let (all_paths, videos) = scan_folder(0, &source, &extensions).expect("scan fixture");
    assert_eq!(all_paths.len(), 2);
    assert_eq!(videos.len(), 2);
    assert!(videos.iter().any(|video| {
      video.file_name == "root.mp4" && video.partial_path == "/"
    }));
    assert!(videos.iter().any(|video| {
      video.file_name == "nested.MKV" && video.partial_path == "/sub"
    }));

    fs::remove_dir_all(root).expect("remove scan fixture");
  }

  #[test]
  fn acceptable_extensions_cover_existing_formats() {
    assert!(ACCEPTABLE_EXTENSIONS.contains(&"mp4"));
    assert!(ACCEPTABLE_EXTENSIONS.contains(&"mkv"));
    assert!(ACCEPTABLE_EXTENSIONS.contains(&"webm"));
  }
}
