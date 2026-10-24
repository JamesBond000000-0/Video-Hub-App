use std::{
  collections::HashSet,
  fs,
  path::{Path, PathBuf},
};

use serde_json::{Map, Number, Value, json};

use crate::storage::write_json_with_backup;

const TEMPORARY_IMAGE_FIELDS: [&str; 7] = [
  "durationDisplay",
  "fileSizeDisplay",
  "index",
  "resBucket",
  "resolution",
  "selected",
  "uuid",
];

pub struct LoadedHub {
  pub final_object: Value,
  pub output_folder: PathBuf,
  pub asset_folder: PathBuf,
}

fn finite_number(value: f64) -> Value {
  Number::from_f64(value)
    .map(Value::Number)
    .unwrap_or_else(|| Value::Number(Number::from(0)))
}

fn number_field(image: &Map<String, Value>, key: &str) -> f64 {
  image.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn string_field(image: &Map<String, Value>, key: &str) -> String {
  image
    .get(key)
    .and_then(Value::as_str)
    .unwrap_or_default()
    .to_string()
}

fn input_source_key(image: &Map<String, Value>) -> String {
  match image.get("inputSource") {
    Some(Value::Number(number)) => number.to_string(),
    Some(Value::String(source)) => source.clone(),
    _ => String::new(),
  }
}

fn duration_display(seconds: f64) -> String {
  if seconds == 0.0 || !seconds.is_finite() {
    return String::new();
  }

  let hours = (seconds / 3600.0).floor() as i64;
  let minutes = (seconds / 60.0).floor() as i64 % 60;
  let seconds = seconds.floor() as i64 % 60;

  if hours == 0 {
    format!("{minutes:02}:{seconds:02}")
  } else {
    format!("{hours}:{minutes:02}:{seconds:02}")
  }
}

fn file_size_display(size_in_bytes: f64) -> String {
  if size_in_bytes == 0.0 || !size_in_bytes.is_finite() {
    return String::new();
  }

  let (rounded, gb_divisor) = if cfg!(target_os = "macos") {
    ((size_in_bytes / 1_000_000.0).round(), 1000.0)
  } else {
    ((size_in_bytes / 1_048_576.0).round(), 1024.0)
  };

  if rounded > 999.0 {
    format!("{:.1} GB", rounded / gb_divisor)
  } else {
    format!("{rounded} MB")
  }
}

fn resolution(width: i64, height: i64) -> (&'static str, f64) {
  match (width, height) {
    (3840, 2160) => ("4K", 3.5),
    (1920, 1080) => ("1080", 2.5),
    (1280, 720) => ("720", 1.5),
    _ if width > 3840 => ("4K+", 3.5),
    _ if width > 1920 => ("1080+", 2.5),
    _ if width > 1280 => ("720+", 1.5),
    _ => ("", 0.5),
  }
}

pub fn compute_number_of_screenshots(settings: &Value, duration: f64) -> i64 {
  let fixed = settings
    .get("fixed")
    .and_then(Value::as_bool)
    .unwrap_or(true);
  let mut n = settings.get("n").and_then(Value::as_f64).unwrap_or(10.0);
  if n <= 0.0 || !n.is_finite() {
    n = 10.0;
  }

  let mut total = if fixed {
    n
  } else {
    (duration / 60.0 / n).ceil()
  };
  total = total.max(3.0);

  let height = settings
    .get("height")
    .and_then(Value::as_f64)
    .unwrap_or(288.0);
  let screen_width = height * (16.0 / 9.0);
  if screen_width > 0.0 && total * screen_width > 65535.0 {
    total = (65535.0 / screen_width).floor();
  }
  if duration < total {
    total = duration.floor().max(2.0);
  }

  if total.is_finite() && total >= 0.0 {
    total as i64
  } else {
    0
  }
}

fn upgrade_to_version_3(final_object: &mut Value) -> Result<(), String> {
  if final_object.get("version").and_then(Value::as_i64) != Some(2) {
    return Ok(());
  }

  let input_dir = final_object
    .get("inputDir")
    .cloned()
    .unwrap_or_else(|| Value::String(String::new()));
  let screenshot_settings = final_object
    .get("screenshotSettings")
    .cloned()
    .unwrap_or(Value::Null);

  let object = final_object
    .as_object_mut()
    .ok_or_else(|| "Hub root must be a JSON object".to_string())?;
  object.insert(
    "inputDirs".to_string(),
    json!({ "0": { "path": input_dir, "watch": false } }),
  );
  object.insert("version".to_string(), Value::Number(Number::from(3)));

  let images = object
    .get_mut("images")
    .and_then(Value::as_array_mut)
    .ok_or_else(|| "Hub images must be an array".to_string())?;
  for image in images {
    let image = image
      .as_object_mut()
      .ok_or_else(|| "Every hub image must be an object".to_string())?;
    image.insert("inputSource".to_string(), Value::Number(Number::from(0)));
    let duration = number_field(image, "duration");
    image.insert(
      "screens".to_string(),
      Value::Number(Number::from(compute_number_of_screenshots(
        &screenshot_settings,
        duration,
      ))),
    );
  }

  Ok(())
}

pub fn insert_temporary_fields_single(image: &mut Value, index: usize) -> Result<(), String> {
  let image = image
    .as_object_mut()
    .ok_or_else(|| "Every hub image must be an object".to_string())?;

  if number_field(image, "lastPlayed") == 0.0 {
    image.insert("lastPlayed".to_string(), Value::Number(Number::from(0)));
  }

  let duration = number_field(image, "duration");
  let file_size = number_field(image, "fileSize");
  let width = number_field(image, "width") as i64;
  let height = number_field(image, "height") as i64;
  let (resolution_label, resolution_bucket) = resolution(width, height);
  let bitrate = if duration == 0.0 {
    0.0
  } else {
    let value = ((file_size / 1000.0) / duration) / 1000.0;
    (value * 100.0).round() / 100.0
  };
  let hash = string_field(image, "hash");

  image.insert(
    "durationDisplay".to_string(),
    Value::String(duration_display(duration)),
  );
  image.insert(
    "fileSizeDisplay".to_string(),
    Value::String(file_size_display(file_size)),
  );
  image.insert("bitrate".to_string(), finite_number(bitrate));
  image.insert("resBucket".to_string(), finite_number(resolution_bucket));
  image.insert(
    "resolution".to_string(),
    Value::String(resolution_label.to_string()),
  );
  image.insert(
    "uuid".to_string(),
    Value::String(format!("{hash}-{index}")),
  );
  image.insert("index".to_string(), Value::Number(Number::from(index)));
  Ok(())
}

fn insert_temporary_fields(final_object: &mut Value) -> Result<(), String> {
  let images = final_object
    .get_mut("images")
    .and_then(Value::as_array_mut)
    .ok_or_else(|| "Hub images must be an array".to_string())?;

  for (index, image) in images.iter_mut().enumerate() {
    insert_temporary_fields_single(image, index)?;
  }

  Ok(())
}

pub fn validate_hub_name(hub_name: &str) -> Result<(), String> {
  if hub_name.is_empty()
    || hub_name == "."
    || hub_name == ".."
    || hub_name.contains('/')
    || hub_name.contains('\\')
    || hub_name.contains('\0')
  {
    Err("Hub has an invalid hubName".to_string())
  } else {
    Ok(())
  }
}

pub fn load(path: &Path) -> Result<LoadedHub, String> {
  let bytes = fs::read(path)
    .map_err(|error| format!("Cannot read {}: {error}", path.display()))?;
  let mut final_object: Value = serde_json::from_slice(&bytes)
    .map_err(|error| format!("Cannot parse {}: {error}", path.display()))?;

  if !final_object.is_object() {
    return Err("Hub root must be a JSON object".to_string());
  }
  upgrade_to_version_3(&mut final_object)?;
  insert_temporary_fields(&mut final_object)?;

  let hub_name = final_object
    .get("hubName")
    .and_then(Value::as_str)
    .ok_or_else(|| "Hub is missing its hubName".to_string())?
    .to_string();
  validate_hub_name(&hub_name)?;

  let output_folder = path
    .parent()
    .ok_or_else(|| format!("Cannot determine the hub folder for {}", path.display()))?
    .to_path_buf();
  let asset_folder = output_folder.join(format!("vha-{hub_name}"));

  Ok(LoadedHub {
    final_object,
    output_folder,
    asset_folder,
  })
}

fn prepare_for_save(final_object: &mut Value) -> Result<(), String> {
  let valid_sources: HashSet<String> = final_object
    .get("inputDirs")
    .and_then(Value::as_object)
    .map(|sources| sources.keys().cloned().collect())
    .ok_or_else(|| "Hub inputDirs must be an object".to_string())?;

  let images_value = final_object
    .get_mut("images")
    .ok_or_else(|| "Hub is missing images".to_string())?;
  let Value::Array(raw_images) = std::mem::take(images_value) else {
    return Err("Hub images must be an array".to_string());
  };

  let mut images: Vec<Value> = raw_images
    .into_iter()
    .filter(|image| {
      image
        .get("deleted")
        .and_then(Value::as_bool)
        != Some(true)
    })
    .filter(|image| {
      image
        .as_object()
        .map(input_source_key)
        .is_some_and(|source| valid_sources.contains(&source))
    })
    .collect();

  for image in &mut images {
    let image = image
      .as_object_mut()
      .ok_or_else(|| "Every hub image must be an object".to_string())?;
    for field in TEMPORARY_IMAGE_FIELDS {
      image.remove(field);
    }
  }

  images.sort_by(|left, right| {
    let left = left.as_object().map(|image| {
      (
        string_field(image, "partialPath").to_lowercase(),
        string_field(image, "fileName").to_lowercase(),
      )
    });
    let right = right.as_object().map(|image| {
      (
        string_field(image, "partialPath").to_lowercase(),
        string_field(image, "fileName").to_lowercase(),
      )
    });
    left.cmp(&right)
  });

  let mut previous: Option<(String, String, String)> = None;
  let mut unique_images = Vec::with_capacity(images.len());
  for image in images {
    let key = image.as_object().map(|image| {
      (
        string_field(image, "fileName"),
        string_field(image, "partialPath"),
        input_source_key(image),
      )
    });
    if key.is_some() && key == previous {
      continue;
    }
    previous = key;
    unique_images.push(image);
  }

  let folders: HashSet<String> = unique_images
    .iter()
    .filter_map(Value::as_object)
    .map(|image| string_field(image, "partialPath"))
    .collect();
  let folder_count = folders.len();

  let object = final_object
    .as_object_mut()
    .ok_or_else(|| "Hub root must be a JSON object".to_string())?;
  object.insert("images".to_string(), Value::Array(unique_images));
  object.insert(
    "numOfFolders".to_string(),
    Value::Number(Number::from(folder_count)),
  );

  Ok(())
}

pub fn save(path: &Path, final_object: &Value) -> Result<(), String> {
  if !final_object.is_object() {
    return Err("Hub payload must be a JSON object".to_string());
  }

  let mut final_object = final_object.clone();
  prepare_for_save(&mut final_object)?;
  write_json_with_backup(path, &final_object)
}

pub fn video_path(source_folder: &Path, partial_path: &str, file_name: &str) -> PathBuf {
  let mut path = source_folder.to_path_buf();
  for component in partial_path
    .split(|character| character == '/' || character == '\\')
    .filter(|component| !component.is_empty())
  {
    path.push(component);
  }
  path.push(file_name);
  path
}

pub fn known_paths(final_object: &Value) -> HashSet<PathBuf> {
  let mut result = HashSet::new();
  let Some(sources) = final_object.get("inputDirs").and_then(Value::as_object) else {
    return result;
  };
  let Some(images) = final_object.get("images").and_then(Value::as_array) else {
    return result;
  };

  for image in images.iter().filter_map(Value::as_object) {
    let source = sources
      .get(&input_source_key(image))
      .and_then(|source| source.get("path"))
      .and_then(Value::as_str);
    if let Some(source) = source {
      result.insert(video_path(
        Path::new(source),
        &string_field(image, "partialPath"),
        &string_field(image, "fileName"),
      ));
    }
  }

  result
}

pub fn connected_sources(final_object: &Value) -> Vec<(u64, String)> {
  let Some(sources) = final_object.get("inputDirs").and_then(Value::as_object) else {
    return Vec::new();
  };

  sources
    .iter()
    .filter_map(|(key, source)| {
      let index = key.parse::<u64>().ok()?;
      let path = source.get("path").and_then(Value::as_str)?;
      if Path::new(path).is_dir() {
        Some((index, path.to_string()))
      } else {
        None
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn upgrades_old_hubs_and_adds_display_fields() {
    let mut hub = json!({
      "version": 2,
      "inputDir": "/videos",
      "hubName": "demo",
      "inputDirs": {},
      "images": [{
        "duration": 65.0,
        "fileSize": 10485760.0,
        "width": 1920,
        "height": 1080,
        "hash": "abc",
        "partialPath": "",
        "fileName": "movie.mp4"
      }],
      "screenshotSettings": {
        "fixed": true,
        "height": 288,
        "n": 10
      }
    });

    upgrade_to_version_3(&mut hub).unwrap();
    insert_temporary_fields(&mut hub).unwrap();

    assert_eq!(hub["version"], 3);
    assert_eq!(hub["inputDirs"]["0"]["path"], "/videos");
    assert_eq!(hub["images"][0]["inputSource"], 0);
    assert_eq!(hub["images"][0]["durationDisplay"], "01:05");
    assert_eq!(hub["images"][0]["resolution"], "1080");
    assert_eq!(hub["images"][0]["lastPlayed"], 0);
  }

  #[test]
  fn strips_temporary_fields_and_adjacent_duplicates_when_saving() {
    let mut hub = json!({
      "inputDirs": { "0": { "path": "/videos" } },
      "images": [
        { "fileName": "a.mp4", "partialPath": "", "inputSource": 0, "index": 4 },
        { "fileName": "a.mp4", "partialPath": "", "inputSource": 0, "index": 5 },
        { "fileName": "b.mp4", "partialPath": "folder", "inputSource": 0, "deleted": true }
      ]
    });

    prepare_for_save(&mut hub).unwrap();

    assert_eq!(hub["images"].as_array().unwrap().len(), 1);
    assert!(hub["images"][0].get("index").is_none());
    assert_eq!(hub["numOfFolders"], 1);
  }
}
