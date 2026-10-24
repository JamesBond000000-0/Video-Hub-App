use std::{
  fs,
  io::Write,
  path::{Path, PathBuf},
};

use atomicwrites::{AtomicFile, OverwriteBehavior};
use serde_json::Value;

pub const IPC_MAP_KEY: &str = "__video_hub_app_ipc_map__";

fn backup_path(path: &Path) -> PathBuf {
  let mut backup = path.as_os_str().to_os_string();
  backup.push(".bak");
  PathBuf::from(backup)
}

/// Write bytes in the destination directory and atomically replace the target.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
  let parent = path
    .parent()
    .ok_or_else(|| format!("Cannot determine parent directory for {}", path.display()))?;
  fs::create_dir_all(parent)
    .map_err(|error| format!("Cannot create {}: {error}", parent.display()))?;

  AtomicFile::new(path, OverwriteBehavior::AllowOverwrite)
    .write(|file| file.write_all(bytes))
    .map_err(|error| format!("Cannot write {}: {error}", path.display()))
}

/// Keep the previous version alongside the destination, then replace it atomically.
pub fn write_json_with_backup(path: &Path, value: &Value) -> Result<(), String> {
  let bytes = serde_json::to_vec(value)
    .map_err(|error| format!("Cannot serialize {}: {error}", path.display()))?;

  if path.exists() {
    let previous = fs::read(path)
      .map_err(|error| format!("Cannot back up {}: {error}", path.display()))?;
    write_atomic(&backup_path(path), &previous)?;
  }

  write_atomic(path, &bytes)
}

/// Convert Maps tagged by the TypeScript IPC bridge into JSON objects for storage.
/// Existing VHA files and settings use plain objects when written by Electron.
pub fn tagged_maps_to_objects(value: &mut Value) -> Result<(), String> {
  match value {
    Value::Array(items) => {
      for item in items {
        tagged_maps_to_objects(item)?;
      }
    }
    Value::Object(object) => {
      if object.len() == 1 && object.contains_key(IPC_MAP_KEY) {
        let entries = object
          .remove(IPC_MAP_KEY)
          .and_then(|value| value.as_array().cloned())
          .ok_or_else(|| "Invalid tagged Map in IPC data".to_string())?;

        let mut converted = serde_json::Map::new();
        for entry in entries {
          let pair = entry
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| "Invalid tagged Map entry in IPC data".to_string())?;
          let key = pair[0]
            .as_str()
            .ok_or_else(|| "Only string Map keys can be saved as JSON".to_string())?
            .to_string();
          let mut entry_value = pair[1].clone();
          tagged_maps_to_objects(&mut entry_value)?;
          converted.insert(key, entry_value);
        }
        *value = Value::Object(converted);
        return Ok(());
      }

      for entry_value in object.values_mut() {
        tagged_maps_to_objects(entry_value)?;
      }
    }
    _ => {}
  }

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  #[test]
  fn converts_tagged_shortcut_maps_to_plain_objects() {
    let mut value = json!({
      "shortcuts": {
        "__video_hub_app_ipc_map__": [["q", "quit"], ["d", "darkMode"]]
      }
    });

    tagged_maps_to_objects(&mut value).unwrap();

    assert_eq!(value["shortcuts"]["q"], "quit");
    assert_eq!(value["shortcuts"]["d"], "darkMode");
  }
}
