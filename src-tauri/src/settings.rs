use std::{fs, io::ErrorKind, path::Path};

use serde_json::Value;

use crate::storage::{tagged_maps_to_objects, write_json_with_backup};

pub fn load(path: &Path) -> Result<Option<Value>, String> {
  let bytes = match fs::read(path) {
    Ok(bytes) => bytes,
    Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
    Err(error) => return Err(format!("Cannot read {}: {error}", path.display())),
  };

  let settings = serde_json::from_slice(&bytes)
    .map_err(|error| format!("Cannot parse {}: {error}", path.display()))?;
  Ok(Some(settings))
}

pub fn save(path: &Path, settings: &Value) -> Result<(), String> {
  if !settings.is_object() {
    return Err("Settings payload must be a JSON object".to_string());
  }

  let mut settings = settings.clone();
  tagged_maps_to_objects(&mut settings)?;
  write_json_with_backup(path, &settings)
}
