use std::{
  collections::{HashMap, HashSet},
  path::PathBuf,
  process::Child,
  sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
  },
};

use serde_json::{Value, json};

#[derive(Clone)]
pub struct CurrentHub {
  pub path: PathBuf,
  pub asset_folder: PathBuf,
  pub input_dirs: Value,
  pub screenshot_settings: Value,
  pub known_paths: HashSet<PathBuf>,
}

pub struct VhaState {
  settings_file: PathBuf,
  face_settings: Mutex<Value>,
  face_provider_info: Mutex<Option<Value>>,
  current_hub: Mutex<Option<CurrentHub>>,
  system_messages: Mutex<Value>,
  additional_extensions: Mutex<Vec<String>>,
  active_children: Mutex<HashMap<u64, Arc<Mutex<Child>>>>,
  next_child_id: AtomicU64,
  background_generation: AtomicU64,
  frontend_ready: AtomicBool,
  ready_to_quit: AtomicBool,
}

impl VhaState {
  pub fn new() -> Result<Self, String> {
    let settings_directory = if let Some(portable) = std::env::var_os("PORTABLE_EXECUTABLE_DIR") {
      PathBuf::from(portable)
    } else {
      dirs::config_dir()
        .ok_or_else(|| "Cannot determine the operating system configuration directory".to_string())?
        .join("video-hub-app-2")
    };

    let english: Value = serde_json::from_str(include_str!("../../i18n/en.json"))
      .map_err(|error| format!("Cannot load English system messages: {error}"))?;
    let system_messages = english
      .get("SYSTEM")
      .cloned()
      .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

    Ok(Self {
      settings_file: settings_directory.join("settings.json"),
      face_settings: Mutex::new(crate::face::default_face_settings()),
      face_provider_info: Mutex::new(None),
      current_hub: Mutex::new(None),
      system_messages: Mutex::new(system_messages),
      additional_extensions: Mutex::new(Vec::new()),
      active_children: Mutex::new(HashMap::new()),
      next_child_id: AtomicU64::new(0),
      background_generation: AtomicU64::new(0),
      frontend_ready: AtomicBool::new(false),
      ready_to_quit: AtomicBool::new(false),
    })
  }

  pub fn settings_file(&self) -> PathBuf {
    self.settings_file.clone()
  }

  /// The face settings of the open hub (Electron's GLOBALS.faceSettings):
  /// read from the hub's FinalObject on open, partially merged on
  /// `face-settings-updated`.
  pub fn face_settings(&self) -> Result<Value, String> {
    Ok(
      self
        .face_settings
        .lock()
        .map_err(|_| "Face settings state is unavailable".to_string())?
        .clone(),
    )
  }

  pub fn set_face_settings_from_final_object(&self, final_object: &Value) -> Result<(), String> {
    *self
      .face_settings
      .lock()
      .map_err(|_| "Face settings state is unavailable".to_string())? =
      crate::face::face_settings_with_defaults(final_object.get("faceSettings"));
    Ok(())
  }

  /// The provider resolution reported by the most recent face engine
  /// run in this process (Electron reports the worker's resolution in
  /// the model status the same way).
  pub fn face_provider_info(&self) -> Result<Option<Value>, String> {
    Ok(
      self
        .face_provider_info
        .lock()
        .map_err(|_| "Face provider state is unavailable".to_string())?
        .clone(),
    )
  }

  pub fn set_face_provider_info(&self, info: Value) -> Result<(), String> {
    *self
      .face_provider_info
      .lock()
      .map_err(|_| "Face provider state is unavailable".to_string())? = Some(info);
    Ok(())
  }

  pub fn update_face_settings(&self, patch: &Value) -> Result<Value, String> {
    let mut current = self
      .face_settings
      .lock()
      .map_err(|_| "Face settings state is unavailable".to_string())?;
    if let (Some(target), Some(source)) = (current.as_object_mut(), patch.as_object()) {
      for (key, value) in source {
        target.insert(key.clone(), value.clone());
      }
    }
    Ok(current.clone())
  }

  pub fn set_current_hub(&self, hub: CurrentHub) -> Result<(), String> {
    *self
      .current_hub
      .lock()
      .map_err(|_| "Current hub state is unavailable".to_string())? = Some(hub);
    Ok(())
  }

  pub fn current_hub(&self) -> Result<Option<CurrentHub>, String> {
    Ok(
      self
        .current_hub
        .lock()
        .map_err(|_| "Current hub state is unavailable".to_string())?
        .clone(),
    )
  }

  pub fn current_vha_file(&self) -> Result<Option<PathBuf>, String> {
    Ok(self.current_hub()?.map(|hub| hub.path))
  }

  pub fn update_source_folder(
    &self,
    source_index: u64,
    path: PathBuf,
    watch: bool,
  ) -> Result<(), String> {
    let mut current = self
      .current_hub
      .lock()
      .map_err(|_| "Current hub state is unavailable".to_string())?;
    let hub = current
      .as_mut()
      .ok_or_else(|| "No hub is currently open".to_string())?;
    let sources = hub
      .input_dirs
      .as_object_mut()
      .ok_or_else(|| "Current hub inputDirs must be an object".to_string())?;
    sources.insert(
      source_index.to_string(),
      json!({ "path": path.to_string_lossy(), "watch": watch }),
    );
    Ok(())
  }

  pub fn set_source_watch(&self, source_index: u64, watch: bool) -> Result<(), String> {
    let mut current = self
      .current_hub
      .lock()
      .map_err(|_| "Current hub state is unavailable".to_string())?;
    let hub = current
      .as_mut()
      .ok_or_else(|| "No hub is currently open".to_string())?;
    let source = hub
      .input_dirs
      .as_object_mut()
      .and_then(|sources| sources.get_mut(&source_index.to_string()))
      .and_then(Value::as_object_mut)
      .ok_or_else(|| format!("Unknown input source {source_index}"))?;
    source.insert("watch".to_string(), Value::Bool(watch));
    Ok(())
  }

  pub fn add_known_path(&self, path: PathBuf) -> Result<(), String> {
    let mut current = self
      .current_hub
      .lock()
      .map_err(|_| "Current hub state is unavailable".to_string())?;
    if let Some(hub) = current.as_mut() {
      hub.known_paths.insert(path);
    }
    Ok(())
  }

  pub fn set_system_messages(&self, messages: Value) -> Result<(), String> {
    *self
      .system_messages
      .lock()
      .map_err(|_| "System message state is unavailable".to_string())? = messages;
    Ok(())
  }

  pub fn system_message(&self, key: &str, fallback: &str) -> String {
    self
      .system_messages
      .lock()
      .ok()
      .and_then(|messages| messages.get(key).and_then(Value::as_str).map(str::to_string))
      .unwrap_or_else(|| fallback.to_string())
  }

  pub fn set_additional_extensions(&self, extensions: Vec<String>) -> Result<(), String> {
    *self
      .additional_extensions
      .lock()
      .map_err(|_| "Additional extension state is unavailable".to_string())? = extensions;
    Ok(())
  }

  pub fn additional_extensions(&self) -> Result<Vec<String>, String> {
    Ok(
      self
        .additional_extensions
        .lock()
        .map_err(|_| "Additional extension state is unavailable".to_string())?
        .clone(),
    )
  }

  pub fn register_child(&self, child: Arc<Mutex<Child>>) -> Result<u64, String> {
    let child_id = self.next_child_id.fetch_add(1, Ordering::AcqRel) + 1;
    self
      .active_children
      .lock()
      .map_err(|_| "Active process state is unavailable".to_string())?
      .insert(child_id, child);
    Ok(child_id)
  }

  pub fn unregister_child(&self, child_id: u64) {
    if let Ok(mut children) = self.active_children.lock() {
      children.remove(&child_id);
    }
  }

  fn kill_active_children(&self) {
    let children: Vec<Arc<Mutex<Child>>> = match self.active_children.lock() {
      Ok(children) => children.values().cloned().collect(),
      Err(_) => return,
    };
    for child in children {
      if let Ok(mut child) = child.lock() {
        let _ = child.kill();
      }
    }
  }

  /// Invalidate all previous scan/import jobs and return the new generation.
  pub fn begin_background_work(&self) -> u64 {
    let generation = self.background_generation.fetch_add(1, Ordering::AcqRel) + 1;
    self.kill_active_children();
    generation
  }

  pub fn cancel_background_work(&self) {
    self.background_generation.fetch_add(1, Ordering::AcqRel);
    self.kill_active_children();
  }

  pub fn background_work_is_current(&self, generation: u64) -> bool {
    self.background_generation.load(Ordering::Acquire) == generation
  }

  pub fn mark_frontend_ready(&self) {
    self.frontend_ready.store(true, Ordering::Release);
  }

  pub fn frontend_ready(&self) -> bool {
    self.frontend_ready.load(Ordering::Acquire)
  }

  pub fn allow_quit(&self) {
    self.cancel_background_work();
    self.ready_to_quit.store(true, Ordering::Release);
  }

  pub fn ready_to_quit(&self) -> bool {
    self.ready_to_quit.load(Ordering::Acquire)
  }
}
