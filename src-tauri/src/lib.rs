mod face;
mod face_cluster;
mod face_infer;
mod hub;
mod ipc;
mod media;
mod settings;
mod state;
mod storage;

use serde_json::Value;
use state::VhaState;
use tauri::{Manager, WindowEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_opener::init())
    .invoke_handler(tauri::generate_handler![ipc::ipc_send])
    .setup(|app| {
      let state = VhaState::new()
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
      app.manage(state);

      // The bundled ONNX Runtime library lives in the resource dir.
      if let Ok(resource_dir) = app.path().resource_dir() {
        face_infer::set_resource_dir_hint(resource_dir);
      }

      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .on_window_event(|window, event| match event {
      WindowEvent::CloseRequested { api, .. } => {
        let needs_angular_save = window
          .try_state::<VhaState>()
          .map(|state| state.frontend_ready() && !state.ready_to_quit())
          .unwrap_or(false);

        if needs_angular_save {
          api.prevent_close();
          if let Err(error) = ipc::emit_to_angular(window, "please-shut-down-ASAP", Vec::new()) {
            eprintln!("Could not request application shutdown: {error}");
          }
        }
      }
      WindowEvent::ThemeChanged(theme) => {
        let mode = if matches!(theme, tauri::Theme::Dark) {
          "dark"
        } else {
          "light"
        };
        if let Err(error) = ipc::emit_to_angular(
          window,
          "os-dark-mode-change",
          vec![Value::String(mode.to_string())],
        ) {
          eprintln!("Could not notify theme change: {error}");
        }
      }
      _ => {}
    })
    .run(tauri::generate_context!())
    .expect("error while building tauri application");
}
