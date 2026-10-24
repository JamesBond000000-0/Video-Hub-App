use std::{fs, path::PathBuf};

use serde_json::{Value, json};
use tauri::{Emitter, Manager, State, Window};

use crate::{
  hub, media, settings,
  state::{CurrentHub, VhaState},
};

/// Emit the same argument list shape as Electron's `event.sender.send(...)`.
/// The TypeScript bridge spreads this array when it invokes Angular callbacks.
pub(crate) fn emit_to_angular(
  window: &Window,
  channel: &str,
  args: Vec<Value>,
) -> Result<(), String> {
  window
    .emit(channel, Value::Array(args))
    .map_err(|error| error.to_string())
}

fn required_arg<'a>(args: &'a [Value], index: usize) -> Result<&'a Value, String> {
  args
    .get(index)
    .ok_or_else(|| format!("Missing IPC argument at index {index}"))
}

fn path_arg(args: &[Value], index: usize) -> Result<PathBuf, String> {
  required_arg(args, index)?
    .as_str()
    .filter(|path| !path.is_empty())
    .map(PathBuf::from)
    .ok_or_else(|| format!("IPC argument {index} must be a non-empty path"))
}

fn source_index_arg(args: &[Value], index: usize) -> Result<u64, String> {
  match required_arg(args, index)? {
    Value::Number(number) => number
      .as_u64()
      .ok_or_else(|| "Input source index must be a non-negative integer".to_string()),
    Value::String(number) => number
      .parse()
      .map_err(|_| "Input source index must be a non-negative integer".to_string()),
    _ => Err("Input source index must be a non-negative integer".to_string()),
  }
}

fn default_screenshot_settings() -> Value {
  json!({
    "height": 288,
    "fixed": false,
    "n": 5,
    "clipHeight": 144,
    "clipSnippetLength": 1,
    "clipSnippets": 0
  })
}

fn screenshot_settings_from_wizard(wizard: &Value) -> Value {
  let fixed = wizard
    .get("isFixedNumberOfScreenshots")
    .and_then(Value::as_bool)
    .unwrap_or(false);
  let number_of_screenshots = if fixed { "ssConstant" } else { "ssVariable" };
  let clip_snippets = if wizard
    .get("extractClips")
    .and_then(Value::as_bool)
    .unwrap_or(false)
  {
    wizard.get("clipSnippets").cloned().unwrap_or_else(|| Value::from(0))
  } else {
    Value::from(0)
  };

  json!({
    "clipHeight": wizard.get("clipHeight").cloned().unwrap_or_else(|| Value::from(144)),
    "clipSnippetLength": wizard.get("clipSnippetLength").cloned().unwrap_or_else(|| Value::from(1)),
    "clipSnippets": clip_snippets,
    "fixed": fixed,
    "height": wizard.get("screenshotSizeForImport").cloned().unwrap_or_else(|| Value::from(288)),
    "n": wizard.get(number_of_screenshots).cloned().unwrap_or_else(|| Value::from(5))
  })
}

fn system_locale() -> String {
  sys_locale::get_locale().unwrap_or_else(|| "en".to_string())
}

fn show_error(window: &Window, state: &VhaState, content: &str, details: &str) -> Result<(), String> {
  emit_to_angular(
    window,
    "show-msg-dialog",
    vec![
      Value::String(state.system_message("error", "Error")),
      Value::String(content.to_string()),
      Value::String(details.to_string()),
    ],
  )
}

fn with_error_dialog(
  window: &Window,
  state: &VhaState,
  content: &str,
  result: Result<(), String>,
) -> Result<(), String> {
  match result {
    Ok(()) => Ok(()),
    Err(error) => {
      show_error(window, state, content, &error)?;
      Err(error)
    }
  }
}

fn finish_import(window: &Window) {
  let _ = emit_to_angular(
    window,
    "import-progress-update",
    vec![Value::from(1), Value::from(1), Value::String("done".to_string())],
  );
}

fn open_hub(window: &Window, state: &VhaState, path: PathBuf) -> Result<(), String> {
  let loaded = hub::load(&path)?;

  // Only expose this hub's generated media directory, not the whole filesystem
  // or even unrelated siblings in the output directory.
  window
    .app_handle()
    .asset_protocol_scope()
    .allow_directory(&loaded.asset_folder, true)
    .map_err(|error| format!("Cannot allow hub media access: {error}"))?;

  let generation = state.begin_background_work();
  finish_import(window);

  let path_string = path.to_string_lossy().to_string();
  let output_folder = loaded.output_folder.to_string_lossy().to_string();
  let final_object = loaded.final_object;
  let connected_sources = hub::connected_sources(&final_object);
  let images_empty = final_object
    .get("images")
    .and_then(Value::as_array)
    .map(|images| images.is_empty())
    .unwrap_or(true);
  let screenshot_settings = final_object
    .get("screenshotSettings")
    .cloned()
    .unwrap_or_else(default_screenshot_settings);
  let input_dirs = final_object
    .get("inputDirs")
    .cloned()
    .unwrap_or_else(|| json!({}));
  let known_paths = hub::known_paths(&final_object);

  let scan_sources: Vec<u64> = connected_sources
    .iter()
    .filter(|(source_index, _)| {
      images_empty
        || input_dirs
          .get(source_index.to_string().as_str())
          .and_then(|source| source.get("watch"))
          .and_then(Value::as_bool)
          .unwrap_or(false)
    })
    .map(|(source_index, _)| *source_index)
    .collect();

  state.set_current_hub(CurrentHub {
    path,
    asset_folder: loaded.asset_folder,
    input_dirs,
    screenshot_settings,
    known_paths,
  })?;

  emit_to_angular(
    window,
    "final-object-returning",
    vec![
      final_object,
      Value::String(path_string),
      Value::String(output_folder),
    ],
  )?;

  for (source_index, source_path) in connected_sources {
    emit_to_angular(
      window,
      "directory-now-connected",
      vec![Value::from(source_index), Value::String(source_path)],
    )?;
  }

  // A watch=true source currently receives an initial scan only. Continuous
  // native directory watching remains a separate migration increment.
  if !scan_sources.is_empty() {
    media::spawn_scan(window.app_handle().clone(), generation, Some(scan_sources));
  }

  Ok(())
}

fn open_hub_with_feedback(window: &Window, state: &VhaState, path: PathBuf) -> Result<(), String> {
  match open_hub(window, state, path.clone()) {
    Ok(()) => Ok(()),
    Err(error) => {
      let content = state.system_message("noSuchFileFound", "Could not open the hub file");
      show_error(window, state, &content, &error)?;
      emit_to_angular(window, "please-open-wizard", Vec::new())?;
      Err(error)
    }
  }
}

fn just_started(window: &Window, state: &VhaState) -> Result<(), String> {
  state.mark_frontend_ready();
  let locale = system_locale();

  if let Ok(theme) = window.theme() {
    let mode = if matches!(theme, tauri::Theme::Dark) { "dark" } else { "light" };
    emit_to_angular(
      window,
      "os-dark-mode-change",
      vec![Value::String(mode.to_string())],
    )?;
  }

  match settings::load(&state.settings_file()) {
    Ok(Some(saved_settings)) => {
      if let Some(extensions) = saved_settings
        .pointer("/appState/addtionalExtensions")
        .and_then(Value::as_str)
      {
        state.set_additional_extensions(
          extensions.split(',').map(|item| item.trim().to_string()).collect(),
        )?;
      }

      emit_to_angular(
        window,
        "settings-returning",
        vec![saved_settings, Value::String(locale)],
      )
    }
    Ok(None) => {
      emit_to_angular(
        window,
        "set-language-based-off-system-locale",
        vec![Value::String(locale)],
      )?;
      emit_to_angular(window, "please-open-wizard", vec![Value::Bool(true)])
    }
    Err(error) => {
      eprintln!("{error}");
      show_error(window, state, "Could not restore application settings", &error)?;
      emit_to_angular(window, "please-open-wizard", vec![Value::Bool(false)])
    }
  }
}

fn current_hub_from_settings(settings: &Value) -> Option<PathBuf> {
  settings
    .pointer("/appState/currentVhaFile")
    .and_then(Value::as_str)
    .filter(|path| !path.is_empty())
    .map(PathBuf::from)
}

fn close_with_save(window: &Window, state: &VhaState, args: &[Value]) -> Result<(), String> {
  state.cancel_background_work();
  finish_import(window);
  let saved_settings = required_arg(args, 0)?;
  let final_object = args.get(1);

  let result = (|| {
    settings::save(&state.settings_file(), saved_settings)?;

    if let Some(final_object) = final_object.filter(|value| !value.is_null()) {
      let current_path = state
        .current_vha_file()?
        .or_else(|| current_hub_from_settings(saved_settings))
        .ok_or_else(|| "Cannot determine which hub file to save".to_string())?;
      hub::save(&current_path, final_object)?;
    }

    Ok::<(), String>(())
  })();

  if let Err(error) = result {
    show_error(
      window,
      state,
      "Changes could not be saved. The app will stay open.",
      &error,
    )?;
    return Err(error);
  }

  state.allow_quit();
  window.close().map_err(|error| error.to_string())
}

fn load_this_vha_file(window: &Window, state: &VhaState, args: &[Value]) -> Result<(), String> {
  state.cancel_background_work();
  finish_import(window);
  let path = path_arg(args, 0)?;

  if let Some(final_object) = args.get(1).filter(|value| !value.is_null()) {
    let save_result = (|| {
      let current_path = state
        .current_vha_file()?
        .ok_or_else(|| "Cannot determine which hub to save before opening another".to_string())?;
      hub::save(&current_path, final_object)
    })();

    if let Err(error) = save_result {
      show_error(
        window,
        state,
        "The current hub could not be saved. The other hub was not opened.",
        &error,
      )?;
      return Err(error);
    }
  }

  open_hub_with_feedback(window, state, path)
}

fn start_import(window: &Window, state: &VhaState, args: &[Value]) -> Result<(), String> {
  let wizard = required_arg(args, 0)?;
  let hub_name = wizard
    .get("futureHubName")
    .and_then(Value::as_str)
    .ok_or_else(|| "The import wizard is missing futureHubName".to_string())?;
  hub::validate_hub_name(hub_name)?;
  let output_folder = wizard
    .get("selectedOutputFolder")
    .and_then(Value::as_str)
    .filter(|path| !path.is_empty())
    .map(PathBuf::from)
    .ok_or_else(|| "The import wizard is missing selectedOutputFolder".to_string())?;
  let hub_path = output_folder.join(format!("{hub_name}.vha2"));

  if hub_path.exists() {
    show_error(
      window,
      state,
      &state.system_message("hubAlreadyExists", "A hub with this name already exists"),
      &state.system_message("pleaseChangeName", "Please choose a different hub name"),
    )?;
    emit_to_angular(window, "please-fix-hub-name", Vec::new())?;
    return Ok(());
  }

  let input_dirs = wizard
    .get("selectedSourceFolder")
    .cloned()
    .filter(Value::is_object)
    .ok_or_else(|| "The import wizard is missing selectedSourceFolder".to_string())?;
  let screenshot_settings = screenshot_settings_from_wizard(wizard);
  let final_object = json!({
    "addTags": [],
    "hubName": hub_name,
    "images": [],
    "inputDirs": input_dirs.clone(),
    "numOfFolders": 0,
    "removeTags": [],
    "screenshotSettings": screenshot_settings.clone(),
    "version": 3
  });

  let asset_folder = output_folder.join(format!("vha-{hub_name}"));
  for folder in ["thumbnails", "filmstrips", "clips"] {
    let path = asset_folder.join(folder);
    fs::create_dir_all(&path)
      .map_err(|error| format!("Cannot create {}: {error}", path.display()))?;
  }
  // Electron creates the initial VHA file before starting extraction. Keep the
  // same recoverability guarantee, but use atomic persistence and backups.
  hub::save(&hub_path, &final_object)?;

  window
    .app_handle()
    .asset_protocol_scope()
    .allow_directory(&asset_folder, true)
    .map_err(|error| format!("Cannot allow hub media access: {error}"))?;

  let generation = state.begin_background_work();
  finish_import(window);
  state.set_current_hub(CurrentHub {
    path: hub_path.clone(),
    asset_folder,
    input_dirs,
    screenshot_settings,
    known_paths: Default::default(),
  })?;

  emit_to_angular(
    window,
    "final-object-returning",
    vec![
      final_object,
      Value::String(hub_path.to_string_lossy().to_string()),
      Value::String(output_folder.to_string_lossy().to_string()),
    ],
  )?;
  media::spawn_scan(window.app_handle().clone(), generation, None);
  Ok(())
}

fn start_watching_folder(
  window: &Window,
  state: &VhaState,
  args: &[Value],
) -> Result<(), String> {
  let source_index = source_index_arg(args, 0)?;
  let source_path = path_arg(args, 1)?;
  let watch = args
    .get(2)
    .and_then(Value::as_bool)
    .unwrap_or(false);
  state.update_source_folder(source_index, source_path, watch)?;
  let generation = state.begin_background_work();
  media::spawn_scan(
    window.app_handle().clone(),
    generation,
    Some(vec![source_index]),
  );
  Ok(())
}

fn add_missing_thumbnails(
  window: &Window,
  state: &VhaState,
  args: &[Value],
) -> Result<(), String> {
  let images = required_arg(args, 0)?
    .as_array()
    .cloned()
    .ok_or_else(|| "Missing-thumbnail input must be an array".to_string())?;
  let generation = state.begin_background_work();
  media::spawn_missing_thumbnails(window.app_handle().clone(), generation, images);
  Ok(())
}

fn unsupported_channel(window: &Window, state: &VhaState, channel: &str) -> Result<(), String> {
  let error = format!("IPC channel '{channel}' has not been ported to the Tauri backend yet");
  show_error(
    window,
    state,
    "This feature has not been ported to the Tauri backend yet.",
    channel,
  )?;
  Err(error)
}

/// Compatibility endpoint used by `window.myElectron.sendToMain` in Angular.
/// Unknown channels fail explicitly instead of silently pretending to succeed.
#[tauri::command]
pub fn ipc_send(
  window: Window,
  state: State<'_, VhaState>,
  channel: String,
  args: Vec<Value>,
) -> Result<(), String> {
  match channel.as_str() {
    "minimize-window" => window.minimize().map_err(|error| error.to_string()),
    "maximize-window" => window.maximize().map_err(|error| error.to_string()),
    "un-maximize-window" => window.unmaximize().map_err(|error| error.to_string()),
    "close-window" => close_with_save(&window, &state, &args),
    "just-started" => just_started(&window, &state),
    "load-this-vha-file" => load_this_vha_file(&window, &state, &args),
    "open-file" => open_hub_with_feedback(&window, &state, path_arg(&args, 0)?),
    "start-the-import" => with_error_dialog(
      &window,
      &state,
      "Could not create and import the new hub",
      start_import(&window, &state, &args),
    ),
    "start-watching-folder" => with_error_dialog(
      &window,
      &state,
      "Could not scan the selected source folder",
      start_watching_folder(&window, &state, &args),
    ),
    "stop-watching-folder" => {
      state.set_source_watch(source_index_arg(&args, 0)?, false)
    }
    "update-source-folder" => state.update_source_folder(
      source_index_arg(&args, 0)?,
      path_arg(&args, 1)?,
      args.get(2).and_then(Value::as_bool).unwrap_or(false),
    ),
    "add-missing-thumbnails" => with_error_dialog(
      &window,
      &state,
      "Could not generate the missing thumbnails",
      add_missing_thumbnails(&window, &state, &args),
    ),
    "cancel-current-import" => {
      state.cancel_background_work();
      finish_import(&window);
      Ok(())
    }
    "system-messages-updated" => {
      state.set_system_messages(required_arg(&args, 0)?.clone())
    }
    "update-additional-extensions" => {
      let extensions = required_arg(&args, 0)?
        .as_str()
        .ok_or_else(|| "Additional extensions must be a string".to_string())?
        .split(',')
        .map(|item| item.trim().to_string())
        .collect();
      state.set_additional_extensions(extensions)
    }
    // Touch Bar integration is macOS/Electron-specific. Settings restoration
    // sends these messages even on platforms where there is nothing to update.
    "app-to-touchBar" => Ok(()),
    // No Tauri remote server exists yet, so stopping it is already a no-op.
    // Angular emits this every time a hub is opened.
    "stop-server" => Ok(()),
    _ => unsupported_channel(&window, &state, &channel),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn wizard_settings_match_the_existing_final_object_contract() {
    let fixed = screenshot_settings_from_wizard(&json!({
      "clipHeight": 216,
      "clipSnippetLength": 2,
      "clipSnippets": 8,
      "extractClips": false,
      "isFixedNumberOfScreenshots": true,
      "screenshotSizeForImport": 288,
      "ssConstant": 12,
      "ssVariable": 5
    }));
    assert_eq!(fixed["fixed"], Value::Bool(true));
    assert_eq!(fixed["n"], Value::from(12));
    assert_eq!(fixed["clipSnippets"], Value::from(0));
    assert_eq!(fixed["clipHeight"], Value::from(216));

    let variable = screenshot_settings_from_wizard(&json!({
      "clipSnippets": 8,
      "extractClips": true,
      "isFixedNumberOfScreenshots": false,
      "ssConstant": 12,
      "ssVariable": 5
    }));
    assert_eq!(variable["fixed"], Value::Bool(false));
    assert_eq!(variable["n"], Value::from(5));
    assert_eq!(variable["clipSnippets"], Value::from(8));
  }
}
