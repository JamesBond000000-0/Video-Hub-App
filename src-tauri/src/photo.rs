//! Photo support (docs/PHOTO_SUPPORT_DESIGN.md): hub-level settings and
//! the candidate extension set. Photos live in the hub's `images` array
//! with `kind: "photo"` (decision P6 — no separate array).

use serde_json::{Value, json};

/// Extensions treated as photo candidates during a folder walk. The
/// list is deliberately wide: whether a file is actually importable is
/// decided by ffprobe returning real dimensions (decision P3), so a
/// format the local ffmpeg build cannot decode is skipped at probe
/// time instead of failing the scan.
pub const PHOTO_EXTENSIONS: [&str; 13] = [
  "jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff", "avif", "heic", "heif", "jxl", "svg",
];

/// Long edge (px) of the preview copy generated for the built-in photo
/// viewer (Phase 3 of the design). The original file stays in the
/// source folder; the preview lives in the hub's `previews/` folder.
pub const PHOTO_PREVIEW_LONG_EDGE: u32 = 1920;

pub fn is_photo_extension(extension: &str) -> bool {
  PHOTO_EXTENSIONS.contains(&extension.to_lowercase().as_str())
}

pub fn is_photo_element(element: &Value) -> bool {
  element.get("kind").and_then(Value::as_str) == Some("photo")
}

/// Defaults for the hub's PhotoSettings (design decision P2: opt-in,
/// default off; the toggle only gates importing new photos).
pub fn default_photo_settings() -> Value {
  json!({ "enabled": false })
}

/// Fill any missing keys of a saved PhotoSettings object with the
/// defaults, the way face settings are merged on open.
pub fn photo_settings_with_defaults(saved: Option<&Value>) -> Value {
  let mut settings = default_photo_settings();
  if let (Some(target), Some(source)) = (settings.as_object_mut(), saved.and_then(Value::as_object)) {
    for (key, value) in source {
      target.insert(key.clone(), value.clone());
    }
  }
  settings
}

pub fn photo_import_enabled(settings: &Value) -> bool {
  settings
    .get("enabled")
    .and_then(Value::as_bool)
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn photo_extensions_are_case_insensitive_and_exclude_videos() {
    assert!(is_photo_extension("JPG"));
    assert!(is_photo_extension("heic"));
    assert!(!is_photo_extension("mp4"));
    assert!(!is_photo_extension(""));
  }

  #[test]
  fn photo_settings_default_to_disabled() {
    assert!(!photo_import_enabled(&default_photo_settings()));
    let merged = photo_settings_with_defaults(Some(&json!({ "enabled": true })));
    assert!(photo_import_enabled(&merged));
    let absent = photo_settings_with_defaults(None);
    assert!(!photo_import_enabled(&absent));
  }
}
