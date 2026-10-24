//! Face recognition backend, Phase A (see docs/TAURI_FACE_PORT_DESIGN.md):
//! the read-only parts of the Electron implementation in node/face —
//! model-pack table, model placement status, and the face index payload.
//! On-disk formats (faces/index.json v1, embeddings.bin) are unchanged.

use std::{
  collections::BTreeSet,
  fs,
  path::{Path, PathBuf},
};

use serde_json::{Map, Value, json};

pub struct FacePack {
  pub model_id: &'static str,
  pub det_file: &'static str,
  pub rec_file: &'static str,
}

/// Mirror of FACE_MODEL_PACKS in node/face/face-types.ts.
pub fn pack_info(pack: &str) -> FacePack {
  match pack {
    "light" => FacePack {
      model_id: "buffalo_sc-mbf",
      det_file: "det_500m.onnx",
      rec_file: "w600k_mbf.onnx",
    },
    "lvface" => FacePack {
      model_id: "lvface-t-glint360k",
      det_file: "det_10g.onnx",
      rec_file: "LVFace-T_Glint360K.onnx",
    },
    _ => FacePack {
      model_id: "buffalo_l-r50",
      det_file: "det_10g.onnx",
      rec_file: "w600k_r50.onnx",
    },
  }
}

/// Defaults identical to GLOBALS in node/main-globals.ts.
pub fn default_face_settings() -> Value {
  json!({
    "enabled": false,
    "modelPack": "quality",
    "maxFramesPerVideo": 60,
    "sampleIntervalSec": 30,
    "minFaceSizePx": 40,
    "clusterThreshold": 0.5,
    "autoScanNew": true,
    "attributesEnabled": false,
    "attributeSamplesPerGroup": 5,
    "provider": "auto"
  })
}

/// Fill any missing keys of a saved FaceSettings object with the defaults,
/// the way Electron merges `{ ...GLOBALS.faceSettings, ...saved }`.
pub fn face_settings_with_defaults(saved: Option<&Value>) -> Value {
  let mut settings = default_face_settings();
  if let (Some(target), Some(source)) = (settings.as_object_mut(), saved.and_then(Value::as_object)) {
    for (key, value) in source {
      target.insert(key.clone(), value.clone());
    }
  }
  settings
}

fn pack_name(settings: &Value) -> String {
  settings
    .get("modelPack")
    .and_then(Value::as_str)
    .filter(|pack| ["quality", "light", "lvface"].contains(pack))
    .unwrap_or("quality")
    .to_string()
}

/// Mirror of normalizeProviderSetting in node/face/ort-provider.ts.
pub fn normalize_provider(settings: &Value) -> String {
  match settings.get("provider").and_then(Value::as_str) {
    Some(provider @ ("cpu" | "cuda" | "directml" | "coreml" | "auto")) => provider.to_string(),
    _ => "auto".to_string(),
  }
}

/// The face-models root must resolve to the same folder Electron uses
/// (design decision T3): <userData>/face-models, where userData is the OS
/// config directory joined with the product name "Video Hub App 3".
pub fn face_models_root() -> PathBuf {
  dirs::config_dir()
    .unwrap_or_default()
    .join("Video Hub App 3")
    .join("face-models")
}

fn genderage_path(settings: &Value, models_root: &Path) -> PathBuf {
  let pack = pack_name(settings);
  let own = models_root.join(&pack).join("genderage.onnx");
  if own.exists() {
    return own;
  }
  if pack != "quality" {
    let borrowed = models_root.join("quality").join("genderage.onnx");
    if borrowed.exists() {
      return borrowed;
    }
  }
  PathBuf::new()
}

/// Mirror of getFaceModelStatus in node/face/main-face.ts. Until the Rust
/// inference worker exists (Phase C) the provider block reports "not
/// resolved yet", exactly like Electron before its first worker report.
pub fn model_status(settings: &Value) -> Value {
  model_status_in(settings, &face_models_root())
}

pub fn model_status_in(settings: &Value, models_root: &Path) -> Value {
  let pack = pack_name(settings);
  let info = pack_info(&pack);
  let models_dir = models_root.join(&pack);
  let files: Vec<Value> = [info.det_file, info.rec_file]
    .iter()
    .map(|name| json!({ "name": name, "present": models_dir.join(name).exists() }))
    .collect();
  let ready = files.iter().all(|file| file["present"] == Value::Bool(true));
  let genderage = genderage_path(settings, models_root);
  let genderage_found = !genderage.as_os_str().is_empty();

  json!({
    "pack": pack,
    "modelId": info.model_id,
    "modelsDir": models_dir.to_string_lossy(),
    "files": files,
    "ready": ready,
    "genderagePath": if genderage_found { genderage.to_string_lossy().into_owned() } else { String::new() },
    "genderagePresent": genderage_found,
    "providerInfo": {
      "requested": normalize_provider(settings),
      "effective": Value::Null,
      "attempts": [],
      "cudaLibsPresent": false,
      "ortBinDir": ""
    }
  })
}

fn empty_index(model_id: &str, cluster_threshold: f64) -> Value {
  json!({
    "version": 1,
    "modelId": model_id,
    "clusterThreshold": cluster_threshold,
    "videos": {},
    "persons": [],
    "nextPersonSeq": 1
  })
}

/// Mirror of computeFaceStats in node/face/face-types.ts.
fn compute_stats(index: Option<&Value>) -> Value {
  let mut stats = json!({
    "scannedVideos": 0,
    "videosWithFaces": 0,
    "totalFaces": 0,
    "totalGroups": 0,
    "totalPersons": 0
  });
  let Some(index) = index else {
    return stats;
  };
  stats["totalPersons"] = Value::from(
    index
      .get("persons")
      .and_then(Value::as_array)
      .map(|persons| persons.len())
      .unwrap_or(0),
  );
  let mut scanned = 0u64;
  let mut with_faces = 0u64;
  let mut total_faces = 0u64;
  let mut total_groups = 0u64;
  if let Some(videos) = index.get("videos").and_then(Value::as_object) {
    for entry in videos.values() {
      if entry.get("status").and_then(Value::as_str) == Some("error") {
        continue;
      }
      scanned += 1;
      let faces = entry.get("faces").and_then(Value::as_array);
      let face_count = faces.map(|faces| faces.len()).unwrap_or(0);
      if face_count > 0 {
        with_faces += 1;
        total_faces += face_count as u64;
        let mut groups = BTreeSet::new();
        if let Some(faces) = faces {
          for face in faces {
            if let Some(group_id) = face.get("groupId").and_then(Value::as_i64) {
              if group_id >= 0 {
                groups.insert(group_id);
              }
            }
          }
        }
        total_groups += groups.len() as u64;
      }
    }
  }
  stats["scannedVideos"] = Value::from(scanned);
  stats["videosWithFaces"] = Value::from(with_faces);
  stats["totalFaces"] = Value::from(total_faces);
  stats["totalGroups"] = Value::from(total_groups);
  stats
}

/// Mirror of FaceStore.personsByVideo in node/face/face-store.ts.
fn persons_by_video(index: &Value) -> Value {
  let mut out = Map::new();
  if let Some(videos) = index.get("videos").and_then(Value::as_object) {
    for (hash, entry) in videos {
      let mut ids = BTreeSet::new();
      if let Some(faces) = entry.get("faces").and_then(Value::as_array) {
        for face in faces {
          if let Some(person_id) = face.get("personId").and_then(Value::as_str) {
            ids.insert(person_id.to_string());
          }
        }
      }
      if !ids.is_empty() {
        out.insert(
          hash.clone(),
          Value::Array(ids.into_iter().map(Value::from).collect()),
        );
      }
    }
  }
  Value::Object(out)
}

/// Mirror of FaceStore.load + payload() for read-only use: builds the
/// `face-index-returning` payload for the hub's faces directory.
pub fn index_payload(faces_dir: &Path, settings: &Value) -> Value {
  let pack = pack_name(settings);
  let info = pack_info(&pack);
  let cluster_threshold = settings
    .get("clusterThreshold")
    .and_then(Value::as_f64)
    .unwrap_or(0.5);

  let index_path = faces_dir.join("index.json");
  let has_data = index_path.exists();
  let mut index = empty_index(info.model_id, cluster_threshold);
  let mut mismatch = false;
  if has_data {
    if let Ok(bytes) = fs::read(&index_path) {
      if let Ok(parsed) = serde_json::from_slice::<Value>(&bytes) {
        let valid = parsed.get("version").and_then(Value::as_u64) == Some(1)
          && parsed
            .get("videos")
            .map(Value::is_object)
            .unwrap_or(false);
        if valid {
          mismatch = parsed.get("modelId").and_then(Value::as_str) != Some(info.model_id);
          index = parsed;
        }
      }
    }
  }

  let has_videos = index
    .get("videos")
    .and_then(Value::as_object)
    .map(|videos| !videos.is_empty())
    .unwrap_or(false);
  let payload_index = if has_data || has_videos {
    index.clone()
  } else {
    Value::Null
  };

  json!({
    "index": payload_index,
    "modelMismatch": mismatch,
    "stats": compute_stats(if mismatch { None } else { Some(&index) }),
    "personsByVideo": if mismatch { json!({}) } else { persons_by_video(&index) }
  })
}

/// The payload for a hub: faces live in <asset folder>/faces.
pub fn hub_index_payload(asset_folder: &Path, settings: &Value) -> Value {
  index_payload(&asset_folder.join("faces"), settings)
}

/// The payload when no hub is open: no face data, without touching the disk.
pub fn empty_index_payload() -> Value {
  json!({
    "index": Value::Null,
    "modelMismatch": false,
    "stats": compute_stats(None),
    "personsByVideo": {}
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::atomic::{AtomicU64, Ordering};

  static COUNTER: AtomicU64 = AtomicU64::new(0);

  fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "vha-face-test-{}-{}-{tag}",
      std::process::id(),
      COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
  }

  fn settings(pack: &str) -> Value {
    let mut value = default_face_settings();
    value["modelPack"] = Value::String(pack.to_string());
    value
  }

  #[test]
  fn model_status_reports_missing_models() {
    let root = temp_dir("models-empty");
    let status = model_status_in(&settings("quality"), &root);
    assert_eq!(status["pack"], "quality");
    assert_eq!(status["modelId"], "buffalo_l-r50");
    assert_eq!(status["ready"], Value::Bool(false));
    assert_eq!(status["files"][0]["name"], "det_10g.onnx");
    assert_eq!(status["files"][1]["name"], "w600k_r50.onnx");
    assert_eq!(status["genderagePresent"], Value::Bool(false));
    assert_eq!(status["providerInfo"]["requested"], "auto");
    assert_eq!(status["providerInfo"]["effective"], Value::Null);
    fs::remove_dir_all(root).ok();
  }

  #[test]
  fn model_status_ready_and_genderage_borrowed_from_quality() {
    let root = temp_dir("models-ready");
    let light = root.join("light");
    fs::create_dir_all(&light).expect("mkdir light");
    fs::write(light.join("det_500m.onnx"), b"x").expect("det");
    fs::write(light.join("w600k_mbf.onnx"), b"x").expect("rec");
    let quality = root.join("quality");
    fs::create_dir_all(&quality).expect("mkdir quality");
    fs::write(quality.join("genderage.onnx"), b"x").expect("genderage");

    let status = model_status_in(&settings("light"), &root);
    assert_eq!(status["ready"], Value::Bool(true));
    assert_eq!(status["modelId"], "buffalo_sc-mbf");
    assert_eq!(status["genderagePresent"], Value::Bool(true));
    assert!(status["genderagePath"]
      .as_str()
      .expect("path")
      .ends_with("quality/genderage.onnx"));
    fs::remove_dir_all(root).ok();
  }

  #[test]
  fn payload_for_hub_without_face_data() {
    let dir = temp_dir("faces-empty");
    let payload = index_payload(&dir, &settings("quality"));
    assert_eq!(payload["index"], Value::Null);
    assert_eq!(payload["modelMismatch"], Value::Bool(false));
    assert_eq!(payload["stats"]["totalPersons"], 0);
    assert_eq!(payload["personsByVideo"], json!({}));
    fs::remove_dir_all(dir).ok();
  }

  fn sample_index(model_id: &str) -> Value {
    json!({
      "version": 1,
      "modelId": model_id,
      "clusterThreshold": 0.5,
      "videos": {
        "hashA": {
          "status": "done",
          "scannedAt": 1,
          "frameCount": 10,
          "faces": [
            { "faceId": "hashA-0000", "groupId": 0, "personId": "p000001" },
            { "faceId": "hashA-0001", "groupId": 0, "personId": "p000001" },
            { "faceId": "hashA-0002", "groupId": 1 }
          ]
        },
        "hashB": { "status": "error", "scannedAt": 1, "frameCount": 0, "faces": [], "error": "x" },
        "hashC": { "status": "none", "scannedAt": 1, "frameCount": 5, "faces": [] }
      },
      "persons": [
        { "personId": "p000001", "name": "", "representativeFaceId": "hashA-0000",
          "videoHashes": ["hashA"], "faceCount": 2, "hidden": false }
      ],
      "nextPersonSeq": 2
    })
  }

  #[test]
  fn payload_computes_stats_and_persons_by_video() {
    let dir = temp_dir("faces-sample");
    fs::write(dir.join("index.json"), sample_index("buffalo_l-r50").to_string())
      .expect("write index");
    let payload = index_payload(&dir, &settings("quality"));
    assert_eq!(payload["modelMismatch"], Value::Bool(false));
    assert_eq!(payload["stats"]["scannedVideos"], 2);
    assert_eq!(payload["stats"]["videosWithFaces"], 1);
    assert_eq!(payload["stats"]["totalFaces"], 3);
    assert_eq!(payload["stats"]["totalGroups"], 2);
    assert_eq!(payload["stats"]["totalPersons"], 1);
    assert_eq!(payload["personsByVideo"], json!({ "hashA": ["p000001"] }));
    fs::remove_dir_all(dir).ok();
  }

  #[test]
  fn payload_flags_model_mismatch_and_suppresses_derived_data() {
    let dir = temp_dir("faces-mismatch");
    fs::write(dir.join("index.json"), sample_index("buffalo_sc-mbf").to_string())
      .expect("write index");
    let payload = index_payload(&dir, &settings("quality"));
    assert_eq!(payload["modelMismatch"], Value::Bool(true));
    assert_eq!(payload["stats"]["totalPersons"], 0);
    assert_eq!(payload["personsByVideo"], json!({}));
    // The stored index itself is still reported, like the Electron store.
    assert_eq!(payload["index"]["modelId"], "buffalo_sc-mbf");
    fs::remove_dir_all(dir).ok();
  }

  #[test]
  fn settings_defaults_fill_missing_keys_and_keep_saved_values() {
    let saved = json!({ "enabled": true, "modelPack": "lvface" });
    let merged = face_settings_with_defaults(Some(&saved));
    assert_eq!(merged["enabled"], Value::Bool(true));
    assert_eq!(merged["modelPack"], "lvface");
    assert_eq!(merged["clusterThreshold"], 0.5);
    assert_eq!(merged["provider"], "auto");
  }
}
