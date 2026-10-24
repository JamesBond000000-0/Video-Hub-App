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

/// The resolved genderage.onnx path for the current pack (its own copy
/// first, else the quality pack's), when one is placed.
pub fn genderage_model_path(settings: &Value) -> Option<PathBuf> {
  let path = genderage_path(settings, &face_models_root());
  if path.as_os_str().is_empty() {
    None
  } else {
    Some(path)
  }
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

/// Mirror of getFaceModelStatus in node/face/main-face.ts. The provider
/// block reports the last engine resolution when a scan or attribute
/// job has run in this process, and "not resolved yet" before that,
/// exactly like Electron before its first worker report.
pub fn model_status(settings: &Value) -> Value {
  model_status_with(settings, &face_models_root(), None)
}

pub fn model_status_with(
  settings: &Value,
  models_root: &Path,
  provider_info: Option<&Value>,
) -> Value {
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
    "providerInfo": provider_info.cloned().unwrap_or_else(|| json!({
      "requested": normalize_provider(settings),
      "effective": Value::Null,
      "attempts": [],
      "cudaLibsPresent": false,
      "ortBinDir": ""
    }))
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
  load_store(faces_dir, settings).payload()
}

fn load_store(faces_dir: &Path, settings: &Value) -> FaceStore {
  let pack = pack_name(settings);
  let info = pack_info(&pack);
  let cluster_threshold = settings
    .get("clusterThreshold")
    .and_then(Value::as_f64)
    .unwrap_or(0.5);
  FaceStore::load(faces_dir, info.model_id, cluster_threshold)
}

/// The store for the currently open hub (faces in <asset folder>/faces).
pub fn hub_store(asset_folder: &Path, settings: &Value) -> FaceStore {
  load_store(&asset_folder.join("faces"), settings)
}

pub const EMBEDDING_DIM: usize = 512;
pub const EMBEDDING_BYTES: usize = EMBEDDING_DIM * 4;

/// Port of FaceStore (node/face/face-store.ts): persistence and edits for
/// one hub's face index. Scan-time writes (setVideo) arrive with Phase C.
pub struct FaceStore {
  faces_dir: PathBuf,
  index: Value,
  /// Records in embeddings.bin; consumed by the Phase C scan writes.
  #[allow(dead_code)]
  embeddings_stored: u64,
  pub model_mismatch: bool,
}

impl FaceStore {
  pub fn load(faces_dir: &Path, model_id: &str, cluster_threshold: f64) -> FaceStore {
    let mut index = empty_index(model_id, cluster_threshold);
    let mut mismatch = false;
    let index_path = faces_dir.join("index.json");
    if index_path.exists() {
      if let Ok(bytes) = fs::read(&index_path) {
        if let Ok(parsed) = serde_json::from_slice::<Value>(&bytes) {
          let valid = parsed.get("version").and_then(Value::as_u64) == Some(1)
            && parsed.get("videos").map(Value::is_object).unwrap_or(false);
          if valid {
            mismatch = parsed.get("modelId").and_then(Value::as_str) != Some(model_id);
            index = parsed;
          }
        }
      }
    }
    let embeddings_stored = fs::metadata(faces_dir.join("embeddings.bin"))
      .map(|meta| meta.len() / EMBEDDING_BYTES as u64)
      .unwrap_or(0);
    FaceStore {
      faces_dir: faces_dir.to_path_buf(),
      index,
      embeddings_stored,
      model_mismatch: mismatch,
    }
  }

  pub fn has_data(faces_dir: &Path) -> bool {
    faces_dir.join("index.json").exists()
  }

  pub fn store_has_data(&self) -> bool {
    Self::has_data(&self.faces_dir)
  }

  /// The loaded index (getIndex in TS); the Phase C scan flow reads it.
  #[allow(dead_code)]
  pub fn index(&self) -> &Value {
    &self.index
  }

  /// The hub's faces directory this store persists to.
  pub fn faces_dir_path(&self) -> &Path {
    &self.faces_dir
  }

  /// One stored embedding (512 float32 at embeddingIdx * 2048). The TS
  /// implementation throws on a missing/truncated file; callers there
  /// treat that as a failed (re)clustering. Here a zero vector is
  /// returned instead, which simply never matches any cluster.
  pub fn read_embedding(&self, embedding_idx: i64) -> Vec<f32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut out = vec![0.0f32; EMBEDDING_DIM];
    if embedding_idx < 0 {
      return out;
    }
    let Ok(mut file) = fs::File::open(self.faces_dir.join("embeddings.bin")) else {
      return out;
    };
    if file
      .seek(SeekFrom::Start(embedding_idx as u64 * EMBEDDING_BYTES as u64))
      .is_err()
    {
      return out;
    }
    let mut buffer = vec![0u8; EMBEDDING_BYTES];
    if file.read_exact(&mut buffer).is_err() {
      return out;
    }
    for i in 0..EMBEDDING_DIM {
      out[i] = f32::from_le_bytes(buffer[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
    }
    out
  }

  pub fn payload(&self) -> Value {
    let has_videos = self
      .index
      .get("videos")
      .and_then(Value::as_object)
      .map(|videos| !videos.is_empty())
      .unwrap_or(false);
    json!({
      "index": if Self::has_data(&self.faces_dir) || has_videos {
        self.index.clone()
      } else {
        Value::Null
      },
      "modelMismatch": self.model_mismatch,
      "stats": compute_stats(if self.model_mismatch { None } else { Some(&self.index) }),
      "personsByVideo": if self.model_mismatch {
        json!({})
      } else {
        persons_by_video(&self.index)
      }
    })
  }

  fn save_index(&self) -> Result<(), String> {
    fs::create_dir_all(&self.faces_dir)
      .map_err(|error| format!("Cannot create {}: {error}", self.faces_dir.display()))?;
    let index_path = self.faces_dir.join("index.json");
    let tmp_path = self.faces_dir.join("index.json.tmp");
    let text = serde_json::to_string(&self.index)
      .map_err(|error| format!("Cannot serialize face index: {error}"))?;
    fs::write(&tmp_path, text)
      .map_err(|error| format!("Cannot write {}: {error}", tmp_path.display()))?;
    fs::rename(&tmp_path, &index_path)
      .map_err(|error| format!("Cannot replace {}: {error}", index_path.display()))?;
    Ok(())
  }

  fn assert_writable(&self) -> Result<(), String> {
    if self.model_mismatch {
      return Err(
        "face index was built with a different model pack; reset required before writing"
          .to_string(),
      );
    }
    Ok(())
  }

  /// Apply a full re-clustering (applyClustering): rewrite personIds,
  /// persons, threshold, sequence, and the pruned manual-edit pins.
  pub fn apply_clustering(
    &mut self,
    result: &crate::face_cluster::ClusterizeResult,
    threshold: f64,
  ) -> Result<(), String> {
    self.assert_writable()?;
    if let Some(videos) = self.index.get_mut("videos").and_then(Value::as_object_mut) {
      for entry in videos.values_mut() {
        if let Some(faces) = entry.get_mut("faces").and_then(Value::as_array_mut) {
          for face in faces.iter_mut() {
            let face_id = face
              .get("faceId")
              .and_then(Value::as_str)
              .unwrap_or_default()
              .to_string();
            match result.face_person.get(&face_id) {
              Some(person_id) => {
                face["personId"] = Value::String(person_id.clone());
              }
              None => {
                if let Some(obj) = face.as_object_mut() {
                  obj.remove("personId");
                }
              }
            }
          }
        }
      }
    }
    self.index["persons"] = Value::Array(result.persons.clone());
    self.index["clusterThreshold"] = json!(threshold);
    self.index["nextPersonSeq"] = json!(result.next_person_seq);
    fn write_back(
      index: &mut Value,
      key: &str,
      value: Value,
      is_empty: bool,
    ) {
      if is_empty {
        if let Some(obj) = index.as_object_mut() {
          obj.remove(key);
        }
      } else {
        index[key] = value;
      }
    }
    write_back(
      &mut self.index,
      "manualGroups",
      Value::Object(result.manual_groups.clone()),
      result.manual_groups.is_empty(),
    );
    write_back(
      &mut self.index,
      "manualUnassigned",
      Value::Array(
        result
          .manual_unassigned
          .iter()
          .map(|key| Value::String(key.clone()))
          .collect(),
      ),
      result.manual_unassigned.is_empty(),
    );
    write_back(
      &mut self.index,
      "manualRepFaces",
      Value::Object(result.manual_rep_faces.clone()),
      result.manual_rep_faces.is_empty(),
    );
    self.save_index()
  }

  /// Re-run clustering over the current index (reclusterNow in the store
  /// uses the stored threshold; the IPC handler passes the settings one,
  /// mirroring reclusterFaces in main-face.ts).
  pub fn recluster(&mut self, threshold: f64) -> Result<(), String> {
    let result = crate::face_cluster::clusterize_index(
      &self.index,
      &|idx| self.read_embedding(idx),
      threshold,
    );
    self.apply_clustering(&result, threshold)
  }

  pub fn stored_threshold(&self) -> f64 {
    self
      .index
      .get("clusterThreshold")
      .and_then(Value::as_f64)
      .unwrap_or(0.5)
  }

  /// Remove all attribute aggregates (clearPersonsAttributes).
  pub fn clear_persons_attributes(&mut self) -> Result<(), String> {
    let mut changed = false;
    if let Some(persons) = self.index.get_mut("persons").and_then(Value::as_array_mut) {
      for person in persons.iter_mut() {
        if let Some(obj) = person.as_object_mut() {
          if obj.remove("attributes").is_some() {
            changed = true;
          }
        }
      }
    }
    if changed {
      self.save_index()?;
    }
    Ok(())
  }

  pub fn rename_person(&mut self, person_id: &str, name: &str) -> bool {
    let mut found = false;
    if let Some(persons) = self.index.get_mut("persons").and_then(Value::as_array_mut) {
      for person in persons.iter_mut() {
        if person.get("personId").and_then(Value::as_str) == Some(person_id) {
          person["name"] = Value::String(name.to_string());
          found = true;
          break;
        }
      }
    }
    if found {
      let _ = self.save_index();
    }
    found
  }

  pub fn set_person_hidden(&mut self, person_id: &str, hidden: bool) -> bool {
    let mut found = false;
    if let Some(persons) = self.index.get_mut("persons").and_then(Value::as_array_mut) {
      for person in persons.iter_mut() {
        if person.get("personId").and_then(Value::as_str) == Some(person_id) {
          person["hidden"] = Value::Bool(hidden);
          found = true;
          break;
        }
      }
    }
    if found {
      let _ = self.save_index();
    }
    found
  }

  pub fn merge_persons(&mut self, person_ids: Vec<String>) -> Result<Option<String>, String> {
    self.assert_writable()?;
    let survivor = crate::face_cluster::merge_persons_in_index(&mut self.index, &person_ids);
    match survivor {
      None => Ok(None),
      Some(survivor) => {
        let threshold = self.stored_threshold();
        self.recluster(threshold)?;
        Ok(Some(survivor))
      }
    }
  }

  pub fn split_person(
    &mut self,
    person_id: &str,
    group_keys: Vec<String>,
  ) -> Result<Option<String>, String> {
    self.assert_writable()?;
    let mut seq = self.index.get("nextPersonSeq").and_then(Value::as_i64).unwrap_or(0);
    if seq == 0 {
      seq = 1;
    }
    if let Some(persons) = self.index.get("persons").and_then(Value::as_array) {
      for person in persons {
        let id = person
          .get("personId")
          .and_then(Value::as_str)
          .unwrap_or_default();
        let n = crate::face_cluster::parse_int_prefix(&id.chars().skip(1).collect::<String>());
        if let Some(n) = n {
          seq = seq.max(n + 1);
        }
      }
    }
    let new_person_id = format!("p{seq:06}");
    if !crate::face_cluster::split_person_in_index(
      &mut self.index,
      person_id,
      &group_keys,
      &new_person_id,
    ) {
      return Ok(None);
    }
    self.index["nextPersonSeq"] = json!(seq + 1);
    let threshold = self.stored_threshold();
    self.recluster(threshold)?;
    Ok(Some(new_person_id))
  }

  pub fn move_groups(
    &mut self,
    group_keys: Vec<String>,
    target_person_id: Option<String>,
  ) -> Result<bool, String> {
    self.assert_writable()?;
    if !crate::face_cluster::move_groups_in_index(
      &mut self.index,
      &group_keys,
      target_person_id.as_deref(),
    ) {
      return Ok(false);
    }
    let threshold = self.stored_threshold();
    self.recluster(threshold)?;
    Ok(true)
  }

  pub fn set_person_representative(
    &mut self,
    person_id: &str,
    face_id: &str,
  ) -> Result<bool, String> {
    self.assert_writable()?;
    if !crate::face_cluster::set_person_representative_in_index(
      &mut self.index,
      person_id,
      face_id,
    ) {
      return Ok(false);
    }
    self.save_index()?;
    Ok(true)
  }

  /// True when a non-error entry exists for the hash (isScanned in TS;
  /// errored videos are retried by later scan requests).
  pub fn is_scanned(&self, hash: &str) -> bool {
    self
      .index
      .get("videos")
      .and_then(|videos| videos.get(hash))
      .map(|entry| entry.get("status").and_then(Value::as_str) != Some("error"))
      .unwrap_or(false)
  }

  /// Persist one video's scan result (setVideo in TS): appends the
  /// embeddings to embeddings.bin, assigning each face its index, and
  /// rewrites index.json atomically.
  pub fn set_video(
    &mut self,
    hash: &str,
    status: &str,
    frame_count: i64,
    faces: &[ScannedFace],
    error: Option<&str>,
  ) -> Result<(), String> {
    self.assert_writable()?;
    let mut records = Vec::with_capacity(faces.len());
    if !faces.is_empty() {
      use std::io::Write;
      fs::create_dir_all(&self.faces_dir)
        .map_err(|e| format!("Cannot create {}: {e}", self.faces_dir.display()))?;
      let mut bin = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(self.faces_dir.join("embeddings.bin"))
        .map_err(|e| format!("Cannot open embeddings.bin: {e}"))?;
      for (offset, face) in faces.iter().enumerate() {
        records.push(json!({
          "faceId": face.face_id,
          "t": face.t,
          "bbox": face.bbox,
          "score": face.score,
          "quality": face.quality,
          "embeddingIdx": self.embeddings_stored + offset as u64,
          "groupId": face.group_id,
          "representative": face.representative
        }));
        let mut bytes = Vec::with_capacity(EMBEDDING_BYTES);
        for value in &face.embedding {
          bytes.extend_from_slice(&value.to_le_bytes());
        }
        bin
          .write_all(&bytes)
          .map_err(|e| format!("Cannot append embeddings: {e}"))?;
      }
      self.embeddings_stored += faces.len() as u64;
    }
    let mut entry = json!({
      "status": status,
      "scannedAt": now_millis(),
      "frameCount": frame_count,
      "faces": records
    });
    if let Some(error) = error {
      entry["error"] = Value::String(error.to_string());
    }
    if let Some(videos) = self.index.get_mut("videos").and_then(Value::as_object_mut) {
      videos.insert(hash.to_string(), entry);
    }
    self.save_index()
  }

  /// Record a failed scan without embeddings (setVideoError in TS).
  pub fn set_video_error(
    &mut self,
    hash: &str,
    frame_count: i64,
    error: &str,
  ) -> Result<(), String> {
    self.set_video(hash, "error", frame_count, &[], Some(error))
  }

  /// Store freshly computed attribute aggregates (setPersonsAttributes;
  /// persons not listed keep theirs).
  pub fn set_persons_attributes(
    &mut self,
    results: &[(String, Value)],
  ) -> Result<(), String> {
    let mut changed = false;
    if let Some(persons) = self.index.get_mut("persons").and_then(Value::as_array_mut) {
      for person in persons.iter_mut() {
        let id = person
          .get("personId")
          .and_then(Value::as_str)
          .unwrap_or_default()
          .to_string();
        if let Some((_, attributes)) = results.iter().find(|(pid, _)| *pid == id) {
          person["attributes"] = attributes.clone();
          changed = true;
        }
      }
    }
    if changed {
      self.save_index()?;
    }
    Ok(())
  }

  /// Wipe all face data for a hub (FaceStore.reset in TS; used when the
  /// model pack changed and stored embeddings are incompatible).
  pub fn reset(faces_dir: &Path, model_id: &str, cluster_threshold: f64) -> FaceStore {
    for target in [
      faces_dir.join("index.json"),
      faces_dir.join("embeddings.bin"),
      faces_dir.join("aligned"),
      faces_dir.join("crops"),
    ] {
      let _ = if target.is_dir() {
        fs::remove_dir_all(&target)
      } else {
        fs::remove_file(&target)
      };
    }
    let store = FaceStore {
      faces_dir: faces_dir.to_path_buf(),
      index: empty_index(model_id, cluster_threshold),
      embeddings_stored: 0,
      model_mismatch: false,
    };
    let _ = store.save_index();
    store
  }
}

/// One face produced by a scan (ScannedFace in face-types.ts), carrying
/// its recognition embedding until the store appends it to
/// embeddings.bin and replaces it with an index.
pub struct ScannedFace {
  pub face_id: String,
  pub t: f64,
  pub bbox: [f64; 4],
  pub score: f64,
  pub quality: f64,
  pub embedding: Vec<f32>,
  pub group_id: i64,
  pub representative: bool,
}

fn now_millis() -> i64 {
  std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .map(|d| d.as_millis() as i64)
    .unwrap_or(0)
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
    let status = model_status_with(&settings("quality"), &root, None);
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

    let status = model_status_with(&settings("light"), &root, None);
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

  /// The Phase B op sequence, with the same synthetic hub used to
  /// cross-check this port against the TypeScript FaceStore (the final
  /// saved index.json was equal in value to the TS result).
  #[test]
  fn store_clustering_and_edits_match_typescript() {
    fn embedding(overrides: &[(usize, f64)]) -> Vec<f32> {
      let mut v = vec![0.02f64; 512];
      for (dim, value) in overrides {
        v[*dim] = *value;
      }
      let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
      v.iter().map(|x| (x / norm) as f32).collect()
    }
    let embeddings = vec![
      embedding(&[(0, 1.0), (1, 0.08)]),
      embedding(&[(0, 1.0), (1, 0.12), (3, 0.05)]),
      embedding(&[(1, 1.0), (0, 0.06)]),
      embedding(&[(0, 1.0), (2, 0.09)]),
      embedding(&[(2, 1.0), (0, 0.04)]),
      embedding(&[(1, 1.0), (2, 0.07)]),
      embedding(&[(0, 1.0), (1, 0.05), (2, 0.03)]),
    ];
    fn face(face_id: &str, emb: usize, group: i64, rep: bool, quality: f64) -> Value {
      json!({
        "faceId": face_id, "t": 1, "bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.99,
        "quality": quality, "embeddingIdx": emb, "groupId": group, "representative": rep
      })
    }
    let root = temp_dir("faces-phase-b");
    let faces_dir = root.join("faces");
    fs::create_dir_all(&faces_dir).expect("faces dir");
    let mut bin = Vec::new();
    for vector in &embeddings {
      for value in vector {
        bin.extend_from_slice(&value.to_le_bytes());
      }
    }
    fs::write(faces_dir.join("embeddings.bin"), bin).expect("embeddings");
    let index = json!({
      "version": 1, "modelId": "buffalo_l-r50", "clusterThreshold": 0.5,
      "videos": {
        "vidA": { "status": "done", "scannedAt": 1, "frameCount": 10, "faces": [
          face("vidA-0000", 0, 0, true, 0.95),
          face("vidA-0001", 1, 0, false, 0.80),
          face("vidA-0002", 2, 1, true, 0.90)
        ]},
        "vidB": { "status": "done", "scannedAt": 1, "frameCount": 8, "faces": [
          face("vidB-0000", 3, 0, true, 0.92),
          face("vidB-0001", 4, 1, true, 0.85)
        ]},
        "vidC": { "status": "done", "scannedAt": 1, "frameCount": 6, "faces": [
          face("vidC-0000", 5, 0, true, 0.88),
          face("vidC-0001", 6, 1, true, 0.86)
        ]}
      },
      "persons": [], "nextPersonSeq": 1
    });
    fs::write(
      faces_dir.join("index.json"),
      serde_json::to_string(&index).expect("serialize"),
    )
    .expect("index");

    let mut store = FaceStore::load(&faces_dir, "buffalo_l-r50", 0.5);
    store.recluster(0.5).expect("recluster");
    let summary = |store: &FaceStore| -> Vec<(String, String, bool, String, i64)> {
      store.payload()["index"]["persons"]
        .as_array()
        .expect("persons")
        .iter()
        .map(|p| {
          (
            p["personId"].as_str().unwrap_or_default().to_string(),
            p["name"].as_str().unwrap_or_default().to_string(),
            p["hidden"].as_bool().unwrap_or(false),
            p["representativeFaceId"]
              .as_str()
              .unwrap_or_default()
              .to_string(),
            p["faceCount"].as_i64().unwrap_or(0),
          )
        })
        .collect()
    };
    assert_eq!(
      summary(&store),
      vec![
        ("p000001".into(), "".into(), false, "vidA-0000".into(), 4),
        ("p000002".into(), "".into(), false, "vidA-0002".into(), 2),
        ("p000003".into(), "".into(), false, "vidB-0001".into(), 1),
      ]
    );

    assert!(store.rename_person("p000001", "Alice"));
    assert!(store.set_person_hidden("p000002", true));
    store.recluster(0.5).expect("recluster after edits");
    // IDs, the name, and the hidden flag survive re-clustering.
    assert_eq!(summary(&store)[0].1, "Alice");
    assert!(summary(&store)[1].2);

    assert_eq!(
      store
        .merge_persons(vec!["p000001".to_string(), "p000003".to_string()])
        .expect("merge"),
      Some("p000001".to_string())
    );
    assert_eq!(
      store
        .split_person("p000002", vec!["vidC#0".to_string()])
        .expect("split"),
      Some("p000004".to_string())
    );
    assert!(
      store
        .move_groups(vec!["vidB#0".to_string()], None)
        .expect("move")
    );
    assert!(
      store
        .set_person_representative("p000001", "vidA-0001")
        .expect("representative")
    );
    assert_eq!(
      summary(&store),
      vec![
        ("p000001".into(), "Alice".into(), false, "vidA-0001".into(), 4),
        ("p000002".into(), "".into(), true, "vidA-0002".into(), 1),
        ("p000004".into(), "".into(), false, "vidC-0000".into(), 1),
      ]
    );
    let payload = store.payload();
    let index = &payload["index"];
    assert_eq!(index["manualUnassigned"], json!(["vidB#0"]));
    assert_eq!(index["manualRepFaces"]["p000001"], json!("vidA-0001"));
    let vid_b0 = &index["videos"]["vidB"]["faces"][0];
    assert!(vid_b0.get("personId").is_none(), "moved group is unassigned");

    // A mismatched store refuses pin edits but still allows renames,
    // exactly like the TS store (assertWritable is not called there).
    let mut mismatched = FaceStore::load(&faces_dir, "buffalo_sc-mbf", 0.5);
    assert!(mismatched.model_mismatch);
    assert!(mismatched
      .merge_persons(vec!["p000001".to_string(), "p000002".to_string()])
      .is_err());
    assert!(mismatched.rename_person("p000001", "Alicia"));
    fs::remove_dir_all(&root).ok();
  }
}
