//! Global clustering and manual-correction pins: a faithful port of
//! node/face/cluster.ts (design doc §5.6 + Phase 3 edits). Embeddings are
//! f32 values read from embeddings.bin; similarity math runs in f64 over
//! those values and vector means keep the Float32Array store rounding of
//! the original, so results match the TypeScript implementation.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};


pub fn dot(a: &[f32], b: &[f32]) -> f64 {
  let mut sum = 0.0f64;
  for i in 0..a.len().min(b.len()) {
    sum += a[i] as f64 * b[i] as f64;
  }
  sum
}

fn l2normalize(v: &[f32]) -> Vec<f32> {
  let mut norm = 0.0f64;
  for x in v {
    norm += *x as f64 * *x as f64;
  }
  norm = norm.sqrt();
  if norm > 0.0 {
    v.iter().map(|x| (*x as f64 / norm) as f32).collect()
  } else {
    vec![0.0; v.len()]
  }
}

fn mean_vec_weighted(vectors: &[Vec<f32>], weights: &[f64]) -> Vec<f32> {
  let dim = vectors[0].len();
  let mut out = vec![0.0f32; dim];
  let mut wsum = 0.0f64;
  for (vi, v) in vectors.iter().enumerate() {
    wsum += weights[vi];
    for i in 0..dim {
      out[i] = (out[i] as f64 + v[i] as f64 * weights[vi]) as f32;
    }
  }
  for i in 0..dim {
    out[i] = (out[i] as f64 / wsum) as f32;
  }
  l2normalize(&out)
}

#[derive(Clone)]
pub struct GlobalGroupItem {
  pub key: String,
  pub video_hash: String,
  #[allow(dead_code)]
  pub group_id: i64,
  pub rep_face_id: String,
  pub rep_embedding: Vec<f32>,
  pub rep_quality: f64,
  pub weight: usize,
  pub face_ids: Vec<String>,
}

fn face_str(face: &Value, key: &str) -> String {
  face.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn face_i64(face: &Value, key: &str) -> i64 {
  face.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn face_f64(face: &Value, key: &str) -> f64 {
  face.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// One item per (video, within-video group), from collectGroupItems.
pub fn collect_group_items<F: Fn(i64) -> Vec<f32>>(index: &Value, read: &F) -> Vec<GlobalGroupItem> {
  let mut items = Vec::new();
  let Some(videos) = index.get("videos").and_then(Value::as_object) else {
    return items;
  };
  for (hash, entry) in videos {
    if entry.get("status").and_then(Value::as_str) == Some("error") {
      continue;
    }
    let Some(faces) = entry.get("faces").and_then(Value::as_array) else {
      continue;
    };
    // Groups in first-appearance order (JS Map semantics).
    let mut group_order: Vec<i64> = Vec::new();
    let mut groups: HashMap<i64, Vec<&Value>> = HashMap::new();
    for face in faces {
      let group_id = face_i64(face, "groupId");
      if group_id < 0 {
        continue;
      }
      if !groups.contains_key(&group_id) {
        group_order.push(group_id);
        groups.insert(group_id, Vec::new());
      }
      groups.get_mut(&group_id).expect("group").push(face);
    }
    for group_id in group_order {
      let members = groups.remove(&group_id).expect("members");
      // The stored representative wins (last flagged member, as in TS);
      // without one, fall back to the highest-quality member.
      let mut rep = members[0];
      for member in &members {
        if member.get("representative").and_then(Value::as_bool) == Some(true) {
          rep = member;
        }
      }
      if rep.get("representative").and_then(Value::as_bool) != Some(true) {
        for member in &members {
          if face_f64(member, "quality") > face_f64(rep, "quality") {
            rep = member;
          }
        }
      }
      items.push(GlobalGroupItem {
        key: format!("{hash}#{group_id}"),
        video_hash: hash.clone(),
        group_id,
        rep_face_id: face_str(rep, "faceId"),
        rep_embedding: read(face_i64(rep, "embeddingIdx")),
        rep_quality: face_f64(rep, "quality"),
        weight: members.len(),
        face_ids: members.iter().map(|m| face_str(m, "faceId")).collect(),
      });
    }
  }
  items
}

pub struct GlobalCluster {
  pub items: Vec<GlobalGroupItem>,
  /// Final centroid, carried for parity with the TS GlobalCluster.
  #[allow(dead_code)]
  pub centroid: Vec<f32>,
}

impl GlobalCluster {
  /// The item that created the cluster is always its first item.
  #[allow(dead_code)]
  pub fn rep_item(&self) -> &GlobalGroupItem {
    &self.items[0]
  }
  pub fn rep_embedding(&self) -> &[f32] {
    &self.items[0].rep_embedding
  }
}

/// Leader-follower assignment from clusterGlobalGroups: an item joins the
/// best cluster only when its similarity clears the threshold against
/// both the cluster representative and the cluster centroid.
pub fn cluster_global_groups(items: Vec<GlobalGroupItem>, threshold: f64) -> Vec<GlobalCluster> {
  let mut sorted = items;
  sorted.sort_by(|a, b| {
    if a.rep_quality != b.rep_quality {
      b.rep_quality.total_cmp(&a.rep_quality)
    } else {
      a.key.cmp(&b.key)
    }
  });

  struct Work {
    items: Vec<GlobalGroupItem>,
    centroid: Vec<f32>,
    rep_embs: Vec<Vec<f32>>,
    weights: Vec<f64>,
  }
  let mut clusters: Vec<Work> = Vec::new();
  for item in sorted {
    let mut best: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, cluster) in clusters.iter().enumerate() {
      let score = dot(&item.rep_embedding, &cluster.items[0].rep_embedding)
        .min(dot(&item.rep_embedding, &cluster.centroid));
      if score > best_score {
        best_score = score;
        best = Some(idx);
      }
    }
    match best {
      Some(idx) if best_score >= threshold => {
        let cluster = &mut clusters[idx];
        cluster.rep_embs.push(item.rep_embedding.clone());
        cluster.weights.push(item.weight as f64);
        cluster.items.push(item);
        cluster.centroid = mean_vec_weighted(&cluster.rep_embs, &cluster.weights);
      }
      _ => {
        let centroid = item.rep_embedding.clone();
        clusters.push(Work {
          rep_embs: vec![item.rep_embedding.clone()],
          weights: vec![item.weight as f64],
          items: vec![item],
          centroid,
        });
      }
    }
  }
  clusters
    .into_iter()
    .map(|work| GlobalCluster {
      items: work.items,
      centroid: work.centroid,
    })
    .collect()
}

pub struct ClusterizeResult {
  pub persons: Vec<Value>,
  pub face_person: HashMap<String, String>,
  pub next_person_seq: i64,
  pub manual_groups: Map<String, Value>,
  pub manual_unassigned: Vec<String>,
  pub manual_rep_faces: Map<String, Value>,
}

/// JS parseInt over the part after the first character ("p000123" -> 123).
fn person_seq_of(person_id: &str) -> i64 {
  let rest = person_id.chars().skip(1).collect::<String>();
  parse_int_prefix(&rest).unwrap_or(0)
}

/// JS parseInt: leading optional sign + digits, None when there are none.
pub fn parse_int_prefix(text: &str) -> Option<i64> {
  let mut chars = text.chars().peekable();
  let mut sign = 1i64;
  if let Some(&c) = chars.peek() {
    if c == '-' || c == '+' {
      sign = if c == '-' { -1 } else { 1 };
      chars.next();
    }
  }
  let mut value: Option<i64> = None;
  for c in chars {
    if let Some(digit) = c.to_digit(10) {
      value = Some(value.unwrap_or(0) * 10 + digit as i64);
    } else {
      break;
    }
  }
  value.map(|v| v * sign)
}

fn format_person_id(seq: i64) -> String {
  format!("p{seq:06}")
}

fn persons_array(index: &Value) -> Vec<Value> {
  index
    .get("persons")
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default()
}

fn group_key(hash: &str, group_id: i64) -> String {
  format!("{hash}#{group_id}")
}

/// Full re-clustering of an index (clusterizeIndex): free clustering with
/// ID inheritance from the previous persons, pinned persons rebuilt
/// verbatim, and pins pruned against what still exists.
pub fn clusterize_index<F: Fn(i64) -> Vec<f32>>(
  index: &Value,
  read: &F,
  threshold: f64,
) -> ClusterizeResult {
  let items = collect_group_items(index, read);

  let manual_groups_in: Map<String, Value> = index
    .get("manualGroups")
    .and_then(Value::as_object)
    .cloned()
    .unwrap_or_default();
  let manual_unassigned_in: HashSet<String> = index
    .get("manualUnassigned")
    .and_then(Value::as_array)
    .map(|keys| {
      keys
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
    })
    .unwrap_or_default();
  let manual_rep_faces_in: Map<String, Value> = index
    .get("manualRepFaces")
    .and_then(Value::as_object)
    .cloned()
    .unwrap_or_default();

  let mut pinned_by_person: Vec<(String, Vec<GlobalGroupItem>)> = Vec::new();
  let mut free_items: Vec<GlobalGroupItem> = Vec::new();
  for item in items.iter() {
    if manual_unassigned_in.contains(&item.key) {
      continue;
    }
    match manual_groups_in.get(&item.key).and_then(Value::as_str) {
      Some(target) => {
        if let Some((_, list)) = pinned_by_person.iter_mut().find(|(id, _)| id == target) {
          list.push(item.clone());
        } else {
          pinned_by_person.push((target.to_string(), vec![item.clone()]));
        }
      }
      None => free_items.push(item.clone()),
    }
  }
  let pinned_person_ids: HashSet<String> =
    pinned_by_person.iter().map(|(id, _)| id.clone()).collect();

  let clusters = cluster_global_groups(free_items, threshold);

  // Face lookup by ID + previous persons with rep embedding and group set.
  struct PrevPerson {
    record: Value,
    rep_embedding: Option<Vec<f32>>,
    group_keys: HashSet<String>,
  }
  let mut face_by_id: HashMap<String, Value> = HashMap::new();
  if let Some(videos) = index.get("videos").and_then(Value::as_object) {
    for entry in videos.values() {
      if let Some(faces) = entry.get("faces").and_then(Value::as_array) {
        for face in faces {
          face_by_id.insert(face_str(face, "faceId"), face.clone());
        }
      }
    }
  }
  let prev_persons: Vec<PrevPerson> = persons_array(index)
    .into_iter()
    .map(|record| {
      let person_id = face_str(&record, "personId");
      let rep_embedding = face_by_id
        .get(&face_str(&record, "representativeFaceId"))
        .map(|face| read(face_i64(face, "embeddingIdx")));
      let mut group_keys = HashSet::new();
      if let Some(videos) = index.get("videos").and_then(Value::as_object) {
        for (hash, entry) in videos {
          if let Some(faces) = entry.get("faces").and_then(Value::as_array) {
            for face in faces {
              if face.get("personId").and_then(Value::as_str) == Some(person_id.as_str()) {
                let gid = face_i64(face, "groupId");
                if gid >= 0 {
                  group_keys.insert(group_key(hash, gid));
                }
              }
            }
          }
        }
      }
      PrevPerson {
        record,
        rep_embedding,
        group_keys,
      }
    })
    .collect();

  // Greedy best-first one-to-one inheritance matching.
  struct Pair {
    cluster_idx: usize,
    prev_idx: usize,
    sim: f64,
  }
  let mut pairs: Vec<Pair> = Vec::new();
  for (cluster_idx, cluster) in clusters.iter().enumerate() {
    for (prev_idx, prev) in prev_persons.iter().enumerate() {
      if pinned_person_ids.contains(&face_str(&prev.record, "personId")) {
        continue;
      }
      if let Some(rep_embedding) = &prev.rep_embedding {
        let sim = dot(cluster.rep_embedding(), rep_embedding);
        if sim >= threshold {
          pairs.push(Pair {
            cluster_idx,
            prev_idx,
            sim,
          });
        }
      }
    }
  }
  pairs.sort_by(|a, b| b.sim.total_cmp(&a.sim));
  let mut cluster_to_prev: HashMap<usize, usize> = HashMap::new();
  let mut used_prev: HashSet<usize> = HashSet::new();
  for pair in &pairs {
    if !cluster_to_prev.contains_key(&pair.cluster_idx) && !used_prev.contains(&pair.prev_idx) {
      cluster_to_prev.insert(pair.cluster_idx, pair.prev_idx);
      used_prev.insert(pair.prev_idx);
    }
  }

  let mut next_seq = index.get("nextPersonSeq").and_then(Value::as_i64).unwrap_or(0);
  if next_seq == 0 {
    next_seq = 1;
  }
  for prev in &prev_persons {
    next_seq = next_seq.max(person_seq_of(&face_str(&prev.record, "personId")) + 1);
  }

  fn build_record(
    person_id: &str,
    prev: Option<&PrevPerson>,
    items: &[GlobalGroupItem],
    rep_override: Option<&str>,
    face_person: &mut HashMap<String, String>,
  ) -> Value {
    let mut video_set: Vec<String> = Vec::new();
    let mut face_count = 0usize;
    for item in items {
      if !video_set.contains(&item.video_hash) {
        video_set.push(item.video_hash.clone());
      }
      face_count += item.face_ids.len();
      for face_id in &item.face_ids {
        face_person.insert(face_id.clone(), person_id.to_string());
      }
    }
    video_set.sort();
    let mut rep_item = &items[0];
    for item in items {
      if item.rep_quality > rep_item.rep_quality {
        rep_item = item;
      }
    }
    let mut representative = rep_item.rep_face_id.clone();
    if let Some(over) = rep_override {
      if items.iter().any(|item| item.face_ids.iter().any(|f| f == over)) {
        representative = over.to_string();
      }
    }
    let mut record = json!({
      "personId": person_id,
      "name": prev.map(|p| face_str(&p.record, "name")).unwrap_or_default(),
      "representativeFaceId": representative,
      "videoHashes": video_set,
      "faceCount": face_count,
      "hidden": prev
        .map(|p| p.record.get("hidden").and_then(Value::as_bool).unwrap_or(false))
        .unwrap_or(false)
    });
    // Attribute aggregates survive only an unchanged group set.
    if let Some(prev) = prev {
      if let Some(attributes) = prev.record.get("attributes") {
        let new_keys: HashSet<String> = items.iter().map(|item| item.key.clone()).collect();
        if new_keys == prev.group_keys {
          record["attributes"] = attributes.clone();
        }
      }
    }
    record
  }

  let mut persons: Vec<Value> = Vec::new();
  let mut face_person: HashMap<String, String> = HashMap::new();

  for (cluster_idx, cluster) in clusters.iter().enumerate() {
    let prev = cluster_to_prev.get(&cluster_idx).map(|idx| &prev_persons[*idx]);
    let person_id = match prev {
      Some(prev) => face_str(&prev.record, "personId"),
      None => {
        let id = format_person_id(next_seq);
        next_seq += 1;
        id
      }
    };
    let rep_override = manual_rep_faces_in
      .get(&person_id)
      .and_then(Value::as_str);
    // build_record picks the highest-quality item as representative; for
    // a free cluster that is always its creator (items are processed in
    // quality order, and ties keep the first item), matching repItem.
    let record = build_record(&person_id, prev, &cluster.items, rep_override, &mut face_person);
    persons.push(record);
  }

  // Pinned persons, in person ID order.
  let mut pinned_sorted = pinned_by_person;
  pinned_sorted.sort_by(|a, b| a.0.cmp(&b.0));
  let prev_by_id: HashMap<String, &PrevPerson> = prev_persons
    .iter()
    .map(|prev| (face_str(&prev.record, "personId"), prev))
    .collect();
  for (person_id, p_items) in &pinned_sorted {
    let prev = prev_by_id.get(person_id).copied();
    let rep_override = manual_rep_faces_in.get(person_id).and_then(Value::as_str);
    let record = build_record(person_id, prev, p_items, rep_override, &mut face_person);
    persons.push(record);
    next_seq = next_seq.max(person_seq_of(person_id) + 1);
  }

  // Prune pins naming groups or persons that no longer exist.
  let result_person_ids: HashSet<String> = persons.iter().map(|p| face_str(p, "personId")).collect();
  let item_keys: HashSet<String> = items.iter().map(|item| item.key.clone()).collect();
  let mut pruned_groups = Map::new();
  for (key, target) in &manual_groups_in {
    if item_keys.contains(key)
      && target
        .as_str()
        .map(|t| result_person_ids.contains(t))
        .unwrap_or(false)
    {
      pruned_groups.insert(key.clone(), target.clone());
    }
  }
  let pruned_unassigned: Vec<String> = index
    .get("manualUnassigned")
    .and_then(Value::as_array)
    .map(|keys| {
      keys
        .iter()
        .filter_map(Value::as_str)
        .filter(|key| item_keys.contains(*key))
        .map(str::to_string)
        .collect()
    })
    .unwrap_or_default();
  let mut pruned_rep_faces = Map::new();
  for (person_id, face_id) in &manual_rep_faces_in {
    if result_person_ids.contains(person_id)
      && face_id
        .as_str()
        .map(|f| face_person.get(f) == Some(person_id))
        .unwrap_or(false)
    {
      pruned_rep_faces.insert(person_id.clone(), face_id.clone());
    }
  }

  ClusterizeResult {
    persons,
    face_person,
    next_person_seq: next_seq,
    manual_groups: pruned_groups,
    manual_unassigned: pruned_unassigned,
    manual_rep_faces: pruned_rep_faces,
  }
}

// ============================================================================
// Manual corrections (Phase 3 edits): pin manipulation only; the store
// materializes every edit by re-running clusterize_index.
// ============================================================================

fn ensure_pins(index: &mut Value) {
  let Some(obj) = index.as_object_mut() else {
    return;
  };
  obj.entry("manualGroups").or_insert_with(|| json!({}));
  obj.entry("manualUnassigned").or_insert_with(|| json!([]));
  obj.entry("manualRepFaces").or_insert_with(|| json!({}));
}

fn unpin_unassigned(index: &mut Value, key: &str) {
  if let Some(list) = index
    .get_mut("manualUnassigned")
    .and_then(Value::as_array_mut)
  {
    list.retain(|k| k.as_str() != Some(key));
  }
}

fn pin_unassigned(index: &mut Value, key: &str) {
  if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
    groups.remove(key);
  }
  let list = index
    .get_mut("manualUnassigned")
    .and_then(Value::as_array_mut);
  if let Some(list) = list {
    if !list.iter().any(|k| k.as_str() == Some(key)) {
      list.push(Value::String(key.to_string()));
    }
  } else if let Some(obj) = index.as_object_mut() {
    obj.insert("manualUnassigned".to_string(), json!([key]));
  }
}

/// Group keys currently assigned to a person (personGroupKeys).
pub fn person_group_keys(index: &Value, person_id: &str) -> Vec<String> {
  let mut keys: Vec<String> = Vec::new();
  if let Some(videos) = index.get("videos").and_then(Value::as_object) {
    for (hash, entry) in videos {
      if let Some(faces) = entry.get("faces").and_then(Value::as_array) {
        for face in faces {
          if face.get("personId").and_then(Value::as_str) == Some(person_id) {
            let gid = face_i64(face, "groupId");
            if gid >= 0 {
              let key = group_key(hash, gid);
              if !keys.contains(&key) {
                keys.push(key);
              }
            }
          }
        }
      }
    }
  }
  keys
}

fn split_group_key(key: &str) -> Option<(String, i64)> {
  let sep = key.rfind('#')?;
  let group_id = parse_int_prefix(&key[sep + 1..])?;
  Some((key[..sep].to_string(), group_id))
}

fn group_exists(index: &Value, key: &str) -> bool {
  let Some((hash, group_id)) = split_group_key(key) else {
    return false;
  };
  index
    .get("videos")
    .and_then(|videos| videos.get(&hash))
    .and_then(|entry| entry.get("faces"))
    .and_then(Value::as_array)
    .map(|faces| faces.iter().any(|face| face_i64(face, "groupId") == group_id))
    .unwrap_or(false)
}

fn group_owner(index: &Value, key: &str) -> Option<String> {
  let (hash, group_id) = split_group_key(key)?;
  let faces = index
    .get("videos")?
    .get(&hash)?
    .get("faces")?
    .as_array()?;
  faces
    .iter()
    .find(|face| {
      face_i64(face, "groupId") == group_id
        && face.get("personId").and_then(Value::as_str).is_some()
    })
    .and_then(|face| face.get("personId"))
    .and_then(Value::as_str)
    .map(str::to_string)
}

fn person_record<'a>(index: &'a Value, person_id: &str) -> Option<&'a Value> {
  index
    .get("persons")
    .and_then(Value::as_array)?
    .iter()
    .find(|p| p.get("personId").and_then(Value::as_str) == Some(person_id))
}

fn find_face(index: &Value, face_id: &str) -> Option<(String, Value)> {
  if let Some(videos) = index.get("videos").and_then(Value::as_object) {
    for (hash, entry) in videos {
      if let Some(faces) = entry.get("faces").and_then(Value::as_array) {
        if let Some(face) = faces
          .iter()
          .find(|f| f.get("faceId").and_then(Value::as_str) == Some(face_id))
        {
          return Some((hash.clone(), face.clone()));
        }
      }
    }
  }
  None
}

/// mergePersonsInIndex: see the TS doc comment for the survivor rule.
pub fn merge_persons_in_index(index: &mut Value, person_ids: &[String]) -> Option<String> {
  let mut ids: Vec<String> = Vec::new();
  for id in person_ids {
    if person_record(index, id).is_some() && !ids.contains(id) {
      ids.push(id.clone());
    }
  }
  if ids.len() < 2 {
    return None;
  }
  let survivor = ids
    .iter()
    .find(|id| {
      person_record(index, id)
        .map(|p| !face_str(p, "name").trim().is_empty())
        .unwrap_or(false)
    })
    .cloned()
    .unwrap_or_else(|| ids[0].clone());
  let sources: Vec<String> = ids.iter().filter(|id| **id != survivor).cloned().collect();

  ensure_pins(index);
  let mut keys: Vec<String> = Vec::new();
  for id in &ids {
    for key in person_group_keys(index, id) {
      if !keys.contains(&key) {
        keys.push(key);
      }
    }
  }
  for key in &keys {
    if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
      groups.insert(key.clone(), Value::String(survivor.clone()));
    }
    unpin_unassigned(index, key);
  }
  if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
    for target in groups.values_mut() {
      if target
        .as_str()
        .map(|t| sources.iter().any(|s| s == t))
        .unwrap_or(false)
      {
        *target = Value::String(survivor.clone());
      }
    }
  }
  for source in &sources {
    if let Some(reps) = index.get_mut("manualRepFaces").and_then(Value::as_object_mut) {
      reps.remove(source);
    }
  }
  if let Some(persons) = index.get_mut("persons").and_then(Value::as_array_mut) {
    persons.retain(|p| {
      !sources
        .iter()
        .any(|s| p.get("personId").and_then(Value::as_str) == Some(s.as_str()))
    });
  }
  Some(survivor)
}

/// splitPersonInIndex: moves a proper subset of a person's groups to the
/// caller-minted new person, freezing both sides with pins.
pub fn split_person_in_index(
  index: &mut Value,
  person_id: &str,
  group_keys: &[String],
  new_person_id: &str,
) -> bool {
  if person_record(index, person_id).is_none() || person_record(index, new_person_id).is_some() {
    return false;
  }
  let owned = person_group_keys(index, person_id);
  let mut moving: Vec<String> = Vec::new();
  for key in group_keys {
    if owned.contains(key) && !moving.contains(key) {
      moving.push(key.clone());
    }
  }
  if moving.is_empty() || moving.len() == owned.len() {
    return false;
  }
  ensure_pins(index);
  for key in &moving {
    if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
      groups.insert(key.clone(), Value::String(new_person_id.to_string()));
    }
    unpin_unassigned(index, key);
  }
  for key in owned.iter().filter(|key| !moving.contains(key)) {
    if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
      groups.insert(key.clone(), Value::String(person_id.to_string()));
    }
  }
  if let Some(persons) = index.get_mut("persons").and_then(Value::as_array_mut) {
    persons.push(json!({
      "personId": new_person_id,
      "name": "",
      "representativeFaceId": "",
      "videoHashes": [],
      "faceCount": 0,
      "hidden": false
    }));
  }
  let rep_override = index
    .get("manualRepFaces")
    .and_then(|reps| reps.get(person_id))
    .and_then(Value::as_str)
    .map(str::to_string);
  if let Some(rep_face_id) = rep_override {
    if let Some((hash, face)) = find_face(index, &rep_face_id) {
      if moving.contains(&group_key(&hash, face_i64(&face, "groupId"))) {
        if let Some(reps) = index.get_mut("manualRepFaces").and_then(Value::as_object_mut) {
          reps.remove(person_id);
        }
      }
    }
  }
  true
}

/// moveGroupsInIndex: pin moved groups to their target (or to unassigned),
/// freezing the source remainders and the target's existing groups.
pub fn move_groups_in_index(
  index: &mut Value,
  group_keys: &[String],
  target_person_id: Option<&str>,
) -> bool {
  if let Some(target) = target_person_id {
    if person_record(index, target).is_none() {
      return false;
    }
  }
  let mut valid: Vec<String> = Vec::new();
  for key in group_keys {
    if group_exists(index, key) && !valid.contains(key) {
      valid.push(key.clone());
    }
  }
  if valid.is_empty() {
    return false;
  }
  ensure_pins(index);
  let mut sources: Vec<String> = Vec::new();
  for key in &valid {
    if let Some(owner) = group_owner(index, key) {
      if !sources.contains(&owner) {
        sources.push(owner);
      }
    }
  }
  for source in &sources {
    if Some(source.as_str()) == target_person_id {
      continue;
    }
    for key in person_group_keys(index, source)
      .into_iter()
      .filter(|key| !valid.contains(key))
    {
      if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
        groups.insert(key.clone(), Value::String(source.clone()));
      }
      unpin_unassigned(index, &key);
    }
  }
  if let Some(target) = target_person_id {
    for key in person_group_keys(index, target) {
      if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
        groups.insert(key.clone(), Value::String(target.to_string()));
      }
      unpin_unassigned(index, &key);
    }
  }
  for key in &valid {
    match target_person_id {
      Some(target) => {
        if let Some(groups) = index.get_mut("manualGroups").and_then(Value::as_object_mut) {
          groups.insert(key.clone(), Value::String(target.to_string()));
        }
        unpin_unassigned(index, key);
      }
      None => pin_unassigned(index, key),
    }
  }
  true
}

/// setPersonRepresentativeInIndex: pin a person's representative face.
pub fn set_person_representative_in_index(
  index: &mut Value,
  person_id: &str,
  face_id: &str,
) -> bool {
  if person_record(index, person_id).is_none() {
    return false;
  }
  let Some((_, face)) = find_face(index, face_id) else {
    return false;
  };
  if face.get("personId").and_then(Value::as_str) != Some(person_id) {
    return false;
  }
  ensure_pins(index);
  if let Some(reps) = index.get_mut("manualRepFaces").and_then(Value::as_object_mut) {
    reps.insert(person_id.to_string(), Value::String(face_id.to_string()));
  }
  if let Some(persons) = index.get_mut("persons").and_then(Value::as_array_mut) {
    for person in persons.iter_mut() {
      if person.get("personId").and_then(Value::as_str) == Some(person_id) {
        person["representativeFaceId"] = Value::String(face_id.to_string());
      }
    }
  }
  true
}
