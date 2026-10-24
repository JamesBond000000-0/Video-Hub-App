// Within-video face grouping (design doc §5.5) and global (cross-video)
// clustering (§5.6), ported from the Phase 0 spike
// (tools/face-spike/src/cluster.ts). Manual corrections (Phase 3) are
// expressed as pins on the index and honored by clusterizeIndex; the
// edit operations at the bottom of this file only manipulate those pins
// (plus person-record bookkeeping) -- the store materializes the result
// by re-running clusterizeIndex.

export interface GroupableFace {
  embedding: Float32Array;
  quality: number;
}

export interface WithinVideoGroup<T extends GroupableFace> {
  members: T[];
  representative: T;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

export function meanVec(vectors: Float32Array[]): Float32Array {
  const dim = vectors[0].length;
  const out = new Float32Array(dim);
  vectors.forEach((v) => {
    for (let i = 0; i < dim; i++) {
      out[i] += v[i];
    }
  });
  for (let i = 0; i < dim; i++) {
    out[i] /= vectors.length;
  }
  return out;
}

export function l2normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) {
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  const out = new Float32Array(v.length);
  if (norm > 0) {
    for (let i = 0; i < v.length; i++) {
      out[i] = v[i] / norm;
    }
  }
  return out;
}

/**
 * Greedy centroid grouping in the given (time) order: a face joins the first
 * group whose centroid similarity is >= threshold, otherwise it starts a new
 * group. The representative is the highest-quality member.
 */
export function groupWithinVideo<T extends GroupableFace>(faces: T[], threshold: number): WithinVideoGroup<T>[] {
  const groups: { members: T[]; centroid: Float32Array }[] = [];
  faces.forEach((face) => {
    let placed = false;
    for (let g = 0; g < groups.length; g++) {
      if (dot(face.embedding, groups[g].centroid) >= threshold) {
        groups[g].members.push(face);
        groups[g].centroid = l2normalize(meanVec(groups[g].members.map((m) => m.embedding)));
        placed = true;
        break;
      }
    }
    if (!placed) {
      groups.push({ members: [face], centroid: face.embedding });
    }
  });
  return groups.map((g) => {
    let representative = g.members[0];
    g.members.forEach((m) => {
      if (m.quality > representative.quality) {
        representative = m;
      }
    });
    return { members: g.members, representative };
  });
}

// ============================================================================
// Global (cross-video) clustering (design doc §5.6)
//
// Every (video, within-video group) is assigned to a person with a
// leader-follower pass in representative-quality order. A group joins a
// person only when its representative embedding is at least `threshold`
// similar to BOTH the person's representative embedding and the person's
// centroid; requiring both is the countermeasure against the centroid
// drift contamination measured in Phase 0 at low thresholds (persons
// absorbing unrelated groups through a drifted centroid even when the
// representatives themselves are unrelated).
// ============================================================================

import { faceGroupKey, type FaceIndexFile, type FaceRecord, type PersonRecord } from './face-types';

export interface GlobalGroupItem {
  key: string;                    // "<videoHash>#<groupId>"
  videoHash: string;
  groupId: number;
  repFaceId: string;
  repEmbedding: Float32Array;
  repQuality: number;
  weight: number;                 // number of faces in the group
  faceIds: string[];
}

export interface GlobalCluster {
  items: GlobalGroupItem[];
  repItem: GlobalGroupItem;       // highest-quality item (created the cluster)
  repEmbedding: Float32Array;     // == repItem.repEmbedding
  centroid: Float32Array;         // weighted mean of member rep embeddings
}

/** Collect one item per (video, group) using the stored representative face. */
export function collectGroupItems(
  index: FaceIndexFile,
  readEmbedding: (embeddingIdx: number) => Float32Array,
): GlobalGroupItem[] {
  const items: GlobalGroupItem[] = [];
  Object.keys(index.videos).forEach((hash) => {
    const entry = index.videos[hash];
    if (!entry || entry.status === 'error') {
      return;
    }
    const byGroup = new Map<number, FaceRecord[]>();
    entry.faces.forEach((face) => {
      if (face.groupId < 0) {
        return;
      }
      if (!byGroup.has(face.groupId)) {
        byGroup.set(face.groupId, []);
      }
      byGroup.get(face.groupId).push(face);
    });
    byGroup.forEach((members, groupId) => {
      let rep = members[0];
      members.forEach((m) => {
        if (m.representative) {
          rep = m;
        }
      });
      if (!rep.representative) {
        members.forEach((m) => {
          if (m.quality > rep.quality) {
            rep = m;
          }
        });
      }
      items.push({
        key: hash + '#' + groupId,
        videoHash: hash,
        groupId,
        repFaceId: rep.faceId,
        repEmbedding: readEmbedding(rep.embeddingIdx),
        repQuality: rep.quality,
        weight: members.length,
        faceIds: members.map((m) => m.faceId),
      });
    });
  });
  return items;
}

/**
 * Leader-follower assignment. Deterministic: items are processed in
 * representative-quality order (ties broken by group key).
 */
export function clusterGlobalGroups(items: GlobalGroupItem[], threshold: number): GlobalCluster[] {
  const sorted = [...items].sort((a, b) => {
    if (b.repQuality !== a.repQuality) {
      return b.repQuality - a.repQuality;
    }
    return a.key < b.key ? -1 : 1;
  });
  const clusters: {
    items: GlobalGroupItem[];
    repItem: GlobalGroupItem;
    repEmbedding: Float32Array;
    centroid: Float32Array;
    repEmbs: Float32Array[];
    weights: number[];
  }[] = [];
  sorted.forEach((item) => {
    let best: (typeof clusters)[number] | null = null;
    let bestScore = -Infinity;
    clusters.forEach((cluster) => {
      const score = Math.min(
        dot(item.repEmbedding, cluster.repEmbedding),
        dot(item.repEmbedding, cluster.centroid),
      );
      if (score > bestScore) {
        bestScore = score;
        best = cluster;
      }
    });
    if (best && bestScore >= threshold) {
      best.items.push(item);
      best.repEmbs.push(item.repEmbedding);
      best.weights.push(item.weight);
      best.centroid = meanVecWeighted(best.repEmbs, best.weights);
    } else {
      clusters.push({
        items: [item],
        repItem: item,
        repEmbedding: item.repEmbedding,
        centroid: item.repEmbedding,
        repEmbs: [item.repEmbedding],
        weights: [item.weight],
      });
    }
  });
  return clusters.map((c) => ({
    items: c.items,
    repItem: c.repItem,
    repEmbedding: c.repEmbedding,
    centroid: c.centroid,
  }));
}

function meanVecWeighted(vectors: Float32Array[], weights: number[]): Float32Array {
  const dim = vectors[0].length;
  const out = new Float32Array(dim);
  let wsum = 0;
  vectors.forEach((v, vi) => {
    wsum += weights[vi];
    for (let i = 0; i < dim; i++) {
      out[i] += v[i] * weights[vi];
    }
  });
  for (let i = 0; i < dim; i++) {
    out[i] /= wsum;
  }
  return l2normalize(out);
}

export interface ClusterizeResult {
  persons: PersonRecord[];
  facePerson: Map<string, string>; // faceId -> personId
  nextPersonSeq: number;
  // Manual-edit pins after pruning (entries naming groups or persons
  // that no longer exist are dropped). The store writes these back.
  manualGroups: Record<string, string>;
  manualUnassigned: string[];
  manualRepFaces: Record<string, string>;
}

function personSeqOf(personId: string): number {
  const n = parseInt(personId.substring(1), 10);
  return isNaN(n) ? 0 : n;
}

function formatPersonId(seq: number): string {
  return 'p' + String(seq).padStart(6, '0');
}

/**
 * Full re-clustering of an index: cluster all (video, group) items, then
 * reconcile with the previous persons so identities survive.
 *
 * ID inheritance: a new cluster takes over a previous person's ID (and
 * with it the user-assigned name and hidden flag) when their
 * representative embeddings are at least `threshold` similar -- greedy
 * best-first, one-to-one. This subsumes the design's "same representative
 * faceId" rule (identical representative faces have similarity 1).
 * Unmatched clusters receive fresh IDs from a monotonically increasing
 * sequence, so IDs of vanished persons are never recycled (stale .vha2
 * caches can never point at a different person). Attribute aggregates are
 * inherited only when the matched person's group set is unchanged.
 */
export function clusterizeIndex(
  index: FaceIndexFile,
  readEmbedding: (embeddingIdx: number) => Float32Array,
  threshold: number,
): ClusterizeResult {
  const items = collectGroupItems(index, readEmbedding);

  // Manual-edit pins (Phase 3): pinned groups are taken out of the free
  // clustering entirely and rebuilt under their pinned person verbatim;
  // unassigned-pinned groups get no person at all.
  const manualGroups = index.manualGroups || {};
  const manualUnassigned = new Set(index.manualUnassigned || []);
  const manualRepFaces = index.manualRepFaces || {};
  const pinnedByPerson = new Map<string, GlobalGroupItem[]>();
  const freeItems: GlobalGroupItem[] = [];
  items.forEach((item) => {
    if (manualUnassigned.has(item.key)) {
      return;
    }
    const target = manualGroups[item.key];
    if (target) {
      if (!pinnedByPerson.has(target)) {
        pinnedByPerson.set(target, []);
      }
      pinnedByPerson.get(target).push(item);
    } else {
      freeItems.push(item);
    }
  });
  const pinnedPersonIds = new Set<string>();
  pinnedByPerson.forEach((_pItems, personId) => pinnedPersonIds.add(personId));

  const clusters = clusterGlobalGroups(freeItems, threshold);

  // Index the stored faces for representative lookups
  const faceById = new Map<string, FaceRecord>();
  Object.keys(index.videos).forEach((hash) => {
    const entry = index.videos[hash];
    if (entry) {
      entry.faces.forEach((face) => faceById.set(face.faceId, face));
    }
  });

  // Previous persons with their representative embedding and group set
  const prevPersons = (index.persons || []).map((p) => {
    const repFace = faceById.get(p.representativeFaceId);
    const groupKeys = new Set<string>();
    Object.keys(index.videos).forEach((hash) => {
      const entry = index.videos[hash];
      if (!entry) {
        return;
      }
      entry.faces.forEach((face) => {
        if (face.personId === p.personId && face.groupId >= 0) {
          groupKeys.add(hash + '#' + face.groupId);
        }
      });
    });
    return {
      record: p,
      repEmbedding: repFace ? readEmbedding(repFace.embeddingIdx) : null,
      groupKeys,
    };
  });

  // Greedy best-first one-to-one matching on representative similarity.
  // Pinned persons are rebuilt from their pins below, so they are not
  // eligible inheritance targets for free clusters.
  const pairs: { clusterIdx: number; prevIdx: number; sim: number }[] = [];
  clusters.forEach((cluster, clusterIdx) => {
    prevPersons.forEach((prev, prevIdx) => {
      if (pinnedPersonIds.has(prev.record.personId)) {
        return;
      }
      if (prev.repEmbedding) {
        const sim = dot(cluster.repEmbedding, prev.repEmbedding);
        if (sim >= threshold) {
          pairs.push({ clusterIdx, prevIdx, sim });
        }
      }
    });
  });
  pairs.sort((a, b) => b.sim - a.sim);
  const clusterToPrev = new Map<number, number>();
  const usedPrev = new Set<number>();
  pairs.forEach((pair) => {
    if (!clusterToPrev.has(pair.clusterIdx) && !usedPrev.has(pair.prevIdx)) {
      clusterToPrev.set(pair.clusterIdx, pair.prevIdx);
      usedPrev.add(pair.prevIdx);
    }
  });

  let nextSeq = index.nextPersonSeq || 1;
  prevPersons.forEach((prev) => {
    nextSeq = Math.max(nextSeq, personSeqOf(prev.record.personId) + 1);
  });

  const persons: PersonRecord[] = [];
  const facePerson = new Map<string, string>();
  clusters.forEach((cluster, clusterIdx) => {
    const prevIdx = clusterToPrev.get(clusterIdx);
    const prev = prevIdx !== undefined ? prevPersons[prevIdx] : null;
    const personId = prev ? prev.record.personId : formatPersonId(nextSeq++);
    const videoSet = new Set<string>();
    let faceCount = 0;
    cluster.items.forEach((item) => {
      videoSet.add(item.videoHash);
      faceCount += item.faceIds.length;
      item.faceIds.forEach((faceId) => facePerson.set(faceId, personId));
    });
    const videoHashes: string[] = [];
    videoSet.forEach((hash) => videoHashes.push(hash));
    videoHashes.sort();
    // A manual representative-face override wins over the cluster's
    // own representative while the chosen face belongs to the person.
    let representativeFaceId = cluster.repItem.repFaceId;
    const repOverride = manualRepFaces[personId];
    if (repOverride && cluster.items.some((item) => item.faceIds.includes(repOverride))) {
      representativeFaceId = repOverride;
    }
    const record: PersonRecord = {
      personId,
      name: prev ? prev.record.name : '',
      representativeFaceId,
      videoHashes,
      faceCount,
      hidden: prev ? prev.record.hidden : false,
    };
    // Keep the aggregates only when the person's membership is unchanged;
    // otherwise they are stale and recomputed by the attribute pass.
    if (prev && prev.record.attributes) {
      const newKeys = new Set<string>();
      cluster.items.forEach((item) => newKeys.add(item.key));
      let same = newKeys.size === prev.groupKeys.size;
      if (same) {
        newKeys.forEach((key) => {
          if (!prev.groupKeys.has(key)) {
            same = false;
          }
        });
      }
      if (same) {
        record.attributes = prev.record.attributes;
      }
    }
    persons.push(record);
  });

  // Pinned persons: rebuilt from their pinned groups verbatim. Name and
  // hidden flag come from the previous record with the same ID (direct
  // lookup, no similarity matching); the representative is the manual
  // override when set, else the highest-quality pinned group's rep.
  const prevById = new Map<string, (typeof prevPersons)[number]>();
  prevPersons.forEach((prev) => prevById.set(prev.record.personId, prev));
  const pinnedIds: string[] = [];
  pinnedByPerson.forEach((_pItems, personId) => pinnedIds.push(personId));
  pinnedIds.sort();
  pinnedIds.forEach((personId) => {
    const pItems = pinnedByPerson.get(personId);
    const prev = prevById.get(personId) || null;
    const videoSet = new Set<string>();
    let faceCount = 0;
    pItems.forEach((item) => {
      videoSet.add(item.videoHash);
      faceCount += item.faceIds.length;
      item.faceIds.forEach((faceId) => facePerson.set(faceId, personId));
    });
    const videoHashes: string[] = [];
    videoSet.forEach((hash) => videoHashes.push(hash));
    videoHashes.sort();
    let repItem = pItems[0];
    pItems.forEach((item) => {
      if (item.repQuality > repItem.repQuality) {
        repItem = item;
      }
    });
    let representativeFaceId = repItem.repFaceId;
    const repOverride = manualRepFaces[personId];
    if (repOverride && pItems.some((item) => item.faceIds.includes(repOverride))) {
      representativeFaceId = repOverride;
    }
    const record: PersonRecord = {
      personId,
      name: prev ? prev.record.name : '',
      representativeFaceId,
      videoHashes,
      faceCount,
      hidden: prev ? prev.record.hidden : false,
    };
    // Same rule as free clusters: aggregates survive only an unchanged
    // group set (a pinned person's set is unchanged by construction,
    // unless a pinned group vanished, e.g. its video was re-scanned).
    if (prev && prev.record.attributes) {
      const newKeys = new Set<string>();
      pItems.forEach((item) => newKeys.add(item.key));
      let same = newKeys.size === prev.groupKeys.size;
      if (same) {
        newKeys.forEach((key) => {
          if (!prev.groupKeys.has(key)) {
            same = false;
          }
        });
      }
      if (same) {
        record.attributes = prev.record.attributes;
      }
    }
    persons.push(record);
    nextSeq = Math.max(nextSeq, personSeqOf(personId) + 1);
  });

  // Prune pins that name groups or persons which no longer exist (a
  // group disappears when its video is re-scanned or removed).
  const resultPersonIds = new Set(persons.map((p) => p.personId));
  const itemKeys = new Set(items.map((item) => item.key));
  const prunedManualGroups: Record<string, string> = {};
  Object.keys(manualGroups).forEach((key) => {
    const target = manualGroups[key];
    if (itemKeys.has(key) && resultPersonIds.has(target)) {
      prunedManualGroups[key] = target;
    }
  });
  const prunedUnassigned: string[] = [];
  manualUnassigned.forEach((key) => {
    if (itemKeys.has(key)) {
      prunedUnassigned.push(key);
    }
  });
  const prunedRepFaces: Record<string, string> = {};
  Object.keys(manualRepFaces).forEach((personId) => {
    const faceId = manualRepFaces[personId];
    if (resultPersonIds.has(personId) && facePerson.get(faceId) === personId) {
      prunedRepFaces[personId] = faceId;
    }
  });

  return {
    persons,
    facePerson,
    nextPersonSeq: nextSeq,
    manualGroups: prunedManualGroups,
    manualUnassigned: prunedUnassigned,
    manualRepFaces: prunedRepFaces,
  };
}

// ============================================================================
// Manual corrections (Phase 3): merges, splits, group reassignment and
// representative-face overrides.
//
// These functions only manipulate the index's pins (and the person
// records an edit unambiguously creates or destroys). They never touch
// face -> person assignments: the store materializes every edit by
// re-running clusterizeIndex, which treats the pins as ground truth.
// That keeps exactly one code path responsible for assignments.
//
// Freezing rule: an edit pins every group of every person it touches,
// so a manually edited person's composition is fixed until the next
// edit. Unpinned persons keep clustering (and inheriting IDs) as usual.
// ============================================================================

function ensurePins(index: FaceIndexFile): void {
  if (!index.manualGroups) {
    index.manualGroups = {};
  }
  if (!index.manualUnassigned) {
    index.manualUnassigned = [];
  }
  if (!index.manualRepFaces) {
    index.manualRepFaces = {};
  }
}

function unpinUnassigned(index: FaceIndexFile, key: string): void {
  if (index.manualUnassigned) {
    index.manualUnassigned = index.manualUnassigned.filter((k) => k !== key);
  }
}

function pinUnassigned(index: FaceIndexFile, key: string): void {
  if (index.manualGroups) {
    delete index.manualGroups[key];
  }
  if (!index.manualUnassigned) {
    index.manualUnassigned = [];
  }
  if (!index.manualUnassigned.includes(key)) {
    index.manualUnassigned.push(key);
  }
}

/** Group keys ("<videoHash>#<groupId>") currently assigned to a person. */
export function personGroupKeys(index: FaceIndexFile, personId: string): string[] {
  const keys = new Set<string>();
  Object.keys(index.videos).forEach((hash) => {
    const entry = index.videos[hash];
    if (!entry) {
      return;
    }
    entry.faces.forEach((face) => {
      if (face.personId === personId && face.groupId >= 0) {
        keys.add(faceGroupKey(hash, face.groupId));
      }
    });
  });
  const out: string[] = [];
  keys.forEach((key) => out.push(key));
  return out;
}

function groupExists(index: FaceIndexFile, key: string): boolean {
  const sep = key.lastIndexOf('#');
  if (sep < 0) {
    return false;
  }
  const entry = index.videos[key.substring(0, sep)];
  if (!entry) {
    return false;
  }
  const groupId = parseInt(key.substring(sep + 1), 10);
  return entry.faces.some((face) => face.groupId === groupId);
}

function groupOwner(index: FaceIndexFile, key: string): string | null {
  const sep = key.lastIndexOf('#');
  if (sep < 0) {
    return null;
  }
  const entry = index.videos[key.substring(0, sep)];
  if (!entry) {
    return null;
  }
  const groupId = parseInt(key.substring(sep + 1), 10);
  const face = entry.faces.filter((f) => f.groupId === groupId && f.personId)[0];
  return face ? face.personId : null;
}

function personRecord(index: FaceIndexFile, personId: string): PersonRecord | undefined {
  return (index.persons || []).filter((p) => p.personId === personId)[0];
}

/**
 * Merge 2+ persons into one. Survivor rule: the first person in
 * `personIds` that has a user-assigned name wins (merging named people
 * keeps a name); when none is named, the first listed person survives.
 * The survivor keeps its name, hidden flag and representative override;
 * the other persons' records and overrides are removed, and their IDs
 * are never reused (the sequence in the index only moves forward).
 * Returns the survivor's ID, or null when fewer than 2 persons exist.
 */
export function mergePersonsInIndex(index: FaceIndexFile, personIds: string[]): string | null {
  const ids: string[] = [];
  personIds.forEach((id) => {
    if (personRecord(index, id) && !ids.includes(id)) {
      ids.push(id);
    }
  });
  if (ids.length < 2) {
    return null;
  }
  const named = ids.filter((id) => (personRecord(index, id).name || '').trim() !== '');
  const survivorId = named.length > 0 ? named[0] : ids[0];
  const sources = ids.filter((id) => id !== survivorId);
  ensurePins(index);
  const keys = new Set<string>();
  ids.forEach((id) => personGroupKeys(index, id).forEach((key) => keys.add(key)));
  keys.forEach((key) => {
    index.manualGroups[key] = survivorId;
    unpinUnassigned(index, key);
  });
  // Pins created by earlier edits may still point at a merged-away
  // person for groups that person no longer owns; follow the merge.
  Object.keys(index.manualGroups).forEach((key) => {
    if (sources.includes(index.manualGroups[key])) {
      index.manualGroups[key] = survivorId;
    }
  });
  sources.forEach((id) => {
    delete index.manualRepFaces[id];
  });
  index.persons = (index.persons || []).filter((p) => !sources.includes(p.personId));
  return survivorId;
}

/**
 * Split the listed groups of a person off into a new person with ID
 * `newPersonId` (minted by the caller from the index sequence). The
 * moved groups are pinned to the new person and the person's remaining
 * groups are pinned to it (both sides frozen). A split must move a
 * proper subset: moving every group is a move, not a split.
 */
export function splitPersonInIndex(
  index: FaceIndexFile,
  personId: string,
  groupKeys: string[],
  newPersonId: string,
): boolean {
  const record = personRecord(index, personId);
  if (!record || personRecord(index, newPersonId)) {
    return false;
  }
  const owned = personGroupKeys(index, personId);
  const moving: string[] = [];
  groupKeys.forEach((key) => {
    if (owned.includes(key) && !moving.includes(key)) {
      moving.push(key);
    }
  });
  if (moving.length === 0 || moving.length === owned.length) {
    return false;
  }
  ensurePins(index);
  moving.forEach((key) => {
    index.manualGroups[key] = newPersonId;
    unpinUnassigned(index, key);
  });
  owned.filter((key) => !moving.includes(key)).forEach((key) => {
    index.manualGroups[key] = personId;
  });
  index.persons.push({
    personId: newPersonId,
    name: '',
    representativeFaceId: '',
    videoHashes: [],
    faceCount: 0,
    hidden: false,
  });
  // A representative override pointing at a face in a moved group no
  // longer belongs to the original person; drop it.
  const repOverride = index.manualRepFaces[personId];
  if (repOverride) {
    const repFace = findFace(index, repOverride);
    if (repFace && moving.includes(faceGroupKey(repFace.hash, repFace.face.groupId))) {
      delete index.manualRepFaces[personId];
    }
  }
  return true;
}

/**
 * Move groups to another person (`targetPersonId`) or back to
 * unassigned (`null`): the moved groups are pinned to their target,
 * and both the source persons' remaining groups and the target
 * person's existing groups are pinned (frozen) as well.
 */
export function moveGroupsInIndex(
  index: FaceIndexFile,
  groupKeys: string[],
  targetPersonId: string | null,
): boolean {
  if (targetPersonId !== null && !personRecord(index, targetPersonId)) {
    return false;
  }
  const valid: string[] = [];
  groupKeys.forEach((key) => {
    if (groupExists(index, key) && !valid.includes(key)) {
      valid.push(key);
    }
  });
  if (valid.length === 0) {
    return false;
  }
  ensurePins(index);
  const sources = new Set<string>();
  valid.forEach((key) => {
    const owner = groupOwner(index, key);
    if (owner) {
      sources.add(owner);
    }
  });
  sources.forEach((sourceId) => {
    if (sourceId === targetPersonId) {
      return;
    }
    personGroupKeys(index, sourceId)
      .filter((key) => !valid.includes(key))
      .forEach((key) => {
        index.manualGroups[key] = sourceId;
        unpinUnassigned(index, key);
      });
  });
  if (targetPersonId) {
    personGroupKeys(index, targetPersonId).forEach((key) => {
      index.manualGroups[key] = targetPersonId;
      unpinUnassigned(index, key);
    });
  }
  valid.forEach((key) => {
    if (targetPersonId) {
      index.manualGroups[key] = targetPersonId;
      unpinUnassigned(index, key);
    } else {
      pinUnassigned(index, key);
    }
  });
  return true;
}

function findFace(index: FaceIndexFile, faceId: string): { hash: string; face: FaceRecord } | null {
  let found: { hash: string; face: FaceRecord } | null = null;
  Object.keys(index.videos).forEach((hash) => {
    const entry = index.videos[hash];
    if (!entry || found) {
      return;
    }
    const face = entry.faces.filter((f) => f.faceId === faceId)[0];
    if (face) {
      found = { hash, face };
    }
  });
  return found;
}

/**
 * Pin a person's representative face. The face must currently belong
 * to the person. The override is applied to the record immediately and
 * re-applied by every later re-clustering while the face stays with
 * the person. Only group-representative faces have a display crop on
 * disk, so the UI offers those; the data layer accepts any face of
 * the person.
 */
export function setPersonRepresentativeInIndex(index: FaceIndexFile, personId: string, faceId: string): boolean {
  const record = personRecord(index, personId);
  if (!record) {
    return false;
  }
  const found = findFace(index, faceId);
  if (!found || found.face.personId !== personId) {
    return false;
  }
  ensurePins(index);
  index.manualRepFaces[personId] = faceId;
  record.representativeFaceId = faceId;
  return true;
}
