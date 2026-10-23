// Within-video face grouping (design doc §5.5), ported from the Phase 0
// spike (tools/face-spike/src/cluster.ts). Global (cross-video) clustering
// (§5.6) is Phase 2 and intentionally not included here.

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

import type { FaceIndexFile, FaceRecord, PersonRecord } from './face-types';

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
  const clusters = clusterGlobalGroups(items, threshold);

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

  // Greedy best-first one-to-one matching on representative similarity
  const pairs: { clusterIdx: number; prevIdx: number; sim: number }[] = [];
  clusters.forEach((cluster, clusterIdx) => {
    prevPersons.forEach((prev, prevIdx) => {
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
    const record: PersonRecord = {
      personId,
      name: prev ? prev.record.name : '',
      representativeFaceId: cluster.repItem.repFaceId,
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

  return { persons, facePerson, nextPersonSeq: nextSeq };
}
