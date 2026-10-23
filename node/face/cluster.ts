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
