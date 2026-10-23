// Clustering for the face spike - the design doc §5.5 / §5.6 algorithms,
// kept as pure functions over embedding arrays so the same code can move
// into the app (node/face/cluster.ts) in Phase 2.
//
// §5.5 within-video: greedy grouping by cosine similarity to the group
// centroid (running mean of member embeddings, renormalized), faces
// considered best-quality first. Representative = highest-quality face.
// §5.6 global: leader-follower. Each (video, group) is compared by its
// representative embedding against person centroids (weighted mean of
// representative embeddings, weight = group size, renormalized). Assign
// to the best person when similarity >= T_global, otherwise create one.

export type Vec = Float32Array | number[];

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function meanVec(vecs: Vec[], weights?: number[]): Float32Array {
  const dim = vecs[0].length;
  const out = new Float32Array(dim);
  let wsum = 0;
  for (let i = 0; i < vecs.length; i++) {
    const w = weights ? weights[i] : 1;
    wsum += w;
    for (let j = 0; j < dim; j++) out[j] += vecs[i][j] * w;
  }
  let norm = 0;
  for (let j = 0; j < dim; j++) {
    out[j] /= wsum;
    norm += out[j] * out[j];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (let j = 0; j < dim; j++) out[j] /= norm;
  return out;
}

export interface FaceItem {
  faceId: string;
  embedding: number[];
  quality: number;
}

export interface VideoGroup<T extends FaceItem> {
  members: T[];
  representative: T;
  centroid: Float32Array;
}

/** §5.5: group faces of one video. Returns groups sorted by size desc. */
export function groupWithinVideo<T extends FaceItem>(
  faces: T[],
  tIn: number,
): VideoGroup<T>[] {
  const sorted = [...faces].sort((a, b) => b.quality - a.quality);
  const groups: VideoGroup<T>[] = [];
  for (const f of sorted) {
    let best: VideoGroup<T> | null = null;
    let bestSim = -Infinity;
    for (const g of groups) {
      const s = dot(f.embedding, g.centroid);
      if (s > bestSim) {
        bestSim = s;
        best = g;
      }
    }
    if (best && bestSim >= tIn) {
      best.members.push(f);
      best.centroid = meanVec(best.members.map((m) => m.embedding));
      if (f.quality > best.representative.quality) best.representative = f;
    } else {
      groups.push({
        members: [f],
        representative: f,
        centroid: meanVec([f.embedding]),
      });
    }
  }
  groups.sort(
    (a, b) =>
      b.members.length - a.members.length ||
      b.representative.quality - a.representative.quality,
  );
  return groups;
}

export interface PersonCluster<G> {
  personId: string; // p000001 ...
  groups: G[];
  centroid: Float32Array;
}

/**
 * §5.6: leader-follower assignment of video groups to persons.
 * `items` must be in deterministic processing order; each item carries
 * the group's representative embedding and its weight (group size).
 */
export function clusterGlobal<G>(
  items: { group: G; repEmbedding: number[]; weight: number }[],
  tGlobal: number,
): PersonCluster<G>[] {
  const persons: PersonCluster<G>[] = [];
  const repEmbs = new Map<PersonCluster<G>, number[][]>();
  const repWeights = new Map<PersonCluster<G>, number[]>();
  for (const it of items) {
    let best: PersonCluster<G> | null = null;
    let bestSim = -Infinity;
    for (const p of persons) {
      const s = dot(it.repEmbedding, p.centroid);
      if (s > bestSim) {
        bestSim = s;
        best = p;
      }
    }
    if (best && bestSim >= tGlobal) {
      best.groups.push(it.group);
      repEmbs.get(best)!.push(it.repEmbedding);
      repWeights.get(best)!.push(it.weight);
      best.centroid = meanVec(repEmbs.get(best)!, repWeights.get(best)!);
    } else {
      const p: PersonCluster<G> = {
        personId: `p${String(persons.length + 1).padStart(6, "0")}`,
        groups: [it.group],
        centroid: meanVec([it.repEmbedding]),
      };
      persons.push(p);
      repEmbs.set(p, [it.repEmbedding]);
      repWeights.set(p, [it.weight]);
    }
  }
  return persons;
}
