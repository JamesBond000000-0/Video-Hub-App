import type { FaceModelPack } from '../../interfaces/final-object.interface';

// ============================================================================
// Face recognition: shared types (design doc §6)
// The face index lives in the hub folder: faces/index.json + embeddings.bin
// ============================================================================

// Which recognizer implementation consumes a pack's recFile. The two
// implementations share the input contract (aligned 112x112 crop,
// (px - 127.5) / 127.5, 512-d L2-normalized output) but their embedding
// spaces are incompatible, hence the distinct modelIds below.
export type RecognizerKind = 'arcface' | 'lvface';

export interface FaceModelPackInfo {
  pack: FaceModelPack;
  modelId: string;          // recorded in index.json; mismatch invalidates the index
  detFile: string;          // SCRFD detector model file name
  recFile: string;          // recognition model file name
  recKind: RecognizerKind;  // recognizer implementation for recFile
}

// Design doc Appendix A: only the det + rec models of each pack are used
export const FACE_MODEL_PACKS: Record<FaceModelPack, FaceModelPackInfo> = {
  quality: {
    pack: 'quality',
    modelId: 'buffalo_l-r50',
    detFile: 'det_10g.onnx',
    recFile: 'w600k_r50.onnx',
    recKind: 'arcface',
  },
  light: {
    pack: 'light',
    modelId: 'buffalo_sc-mbf',
    detFile: 'det_500m.onnx',
    recFile: 'w600k_mbf.onnx',
    recKind: 'arcface',
  },
  // Decision D8: detection is identical to quality (det_10g); recognition
  // is LVFace-T (ByteDance, Glint360K weights). The recFile name is the
  // file name as distributed on HuggingFace (bytedance-research/LVFace)
  // so users can place the download as-is.
  lvface: {
    pack: 'lvface',
    modelId: 'lvface-t-glint360k',
    detFile: 'det_10g.onnx',
    recFile: 'LVFace-T_Glint360K.onnx',
    recKind: 'lvface',
  },
};

export interface FaceModelFileStatus {
  name: string;
  present: boolean;
}

export interface FaceModelStatus {
  pack: FaceModelPack;
  modelId: string;
  modelsDir: string;              // <userData>/face-models/<pack>/ (design D2)
  files: FaceModelFileStatus[];   // files the user must place in modelsDir
  ready: boolean;                 // all required files present
  genderagePath: string;          // resolved genderage.onnx path ('' when not found)
  genderagePresent: boolean;      // genderage.onnx found (needed only for §5.8 attributes)
}

export interface FaceRecord {
  faceId: string;                 // "<videoHash>-<4 digit sequence>"
  t: number;                      // appearance time (seconds into the video)
  bbox: [number, number, number, number]; // relative x, y, w, h (0..1 of frame)
  score: number;                  // detection score 0..1
  quality: number;                // score * sqrt(face area / frame area)
  embeddingIdx: number;           // offset into embeddings.bin (idx * 2048 bytes)
  groupId: number;                // within-video group (§5.5); -1 = ungrouped
  representative: boolean;        // highest-quality face of its within-video group
  personId?: string;              // global cluster (§5.6); absent until clustered
}

// Person-level attribute aggregates (§5.8). Only aggregates are ever stored;
// per-face estimates exist only transiently inside the worker.
export interface PersonAttributes {
  sex: 'M' | 'F';                 // majority vote over the person's samples
  sexAgreement: number;           // share of samples agreeing with the vote (0..1)
  ageMedian: number;              // median estimated age (an estimate, ±10y class)
  sampleCount: number;            // number of faces sampled for the aggregate
}

export interface PersonRecord {
  personId: string;               // "p000001", ... (stable across re-clustering)
  name: string;                   // user-assigned name; '' = unnamed
  representativeFaceId: string;   // faceId of the person's representative face
  videoHashes: string[];          // videos this person appears in (sorted)
  faceCount: number;              // total faces across all of the person's groups
  hidden: boolean;                // hidden from the sidebar / person list
  attributes?: PersonAttributes;  // present only when attributes are enabled and computed
}

export interface FaceVideoEntry {
  status: 'done' | 'none' | 'error';
  scannedAt: number;              // epoch ms
  frameCount: number;             // frames sampled (§5.1)
  faces: FaceRecord[];
  error?: string;
}

export interface FaceIndexFile {
  version: 1;
  modelId: string;
  clusterThreshold: number;
  videos: Record<string, FaceVideoEntry>; // key: ImageElement.hash
  persons: PersonRecord[];                // global clustering result (§5.6)
  nextPersonSeq?: number;                 // next free person number (IDs never reused)
  // Manual corrections (Phase 3). Pins are the source of truth for the
  // groups they name: re-clustering rebuilds pinned persons from them
  // verbatim and never reassigns a pinned group. Editing a person pins
  // ("freezes") its whole composition, so manual work survives any
  // number of re-clusterings until another edit touches the person.
  manualGroups?: Record<string, string>;  // "<videoHash>#<groupId>" -> personId
  manualUnassigned?: string[];            // group keys pinned to "no person"
  manualRepFaces?: Record<string, string>; // personId -> representative faceId override
}

/** Key identifying one (video, within-video group) pair. */
export function faceGroupKey(videoHash: string, groupId: number): string {
  return videoHash + '#' + groupId;
}

export interface FaceIndexPayload {
  index: FaceIndexFile | null;    // null when no face data exists yet
  modelMismatch: boolean;         // index was built with a different model pack
  stats: FaceStats;
  // Derived cache for the .vha2 `ImageElement.persons` field (§6.5):
  // video hash -> person IDs appearing in it. Empty when there is no
  // (valid) face index, in which case the renderer clears the cache.
  personsByVideo: Record<string, string[]>;
}

export interface FaceStats {
  scannedVideos: number;
  videosWithFaces: number;
  totalFaces: number;
  totalGroups: number;
  totalPersons: number;
}

// One scanned face as returned by the worker (embedding not yet persisted)
export interface ScannedFace {
  faceId: string;
  t: number;
  bbox: [number, number, number, number];
  score: number;
  quality: number;
  groupId: number;
  representative: boolean;
  embedding: Float32Array;        // 512-dim, L2-normalized
}

export interface FaceScanResult {
  status: 'done' | 'none' | 'error';
  frameCount: number;
  faces: ScannedFace[];
  error?: string;
}

export interface FaceScanJob {
  videoPath: string;
  duration: number;               // seconds (ImageElement.duration)
  hash: string;
  facesDir: string;               // <hub folder>/faces
  maxFramesPerVideo: number;
  sampleIntervalSec: number;
  minFaceSizePx: number;
  detModelPath: string;
  recModelPath: string;
  recKind: RecognizerKind;
}

// Within-video grouping threshold (§5.5, decision D5: T_in = 0.50)
export const WITHIN_VIDEO_THRESHOLD = 0.5;

export const EMBEDDING_DIM = 512;
export const EMBEDDING_BYTES = EMBEDDING_DIM * 4; // float32

export function emptyFaceIndex(modelId: string, clusterThreshold: number): FaceIndexFile {
  return {
    version: 1,
    modelId,
    clusterThreshold,
    videos: {},
    persons: [],
    nextPersonSeq: 1,
  };
}

export function computeFaceStats(index: FaceIndexFile | null): FaceStats {
  const stats: FaceStats = { scannedVideos: 0, videosWithFaces: 0, totalFaces: 0, totalGroups: 0, totalPersons: 0 };
  if (!index) {
    return stats;
  }
  stats.totalPersons = (index.persons || []).length;
  Object.keys(index.videos).forEach((hash) => {
    const entry = index.videos[hash];
    if (entry.status === 'error') {
      return;
    }
    stats.scannedVideos++;
    if (entry.faces.length > 0) {
      stats.videosWithFaces++;
      stats.totalFaces += entry.faces.length;
      const groups = new Set<number>();
      entry.faces.forEach((face) => {
        if (face.groupId >= 0) {
          groups.add(face.groupId);
        }
      });
      stats.totalGroups += groups.size;
    }
  });
  return stats;
}

// ---------------------------------------------------------------------------
// Attribute estimation (§5.8): pure helpers shared by the worker (which runs
// the model) and the main process (which selects the samples). Per-face
// estimates are transient samples; only the aggregates below are persisted.
// ---------------------------------------------------------------------------

export interface AttributeSample {
  sex: 'M' | 'F';
  age: number;
}

/** Top-`limit` faces of one (video, group) by quality (§5.8 sample rule). */
export function selectGroupSamples<T extends { quality: number }>(faces: T[], limit: number): T[] {
  return [...faces].sort((a, b) => b.quality - a.quality).slice(0, Math.max(0, limit));
}

/** Aggregate per-face samples into the person-level record (§5.8). */
export function aggregateAttributes(samples: AttributeSample[]): PersonAttributes | undefined {
  if (samples.length === 0) {
    return undefined;
  }
  const ages = samples.map((s) => s.age).sort((a, b) => a - b);
  const mid = Math.floor(ages.length / 2);
  const ageMedian = ages.length % 2 === 1 ? ages[mid] : Math.round((ages[mid - 1] + ages[mid]) / 2);
  const fCount = samples.filter((s) => s.sex === 'F').length;
  const sex: 'M' | 'F' = fCount * 2 >= samples.length ? 'F' : 'M';
  const majority = Math.max(fCount, samples.length - fCount);
  return {
    sex,
    sexAgreement: Math.round((majority / samples.length) * 100) / 100,
    ageMedian,
    sampleCount: samples.length,
  };
}

// One sample face for the worker's attribute job: the worker re-extracts
// this frame from the video and runs genderage on it (reference input path;
// the stored aligned thumbnails measurably degrade the estimates).
export interface FaceAttributeSampleRequest {
  faceId: string;
  videoPath: string;
  t: number;                      // appearance time (seconds)
  bbox: [number, number, number, number]; // relative x, y, w, h (0..1 of frame)
}

export interface FaceAttributesJob {
  facesDir: string;               // <hub folder>/faces (temp frame location)
  genderageModelPath: string;
  persons: { personId: string; faces: FaceAttributeSampleRequest[] }[];
}

export interface FaceAttributesResult {
  results: { personId: string; attributes: PersonAttributes }[];
  error?: string;
}
