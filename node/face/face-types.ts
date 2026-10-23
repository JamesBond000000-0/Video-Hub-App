import type { FaceModelPack } from '../../interfaces/final-object.interface';

// ============================================================================
// Face recognition: shared types (design doc §6)
// The face index lives in the hub folder: faces/index.json + embeddings.bin
// ============================================================================

export interface FaceModelPackInfo {
  pack: FaceModelPack;
  modelId: string;          // recorded in index.json; mismatch invalidates the index
  detFile: string;          // SCRFD detector model file name
  recFile: string;          // ArcFace recognition model file name
}

// Design doc Appendix A: only the det + rec models of each pack are used
export const FACE_MODEL_PACKS: Record<FaceModelPack, FaceModelPackInfo> = {
  quality: {
    pack: 'quality',
    modelId: 'buffalo_l-r50',
    detFile: 'det_10g.onnx',
    recFile: 'w600k_r50.onnx',
  },
  light: {
    pack: 'light',
    modelId: 'buffalo_sc-mbf',
    detFile: 'det_500m.onnx',
    recFile: 'w600k_mbf.onnx',
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
  persons: unknown[];                     // Phase 2 (global clustering) fills this
}

export interface FaceIndexPayload {
  index: FaceIndexFile | null;    // null when no face data exists yet
  modelMismatch: boolean;         // index was built with a different model pack
  stats: FaceStats;
}

export interface FaceStats {
  scannedVideos: number;
  videosWithFaces: number;
  totalFaces: number;
  totalGroups: number;
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
  };
}

export function computeFaceStats(index: FaceIndexFile | null): FaceStats {
  const stats: FaceStats = { scannedVideos: 0, videosWithFaces: 0, totalFaces: 0, totalGroups: 0 };
  if (!index) {
    return stats;
  }
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
