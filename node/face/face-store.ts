import * as fs from 'fs';
import * as path from 'path';

import {
  computeFaceStats,
  emptyFaceIndex,
  EMBEDDING_BYTES,
  type FaceIndexFile,
  type FaceIndexPayload,
  type FaceRecord,
  type FaceVideoEntry,
  type ScannedFace,
} from './face-types';

// ============================================================================
// FaceStore: persistence for the face index (design doc §6)
//   <hub>/faces/index.json     - all metadata (atomic tmp+rename writes)
//   <hub>/faces/embeddings.bin - Float32 embeddings, append-only
//   <hub>/faces/aligned/       - 112x112 aligned crops (every face)
//   <hub>/faces/crops/         - display crops (group representatives only)
// ============================================================================

export class FaceStore {
  readonly facesDir: string;
  readonly modelMismatch: boolean;
  private index: FaceIndexFile;
  private embeddingsStored: number;

  private constructor(facesDir: string, index: FaceIndexFile, embeddingsStored: number, modelMismatch: boolean) {
    this.facesDir = facesDir;
    this.index = index;
    this.embeddingsStored = embeddingsStored;
    this.modelMismatch = modelMismatch;
  }

  static indexPath(facesDir: string): string {
    return path.join(facesDir, 'index.json');
  }

  static embeddingsPath(facesDir: string): string {
    return path.join(facesDir, 'embeddings.bin');
  }

  static alignedDir(facesDir: string): string {
    return path.join(facesDir, 'aligned');
  }

  static cropsDir(facesDir: string): string {
    return path.join(facesDir, 'crops');
  }

  /**
   * Load the index for a hub (read-only usage is fine). When the stored
   * modelId differs from `modelId`, the store is flagged as mismatch and
   * setVideo() refuses to write (design §6.1: treat the index as invalid).
   */
  static load(facesDir: string, modelId: string, clusterThreshold: number): FaceStore {
    let index = emptyFaceIndex(modelId, clusterThreshold);
    let mismatch = false;
    const indexPath = FaceStore.indexPath(facesDir);
    if (fs.existsSync(indexPath)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(indexPath, 'utf8')) as FaceIndexFile;
        if (parsed && parsed.version === 1 && parsed.videos) {
          index = parsed;
          mismatch = parsed.modelId !== modelId;
        }
      } catch (err) {
        console.log('face index unreadable, starting fresh:', err);
      }
    }
    let embeddingsStored = 0;
    const binPath = FaceStore.embeddingsPath(facesDir);
    if (fs.existsSync(binPath)) {
      embeddingsStored = Math.floor(fs.statSync(binPath).size / EMBEDDING_BYTES);
    }
    return new FaceStore(facesDir, index, embeddingsStored, mismatch);
  }

  /** Whether any face data exists on disk for this hub. */
  static hasData(facesDir: string): boolean {
    return fs.existsSync(FaceStore.indexPath(facesDir));
  }

  /**
   * Wipe all face data for a hub (used when the model pack changed: embeddings
   * from different recognition models are mutually incompatible, §4.3).
   */
  static reset(facesDir: string, modelId: string, clusterThreshold: number): FaceStore {
    [FaceStore.indexPath(facesDir), FaceStore.embeddingsPath(facesDir),
     FaceStore.alignedDir(facesDir), FaceStore.cropsDir(facesDir)].forEach((target) => {
      try {
        fs.rmSync(target, { recursive: true, force: true });
      } catch (err) {
        console.log('face store reset failed for', target, err);
      }
    });
    const store = new FaceStore(facesDir, emptyFaceIndex(modelId, clusterThreshold), 0, false);
    store.saveIndex();
    return store;
  }

  getIndex(): FaceIndexFile {
    return this.index;
  }

  getVideoEntry(hash: string): FaceVideoEntry | undefined {
    return this.index.videos[hash];
  }

  isScanned(hash: string): boolean {
    const entry = this.index.videos[hash];
    return !!entry && entry.status !== 'error';
  }

  /**
   * Persist one video's scan result: appends embeddings (assigning each face
   * its embeddingIdx) and rewrites index.json atomically.
   */
  setVideo(hash: string, status: 'done' | 'none' | 'error', frameCount: number, faces: ScannedFace[], error?: string): void {
    if (this.modelMismatch) {
      throw new Error('face index was built with a different model pack; reset required before writing');
    }
    const records: FaceRecord[] = [];
    if (faces.length > 0) {
      const buffers: Buffer[] = [];
      faces.forEach((face) => {
        records.push({
          faceId: face.faceId,
          t: face.t,
          bbox: face.bbox,
          score: face.score,
          quality: face.quality,
          embeddingIdx: this.embeddingsStored + buffers.length,
          groupId: face.groupId,
          representative: face.representative,
        });
        buffers.push(Buffer.from(face.embedding.buffer, face.embedding.byteOffset, face.embedding.byteLength));
      });
      fs.mkdirSync(this.facesDir, { recursive: true });
      fs.appendFileSync(FaceStore.embeddingsPath(this.facesDir), Buffer.concat(buffers));
      this.embeddingsStored += buffers.length;
    }
    const entry: FaceVideoEntry = {
      status,
      scannedAt: Date.now(),
      frameCount,
      faces: records,
    };
    if (error) {
      entry.error = error;
    }
    this.index.videos[hash] = entry;
    this.saveIndex();
  }

  /** Record a failed scan without embeddings. */
  setVideoError(hash: string, frameCount: number, error: string): void {
    this.setVideo(hash, 'error', frameCount, [], error);
  }

  payload(): FaceIndexPayload {
    return {
      index: FaceStore.hasData(this.facesDir) || Object.keys(this.index.videos).length > 0 ? this.index : null,
      modelMismatch: this.modelMismatch,
      stats: computeFaceStats(this.modelMismatch ? null : this.index),
    };
  }

  private saveIndex(): void {
    fs.mkdirSync(this.facesDir, { recursive: true });
    const tmpPath = FaceStore.indexPath(this.facesDir) + '.tmp';
    fs.writeFileSync(tmpPath, JSON.stringify(this.index));
    fs.renameSync(tmpPath, FaceStore.indexPath(this.facesDir));
  }
}
