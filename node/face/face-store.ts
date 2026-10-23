import * as fs from 'fs';
import * as path from 'path';

import type { ClusterizeResult } from './cluster';
import {
  computeFaceStats,
  emptyFaceIndex,
  EMBEDDING_BYTES,
  type FaceIndexFile,
  type FaceIndexPayload,
  type FaceRecord,
  type FaceVideoEntry,
  type PersonAttributes,
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

  /** Read one stored embedding (512 float32 at `embeddingIdx * 2048`). */
  readEmbedding(embeddingIdx: number): Float32Array {
    const binPath = FaceStore.embeddingsPath(this.facesDir);
    const buffer = Buffer.alloc(EMBEDDING_BYTES);
    const fd = fs.openSync(binPath, 'r');
    try {
      fs.readSync(fd, buffer, 0, EMBEDDING_BYTES, embeddingIdx * EMBEDDING_BYTES);
    } finally {
      fs.closeSync(fd);
    }
    const out = new Float32Array(EMBEDDING_BYTES / 4);
    for (let i = 0; i < out.length; i++) {
      out[i] = buffer.readFloatLE(i * 4);
    }
    return out;
  }

  /**
   * Apply a full re-clustering (§5.6): rewrite every face's personId from
   * the assignment (faces absent from the map lose theirs) and replace the
   * persons array.
   */
  applyClustering(result: ClusterizeResult, threshold: number): void {
    if (this.modelMismatch) {
      throw new Error('face index was built with a different model pack; reset required before writing');
    }
    Object.keys(this.index.videos).forEach((hash) => {
      const entry = this.index.videos[hash];
      if (!entry) {
        return;
      }
      entry.faces.forEach((face) => {
        const personId = result.facePerson.get(face.faceId);
        if (personId) {
          face.personId = personId;
        } else if (face.personId) {
          delete face.personId;
        }
      });
    });
    this.index.persons = result.persons;
    this.index.clusterThreshold = threshold;
    this.index.nextPersonSeq = result.nextPersonSeq;
    this.saveIndex();
  }

  renamePerson(personId: string, name: string): boolean {
    const person = (this.index.persons || []).filter((p) => p.personId === personId)[0];
    if (!person) {
      return false;
    }
    person.name = name;
    this.saveIndex();
    return true;
  }

  setPersonHidden(personId: string, hidden: boolean): boolean {
    const person = (this.index.persons || []).filter((p) => p.personId === personId)[0];
    if (!person) {
      return false;
    }
    person.hidden = hidden;
    this.saveIndex();
    return true;
  }

  /** Store freshly computed attribute aggregates (persons not listed keep theirs). */
  setPersonsAttributes(results: { personId: string; attributes: PersonAttributes }[]): void {
    const byId = new Map<string, PersonAttributes>();
    results.forEach((r) => byId.set(r.personId, r.attributes));
    let changed = false;
    (this.index.persons || []).forEach((person) => {
      const attributes = byId.get(person.personId);
      if (attributes) {
        person.attributes = attributes;
        changed = true;
      }
    });
    if (changed) {
      this.saveIndex();
    }
  }

  /** Remove all attribute aggregates (attributes feature turned off). */
  clearPersonsAttributes(): void {
    let changed = false;
    (this.index.persons || []).forEach((person) => {
      if (person.attributes) {
        delete person.attributes;
        changed = true;
      }
    });
    if (changed) {
      this.saveIndex();
    }
  }

  /** Person IDs per video hash, derived from the faces' personId (§6.5 cache source). */
  personsByVideo(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    if (this.modelMismatch) {
      return out;
    }
    Object.keys(this.index.videos).forEach((hash) => {
      const ids = new Set<string>();
      this.index.videos[hash].faces.forEach((face) => {
        if (face.personId) {
          ids.add(face.personId);
        }
      });
      if (ids.size > 0) {
        const list: string[] = [];
        ids.forEach((id) => list.push(id));
        out[hash] = list.sort();
      }
    });
    return out;
  }

  payload(): FaceIndexPayload {
    return {
      index: FaceStore.hasData(this.facesDir) || Object.keys(this.index.videos).length > 0 ? this.index : null,
      modelMismatch: this.modelMismatch,
      stats: computeFaceStats(this.modelMismatch ? null : this.index),
      personsByVideo: this.personsByVideo(),
    };
  }

  private saveIndex(): void {
    fs.mkdirSync(this.facesDir, { recursive: true });
    const tmpPath = FaceStore.indexPath(this.facesDir) + '.tmp';
    fs.writeFileSync(tmpPath, JSON.stringify(this.index));
    fs.renameSync(tmpPath, FaceStore.indexPath(this.facesDir));
  }
}
