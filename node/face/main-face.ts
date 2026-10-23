import { app, utilityProcess, type UtilityProcess } from 'electron';
import * as fs from 'fs';
import * as path from 'path';

import type { FaceSettings, ImageElement } from '../../interfaces/final-object.interface';
import { GLOBALS } from '../main-globals';

import { FaceStore } from './face-store';
import {
  FACE_MODEL_PACKS,
  type FaceIndexPayload,
  type FaceModelStatus,
  type FaceScanJob,
  type FaceScanResult,
} from './face-types';

// ============================================================================
// Main-process side of face recognition (design §9.1): owns the face worker
// (Electron utility process), the FaceStore for the currently open hub, and
// the model placement status (design D2: models live in
// <userData>/face-models/<pack>/, placed there manually by the user).
// ============================================================================

export function getFaceModelsRoot(): string {
  return path.join(app.getPath('userData'), 'face-models');
}

export function getFaceModelStatus(settings: FaceSettings): FaceModelStatus {
  const info = FACE_MODEL_PACKS[settings.modelPack];
  const modelsDir = path.join(getFaceModelsRoot(), settings.modelPack);
  const files = [info.detFile, info.recFile].map((name) => ({
    name,
    present: fs.existsSync(path.join(modelsDir, name)),
  }));
  return {
    pack: settings.modelPack,
    modelId: info.modelId,
    modelsDir,
    files,
    ready: files.every((f) => f.present),
  };
}

export function hubFacesDir(): string {
  return path.join(GLOBALS.selectedOutputFolder, 'vha-' + GLOBALS.hubName, 'faces');
}

// ---------------------------------------------------------------------------
// FaceStore access (one open hub at a time; cached per facesDir + model)
// ---------------------------------------------------------------------------

let storeCache: { facesDir: string; modelId: string; store: FaceStore } | null = null;

export function getFaceStore(): FaceStore {
  const settings = GLOBALS.faceSettings;
  const info = FACE_MODEL_PACKS[settings.modelPack];
  const facesDir = hubFacesDir();
  if (!storeCache || storeCache.facesDir !== facesDir || storeCache.modelId !== info.modelId) {
    storeCache = {
      facesDir,
      modelId: info.modelId,
      store: FaceStore.load(facesDir, info.modelId, settings.clusterThreshold),
    };
  }
  return storeCache.store;
}

export function getFaceIndexPayload(): FaceIndexPayload {
  return getFaceStore().payload();
}

export function notifyFaceIndexUpdated(): void {
  if (GLOBALS.angularApp && GLOBALS.angularApp.sender) {
    GLOBALS.angularApp.sender.send('face-index-updated');
  }
}

// ---------------------------------------------------------------------------
// Worker lifecycle (spawned lazily on first scan; killed on pack change/exit)
// ---------------------------------------------------------------------------

let worker: UtilityProcess | null = null;
let workerPack: FaceSettings['modelPack'] | null = null;
let workerReady: Promise<void> | null = null;
let jobCounter = 0;
const pendingJobs = new Map<number, { resolve: (r: FaceScanResult) => void; reject: (e: Error) => void }>();

function killWorker(): void {
  if (worker) {
    try {
      worker.kill();
    } catch (err) { /* already gone */ }
    worker = null;
  }
  workerPack = null;
  workerReady = null;
  pendingJobs.forEach((pending) => pending.reject(new Error('face worker stopped')));
  pendingJobs.clear();
}

function ensureWorker(settings: FaceSettings): Promise<void> {
  if (worker && workerReady && workerPack === settings.modelPack) {
    return workerReady;
  }
  killWorker();
  const info = FACE_MODEL_PACKS[settings.modelPack];
  const modelsDir = path.join(getFaceModelsRoot(), settings.modelPack);
  const child = utilityProcess.fork(path.join(__dirname, 'face-worker.js'), [], {
    serviceName: 'vha-face-worker',
  });
  worker = child;
  workerPack = settings.modelPack;
  workerReady = new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('face worker init timed out')), 120000);
    child.on('message', (message: any) => {
      if (!message) {
        return;
      }
      if (message.type === 'ready') {
        clearTimeout(timeout);
        resolve();
      } else if (message.type === 'init-error') {
        clearTimeout(timeout);
        reject(new Error('face worker init failed: ' + message.message));
      } else if (message.type === 'result') {
        const pending = pendingJobs.get(message.jobId);
        if (pending) {
          pendingJobs.delete(message.jobId);
          pending.resolve(message.result as FaceScanResult);
        }
      }
    });
    child.on('exit', () => {
      if (worker === child) {
        worker = null;
        workerPack = null;
        workerReady = null;
      }
      pendingJobs.forEach((pending) => pending.reject(new Error('face worker exited')));
      pendingJobs.clear();
    });
    child.postMessage({
      type: 'init',
      detModelPath: path.join(modelsDir, info.detFile),
      recModelPath: path.join(modelsDir, info.recFile),
    });
  });
  return workerReady;
}

// ---------------------------------------------------------------------------
// Scanning (called by faceQueue in main-extract-async)
// ---------------------------------------------------------------------------

/**
 * Scan one video's faces unless disabled / already scanned / models missing.
 * Never throws: failures are recorded in the index as status 'error'.
 */
export async function scanVideoFaces(element: ImageElement, videoFolderPath: string, screenshotOutputFolder: string): Promise<void> {
  const settings: FaceSettings = GLOBALS.faceSettings;
  if (!settings || !settings.enabled) {
    return;
  }
  const status = getFaceModelStatus(settings);
  if (!status.ready) {
    console.log('face scan skipped (models not placed):', status.modelsDir);
    return;
  }
  let store = getFaceStore();
  if (store.modelMismatch) {
    // embeddings from another recognition model are incompatible (§4.3):
    // wipe and rebuild the face data for this hub
    console.log('face index model mismatch; resetting face data for this hub');
    store = FaceStore.reset(hubFacesDir(), status.modelId, settings.clusterThreshold);
    storeCache = { facesDir: hubFacesDir(), modelId: status.modelId, store };
  }
  if (store.isScanned(element.hash)) {
    return;
  }
  const job: FaceScanJob = {
    videoPath: path.join(videoFolderPath, element.partialPath, element.fileName),
    duration: element.duration,
    hash: element.hash,
    facesDir: path.join(screenshotOutputFolder, 'faces'),
    maxFramesPerVideo: settings.maxFramesPerVideo,
    sampleIntervalSec: settings.sampleIntervalSec,
    minFaceSizePx: settings.minFaceSizePx,
    detModelPath: path.join(status.modelsDir, FACE_MODEL_PACKS[settings.modelPack].detFile),
    recModelPath: path.join(status.modelsDir, FACE_MODEL_PACKS[settings.modelPack].recFile),
  };
  try {
    await ensureWorker(settings);
    const result = await new Promise<FaceScanResult>((resolve, reject) => {
      const jobId = ++jobCounter;
      pendingJobs.set(jobId, { resolve, reject });
      if (worker) {
        worker.postMessage({ type: 'scan', jobId, job });
      } else {
        pendingJobs.delete(jobId);
        reject(new Error('face worker unavailable'));
      }
    });
    if (result.status === 'error') {
      store.setVideoError(element.hash, result.frameCount, result.error || 'unknown error');
    } else {
      store.setVideo(element.hash, result.status, result.frameCount, result.faces);
    }
  } catch (err) {
    console.log('face scan failed for', element.fileName, err);
    try {
      store.setVideoError(element.hash, 0, String(err));
    } catch (storeErr) {
      console.log('could not record face scan error:', storeErr);
    }
  }
}

// ---------------------------------------------------------------------------
// IPC (registered from main-ipc.ts)
// ---------------------------------------------------------------------------

export function registerFaceIpc(ipc: Electron.IpcMain): void {
  ipc.on('face-get-model-status', (event) => {
    event.sender.send('face-model-status-returning', getFaceModelStatus(GLOBALS.faceSettings));
  });

  ipc.on('face-get-index', (event) => {
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  ipc.on('face-settings-updated', (event, faceSettings: FaceSettings) => {
    const packChanged = !!faceSettings && faceSettings.modelPack !== GLOBALS.faceSettings.modelPack;
    if (faceSettings) {
      GLOBALS.faceSettings = { ...GLOBALS.faceSettings, ...faceSettings };
    }
    if (packChanged) {
      killWorker(); // sessions for the other pack are useless; respawn lazily
    }
    event.sender.send('face-model-status-returning', getFaceModelStatus(GLOBALS.faceSettings));
  });
}
