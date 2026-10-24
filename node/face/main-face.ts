import { app, utilityProcess, type UtilityProcess } from 'electron';
import * as fs from 'fs';
import * as path from 'path';

import type { FaceSettings, ImageElement } from '../../interfaces/final-object.interface';
import { GLOBALS } from '../main-globals';

import { clusterizeIndex } from './cluster';
import { FaceStore } from './face-store';
import {
  FACE_MODEL_PACKS,
  selectGroupSamples,
  type FaceAttributeSampleRequest,
  type FaceAttributesJob,
  type FaceAttributesResult,
  type FaceIndexPayload,
  type FaceModelStatus,
  type FaceProviderInfo,
  type FaceProviderResolution,
  type FaceRecord,
  type FaceScanJob,
  type FaceScanResult,
} from './face-types';
import { cudaProviderLibsPresent, getOrtBinDir, normalizeProviderSetting } from './ort-provider';

// ============================================================================
// Main-process side of face recognition (design §9.1): owns the face worker
// (Electron utility process), the FaceStore for the currently open hub, and
// the model placement status (design D2: models live in
// <userData>/face-models/<pack>/, placed there manually by the user).
// ============================================================================

export function getFaceModelsRoot(): string {
  return path.join(app.getPath('userData'), 'face-models');
}

// genderage.onnx ships only in the buffalo_l pack (design Appendix A), so
// the light and lvface packs borrow the quality pack's copy for attribute
// estimation. (genderage consumes frame crops, not recognition embeddings,
// so it is independent of the recognizer -- see §5.8 / decision D8.)
export function getGenderageModelPath(settings: FaceSettings): string {
  const own = path.join(getFaceModelsRoot(), settings.modelPack, 'genderage.onnx');
  if (fs.existsSync(own)) {
    return own;
  }
  if (settings.modelPack !== 'quality') {
    const borrowed = path.join(getFaceModelsRoot(), 'quality', 'genderage.onnx');
    if (fs.existsSync(borrowed)) {
      return borrowed;
    }
  }
  return '';
}

export function getFaceModelStatus(settings: FaceSettings): FaceModelStatus {
  const info = FACE_MODEL_PACKS[settings.modelPack];
  const modelsDir = path.join(getFaceModelsRoot(), settings.modelPack);
  const files = [info.detFile, info.recFile].map((name) => ({
    name,
    present: fs.existsSync(path.join(modelsDir, name)),
  }));
  const genderagePath = getGenderageModelPath(settings);
  return {
    pack: settings.modelPack,
    modelId: info.modelId,
    modelsDir,
    files,
    ready: files.every((f) => f.present),
    genderagePath,
    genderagePresent: genderagePath !== '',
    providerInfo: getFaceProviderInfo(settings),
  };
}

// ---------------------------------------------------------------------------
// Execution provider state (decision D9): what was requested, what the
// worker actually resolved (kept from its last report), and whether the
// CUDA provider libraries are placed next to the ONNX Runtime binary.
// ---------------------------------------------------------------------------

let lastProviderResolution: FaceProviderResolution | null = null;

export function getFaceProviderInfo(settings: FaceSettings): FaceProviderInfo {
  const requested = normalizeProviderSetting(settings ? settings.provider : undefined);
  const resolution = lastProviderResolution && lastProviderResolution.requested === requested ? lastProviderResolution : null;
  return {
    requested,
    effective: resolution ? resolution.effective : null,
    attempts: resolution ? resolution.attempts : [],
    cudaLibsPresent: cudaProviderLibsPresent(),
    ortBinDir: getOrtBinDir(),
  };
}

function noteProviderResolution(resolution: FaceProviderResolution | undefined): void {
  if (!resolution) {
    return;
  }
  const changed = !lastProviderResolution
    || lastProviderResolution.effective !== resolution.effective
    || lastProviderResolution.attempts.length !== resolution.attempts.length;
  lastProviderResolution = resolution;
  if (changed) {
    const failed = resolution.attempts.filter((a) => !a.ok).map((a) => a.provider).join(', ');
    console.log(
      'face provider: effective =', resolution.effective || 'none',
      '(requested:', resolution.requested + ')',
      failed ? 'unavailable: ' + failed : '',
    );
  }
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
let workerProvider: FaceSettings['provider'] | null = null;
let workerReady: Promise<void> | null = null;
let jobCounter = 0;
const pendingJobs = new Map<number, { resolve: (r: FaceScanResult) => void; reject: (e: Error) => void }>();
const pendingAttributeJobs = new Map<number, { resolve: (r: FaceAttributesResult) => void; reject: (e: Error) => void }>();

// Full paths of videos seen this session (from scans and renderer calls).
// The attribute pass needs them to re-extract sample frames; videos whose
// path is unknown are simply skipped there.
const videoPathRegistry = new Map<string, string>();

function registerVideoPaths(elements: ImageElement[] | undefined): void {
  if (!elements) {
    return;
  }
  elements.forEach((element) => {
    const sourceFolder = GLOBALS.selectedSourceFolders[element.inputSource];
    if (sourceFolder && sourceFolder.path) {
      videoPathRegistry.set(element.hash, path.join(sourceFolder.path, element.partialPath, element.fileName));
    }
  });
}

function killWorker(): void {
  if (worker) {
    try {
      worker.kill();
    } catch (err) { /* already gone */ }
    worker = null;
  }
  workerPack = null;
  workerProvider = null;
  workerReady = null;
  lastProviderResolution = null;
  pendingJobs.forEach((pending) => pending.reject(new Error('face worker stopped')));
  pendingJobs.clear();
  pendingAttributeJobs.forEach((pending) => pending.reject(new Error('face worker stopped')));
  pendingAttributeJobs.clear();
}

function ensureWorker(settings: FaceSettings): Promise<void> {
  const provider = normalizeProviderSetting(settings.provider);
  if (worker && workerReady && workerPack === settings.modelPack && workerProvider === provider) {
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
  workerProvider = provider;
  workerReady = new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('face worker init timed out')), 120000);
    child.on('message', (message: any) => {
      if (!message) {
        return;
      }
      if (message.type === 'ready') {
        clearTimeout(timeout);
        noteProviderResolution(message.providerInfo as FaceProviderResolution | undefined);
        // push the fresh status so an open settings view shows the
        // effective provider without waiting for the next poll
        if (GLOBALS.angularApp && GLOBALS.angularApp.sender) {
          GLOBALS.angularApp.sender.send('face-model-status-returning', getFaceModelStatus(GLOBALS.faceSettings));
        }
        resolve();
      } else if (message.type === 'init-error') {
        clearTimeout(timeout);
        reject(new Error('face worker init failed: ' + message.message));
      } else if (message.type === 'result') {
        noteProviderResolution((message.result as FaceScanResult).providerInfo);
        const pending = pendingJobs.get(message.jobId);
        if (pending) {
          pendingJobs.delete(message.jobId);
          pending.resolve(message.result as FaceScanResult);
        }
      } else if (message.type === 'attributes-result') {
        noteProviderResolution((message.result as FaceAttributesResult).providerInfo);
        const pending = pendingAttributeJobs.get(message.jobId);
        if (pending) {
          pendingAttributeJobs.delete(message.jobId);
          pending.resolve(message.result as FaceAttributesResult);
        }
      }
    });
    child.on('exit', () => {
      if (worker === child) {
        worker = null;
        workerPack = null;
        workerProvider = null;
        workerReady = null;
        lastProviderResolution = null;
      }
      pendingJobs.forEach((pending) => pending.reject(new Error('face worker exited')));
      pendingJobs.clear();
      pendingAttributeJobs.forEach((pending) => pending.reject(new Error('face worker exited')));
      pendingAttributeJobs.clear();
    });
    child.postMessage({
      type: 'init',
      detModelPath: path.join(modelsDir, info.detFile),
      recModelPath: path.join(modelsDir, info.recFile),
      recKind: info.recKind,
      provider,
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
  videoPathRegistry.set(element.hash, path.join(videoFolderPath, element.partialPath, element.fileName));
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
    recKind: FACE_MODEL_PACKS[settings.modelPack].recKind,
    provider: normalizeProviderSetting(settings.provider),
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
// Global clustering (§5.6) and attribute aggregation (§5.8)
// ---------------------------------------------------------------------------

/**
 * Re-run the global clustering over the current index. Person IDs, names
 * and hidden flags are reconciled with the previous persons inside
 * clusterizeIndex. When attributes are disabled, any stored aggregates
 * are removed (the feature must leave no trace while off). Never throws.
 */
export async function reclusterFaces(): Promise<void> {
  const settings: FaceSettings = GLOBALS.faceSettings;
  if (!settings || !settings.enabled) {
    return;
  }
  const store = getFaceStore();
  if (store.modelMismatch || !FaceStore.hasData(store.facesDir)) {
    return;
  }
  try {
    const result = clusterizeIndex(store.getIndex(), (idx) => store.readEmbedding(idx), settings.clusterThreshold);
    store.applyClustering(result, settings.clusterThreshold);
    if (!settings.attributesEnabled) {
      store.clearPersonsAttributes();
    }
    console.log('face re-clustering done:', result.persons.length, 'persons');
  } catch (err) {
    console.log('face re-clustering failed:', err);
  }
}

/**
 * Compute the person-level attribute aggregates (§5.8) in the worker and
 * store them. Runs only when the feature is enabled and genderage.onnx
 * is placed; (video, group)s whose video path is unknown this session
 * are skipped. Never throws.
 */
export async function computeFaceAttributes(elements?: ImageElement[]): Promise<void> {
  const settings: FaceSettings = GLOBALS.faceSettings;
  if (!settings || !settings.enabled || !settings.attributesEnabled) {
    return;
  }
  registerVideoPaths(elements);
  const genderageModelPath = getGenderageModelPath(settings);
  if (!genderageModelPath) {
    console.log('face attributes skipped (genderage.onnx not placed)');
    return;
  }
  const store = getFaceStore();
  if (store.modelMismatch) {
    return;
  }
  const index = store.getIndex();
  const persons = index.persons || [];
  if (persons.length === 0) {
    return;
  }
  // sample faces per person: per (video, group), the top-N by quality
  const jobPersons: FaceAttributesJob['persons'] = persons.map((person) => {
    const faces: FaceAttributeSampleRequest[] = [];
    person.videoHashes.forEach((hash) => {
      const videoPath = videoPathRegistry.get(hash);
      const entry = index.videos[hash];
      if (!videoPath || !entry) {
        return;
      }
      const byGroup = new Map<number, FaceRecord[]>();
      entry.faces.forEach((face) => {
        if (face.personId === person.personId && face.groupId >= 0) {
          if (!byGroup.has(face.groupId)) {
            byGroup.set(face.groupId, []);
          }
          byGroup.get(face.groupId).push(face);
        }
      });
      byGroup.forEach((members) => {
        selectGroupSamples(members, settings.attributeSamplesPerGroup).forEach((face) => {
          faces.push({ faceId: face.faceId, videoPath, t: face.t, bbox: face.bbox });
        });
      });
    });
    return { personId: person.personId, faces };
  }).filter((p) => p.faces.length > 0);
  if (jobPersons.length === 0) {
    return;
  }
  try {
    await ensureWorker(settings);
    const result = await new Promise<FaceAttributesResult>((resolve, reject) => {
      const jobId = ++jobCounter;
      pendingAttributeJobs.set(jobId, { resolve, reject });
      if (worker) {
        const job: FaceAttributesJob = {
          facesDir: store.facesDir,
          genderageModelPath,
          persons: jobPersons,
          provider: normalizeProviderSetting(settings.provider),
        };
        worker.postMessage({ type: 'attributes', jobId, job });
      } else {
        pendingAttributeJobs.delete(jobId);
        reject(new Error('face worker unavailable'));
      }
    });
    if (result.error) {
      console.log('face attributes job reported:', result.error);
    }
    if (result.results.length > 0) {
      store.setPersonsAttributes(result.results);
      console.log('face attributes stored for', result.results.length, 'persons');
    }
  } catch (err) {
    console.log('face attributes failed:', err);
  }
}

/**
 * Post-scan hook (called by the face queue drain in main-extract-async):
 * cluster first, then aggregate attributes when the feature is on.
 */
export async function reclusterAfterScan(): Promise<void> {
  await reclusterFaces();
  await computeFaceAttributes();
  notifyFaceIndexUpdated();
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
    const providerChanged = !!faceSettings
      && normalizeProviderSetting(faceSettings.provider) !== normalizeProviderSetting(GLOBALS.faceSettings.provider);
    if (faceSettings) {
      GLOBALS.faceSettings = { ...GLOBALS.faceSettings, ...faceSettings };
    }
    if (packChanged || providerChanged) {
      killWorker(); // sessions for the other pack / provider are useless; respawn lazily
    }
    event.sender.send('face-model-status-returning', getFaceModelStatus(GLOBALS.faceSettings));
  });

  // Manual re-clustering (persons view / settings). Also recomputes the
  // attribute aggregates when the feature is enabled. The elements ride
  // along so the attribute pass knows the video file paths.
  ipc.on('face-recluster', async (event, elements?: ImageElement[]) => {
    registerVideoPaths(elements);
    await reclusterFaces();
    await computeFaceAttributes(elements);
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  // Recompute only the attribute aggregates (§5.8)
  ipc.on('face-attributes-request', async (event, elements?: ImageElement[]) => {
    await computeFaceAttributes(elements);
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  ipc.on('face-rename-person', (event, personId: string, name: string) => {
    if (getFaceStore().renamePerson(personId, (name || '').trim())) {
      event.sender.send('face-index-returning', getFaceIndexPayload());
    }
  });

  ipc.on('face-set-person-hidden', (event, personId: string, hidden: boolean) => {
    if (getFaceStore().setPersonHidden(personId, !!hidden)) {
      event.sender.send('face-index-returning', getFaceIndexPayload());
    }
  });

  // Manual corrections (Phase 3). Composition-changing edits invalidate
  // the affected persons' attribute aggregates inside the re-clustering
  // the store runs; when attributes are enabled, fresh aggregates are
  // computed right away (persons whose video paths are unknown this
  // session are picked up by the next full attribute pass instead).
  ipc.on('face-merge-persons', async (event, personIds: string[]) => {
    try {
      const survivor = getFaceStore().mergePersons(Array.isArray(personIds) ? personIds : []);
      if (survivor) {
        await computeFaceAttributes();
      }
    } catch (err) {
      console.log('face merge failed:', err);
    }
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  ipc.on('face-split-person', async (event, personId: string, groupKeys: string[]) => {
    try {
      const newPersonId = getFaceStore().splitPerson(personId, Array.isArray(groupKeys) ? groupKeys : []);
      if (newPersonId) {
        await computeFaceAttributes();
      }
    } catch (err) {
      console.log('face split failed:', err);
    }
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  ipc.on('face-move-groups', async (event, groupKeys: string[], targetPersonId: string | null) => {
    try {
      const moved = getFaceStore().moveGroups(
        Array.isArray(groupKeys) ? groupKeys : [],
        typeof targetPersonId === 'string' ? targetPersonId : null,
      );
      if (moved) {
        await computeFaceAttributes();
      }
    } catch (err) {
      console.log('face group move failed:', err);
    }
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });

  ipc.on('face-set-person-representative', (event, personId: string, faceId: string) => {
    try {
      getFaceStore().setPersonRepresentative(personId, faceId);
    } catch (err) {
      console.log('face representative change failed:', err);
    }
    event.sender.send('face-index-returning', getFaceIndexPayload());
  });
}
