// Face worker: entry point for the Electron utility process (design §9.1).
// Owns the ONNX sessions so the main process never blocks on inference.
// Protocol (structured-clone messages over process.parentPort):
//   in : { type: 'init', detModelPath, recModelPath, recKind, provider }
//        { type: 'scan', jobId, job: FaceScanJob }
//        { type: 'attributes', jobId, job: FaceAttributesJob }
//   out: { type: 'ready', providerInfo } | { type: 'init-error', message }
//        { type: 'result', jobId, result: FaceScanResult }
//        { type: 'attributes-result', jobId, result: FaceAttributesResult }
// All sessions share one OrtSessionFactory (decision D9): the provider is
// resolved once and every result reports the resolution back, so the main
// process can show which provider is actually running.

import * as fs from 'fs';
import * as path from 'path';

import type { FaceProviderSetting } from '../../interfaces/final-object.interface';

import { FaceEngine, grabFrameImage } from './face-scan';
import { GenderAge } from './genderage';
import { normalizeProviderSetting, OrtSessionFactory } from './ort-provider';
import type { DetectedFace } from './scrfd';
import {
  aggregateAttributes,
  type AttributeSample,
  type FaceAttributesJob,
  type FaceAttributesResult,
  type FaceProviderResolution,
  type FaceScanJob,
  type FaceScanResult,
} from './face-types';

const parentPort = (process as any).parentPort;

let sessionFactory: OrtSessionFactory | null = null;
let sessionProvider: FaceProviderSetting = 'auto';

let engine: FaceEngine | null = null;
let enginePromise: Promise<FaceEngine> | null = null;
let engineKey = '';

let genderAge: GenderAge | null = null;
let genderAgePath = '';
let genderAgePromise: Promise<GenderAge> | null = null;

/**
 * The one session factory for this worker. When the requested provider
 * changes, the factory and every session built from it are dropped (the
 * main process also kills the worker on a provider change; this is the
 * belt-and-braces path for jobs that carry their own provider value).
 */
function getSessionFactory(provider: FaceProviderSetting | undefined): OrtSessionFactory {
  const normalized = normalizeProviderSetting(provider);
  if (!sessionFactory || sessionProvider !== normalized) {
    sessionFactory = new OrtSessionFactory(normalized);
    sessionProvider = normalized;
    engine = null;
    enginePromise = null;
    engineKey = '';
    genderAge = null;
    genderAgePath = '';
    genderAgePromise = null;
  }
  return sessionFactory;
}

function providerResolution(provider: FaceProviderSetting | undefined): FaceProviderResolution {
  return getSessionFactory(provider).resolution();
}

function post(message: unknown): void {
  if (parentPort) {
    parentPort.postMessage(message);
  }
}

async function getEngine(
  detModelPath: string,
  recModelPath: string,
  recKind: FaceScanJob['recKind'],
  provider: FaceProviderSetting | undefined,
): Promise<FaceEngine> {
  const factory = getSessionFactory(provider);
  const key = [detModelPath, recModelPath, recKind, factory.requested].join('|');
  if (engine && engineKey === key) {
    return engine;
  }
  if (!enginePromise || engineKey !== key) {
    engineKey = key;
    enginePromise = FaceEngine.create(detModelPath, recModelPath, recKind, factory.requested, factory).then((created) => {
      engine = created;
      return created;
    });
  }
  return enginePromise;
}

async function getGenderAge(modelPath: string, provider: FaceProviderSetting | undefined): Promise<GenderAge> {
  const factory = getSessionFactory(provider);
  if (genderAge && genderAgePath === modelPath) {
    return genderAge;
  }
  if (!genderAgePromise || genderAgePath !== modelPath) {
    genderAgePath = modelPath;
    genderAgePromise = GenderAge.create(modelPath, factory).then((created) => {
      genderAge = created;
      return created;
    });
  }
  return genderAgePromise;
}

/**
 * Attribute job (§5.8): for each person, re-extract the sample frames,
 * run genderage on them (reference input path), and return ONLY the
 * per-person aggregates -- the per-face estimates never leave the worker.
 * Frames are cached per (video, time) inside the job since several
 * sample faces can come from the same frame.
 */
async function runAttributesJob(job: FaceAttributesJob): Promise<FaceAttributesResult> {
  const ga = await getGenderAge(job.genderageModelPath, job.provider);
  const tmpDir = path.join(job.facesDir, 'tmp', 'attributes');
  fs.mkdirSync(tmpDir, { recursive: true });
  const frameCache = new Map<string, { width: number; height: number; data: Uint8Array } | null>();
  const results: FaceAttributesResult['results'] = [];
  try {
    for (let pi = 0; pi < job.persons.length; pi++) {
      const person = job.persons[pi];
      const samples: AttributeSample[] = [];
      for (let fi = 0; fi < person.faces.length; fi++) {
        const sample = person.faces[fi];
        const cacheKey = sample.videoPath + '@' + sample.t;
        if (!frameCache.has(cacheKey)) {
          const framePath = path.join(tmpDir, 'frame-' + frameCache.size + '.jpg');
          frameCache.set(cacheKey, await grabFrameImage(sample.videoPath, sample.t, framePath));
        }
        const img = frameCache.get(cacheKey);
        if (!img) {
          continue;
        }
        const det: DetectedFace = {
          bbox: [
            sample.bbox[0] * img.width,
            sample.bbox[1] * img.height,
            (sample.bbox[0] + sample.bbox[2]) * img.width,
            (sample.bbox[1] + sample.bbox[3]) * img.height,
          ],
          score: 1,
          kps: [],
        };
        try {
          samples.push(await ga.get(img, det));
        } catch (err) {
          console.log('genderage failed for face', sample.faceId, err);
        }
      }
      const attributes = aggregateAttributes(samples);
      if (attributes) {
        results.push({ personId: person.personId, attributes });
      }
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
  return { results };
}

if (parentPort) {
  parentPort.on('message', async (event: { data: any }) => {
    const message = event.data;
    if (!message) {
      return;
    }
    if (message.type === 'init') {
      try {
        const eng = await getEngine(message.detModelPath, message.recModelPath, message.recKind || 'arcface', message.provider);
        post({ type: 'ready', providerInfo: eng.providerResolution() });
      } catch (err) {
        post({ type: 'init-error', message: String(err) });
      }
    } else if (message.type === 'scan') {
      const job = message.job as FaceScanJob;
      try {
        const eng = await getEngine(job.detModelPath, job.recModelPath, job.recKind, job.provider);
        const result: FaceScanResult = await eng.scanVideo(job);
        result.providerInfo = eng.providerResolution();
        post({ type: 'result', jobId: message.jobId, result });
      } catch (err) {
        post({
          type: 'result',
          jobId: message.jobId,
          result: {
            status: 'error',
            frameCount: 0,
            faces: [],
            error: String(err),
            providerInfo: providerResolution(job.provider),
          } as FaceScanResult,
        });
      }
    } else if (message.type === 'attributes') {
      const job = message.job as FaceAttributesJob;
      try {
        const result = await runAttributesJob(job);
        result.providerInfo = providerResolution(job.provider);
        post({ type: 'attributes-result', jobId: message.jobId, result });
      } catch (err) {
        post({
          type: 'attributes-result',
          jobId: message.jobId,
          result: { results: [], error: String(err), providerInfo: providerResolution(job.provider) } as FaceAttributesResult,
        });
      }
    }
  });
  post({ type: 'listening' });
}
