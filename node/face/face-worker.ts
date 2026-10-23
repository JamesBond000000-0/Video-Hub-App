// Face worker: entry point for the Electron utility process (design §9.1).
// Owns the ONNX sessions so the main process never blocks on inference.
// Protocol (structured-clone messages over process.parentPort):
//   in : { type: 'init', detModelPath, recModelPath }
//        { type: 'scan', jobId, job: FaceScanJob }
//        { type: 'attributes', jobId, job: FaceAttributesJob }
//   out: { type: 'ready' } | { type: 'init-error', message }
//        { type: 'result', jobId, result: FaceScanResult }
//        { type: 'attributes-result', jobId, result: FaceAttributesResult }

import * as fs from 'fs';
import * as path from 'path';

import { FaceEngine, grabFrameImage } from './face-scan';
import { GenderAge } from './genderage';
import type { DetectedFace } from './scrfd';
import {
  aggregateAttributes,
  type AttributeSample,
  type FaceAttributesJob,
  type FaceAttributesResult,
  type FaceScanJob,
  type FaceScanResult,
} from './face-types';

const parentPort = (process as any).parentPort;

let engine: FaceEngine | null = null;
let enginePromise: Promise<FaceEngine> | null = null;

let genderAge: GenderAge | null = null;
let genderAgePath = '';
let genderAgePromise: Promise<GenderAge> | null = null;

function post(message: unknown): void {
  if (parentPort) {
    parentPort.postMessage(message);
  }
}

async function getEngine(detModelPath: string, recModelPath: string): Promise<FaceEngine> {
  if (engine) {
    return engine;
  }
  if (!enginePromise) {
    enginePromise = FaceEngine.create(detModelPath, recModelPath).then((created) => {
      engine = created;
      return created;
    });
  }
  return enginePromise;
}

async function getGenderAge(modelPath: string): Promise<GenderAge> {
  if (genderAge && genderAgePath === modelPath) {
    return genderAge;
  }
  if (!genderAgePromise || genderAgePath !== modelPath) {
    genderAgePath = modelPath;
    genderAgePromise = GenderAge.create(modelPath).then((created) => {
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
  const ga = await getGenderAge(job.genderageModelPath);
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
        await getEngine(message.detModelPath, message.recModelPath);
        post({ type: 'ready' });
      } catch (err) {
        post({ type: 'init-error', message: String(err) });
      }
    } else if (message.type === 'scan') {
      const job = message.job as FaceScanJob;
      try {
        const eng = await getEngine(job.detModelPath, job.recModelPath);
        const result: FaceScanResult = await eng.scanVideo(job);
        post({ type: 'result', jobId: message.jobId, result });
      } catch (err) {
        post({
          type: 'result',
          jobId: message.jobId,
          result: { status: 'error', frameCount: 0, faces: [], error: String(err) } as FaceScanResult,
        });
      }
    } else if (message.type === 'attributes') {
      const job = message.job as FaceAttributesJob;
      try {
        const result = await runAttributesJob(job);
        post({ type: 'attributes-result', jobId: message.jobId, result });
      } catch (err) {
        post({
          type: 'attributes-result',
          jobId: message.jobId,
          result: { results: [], error: String(err) } as FaceAttributesResult,
        });
      }
    }
  });
  post({ type: 'listening' });
}
