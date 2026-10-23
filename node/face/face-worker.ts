// Face worker: entry point for the Electron utility process (design §9.1).
// Owns the ONNX sessions so the main process never blocks on inference.
// Protocol (structured-clone messages over process.parentPort):
//   in : { type: 'init', detModelPath, recModelPath }
//        { type: 'scan', jobId, job: FaceScanJob }
//   out: { type: 'ready' } | { type: 'init-error', message }
//        { type: 'result', jobId, result: FaceScanResult }

import { FaceEngine } from './face-scan';
import type { FaceScanJob, FaceScanResult } from './face-types';

const parentPort = (process as any).parentPort;

let engine: FaceEngine | null = null;
let enginePromise: Promise<FaceEngine> | null = null;

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
    }
  });
  post({ type: 'listening' });
}
