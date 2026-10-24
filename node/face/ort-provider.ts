// Execution-provider selection for every face ONNX session (decision D9).
//
// All face sessions (SCRFD detection, ArcFace / LVFace recognition and
// genderage) are created through OrtSessionFactory, which walks an ordered
// candidate ladder and keeps the first provider that actually creates a
// session:
//
//   setting 'auto'      win32:  cuda -> directml -> cpu
//                       linux:  cuda -> cpu
//                       darwin: coreml -> cpu
//   setting 'cuda'      cuda -> cpu        (manual pin, CPU is the net)
//   setting 'directml'  directml -> cpu
//   setting 'coreml'    coreml -> cpu
//   setting 'cpu'       cpu
//
// A failed candidate never aborts session creation; the failure is logged
// and recorded in `attempts`, and the provider that won is reported back
// to the main process (FaceProviderResolution) so the settings UI can show
// what is really running (e.g. "CUDA requested, DirectML active").
//
// Why CUDA can fail even on an NVIDIA machine: the npm onnxruntime-node
// package does NOT bundle the CUDA provider libraries. Its runtime loads
// them through the ORT shared-provider bridge from the folder that holds
// the ORT binary (bin/napi-v6/<platform>/<arch>/), so the user places
// onnxruntime_providers_cuda + onnxruntime_providers_shared there (from
// the official ONNX Runtime 1.30.0 GPU package), and the CUDA 13 runtime
// libraries + cuDNN 9 come from the user's own CUDA Toolkit / cuDNN
// installation (decision of 2026-10-24: the CUDA runtime is installed by
// the user, never bundled with the app). DirectML needs none of this: the
// Windows package already bundles DirectML.dll. Measured sizes and the
// loading-chain evidence: tools/face-spike/GPU_PROVIDER_RESULTS.md.

import * as fs from 'fs';
import * as path from 'path';
import * as ort from 'onnxruntime-node';

import type { FaceProviderSetting } from '../../interfaces/final-object.interface';
import type { FaceProviderAttempt, FaceProviderResolution } from './face-types';

/** Canonical provider names (identical to the manual setting values). */
export type FaceProviderName = 'cpu' | 'cuda' | 'directml' | 'coreml';

// Names as onnxruntime-node expects them in SessionOptions.
const ORT_PROVIDER_NAME: Record<FaceProviderName, string> = {
  cpu: 'cpu',
  cuda: 'cuda',
  directml: 'dml',
  coreml: 'coreml',
};

/** Older .vha2 files have no provider field; anything unknown means 'auto'. */
export function normalizeProviderSetting(setting: FaceProviderSetting | undefined | null): FaceProviderSetting {
  switch (setting) {
    case 'cpu':
    case 'cuda':
    case 'directml':
    case 'coreml':
    case 'auto':
      return setting;
    default:
      return 'auto';
  }
}

/** Ordered candidates for a setting on a platform (CPU always last). */
export function providerCandidates(setting: FaceProviderSetting | undefined | null, platform: string = process.platform): FaceProviderName[] {
  const normalized = normalizeProviderSetting(setting);
  if (normalized === 'cpu') {
    return ['cpu'];
  }
  if (normalized === 'cuda') {
    return ['cuda', 'cpu'];
  }
  if (normalized === 'directml') {
    return ['directml', 'cpu'];
  }
  if (normalized === 'coreml') {
    return ['coreml', 'cpu'];
  }
  // auto
  if (platform === 'win32') {
    return ['cuda', 'directml', 'cpu'];
  }
  if (platform === 'darwin') {
    return ['coreml', 'cpu'];
  }
  return ['cuda', 'cpu'];
}

function firstLine(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.split('\n')[0].slice(0, 300);
}

/**
 * Session factory shared by all models of one worker. The first session
 * walks the ladder; later sessions go straight to the winner (and re-walk
 * the remaining candidates if the winner unexpectedly fails for one
 * model). Every attempt is recorded once for reporting.
 */
export class OrtSessionFactory {
  readonly requested: FaceProviderSetting;
  private readonly candidates: FaceProviderName[];
  private resolved: FaceProviderName | null = null;
  private readonly failed = new Map<FaceProviderName, string>();
  readonly attempts: FaceProviderAttempt[] = [];

  constructor(setting: FaceProviderSetting | undefined | null) {
    this.requested = normalizeProviderSetting(setting);
    this.candidates = providerCandidates(this.requested);
  }

  get effective(): FaceProviderName | null {
    return this.resolved;
  }

  resolution(): FaceProviderResolution {
    return {
      requested: this.requested,
      effective: this.resolved,
      attempts: this.attempts.map((a) => ({ ...a })),
    };
  }

  async createSession(modelPath: string): Promise<ort.InferenceSession> {
    const order: FaceProviderName[] = [];
    if (this.resolved) {
      order.push(this.resolved);
    }
    this.candidates.forEach((candidate) => {
      if (!order.includes(candidate) && !this.failed.has(candidate)) {
        order.push(candidate);
      }
    });
    if (order.length === 0) {
      // Everything failed before (e.g. a transient error): allow one retry.
      order.push(...this.candidates);
    }
    let lastError: unknown = null;
    for (let i = 0; i < order.length; i++) {
      const provider = order[i];
      try {
        const session = await ort.InferenceSession.create(modelPath, {
          executionProviders: [ORT_PROVIDER_NAME[provider]],
        } as ort.InferenceSession.SessionOptions);
        if (this.resolved !== provider) {
          this.resolved = provider;
          this.attempts.push({ provider, ok: true });
          console.log(`face ort: provider = ${provider} (requested: ${this.requested})`);
        }
        return session;
      } catch (err) {
        const message = firstLine(err);
        if (!this.failed.has(provider)) {
          this.failed.set(provider, message);
          this.attempts.push({ provider, ok: false, error: message });
        }
        if (this.resolved === provider) {
          this.resolved = null;
        }
        console.log(`face ort: provider ${provider} unavailable: ${message}`);
        lastError = err;
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }
}

// Shared CPU-only factory for callers that do not pass one (the pre-D9
// behaviour, kept so plain-Node harnesses need no provider plumbing).
let cpuFactory: OrtSessionFactory | null = null;

/** Create a session through `factory`, or on CPU when none is given. */
export function createOrtSession(modelPath: string, factory?: OrtSessionFactory): Promise<ort.InferenceSession> {
  if (factory) {
    return factory.createSession(modelPath);
  }
  if (!cpuFactory) {
    cpuFactory = new OrtSessionFactory('cpu');
  }
  return cpuFactory.createSession(modelPath);
}

// ---------------------------------------------------------------------------
// CUDA provider library placement (the shared-provider bridge loads the
// files from the folder that contains the ONNX Runtime binary).
// ---------------------------------------------------------------------------

/** Folder holding the platform's ONNX Runtime binary ('' when unknown). */
export function getOrtBinDir(): string {
  try {
    const packageJson = require.resolve('onnxruntime-node/package.json', { paths: [__dirname] });
    return path
      .join(path.dirname(packageJson), 'bin', 'napi-v6', process.platform, process.arch)
      .replace('app.asar', 'app.asar.unpacked');
  } catch (err) {
    return '';
  }
}

/** File names the user must place in getOrtBinDir() to enable CUDA. */
export function cudaProviderLibFileNames(platform: string = process.platform): string[] {
  return platform === 'win32'
    ? ['onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll']
    : ['libonnxruntime_providers_cuda.so', 'libonnxruntime_providers_shared.so'];
}

/** True when both CUDA provider libraries are placed next to the ORT binary. */
export function cudaProviderLibsPresent(): boolean {
  const dir = getOrtBinDir();
  if (!dir) {
    return false;
  }
  return cudaProviderLibFileNames().every((name) => fs.existsSync(path.join(dir, name)));
}
