/*
 * GPU execution provider verification harness (not part of the app).
 *
 * Verifies decision D9 (provider selection + fallback ladder) through the
 * COMPILED app code (node/face/*.js) on a machine WITHOUT any GPU and
 * WITHOUT the CUDA provider libraries -- i.e. exactly the environment in
 * which every GPU candidate must fail cleanly and land on CPU:
 *   [candidates] the ladder order per setting and platform (pure function)
 *   [ortbin]     ORT binary folder resolution; CUDA libs absent here
 *   [fallback]   FaceEngine creation under auto / cuda / directml /
 *                coreml / cpu never throws, resolves to CPU, and records
 *                the failed attempts (ArcFace and LVFace recognizers)
 *   [scan]       an end-to-end scan with 'cuda' requested completes on CPU
 *   [genderage]  GenderAge through a 'directml' factory falls back to CPU
 *   [worker]     the real face-worker.js driven over a worker_threads
 *                parentPort shim: init/scan/attributes all report
 *                providerInfo with effective = 'cpu'
 *
 * What this harness CANNOT verify (needs real hardware): that CUDA or
 * DirectML sessions actually run on a GPU, their speed, and numeric
 * parity of GPU vs CPU embeddings. See GPU_PROVIDER_RESULTS.md.
 *
 * Usage: node tools/face-spike/gpu-provider-harness.cjs
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { Worker } = require('worker_threads');

const REPO = path.resolve(__dirname, '..', '..');
const CT = path.join(__dirname, 'cluster-test');
const OUT = path.join(__dirname, 'out', 'gpu-provider');
const FACES_DIR = path.join(OUT, 'faces');

const { FaceEngine, grabFrameImage } = require(path.join(REPO, 'node', 'face', 'face-scan.js'));
const { GenderAge } = require(path.join(REPO, 'node', 'face', 'genderage.js'));
const {
  OrtSessionFactory,
  providerCandidates,
  normalizeProviderSetting,
  getOrtBinDir,
  cudaProviderLibsPresent,
  cudaProviderLibFileNames,
} = require(path.join(REPO, 'node', 'face', 'ort-provider.js'));
const ffmpegPath = require(path.join(REPO, 'node_modules', 'ffmpeg-static'));

const BUFFALO_L = path.join(__dirname, 'models', 'buffalo_l');
const DET = path.join(BUFFALO_L, 'det_10g.onnx');
const REC = path.join(BUFFALO_L, 'w600k_r50.onnx');
const REC_LVFACE = path.join(__dirname, 'models', 'lvface', 'LVFace-T_Glint360K.onnx');
const GA = path.join(BUFFALO_L, 'genderage.onnx');
const VIDEO = path.join(CT, 'videos', 'yuXGpUR7fXA.mp4');

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? '  PASS ' : '  FAIL ') + label + (detail ? ' -- ' + detail : ''));
  if (!ok) {
    failures++;
  }
}

function probeDuration(file) {
  let stderr = '';
  try {
    execFileSync(ffmpegPath, ['-i', file], { stdio: ['ignore', 'ignore', 'pipe'] });
  } catch (e) {
    stderr = e.stderr ? e.stderr.toString() : '';
  }
  const m = stderr.match(/Duration: (\d+):(\d+):([\d.]+)/);
  if (!m) {
    throw new Error('could not probe duration of ' + file);
  }
  return parseInt(m[1], 10) * 3600 + parseInt(m[2], 10) * 60 + parseFloat(m[3]);
}

function attemptsSummary(resolution) {
  return resolution.attempts.map((a) => a.provider + (a.ok ? ':ok' : ':fail')).join(' ');
}

async function testCandidates() {
  console.log('[candidates] ladder order per setting / platform');
  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  check('auto/win32 = cuda>directml>cpu', eq(providerCandidates('auto', 'win32'), ['cuda', 'directml', 'cpu']));
  check('auto/linux = cuda>cpu', eq(providerCandidates('auto', 'linux'), ['cuda', 'cpu']));
  check('auto/darwin = coreml>cpu', eq(providerCandidates('auto', 'darwin'), ['coreml', 'cpu']));
  check('cuda pin = cuda>cpu', eq(providerCandidates('cuda', 'win32'), ['cuda', 'cpu']));
  check('directml pin = directml>cpu', eq(providerCandidates('directml', 'linux'), ['directml', 'cpu']));
  check('coreml pin = coreml>cpu', eq(providerCandidates('coreml', 'linux'), ['coreml', 'cpu']));
  check('cpu pin = cpu', eq(providerCandidates('cpu', 'win32'), ['cpu']));
  check('undefined setting normalizes to auto', normalizeProviderSetting(undefined) === 'auto'
    && eq(providerCandidates(undefined, 'linux'), ['cuda', 'cpu']));
  check('unknown setting normalizes to auto', normalizeProviderSetting('bogus') === 'auto');
}

async function testOrtBin() {
  console.log('[ortbin] ORT binary folder + CUDA library placement state');
  const dir = getOrtBinDir();
  check('bin dir resolved', dir !== '' && fs.existsSync(dir), dir);
  check('bin dir is the platform napi-v6 folder', dir.replace(/\\/g, '/').endsWith('bin/napi-v6/' + process.platform + '/' + process.arch));
  check('CUDA provider libs NOT placed in this environment', cudaProviderLibsPresent() === false);
  check('win32 file names', JSON.stringify(cudaProviderLibFileNames('win32'))
    === JSON.stringify(['onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll']));
  check('linux file names', JSON.stringify(cudaProviderLibFileNames('linux'))
    === JSON.stringify(['libonnxruntime_providers_cuda.so', 'libonnxruntime_providers_shared.so']));
}

async function testFallback() {
  console.log('[fallback] engine creation under every setting lands on CPU, nothing throws');
  const expectations = {
    auto: ['cuda:fail', 'cpu:ok'],
    cuda: ['cuda:fail', 'cpu:ok'],
    directml: ['directml:fail', 'cpu:ok'],
    coreml: ['coreml:fail', 'cpu:ok'],
    cpu: ['cpu:ok'],
  };
  for (const setting of Object.keys(expectations)) {
    const engine = await FaceEngine.create(DET, REC, 'arcface', setting);
    const res = engine.providerResolution();
    check(`setting=${setting}: requested recorded`, res.requested === setting, res.requested);
    check(`setting=${setting}: effective is cpu`, res.effective === 'cpu', attemptsSummary(res));
    check(`setting=${setting}: attempts match ladder`, attemptsSummary(res) === expectations[setting].join(' '), attemptsSummary(res));
    if (setting !== 'cpu') {
      const failed = res.attempts.find((a) => !a.ok);
      check(`setting=${setting}: failure carries an error message`, !!failed && typeof failed.error === 'string' && failed.error.length > 0,
        failed ? failed.error : '');
    }
  }
  const lv = await FaceEngine.create(DET, REC_LVFACE, 'lvface', 'cuda');
  check('lvface engine under cuda: effective is cpu', lv.providerResolution().effective === 'cpu', attemptsSummary(lv.providerResolution()));
}

async function testScan() {
  console.log('[scan] end-to-end scan with cuda requested completes on CPU');
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(FACES_DIR, { recursive: true });
  const engine = await FaceEngine.create(DET, REC, 'arcface', 'cuda');
  const duration = probeDuration(VIDEO);
  const result = await engine.scanVideo({
    videoPath: VIDEO,
    duration,
    hash: 'yuXGpUR7fXA',
    facesDir: FACES_DIR,
    maxFramesPerVideo: 10,
    sampleIntervalSec: 30,
    minFaceSizePx: 40,
    detModelPath: DET,
    recModelPath: REC,
    recKind: 'arcface',
    provider: 'cuda',
  });
  check('scan status done', result.status === 'done', result.status + (result.error ? ' ' + result.error : ''));
  check('faces found', result.faces.length > 0, result.faces.length + ' faces');
  check('embeddings are 512-d', result.faces.every((f) => f.embedding.length === 512));
  check('engine effective provider is cpu', engine.providerResolution().effective === 'cpu');
  return result;
}

async function testGenderAge() {
  console.log('[genderage] GenderAge through a directml factory falls back to CPU and runs');
  const factory = new OrtSessionFactory('directml');
  const ga = await GenderAge.create(GA, factory);
  check('factory effective is cpu', factory.effective === 'cpu', attemptsSummary(factory.resolution()));
  const duration = probeDuration(VIDEO);
  const img = await grabFrameImage(VIDEO, duration / 2, path.join(OUT, 'ga-frame.jpg'));
  check('sample frame grabbed', !!img);
  if (img) {
    const det = {
      bbox: [img.width * 0.25, img.height * 0.15, img.width * 0.75, img.height * 0.85],
      score: 1,
      kps: [],
    };
    const sample = await ga.get(img, det);
    check('attribute sample produced', !!sample && (sample.sex === 'M' || sample.sex === 'F') && sample.age > 0,
      sample ? sample.sex + ' ' + sample.age : '');
  }
}

function runWorkerSession(provider, scanResult) {
  return new Promise((resolve, reject) => {
    const childScript = path.join(OUT, 'worker-child-' + provider + '.cjs');
    fs.writeFileSync(childScript, [
      "const { parentPort } = require('worker_threads');",
      'process.parentPort = {',
      '  postMessage: (msg) => parentPort.postMessage(msg),',
      "  on: (event, fn) => parentPort.on('message', (value) => fn({ data: value })),",
      '};',
      'require(' + JSON.stringify(path.join(REPO, 'node', 'face', 'face-worker.js')) + ');',
    ].join('\n'));
    const worker = new Worker(childScript);
    const waiters = [];
    const inbox = [];
    worker.on('message', (msg) => {
      const idx = waiters.findIndex((w) => w.type === msg.type);
      if (idx >= 0) {
        waiters.splice(idx, 1)[0].resolve(msg);
      } else {
        inbox.push(msg);
      }
    });
    worker.on('error', reject);
    const waitFor = (type, timeoutMs) => {
      const found = inbox.findIndex((m) => m.type === type);
      if (found >= 0) {
        return Promise.resolve(inbox.splice(found, 1)[0]);
      }
      return new Promise((res, rej) => {
        const timer = setTimeout(() => rej(new Error('timeout waiting for ' + type)), timeoutMs || 180000);
        waiters.push({ type, resolve: (m) => { clearTimeout(timer); res(m); } });
      });
    };
    (async () => {
      await waitFor('listening');
      worker.postMessage({ type: 'init', detModelPath: DET, recModelPath: REC, recKind: 'arcface', provider });
      const ready = await waitFor('ready');
      const out = { ready };
      if (scanResult) {
        const jobId = 1;
        worker.postMessage({
          type: 'scan',
          jobId,
          job: {
            videoPath: VIDEO,
            duration: probeDuration(VIDEO),
            hash: 'yuXGpUR7fXA',
            facesDir: path.join(OUT, 'faces-worker'),
            maxFramesPerVideo: 10,
            sampleIntervalSec: 30,
            minFaceSizePx: 40,
            detModelPath: DET,
            recModelPath: REC,
            recKind: 'arcface',
            provider,
          },
        });
        out.scan = await waitFor('result');
        const face = out.scan.result.faces[0];
        worker.postMessage({
          type: 'attributes',
          jobId: 2,
          job: {
            facesDir: path.join(OUT, 'faces-worker'),
            genderageModelPath: GA,
            provider,
            persons: [{
              personId: 'p000001',
              faces: [{ faceId: face.faceId, videoPath: VIDEO, t: face.t, bbox: face.bbox }],
            }],
          },
        });
        out.attributes = await waitFor('attributes-result');
      }
      await worker.terminate();
      resolve(out);
    })().catch(async (err) => {
      await worker.terminate();
      reject(err);
    });
  });
}

async function testWorker(scanResult) {
  console.log('[worker] face-worker protocol reports the resolution (cuda and directml requests)');
  const s1 = await runWorkerSession('cuda', scanResult);
  check('init ready with providerInfo', !!s1.ready.providerInfo);
  check('ready: requested cuda', s1.ready.providerInfo.requested === 'cuda');
  check('ready: effective cpu', s1.ready.providerInfo.effective === 'cpu', attemptsSummary(s1.ready.providerInfo));
  check('ready: cuda attempt failed with error',
    s1.ready.providerInfo.attempts[0].provider === 'cuda' && s1.ready.providerInfo.attempts[0].ok === false
      && typeof s1.ready.providerInfo.attempts[0].error === 'string');
  check('scan result done', s1.scan.result.status === 'done', s1.scan.result.status);
  check('scan result providerInfo effective cpu', !!s1.scan.result.providerInfo && s1.scan.result.providerInfo.effective === 'cpu');
  check('attributes result has one person', s1.attributes.result.results.length === 1,
    JSON.stringify(s1.attributes.result.results).slice(0, 120));
  check('attributes providerInfo effective cpu',
    !!s1.attributes.result.providerInfo && s1.attributes.result.providerInfo.effective === 'cpu');

  const s2 = await runWorkerSession('directml', null);
  check('directml worker: ready effective cpu', s2.ready.providerInfo.effective === 'cpu', attemptsSummary(s2.ready.providerInfo));
  check('directml worker: first attempt is directml failure',
    s2.ready.providerInfo.attempts[0].provider === 'directml' && s2.ready.providerInfo.attempts[0].ok === false);
}

async function main() {
  await testCandidates();
  await testOrtBin();
  await testFallback();
  const scanResult = await testScan();
  await testGenderAge();
  await testWorker(scanResult);
  console.log(failures === 0 ? 'HARNESS: ALL CHECKS PASSED' : 'HARNESS: ' + failures + ' CHECK(S) FAILED');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('HARNESS ERROR', err);
  process.exit(2);
});
