/*
 * Phase 1 verification harness (not part of the app).
 *
 * Drives the COMPILED app-side face engine (node/face/*.js, built by
 * `npm run electron:serve-tsc`) directly from Node -- no Electron GUI --
 * over the Phase 0 BBC test videos:
 *   frames -> detection -> thumbnails -> index/embeddings save -> reload
 * and checks the counts against the Phase 0 baseline (63 faces total).
 * Also verifies the enabled=false gate of main-face.scanVideoFaces and the
 * FaceStore model-mismatch behaviour.
 *
 * Usage: node tools/face-spike/phase1-harness.cjs
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const { FaceEngine } = require(path.join(REPO, 'node', 'face', 'face-scan.js'));
const { FaceStore } = require(path.join(REPO, 'node', 'face', 'face-store.js'));
const ffmpegPath = require(path.join(REPO, 'node_modules', 'ffmpeg-static'));

const VIDEOS_DIR = path.join(__dirname, 'cluster-test', 'videos');
const DET = path.join(__dirname, 'models', 'buffalo_l', 'det_10g.onnx');
const REC = path.join(__dirname, 'models', 'buffalo_l', 'w600k_r50.onnx');
const MODEL_ID = 'buffalo_l-r50';
const HUB = '/tmp/phase1-harness';
const FACES_DIR = path.join(HUB, 'faces');

// Phase 0 kept-face counts (cluster-test/data.json stats)
const BASELINE = {
  'yuXGpUR7fXA': 18,
  'dmbpagijVkk': 13,
  'V9DqzY1jPhY': 9,
  '8cPxeyZ6-g0': 9,
  'z0B4KEXVaqo': 7,
  'Gfa88SeNohY': 7,
};

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

async function testDisabledGate() {
  console.log('[gate] enabled=false must be a complete no-op');
  const { GLOBALS } = require(path.join(REPO, 'node', 'main-globals.js'));
  const { scanVideoFaces } = require(path.join(REPO, 'node', 'face', 'main-face.js'));
  check('GLOBALS.faceSettings default enabled=false', GLOBALS.faceSettings.enabled === false);
  const outDir = path.join(HUB, 'disabled-hub');
  const element = { hash: 'disabledtest', fileName: 'x.mp4', partialPath: '', inputSource: 0, duration: 60 };
  await scanVideoFaces(element, '/nonexistent-source', outDir);
  check('no faces folder created', !fs.existsSync(path.join(outDir, 'faces')));
  check('no output folder created at all', !fs.existsSync(outDir));
}

/**
 * Strict engine parity: run the compiled Scrfd over the SAME frames the
 * Phase 0 cluster test sampled for one video ([2%, 98%] spacing) and count
 * faces with the §5.2 filter. Must land within +-2 of the Phase 0 recording
 * (the residual is sub-0.1s duration rounding between ffprobe readings).
 */
async function testParity() {
  console.log('[parity] compiled Scrfd at Phase 0 sample times (yuXGpUR7fXA)');
  const { Scrfd } = require(path.join(REPO, 'node', 'face', 'scrfd.js'));
  const { decodeImage } = require(path.join(REPO, 'node', 'face', 'image-ops.js'));
  const { execFileSync } = require('child_process');
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'cluster-test', 'data.json'), 'utf8'));
  const stat = data.stats.filter((s) => s.videoId === 'yuXGpUR7fXA')[0];
  const duration = stat.duration;
  const n = Math.min(60, Math.max(10, Math.ceil(duration / 30)));
  const lo = duration * 0.02;
  const hi = duration * 0.98;
  const det = await Scrfd.create(DET);
  const file = path.join(VIDEOS_DIR, 'yuXGpUR7fXA.mp4');
  const framePath = '/tmp/phase1-parity-frame.jpg';
  let kept = 0;
  for (let i = 0; i < n; i++) {
    const t = lo + ((hi - lo) * i) / (n - 1);
    execFileSync(ffmpegPath, ['-v', 'error', '-ss', t.toFixed(3), '-i', file,
      '-frames:v', '1', '-vf', "scale='min(1280,iw)':-2", '-q:v', '2', '-y', framePath]);
    const img = await decodeImage(framePath);
    let dets = await det.detect(img, [640, 640]);
    if (dets.length === 0) {
      const retry = await det.detect(img, [320, 320]);
      if (retry.length > 0) {
        dets = retry;
      }
    }
    kept += dets.filter((d) => Math.min(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) >= 40).length;
  }
  check('kept faces at identical frames ~= Phase 0 (' + stat.keptFaces + ')',
    Math.abs(kept - stat.keptFaces) <= 2, kept + ' vs ' + stat.keptFaces);
}

async function main() {
  await testDisabledGate();
  await testParity();

  console.log('[scan] driving FaceEngine over the Phase 0 videos');
  fs.rmSync(HUB, { recursive: true, force: true });
  fs.mkdirSync(FACES_DIR, { recursive: true });
  const store = FaceStore.load(FACES_DIR, MODEL_ID, 0.5);
  const engine = await FaceEngine.create(DET, REC);

  const videoIds = Object.keys(BASELINE).sort();
  let totalFaces = 0;
  let firstEmbedding = null;
  let firstFace = null;
  let firstHash = null;
  const t0 = Date.now();
  for (const id of videoIds) {
    const videoPath = path.join(VIDEOS_DIR, id + '.mp4');
    const duration = probeDuration(videoPath);
    const start = Date.now();
    const result = await engine.scanVideo({
      videoPath,
      duration,
      hash: id,
      facesDir: FACES_DIR,
      maxFramesPerVideo: 60,
      sampleIntervalSec: 30,
      minFaceSizePx: 40,
      detModelPath: DET,
      recModelPath: REC,
    });
    store.setVideo(id, result.status, result.frameCount, result.faces);
    const groups = new Set(result.faces.map((f) => f.groupId)).size;
    totalFaces += result.faces.length;
    if (!firstEmbedding && result.faces.length > 0) {
      firstEmbedding = result.faces[0].embedding;
      firstFace = result.faces[0];
      firstHash = id;
    }
    console.log(
      '  ' + id + ': status=' + result.status + ' frames=' + result.frameCount +
      ' faces=' + result.faces.length + ' (Phase 0: ' + BASELINE[id] + ')' +
      ' groups=' + groups + ' in ' + Math.round((Date.now() - start) / 100) / 10 + 's'
    );
    // NOTE: the app samples segment midpoints over [0, duration] (design
    // §5.1), while the Phase 0 cluster test sampled [2%, 98%] -- different
    // frames, so counts differ per video. The band below (+-5 per video,
    // +-25% total) is the "no large divergence" criterion; strict engine
    // parity at identical frames is verified in the [parity] section.
    check(id + ' face count within band of Phase 0', Math.abs(result.faces.length - BASELINE[id]) <= 5,
      result.faces.length + ' vs ' + BASELINE[id]);
  }
  console.log('  total faces=' + totalFaces + ' (Phase 0: 63), wall time ' + Math.round((Date.now() - t0) / 1000) + 's');
  check('total faces within 25% of Phase 0 (63)', Math.abs(totalFaces - 63) <= 15, String(totalFaces));

  console.log('[persist] index / embeddings / images on disk');
  const alignedCount = fs.readdirSync(path.join(FACES_DIR, 'aligned')).length;
  const cropsCount = fs.readdirSync(path.join(FACES_DIR, 'crops')).length;
  const binSize = fs.statSync(path.join(FACES_DIR, 'embeddings.bin')).size;
  check('aligned thumbnails == total faces', alignedCount === totalFaces, alignedCount + ' vs ' + totalFaces);
  check('embeddings.bin size == faces * 2048', binSize === totalFaces * 2048, String(binSize));
  check('display crops exist (>= 6, one per group)', cropsCount >= 6, String(cropsCount));
  check('tmp frames cleaned up', !fs.existsSync(path.join(FACES_DIR, 'tmp')) ||
    fs.readdirSync(path.join(FACES_DIR, 'tmp')).length === 0);

  console.log('[reload] FaceStore.load round trip');
  const store2 = FaceStore.load(FACES_DIR, MODEL_ID, 0.5);
  const payload = store2.payload();
  check('no model mismatch for same pack', payload.modelMismatch === false);
  check('stats: scanned videos == 6', payload.stats.scannedVideos === 6, String(payload.stats.scannedVideos));
  check('stats: total faces match', payload.stats.totalFaces === totalFaces,
    payload.stats.totalFaces + ' vs ' + totalFaces);
  check('stats: groups == crops files', payload.stats.totalGroups === cropsCount,
    payload.stats.totalGroups + ' vs ' + cropsCount);

  const entry = firstHash ? store2.getVideoEntry(firstHash) : undefined;
  const rec = entry && entry.faces.filter((f) => f.faceId === firstFace.faceId)[0];
  check('first face record found after reload', !!rec);
  if (rec && firstEmbedding) {
    const bin = fs.readFileSync(path.join(FACES_DIR, 'embeddings.bin'));
    const loaded = new Float32Array(bin.buffer, bin.byteOffset + rec.embeddingIdx * 2048, 512);
    let dotProd = 0;
    for (let i = 0; i < 512; i++) {
      dotProd += loaded[i] * firstEmbedding[i];
    }
    check('embedding round trip cosine == 1', Math.abs(dotProd - 1) < 1e-4, dotProd.toFixed(6));
    check('representative flag persisted', entry.faces.some((f) => f.representative));
  }

  console.log('[mismatch] other pack invalidates the index');
  const store3 = FaceStore.load(FACES_DIR, 'buffalo_sc-mbf', 0.5);
  check('modelMismatch flagged', store3.modelMismatch === true);
  let threw = false;
  try {
    store3.setVideo('whatever', 'none', 0, []);
  } catch (e) {
    threw = true;
  }
  check('setVideo refuses to write on mismatch', threw);

  console.log(failures === 0 ? 'HARNESS: ALL CHECKS PASSED' : 'HARNESS: ' + failures + ' CHECK(S) FAILED');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('HARNESS ERROR', err);
  process.exit(2);
});
