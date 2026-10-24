/*
 * LVFace pack verification harness (not part of the app).
 *
 * Verifies the 'lvface' model pack (decision D8) end to end through the
 * COMPILED app code (node/face/*.js), over the Phase 0 BBC test videos:
 *   [pack]       FACE_MODEL_PACKS.lvface wiring (files, kind, modelId)
 *                and the spike model files exist under the exact names
 *                the app expects users to place.
 *   [scan]       FaceEngine scans with the lvface pack (det_10g +
 *                LVFace-T) and, as the correspondence reference, with the
 *                quality pack (det_10g + w600k_r50).
 *   [parity]     detection is identical between the packs (same §5.1
 *                sampling, same detector): the face sequences (faceId,
 *                t, bbox, score) must match face-for-face, and the total
 *                must equal the Phase 1/2 quality baseline (72 faces).
 *   [mismatch]   the two packs' indexes reject each other's modelId.
 *   [cluster]    global clustering of the LVFace index at T=0.50.
 *                Labels are transferred to quality groups exactly like
 *                the Phase 2 harness (representative-embedding match
 *                against Phase 0 groups), then carried to LVFace faces
 *                via the face-for-face correspondence. Gates: no LVFace
 *                person mixes two definitely-labelled people, and the
 *                Davies cross-video merge (the fully-labelled case)
 *                holds in the LVFace index (Phase 2 merge semantics,
 *                with the quality index as control; the remaining
 *                cross-video merges are reported, not gated).
 *   [attributes] genderage aggregates over LVFace persons through the
 *                reference path (re-extracted frames): the model takes
 *                frame crops, not recognition embeddings, so it must
 *                work unchanged for the lvface pack.
 *
 * Usage: node tools/face-spike/lvface-pack-harness.cjs
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const CT = path.join(__dirname, 'cluster-test');
const { FaceEngine, grabFrameImage } = require(path.join(REPO, 'node', 'face', 'face-scan.js'));
const { FaceStore } = require(path.join(REPO, 'node', 'face', 'face-store.js'));
const { clusterizeIndex, collectGroupItems } = require(path.join(REPO, 'node', 'face', 'cluster.js'));
const { GenderAge } = require(path.join(REPO, 'node', 'face', 'genderage.js'));
const { FACE_MODEL_PACKS, selectGroupSamples, aggregateAttributes } = require(path.join(REPO, 'node', 'face', 'face-types.js'));
const ffmpegPath = require(path.join(REPO, 'node_modules', 'ffmpeg-static'));

const VIDEOS_DIR = path.join(CT, 'videos');
const BUFFALO_L = path.join(__dirname, 'models', 'buffalo_l');
const DET = path.join(BUFFALO_L, 'det_10g.onnx');
const REC_QUALITY = path.join(BUFFALO_L, 'w600k_r50.onnx');
const REC_LVFACE = path.join(__dirname, 'models', 'lvface', 'LVFace-T_Glint360K.onnx');
const GA = path.join(BUFFALO_L, 'genderage.onnx');
const T = 0.5;
const QUALITY_BASELINE_FACES = 72; // Phase 1/2 harness, app §5.1 sampling

const VIDEO_IDS = ['yuXGpUR7fXA', 'dmbpagijVkk', 'V9DqzY1jPhY', '8cPxeyZ6-g0', 'z0B4KEXVaqo', 'Gfa88SeNohY'];
const KNOWN_SEX = {
  'Greg Davies': 'M', 'Russell Crowe': 'M', 'Ryan Gosling': 'M', 'Jodie Whittaker': 'F',
  'Lee Mack': 'M', 'John Cleese': 'M', 'Colin Farrell': 'M', 'Graham Norton': 'M',
  'David Attenborough': 'M',
};
const JUNK_LABELS = ['(ape - false positive)', 'Unknown (face not visible)'];
const CROSS_VIDEO_LABELS = ['Greg Davies', 'Ryan Gosling', 'David Attenborough'];

let failures = 0;
function check(label, ok, detail) {
  console.log((ok ? '  PASS ' : '  FAIL ') + label + (detail ? ' -- ' + detail : ''));
  if (!ok) {
    failures++;
  }
}
function report(label, detail) {
  console.log('  .... ' + label + (detail ? ' -- ' + detail : ''));
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

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

async function scanAll(tag, recPath, recKind, modelId) {
  const hub = '/tmp/lvface-pack-harness-' + tag;
  const facesDir = path.join(hub, 'faces');
  fs.rmSync(hub, { recursive: true, force: true });
  fs.mkdirSync(facesDir, { recursive: true });
  const store = FaceStore.load(facesDir, modelId, T);
  const engine = await FaceEngine.create(DET, recPath, recKind);
  let totalFaces = 0;
  for (const id of VIDEO_IDS) {
    const videoPath = path.join(VIDEOS_DIR, id + '.mp4');
    const result = await engine.scanVideo({
      videoPath,
      duration: probeDuration(videoPath),
      hash: id,
      facesDir,
      maxFramesPerVideo: 60,
      sampleIntervalSec: 30,
      minFaceSizePx: 40,
      detModelPath: DET,
      recModelPath: recPath,
      recKind,
    });
    store.setVideo(id, result.status, result.frameCount, result.faces);
    totalFaces += result.faces.length;
    console.log('  [' + tag + '] ' + id + ': faces=' + result.faces.length +
      ' groups=' + new Set(result.faces.map((f) => f.groupId)).size);
  }
  console.log('  [' + tag + '] total faces=' + totalFaces);
  return { store, facesDir, totalFaces };
}

async function main() {
  console.log('[pack] FACE_MODEL_PACKS wiring');
  const lv = FACE_MODEL_PACKS.lvface;
  check('lvface pack registered', !!lv);
  check('lvface detFile is det_10g.onnx (same detector as quality)', lv.detFile === 'det_10g.onnx', lv.detFile);
  check('lvface recFile is the HuggingFace file name', lv.recFile === 'LVFace-T_Glint360K.onnx', lv.recFile);
  check('lvface recKind is lvface', lv.recKind === 'lvface', lv.recKind);
  check('lvface modelId differs from both ArcFace packs',
    lv.modelId !== FACE_MODEL_PACKS.quality.modelId && lv.modelId !== FACE_MODEL_PACKS.light.modelId, lv.modelId);
  check('quality/light still arcface kind',
    FACE_MODEL_PACKS.quality.recKind === 'arcface' && FACE_MODEL_PACKS.light.recKind === 'arcface');
  check('spike LVFace-T file exists under the expected name', fs.existsSync(REC_LVFACE), REC_LVFACE);
  check('spike det_10g file exists under the expected name', fs.existsSync(DET), DET);

  console.log('[scan] lvface pack');
  const lvRun = await scanAll('lvface', REC_LVFACE, lv.recKind, lv.modelId);
  console.log('[scan] quality pack (correspondence reference)');
  const qRun = await scanAll('quality', REC_QUALITY, FACE_MODEL_PACKS.quality.recKind, FACE_MODEL_PACKS.quality.modelId);

  console.log('[parity] detection identical between packs');
  check('quality total faces == Phase 1/2 baseline', qRun.totalFaces === QUALITY_BASELINE_FACES,
    qRun.totalFaces + ' vs ' + QUALITY_BASELINE_FACES);
  check('lvface total faces == quality total faces', lvRun.totalFaces === qRun.totalFaces,
    lvRun.totalFaces + ' vs ' + qRun.totalFaces);
  const qIndex = qRun.store.getIndex();
  const lvIndexPre = lvRun.store.getIndex();
  let faceParity = true;
  VIDEO_IDS.forEach((id) => {
    const qFaces = qIndex.videos[id].faces;
    const lFaces = lvIndexPre.videos[id].faces;
    if (qFaces.length !== lFaces.length) {
      faceParity = false;
      console.log('    count differs for ' + id + ': ' + lFaces.length + ' vs ' + qFaces.length);
      return;
    }
    qFaces.forEach((qf, i) => {
      const lf = lFaces[i];
      if (qf.faceId !== lf.faceId || qf.t !== lf.t || qf.score !== lf.score ||
        JSON.stringify(qf.bbox) !== JSON.stringify(lf.bbox)) {
        faceParity = false;
        console.log('    face differs: ' + id + ' #' + i);
      }
    });
  });
  check('face sequences (faceId, t, bbox, score) identical face-for-face', faceParity);

  console.log('[mismatch] packs reject each other\'s modelId');
  const crossA = FaceStore.load(lvRun.facesDir, FACE_MODEL_PACKS.quality.modelId, T);
  check('lvface index flagged as mismatch under quality modelId', crossA.modelMismatch === true);
  const crossB = FaceStore.load(qRun.facesDir, lv.modelId, T);
  check('quality index flagged as mismatch under lvface modelId', crossB.modelMismatch === true);
  let threw = false;
  try {
    crossA.setVideo('whatever', 'none', 0, []);
  } catch (e) {
    threw = true;
  }
  check('setVideo refuses to write on mismatch', threw);
  const lvReload = FaceStore.load(lvRun.facesDir, lv.modelId, T);
  check('lvface index loads clean under its own modelId', lvReload.modelMismatch === false);

  console.log('[cluster] LVFace global clustering at T=' + T);
  const lvResult = clusterizeIndex(lvReload.getIndex(), (idx) => lvReload.readEmbedding(idx), T);
  lvReload.applyClustering(lvResult, T);
  const qResult = clusterizeIndex(qRun.store.getIndex(), (idx) => qRun.store.readEmbedding(idx), T);
  qRun.store.applyClustering(qResult, T);
  const lvStore = FaceStore.load(lvRun.facesDir, lv.modelId, T);
  const lvIndex = lvStore.getIndex();
  const qIndex2 = FaceStore.load(qRun.facesDir, FACE_MODEL_PACKS.quality.modelId, T).getIndex();
  report('persons: lvface=' + lvIndex.persons.length + ' quality=' + qIndex2.persons.length);
  check('lvface persons persisted after reload', lvIndex.persons.length === lvResult.persons.length && lvIndex.persons.length > 0,
    String(lvIndex.persons.length));
  const lvFacesTotal = Object.values(lvIndex.videos).reduce((n, v) => n + v.faces.length, 0);
  const lvFacesWithPerson = Object.values(lvIndex.videos).reduce(
    (n, v) => n + v.faces.filter((f) => f.personId).length, 0);
  check('every grouped lvface face has a personId', lvFacesWithPerson === lvFacesTotal,
    lvFacesWithPerson + '/' + lvFacesTotal);

  // --- label transfer (Phase 2 method, on the quality run) ---
  const p0Groups = JSON.parse(fs.readFileSync(path.join(CT, 'groups.json'), 'utf8'));
  const labels = JSON.parse(fs.readFileSync(path.join(CT, 'labels.json'), 'utf8')).groups;
  const p0ByVideo = {};
  p0Groups.forEach((g) => {
    (p0ByVideo[g.videoId] = p0ByVideo[g.videoId] || []).push(g);
  });
  const qStore2 = FaceStore.load(qRun.facesDir, FACE_MODEL_PACKS.quality.modelId, T);
  const qItems = collectGroupItems(qIndex2, (idx) => qStore2.readEmbedding(idx));
  // quality group key -> transferred label (Phase 2 rule: best Phase 0
  // group in the same video at sim >= 0.45; tentative kept separate)
  const qGroupLabel = new Map();
  qItems.forEach((item) => {
    let best = null;
    let bestSim = -Infinity;
    (p0ByVideo[item.videoHash] || []).forEach((g) => {
      const s = dot(item.repEmbedding, g.repEmbedding);
      if (s > bestSim) {
        bestSim = s;
        best = g;
      }
    });
    const lab = best && bestSim >= 0.45 ? (labels[best.groupKey] || {}) : {};
    qGroupLabel.set(item.key, {
      label: lab.person || '',
      tentative: !!lab.tentative,
      p0Key: best ? best.groupKey : '',
      sim: bestSim,
    });
    console.log('    transfer ' + item.key + ' -> ' + (best ? best.groupKey : 'none') +
      ' sim=' + bestSim.toFixed(3) + ' label=' + (lab.person || '(none)'));
  });
  // faceId -> label, via the quality run (faces correspond 1:1 between
  // the packs, so a face's label is the label of its quality group)
  const faceLabel = new Map();
  Object.entries(qIndex2.videos).forEach(([hash, v]) => v.faces.forEach((f) => {
    faceLabel.set(f.faceId, qGroupLabel.get(hash + '#' + f.groupId));
  }));

  // LVFace person -> labels of its faces
  const lvPersonLabels = new Map(); // personId -> [{label, tentative, videoHash}]
  Object.entries(lvIndex.videos).forEach(([hash, v]) => v.faces.forEach((f) => {
    if (!f.personId) {
      return;
    }
    if (!lvPersonLabels.has(f.personId)) {
      lvPersonLabels.set(f.personId, []);
    }
    const gl = faceLabel.get(f.faceId);
    if (gl && gl.label) {
      lvPersonLabels.get(f.personId).push({ ...gl, videoHash: hash });
    }
  }));

  let contamination = 0;
  lvPersonLabels.forEach((entries, personId) => {
    const definite = new Set(entries.filter((e) => !e.tentative && !JUNK_LABELS.includes(e.label)).map((e) => e.label));
    if (definite.size > 1) {
      contamination++;
      console.log('    CONTAMINATED ' + personId + ': ' + [...definite].join(' + '));
    }
  });
  check('no LVFace person mixes two definitely-labelled people', contamination === 0, String(contamination));

  CROSS_VIDEO_LABELS.forEach((label) => {
    const videosWithLabel = new Set();
    let mergedVideos = new Set();
    lvPersonLabels.forEach((entries) => {
      const mine = entries.filter((e) => e.label === label && !e.tentative);
      mine.forEach((e) => videosWithLabel.add(e.videoHash));
      if (mine.length > 0) {
        const vs = new Set(mine.map((e) => e.videoHash));
        if (vs.size > mergedVideos.size) {
          mergedVideos = vs;
        }
      }
    });
    report(label + ': labelled faces span videos ' + [...videosWithLabel].join(',') +
      '; largest single LVFace person spans ' + [...mergedVideos].join(','));
  });

  // Cross-video merges, Phase 2 semantics: for each Phase 0 person at
  // T=0.50 whose groups span two videos, take the app faces whose
  // quality group transferred into one of that person's Phase 0 groups
  // (sim >= 0.45, labelled) and see which persons of the target index
  // they sit in. Run against the LVFace index and the quality index
  // (control). Gated only for Davies, the one cross-video person with
  // full label coverage in both videos and an 11-face sample; the other
  // merges are reported, because split-side differences between
  // recognizers are expected and documented (LVFACE_PACK_RESULTS.md),
  // while contamination (the hard invariant) is gated above. When the
  // transferred faces span only one video the sample cannot judge the
  // merge at all (the Phase 2 harness has the same blind spot).
  const sweep = JSON.parse(fs.readFileSync(path.join(CT, 'sweep.json'), 'utf8'))['0.50'];
  const faceP0 = new Map(); // faceId -> quality-group transfer entry
  Object.entries(qIndex2.videos).forEach(([hash, v]) => v.faces.forEach((f) => {
    faceP0.set(f.faceId, qGroupLabel.get(hash + '#' + f.groupId));
  }));
  const personIdOfFace = (index) => {
    const m = new Map();
    Object.values(index.videos).forEach((v) => v.faces.forEach((f) => {
      if (f.personId) {
        m.set(f.faceId, f.personId);
      }
    }));
    return m;
  };
  const crossVideoP0 = sweep.filter((p) => new Set(p.groupKeys.map((k) => k.split('#')[0])).size > 1);
  const checkMerges = (index, tag, gate) => {
    const byFace = personIdOfFace(index);
    crossVideoP0.forEach((p) => {
      const pLabels = [...new Set(p.groupKeys.map((k) => (labels[k] || {}).person).filter(Boolean))];
      const persons = new Set();
      const videos = new Set();
      Object.entries(qIndex2.videos).forEach(([hash, v]) => v.faces.forEach((f) => {
        const tr = faceP0.get(f.faceId);
        if (tr && tr.sim >= 0.45 && tr.label && p.groupKeys.includes(tr.p0Key)) {
          const pid = byFace.get(f.faceId);
          if (pid) {
            persons.add(pid);
          }
          videos.add(hash);
        }
      }));
      const name = p.personId + ' (' + pLabels.join('/') + ')';
      if (videos.size < 2) {
        report('[' + tag + '] ' + name + ': transferred faces span only ' +
          [...videos].join(',') + ' -- merge not judgeable from labels');
        return;
      }
      const ok = persons.size === 1;
      if (gate && pLabels.every((l) => l === 'Greg Davies')) {
        check('[' + tag + '] Phase 0 ' + name + ' merged into one person', ok, [...persons].join(','));
      } else {
        report('[' + tag + '] Phase 0 ' + name + ' -> persons ' + [...persons].join(',') + (ok ? ' (merged)' : ' (SPLIT)'));
      }
    });
  };
  checkMerges(qIndex2, 'quality-control', false);
  checkMerges(lvIndex, 'lvface', true);

  // Per-person label summary vs the quality run, for the results doc
  console.log('  LVFace persons and their transferred labels:');
  lvPersonLabels.forEach((entries, personId) => {
    const counts = {};
    entries.forEach((e) => {
      counts[e.label + (e.tentative ? ' (tentative)' : '')] = (counts[e.label + (e.tentative ? ' (tentative)' : '')] || 0) + 1;
    });
    console.log('    ' + personId + ': ' + Object.entries(counts).map(([k, n]) => k + ' x' + n).join(' | '));
  });

  console.log('[attributes] genderage aggregates over LVFace persons');
  const ga = await GenderAge.create(GA);
  const frameCache = new Map();
  const attrResults = [];
  for (const person of lvIndex.persons) {
    const samples = [];
    for (const hash of person.videoHashes) {
      const entry = lvIndex.videos[hash];
      const byGroup = new Map();
      entry.faces.forEach((face) => {
        if (face.personId === person.personId && face.groupId >= 0) {
          if (!byGroup.has(face.groupId)) {
            byGroup.set(face.groupId, []);
          }
          byGroup.get(face.groupId).push(face);
        }
      });
      for (const members of byGroup.values()) {
        for (const face of selectGroupSamples(members, 5)) {
          const cacheKey = hash + '@' + face.t;
          if (!frameCache.has(cacheKey)) {
            frameCache.set(cacheKey, await grabFrameImage(
              path.join(VIDEOS_DIR, hash + '.mp4'), face.t,
              '/tmp/lvface-pack-harness-attr-' + frameCache.size + '.jpg'));
          }
          const img = frameCache.get(cacheKey);
          if (!img) {
            continue;
          }
          samples.push(await ga.get(img, {
            bbox: [face.bbox[0] * img.width, face.bbox[1] * img.height,
                   (face.bbox[0] + face.bbox[2]) * img.width, (face.bbox[1] + face.bbox[3]) * img.height],
            score: face.score,
            kps: [],
          }));
        }
      }
    }
    const attributes = aggregateAttributes(samples);
    if (attributes) {
      attrResults.push({ personId: person.personId, attributes });
    }
  }
  report('aggregates computed for ' + attrResults.length + '/' + lvIndex.persons.length + ' LVFace persons');
  let sexChecked = 0;
  let sexOk = 0;
  attrResults.forEach(({ personId, attributes }) => {
    const entries = lvPersonLabels.get(personId) || [];
    const counts = {};
    entries.forEach((e) => {
      if (!e.tentative && KNOWN_SEX[e.label]) {
        counts[e.label] = (counts[e.label] || 0) + 1;
      }
    });
    const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (ranked.length === 0) {
      return;
    }
    const total = ranked.reduce((n, [, c]) => n + c, 0);
    if (ranked[0][1] * 2 <= total) {
      return; // no strict-majority label: not a judgeable person
    }
    const label = ranked[0][0];
    sexChecked++;
    const ok = attributes.sex === KNOWN_SEX[label];
    if (ok) {
      sexOk++;
    }
    console.log('    ' + personId + ' [' + label + '] sex=' + attributes.sex + ' (known ' + KNOWN_SEX[label] + ')' +
      ' ageMedian=' + attributes.ageMedian + ' samples=' + attributes.sampleCount + ' agreement=' + attributes.sexAgreement);
    check('sex matches for ' + label + ' (' + personId + ')', ok, attributes.sex);
  });
  check('sex checked for at least 4 labelled LVFace persons', sexChecked >= 4,
    sexChecked + ' persons, ' + sexOk + ' ok');

  lvStore.setPersonsAttributes(attrResults);
  const rawIndex = fs.readFileSync(path.join(lvRun.facesDir, 'index.json'), 'utf8');
  const parsed = JSON.parse(rawIndex);
  const anyFaceAttr = Object.values(parsed.videos).some((v) => v.faces.some((f) => 'attr' in f || 'age' in f || 'sex' in f));
  check('no face record carries attribute fields (lvface index)', !anyFaceAttr);
  check('person-level aggregates persisted (lvface index)',
    parsed.persons.filter((p) => p.attributes).length === attrResults.length,
    parsed.persons.filter((p) => p.attributes).length + '/' + attrResults.length);

  console.log(failures === 0 ? 'HARNESS: ALL CHECKS PASSED' : 'HARNESS: ' + failures + ' CHECK(S) FAILED');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('HARNESS ERROR', err);
  process.exit(2);
});
