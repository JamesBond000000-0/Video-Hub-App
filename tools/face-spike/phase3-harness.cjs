/*
 * Phase 3 verification harness (not part of the app).
 *
 * Drives the COMPILED app code (node/face/*.js) over the Phase 0 BBC
 * test videos -- same scan as the Phase 1/2 harnesses -- and then
 * exercises the manual corrections through the FaceStore API:
 *   [merge]      the split Gosling persons merge into the named one
 *   [split]      Davies splits by video; the new person gets a fresh
 *                sequence ID (the merged-away ID is not recycled)
 *   [move]       a group moves to unassigned and then to another
 *                person; empty persons disappear
 *   [representative] a person's representative face is overridden
 *   [attributes] composition-changing edits invalidate aggregates;
 *                untouched persons keep theirs; recomputation after
 *                the edit reflects the new composition
 *   [recluster]  THE core acceptance: after all edits, a full
 *                re-clustering preserves every edit (assignment, IDs,
 *                names, representative override, person count), and a
 *                second re-clustering is a no-op
 *   [cache]      personsByVideo (the .vha2 cache source) stays
 *                consistent with the persons array throughout
 *
 * Usage: node tools/face-spike/phase3-harness.cjs
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
const { selectGroupSamples, aggregateAttributes } = require(path.join(REPO, 'node', 'face', 'face-types.js'));
const ffmpegPath = require(path.join(REPO, 'node_modules', 'ffmpeg-static'));

const VIDEOS_DIR = path.join(CT, 'videos');
const MODELS = path.join(__dirname, 'models', 'buffalo_l');
const DET = path.join(MODELS, 'det_10g.onnx');
const REC = path.join(MODELS, 'w600k_r50.onnx');
const GA = path.join(MODELS, 'genderage.onnx');
const MODEL_ID = 'buffalo_l-r50';
const T = 0.5;
const HUB = '/tmp/phase3-harness';
const FACES_DIR = path.join(HUB, 'faces');

const VIDEO_IDS = ['yuXGpUR7fXA', 'dmbpagijVkk', 'V9DqzY1jPhY', '8cPxeyZ6-g0', 'z0B4KEXVaqo', 'Gfa88SeNohY'];

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

async function scanAll() {
  console.log('[scan] driving FaceEngine over the Phase 0 videos');
  fs.rmSync(HUB, { recursive: true, force: true });
  fs.mkdirSync(FACES_DIR, { recursive: true });
  const store = FaceStore.load(FACES_DIR, MODEL_ID, T);
  const engine = await FaceEngine.create(DET, REC);
  let totalFaces = 0;
  for (const id of VIDEO_IDS) {
    const videoPath = path.join(VIDEOS_DIR, id + '.mp4');
    const result = await engine.scanVideo({
      videoPath,
      duration: probeDuration(videoPath),
      hash: id,
      facesDir: FACES_DIR,
      maxFramesPerVideo: 60,
      sampleIntervalSec: 30,
      minFaceSizePx: 40,
      detModelPath: DET,
      recModelPath: REC,
    });
    store.setVideo(id, result.status, result.frameCount, result.faces);
    totalFaces += result.faces.length;
  }
  console.log('  total faces=' + totalFaces);
  return store;
}

function loadStore() {
  return FaceStore.load(FACES_DIR, MODEL_ID, T);
}

function personOf(index, personId) {
  return index.persons.filter((p) => p.personId === personId)[0];
}

// group keys of a person, derived from the index (hash from video key)
function keysOfPerson(index, personId) {
  const keys = new Set();
  Object.keys(index.videos).forEach((hash) => {
    index.videos[hash].faces.forEach((f) => {
      if (f.personId === personId && f.groupId >= 0) {
        keys.add(hash + '#' + f.groupId);
      }
    });
  });
  return [...keys];
}

function facesOfGroup(index, key) {
  const sep = key.lastIndexOf('#');
  const hash = key.substring(0, sep);
  const groupId = parseInt(key.substring(sep + 1), 10);
  const entry = index.videos[hash];
  return entry ? entry.faces.filter((f) => f.groupId === groupId) : [];
}

function assignmentSnapshot(index) {
  const map = new Map();
  Object.values(index.videos).forEach((v) => v.faces.forEach((f) => map.set(f.faceId, f.personId || '')));
  return map;
}

async function computeAttributes(store, index, personIds, ga, frameCache) {
  const results = [];
  for (const person of index.persons) {
    if (personIds && !personIds.includes(person.personId)) {
      continue;
    }
    const samples = [];
    for (const hash of person.videoHashes) {
      const entry = index.videos[hash];
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
              path.join(HUB, 'attr-frame-' + frameCache.size + '.jpg')));
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
      results.push({ personId: person.personId, attributes });
    }
  }
  return results;
}

async function main() {
  const store = await scanAll();

  // ---------------------------------------------------------------
  console.log('[baseline] cluster at T=' + T);
  const baseResult = clusterizeIndex(store.getIndex(), (idx) => store.readEmbedding(idx), T);
  store.applyClustering(baseResult, T);
  let s = loadStore();
  let index = s.getIndex();
  check('baseline persons = 24 (Phase 2 state)', index.persons.length === 24, String(index.persons.length));
  const baselineNextSeq = index.nextPersonSeq;
  const baselineIds = new Set(index.persons.map((p) => p.personId));
  report('baseline nextPersonSeq', String(baselineNextSeq));

  // ---------------------------------------------------------------
  console.log('[labels] transfer Phase 0 labels to identify the test persons');
  const p0Groups = JSON.parse(fs.readFileSync(path.join(CT, 'groups.json'), 'utf8'));
  const labels = JSON.parse(fs.readFileSync(path.join(CT, 'labels.json'), 'utf8')).groups;
  const p0ByVideo = {};
  p0Groups.forEach((g) => {
    (p0ByVideo[g.videoId] = p0ByVideo[g.videoId] || []).push(g);
  });
  const items = collectGroupItems(index, (idx) => s.readEmbedding(idx));
  const labelOfItem = new Map();
  items.forEach((item) => {
    let best = null;
    let bestSim = -Infinity;
    (p0ByVideo[item.videoHash] || []).forEach((g) => {
      const sim = dot(item.repEmbedding, g.repEmbedding);
      if (sim > bestSim) {
        bestSim = sim;
        best = g;
      }
    });
    const entry = best ? (labels[best.groupKey] || {}) : {};
    // Identification only needs the GROUP match to be certain (sim >=
    // 0.45 against the Phase 0 group of the same video). The Phase 0
    // label's tentative flag described Phase 0's uncertainty about the
    // person's identity (e.g. the degraded Gosling fragment, matched
    // here at sim 0.928), not about the group correspondence.
    labelOfItem.set(item.key, {
      label: best && bestSim >= 0.45 ? entry.person || '' : '',
    });
  });
  const personLabels = new Map(); // personId -> Map(label -> group keys)
  items.forEach((item) => {
    const face = facesOfGroup(index, item.key).filter((f) => f.faceId === item.repFaceId)[0];
    const personId = face ? face.personId : '';
    if (!personLabels.has(personId)) {
      personLabels.set(personId, new Map());
    }
    const m = personLabels.get(personId);
    const label = labelOfItem.get(item.key).label;
    if (label) {
      if (!m.has(label)) {
        m.set(label, []);
      }
      m.get(label).push(item.key);
    }
  });
  function personsWithLabel(label) {
    const out = [];
    personLabels.forEach((m, personId) => {
      if (m.has(label)) {
        out.push({ personId, keys: m.get(label) });
      }
    });
    return out;
  }
  const goslings = personsWithLabel('Ryan Gosling');
  check('two Gosling persons at baseline (main + fragment)', goslings.length === 2,
    goslings.map((g) => g.personId).join(','));
  goslings.sort((a, b) => (personOf(index, b.personId).faceCount - personOf(index, a.personId).faceCount));
  const goslingMain = goslings[0].personId;
  const goslingFragment = goslings[1].personId;
  const davies = personsWithLabel('Greg Davies');
  check('one Davies person spanning two videos', davies.length === 1 &&
    new Set(davies[0].keys.map((k) => k.split('#')[0])).size === 2,
    davies.map((d) => d.personId + ':' + d.keys.join(',')).join(' '));
  const daviesId = davies[0].personId;
  const daviesV9Key = davies[0].keys.filter((k) => k.startsWith('V9DqzY1jPhY#'))[0];
  check('Davies has a V9Dq group to split off', !!daviesV9Key, daviesV9Key || 'none');
  const attens = personsWithLabel('David Attenborough');
  const attenOld = attens.filter((a) => new Set(a.keys.map((k) => k.split('#')[0])).size === 2)
    .map((a) => a.personId)[0];
  check('one Attenborough person spanning z0B4 + Gfa88', !!attenOld, attenOld || 'none');
  const nortons = personsWithLabel('Graham Norton');
  nortons.sort((a, b) => keysOfPerson(index, b.personId).length - keysOfPerson(index, a.personId).length);
  const nortonId = nortons.length > 0 ? nortons[0].personId : '';
  check('a Norton person with >= 2 groups (representative test)',
    !!nortonId && keysOfPerson(index, nortonId).length >= 2,
    nortonId + ' groups=' + (nortonId ? keysOfPerson(index, nortonId).length : 0));
  const mackId = (personsWithLabel('Lee Mack')[0] || {}).personId || '';
  check('a Mack person exists (untouched control)', !!mackId, mackId);
  if (failures > 0) {
    console.log('HARNESS: identification failed, aborting before the edit tests');
    process.exit(1);
  }

  // ---------------------------------------------------------------
  console.log('[attributes] compute aggregates for everyone BEFORE the edits');
  const ga = await GenderAge.create(GA);
  const frameCache = new Map();
  const attrResults = await computeAttributes(s, index, null, ga, frameCache);
  s.setPersonsAttributes(attrResults);
  s = loadStore();
  index = s.getIndex();
  check('control persons carry attributes before edits',
    !!(personOf(index, mackId).attributes && personOf(index, goslingMain).attributes &&
       personOf(index, goslingFragment).attributes && personOf(index, daviesId).attributes &&
       personOf(index, attenOld).attributes),
    'computed for ' + attrResults.length + ' persons');

  // ---------------------------------------------------------------
  console.log('[merge] merge the Gosling fragment into the named main person');
  s.renamePerson(goslingMain, 'Ryan Gosling');
  const fragmentFaceCount = personOf(index, goslingFragment).faceCount;
  const mainFaceCount = personOf(index, goslingMain).faceCount;
  const survivor = s.mergePersons([goslingFragment, goslingMain]); // fragment listed first on purpose
  check('survivor is the named person', survivor === goslingMain, String(survivor));
  s = loadStore();
  index = s.getIndex();
  check('person count 24 -> 23', index.persons.length === 23, String(index.persons.length));
  check('fragment person is gone', !personOf(index, goslingFragment));
  const merged = personOf(index, goslingMain);
  check('merged person keeps the name', merged.name === 'Ryan Gosling', merged.name);
  check('merged faceCount is the sum', merged.faceCount === mainFaceCount + fragmentFaceCount,
    merged.faceCount + ' = ' + mainFaceCount + ' + ' + fragmentFaceCount);
  check('merged person attributes invalidated', !merged.attributes);
  check('untouched Mack keeps attributes through merge', !!personOf(index, mackId).attributes);
  const byVideoAfterMerge = s.personsByVideo();
  check('personsByVideo no longer lists the fragment anywhere',
    !Object.values(byVideoAfterMerge).some((ids) => ids.includes(goslingFragment)));

  // ---------------------------------------------------------------
  console.log('[split] split the Davies V9Dq group into a new person');
  const newId = s.splitPerson(daviesId, [daviesV9Key]);
  check('split returned a new person ID', !!newId, String(newId));
  check('new ID is minted from the sequence (not a recycled one)',
    newId === 'p' + String(baselineNextSeq).padStart(6, '0') && !baselineIds.has(newId) && newId !== goslingFragment,
    newId + ' vs expected p' + String(baselineNextSeq).padStart(6, '0'));
  s = loadStore();
  index = s.getIndex();
  check('person count 23 -> 24', index.persons.length === 24, String(index.persons.length));
  check('Davies keeps only the yuXG video',
    JSON.stringify(personOf(index, daviesId).videoHashes) === JSON.stringify(['yuXGpUR7fXA']),
    personOf(index, daviesId).videoHashes.join(','));
  check('new person holds only the V9Dq video',
    !!personOf(index, newId) && JSON.stringify(personOf(index, newId).videoHashes) === JSON.stringify(['V9DqzY1jPhY']));
  check('split faces moved to the new person',
    facesOfGroup(index, daviesV9Key).every((f) => f.personId === newId));
  check('Davies attributes invalidated by the split', !personOf(index, daviesId).attributes);

  // ---------------------------------------------------------------
  console.log('[move] move the split-off group to unassigned, then to Attenborough');
  check('move to unassigned applies', s.moveGroups([daviesV9Key], null) === true);
  s = loadStore();
  index = s.getIndex();
  check('emptied new person disappears', !personOf(index, newId));
  check('person count back to 23', index.persons.length === 23, String(index.persons.length));
  check('moved group faces are unassigned',
    facesOfGroup(index, daviesV9Key).every((f) => !f.personId));
  check('personsByVideo drops the video for the removed person',
    !(s.personsByVideo()['V9DqzY1jPhY'] || []).includes(newId));
  check('move to Attenborough applies', s.moveGroups([daviesV9Key], attenOld) === true);
  s = loadStore();
  index = s.getIndex();
  check('moved group faces now belong to Attenborough',
    facesOfGroup(index, daviesV9Key).every((f) => f.personId === attenOld));
  check('Attenborough now spans the V9Dq video too',
    personOf(index, attenOld).videoHashes.includes('V9DqzY1jPhY'),
    personOf(index, attenOld).videoHashes.join(','));
  check('Attenborough attributes invalidated by the move', !personOf(index, attenOld).attributes);
  check('person count still 23', index.persons.length === 23, String(index.persons.length));

  // ---------------------------------------------------------------
  console.log('[representative] override the Norton representative face');
  const nortonGroups = keysOfPerson(index, nortonId)
    .map((key) => ({ key, rep: facesOfGroup(index, key).filter((f) => f.representative)[0] }))
    .filter((g) => g.rep);
  const nortonBefore = personOf(index, nortonId).representativeFaceId;
  const otherRep = nortonGroups.map((g) => g.rep.faceId).filter((id) => id !== nortonBefore)[0];
  check('another group representative exists', !!otherRep, otherRep || 'none');
  check('override applies', s.setPersonRepresentative(nortonId, otherRep) === true);
  check('override of a foreign face is rejected',
    s.setPersonRepresentative(nortonId, facesOfGroup(index, daviesV9Key)[0].faceId) === false);
  s = loadStore();
  index = s.getIndex();
  check('representative face changed', personOf(index, nortonId).representativeFaceId === otherRep,
    personOf(index, nortonId).representativeFaceId);

  // snapshot of the fully edited state
  const editedAssignment = assignmentSnapshot(index);
  const editedPersonIds = index.persons.map((p) => p.personId).sort();
  const editedCount = index.persons.length;

  // ---------------------------------------------------------------
  console.log('[recluster] full re-clustering must preserve every edit');
  const rc = clusterizeIndex(index, (idx) => s.readEmbedding(idx), T);
  s.applyClustering(rc, T);
  s = loadStore();
  index = s.getIndex();
  check('person count preserved', index.persons.length === editedCount,
    index.persons.length + ' vs ' + editedCount);
  check('person ID set preserved',
    JSON.stringify(index.persons.map((p) => p.personId).sort()) === JSON.stringify(editedPersonIds));
  let assignmentSame = true;
  Object.values(index.videos).forEach((v) => v.faces.forEach((f) => {
    if (editedAssignment.get(f.faceId) !== (f.personId || '')) {
      assignmentSame = false;
      console.log('    assignment changed: ' + f.faceId + ' ' + editedAssignment.get(f.faceId) + ' -> ' + (f.personId || ''));
    }
  }));
  check('face->person assignment identical after re-clustering', assignmentSame);
  check('merged Gosling keeps ID and name',
    !!personOf(index, goslingMain) && personOf(index, goslingMain).name === 'Ryan Gosling');
  check('merged-away fragment ID did not come back', !personOf(index, goslingFragment));
  check('split-off-then-moved person ID did not come back', !personOf(index, newId));
  check('Davies did not regain the moved group',
    facesOfGroup(index, daviesV9Key).every((f) => f.personId === attenOld));
  check('representative override preserved',
    personOf(index, nortonId).representativeFaceId === otherRep,
    personOf(index, nortonId).representativeFaceId);
  check('untouched Mack still has attributes', !!personOf(index, mackId).attributes);
  check('edited persons have no stale attributes',
    !personOf(index, goslingMain).attributes && !personOf(index, attenOld).attributes &&
    !personOf(index, daviesId).attributes);
  check('pins recorded: moved group pinned to Attenborough',
    !!index.manualGroups && index.manualGroups[daviesV9Key] === attenOld,
    index.manualGroups ? index.manualGroups[daviesV9Key] : 'no pins');
  check('pins recorded: representative override stored',
    !!index.manualRepFaces && index.manualRepFaces[nortonId] === otherRep);

  console.log('[recluster] second re-clustering is a no-op');
  const before2 = assignmentSnapshot(index);
  const rc2 = clusterizeIndex(index, (idx) => s.readEmbedding(idx), T);
  s.applyClustering(rc2, T);
  s = loadStore();
  index = s.getIndex();
  let same2 = index.persons.length === editedCount;
  Object.values(index.videos).forEach((v) => v.faces.forEach((f) => {
    if (before2.get(f.faceId) !== (f.personId || '')) {
      same2 = false;
    }
  }));
  check('assignment and person count stable over a second re-clustering', same2);

  // ---------------------------------------------------------------
  console.log('[cache] personsByVideo consistency after all edits');
  const byVideo = s.personsByVideo();
  let cacheOk = true;
  index.persons.forEach((person) => {
    person.videoHashes.forEach((hash) => {
      if (!(byVideo[hash] || []).includes(person.personId)) {
        cacheOk = false;
        console.log('    personsByVideo[' + hash + '] misses ' + person.personId);
      }
    });
  });
  check('personsByVideo contains every person for its videos', cacheOk);
  const totalFaces = Object.values(index.videos).reduce((n, v) => n + v.faces.length, 0);
  check('face total unchanged (72)', totalFaces === 72, String(totalFaces));

  // ---------------------------------------------------------------
  console.log('[attributes] recompute after the edits reflects the new composition');
  const recomputed = await computeAttributes(s, index, [goslingMain], ga, frameCache);
  check('aggregate recomputed for the merged person', recomputed.length === 1,
    recomputed.length + ' results');
  if (recomputed.length === 1) {
    check('merged Gosling aggregate: sex M', recomputed[0].attributes.sex === 'M',
      recomputed[0].attributes.sex + ', samples=' + recomputed[0].attributes.sampleCount);
    s.setPersonsAttributes(recomputed);
    s = loadStore();
    index = s.getIndex();
    check('recomputed aggregate persisted on the merged person',
      !!personOf(index, goslingMain).attributes);
  }
  const rawIndex = fs.readFileSync(path.join(FACES_DIR, 'index.json'), 'utf8');
  check('no per-face "attr" persisted in index.json', !rawIndex.includes('"attr"'));

  console.log(failures === 0 ? 'HARNESS: ALL CHECKS PASSED' : 'HARNESS: ' + failures + ' CHECK(S) FAILED');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('HARNESS ERROR', err);
  process.exit(2);
});
