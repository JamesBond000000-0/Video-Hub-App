/*
 * Phase 2 verification harness (not part of the app).
 *
 * Drives the COMPILED app code (node/face/*.js) over the Phase 0 BBC test
 * videos, exactly like the Phase 1 harness, and then verifies:
 *   [cluster]   global clustering at T=0.50 against the Phase 0 ground
 *               truth (labels.json / groups.json / sweep.json): the three
 *               cross-video persons (Davies, Gosling, Attenborough) merge,
 *               and no person mixes two different labelled people.
 *   [stability] re-clustering preserves a renamed person's ID and the
 *               exact face->person assignment.
 *   [attributes] genderage aggregates via the worker's reference path
 *               (re-extracted frames): sex matches the known persons, age
 *               medians stay near the Phase 0 aggregates, and no per-face
 *               attribute values are persisted anywhere.
 *   [cache]     personsByVideo (the .vha2 cache source) is consistent
 *               with the persons array.
 *
 * Usage: node tools/face-spike/phase2-harness.cjs
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
const HUB = '/tmp/phase2-harness';
const FACES_DIR = path.join(HUB, 'faces');

const VIDEO_IDS = ['yuXGpUR7fXA', 'dmbpagijVkk', 'V9DqzY1jPhY', '8cPxeyZ6-g0', 'z0B4KEXVaqo', 'Gfa88SeNohY'];
const KNOWN_SEX = {
  'Greg Davies': 'M', 'Russell Crowe': 'M', 'Ryan Gosling': 'M', 'Jodie Whittaker': 'F',
  'Lee Mack': 'M', 'John Cleese': 'M', 'Colin Farrell': 'M', 'Graham Norton': 'M',
  'David Attenborough': 'M',
};
const JUNK_LABELS = ['(ape - false positive)', 'Unknown (face not visible)'];

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
    console.log('  ' + id + ': faces=' + result.faces.length + ' groups=' + new Set(result.faces.map((f) => f.groupId)).size);
  }
  console.log('  total faces=' + totalFaces);
  return store;
}

async function main() {
  const store = await scanAll();

  console.log('[crops] display crops must not be black (Phase 1 regression check)');
  const { decodeImage } = require(path.join(REPO, 'node', 'face', 'image-ops.js'));
  const cropFiles = fs.readdirSync(path.join(FACES_DIR, 'crops'));
  let blackCrops = 0;
  for (const file of cropFiles) {
    const img = await decodeImage(path.join(FACES_DIR, 'crops', file));
    let sum = 0;
    for (let i = 0; i < img.data.length; i += 99) {
      sum += img.data[i];
    }
    if (sum / (img.data.length / 99) < 5) {
      blackCrops++;
    }
  }
  check('all display crops have real content', blackCrops === 0,
    cropFiles.length + ' crops, ' + blackCrops + ' black');

  // ---------------------------------------------------------------
  console.log('[cluster] global clustering at T=' + T);
  const result = clusterizeIndex(store.getIndex(), (idx) => store.readEmbedding(idx), T);
  store.applyClustering(result, T);
  const store2 = FaceStore.load(FACES_DIR, MODEL_ID, T);
  const index = store2.getIndex();
  const persons = index.persons;
  console.log('  persons=' + persons.length + ' (Phase 0 at T=0.50: 18, from 21 groups; app has more groups)');
  check('persons persisted after reload', persons.length === result.persons.length && persons.length > 0,
    String(persons.length));
  const facesWithPerson = Object.values(index.videos).reduce(
    (n, v) => n + v.faces.filter((f) => f.personId).length, 0);
  const totalFaces = Object.values(index.videos).reduce((n, v) => n + v.faces.length, 0);
  check('every grouped face has a personId', facesWithPerson === totalFaces,
    facesWithPerson + '/' + totalFaces);

  // ---------------------------------------------------------------
  console.log('[labels] transfer Phase 0 labels via representative embeddings');
  const p0Groups = JSON.parse(fs.readFileSync(path.join(CT, 'groups.json'), 'utf8'));
  const labels = JSON.parse(fs.readFileSync(path.join(CT, 'labels.json'), 'utf8')).groups;
  const sweep = JSON.parse(fs.readFileSync(path.join(CT, 'sweep.json'), 'utf8'))['0.50'];
  const p0ByVideo = {};
  p0Groups.forEach((g) => {
    (p0ByVideo[g.videoId] = p0ByVideo[g.videoId] || []).push(g);
  });
  const items = collectGroupItems(index, (idx) => store2.readEmbedding(idx));
  // app group key -> { p0Key, label, sim } (best Phase 0 group in same video)
  const transfer = new Map();
  items.forEach((item) => {
    let best = null;
    let bestSim = -Infinity;
    (p0ByVideo[item.videoHash] || []).forEach((g) => {
      const s = dot(item.repEmbedding, g.repEmbedding);
      if (s > bestSim) {
        bestSim = s;
        best = g;
      }
    });
    transfer.set(item.key, {
      p0Key: best ? best.groupKey : '',
      label: best && bestSim >= 0.45 ? (labels[best.groupKey] || {}).person || '' : '',
      tentative: best ? !!((labels[best.groupKey] || {}).tentative) : true,
      sim: bestSim,
    });
  });
  const matched = [...transfer.values()].filter((t) => t.label).length;
  // Coverage is a report, not a gate: the app samples different frames
  // than Phase 0, so some app groups show people/faces Phase 0 never
  // captured (e.g. the interviewer in the Attenborough video, Norton in
  // wide shots) and legitimately have no Phase 0 counterpart.
  report('label transfer coverage', matched + '/' + transfer.size + ' app groups matched a Phase 0 group at sim>=0.45');

  // app person -> its groups with labels
  const personGroups = new Map(); // personId -> [{key, label, tentative}]
  items.forEach((item) => {
    const face = Object.values(index.videos).reduce((acc, v) => acc || v.faces.filter((f) => f.faceId === item.repFaceId)[0], null);
    const personId = face ? face.personId : '';
    if (!personGroups.has(personId)) {
      personGroups.set(personId, []);
    }
    personGroups.get(personId).push({ key: item.key, ...transfer.get(item.key) });
  });

  console.log('  app persons and their transferred labels:');
  personGroups.forEach((groups, personId) => {
    const summary = groups.map((g) => g.label || '(unmatched ' + g.sim.toFixed(2) + ')').join(' | ');
    console.log('    ' + personId + ': ' + summary);
  });

  // contamination: a person must not contain two different definite labels.
  // Tentative Phase 0 labels (the "Unknown guest B" pair) are excluded from
  // the gate -- Phase 0 itself was unsure about them -- but reported.
  let contamination = 0;
  personGroups.forEach((groups, personId) => {
    const definite = new Set(groups.filter((g) => g.label && !g.tentative && !JUNK_LABELS.includes(g.label)).map((g) => g.label));
    if (definite.size > 1) {
      contamination++;
      console.log('    CONTAMINATED ' + personId + ': ' + [...definite].join(' + '));
    }
    const tentative = new Set(groups.filter((g) => g.label && g.tentative).map((g) => g.label));
    if (tentative.size > 0 && definite.size > 0) {
      report('person ' + personId + ' also holds tentative-labelled groups', [...tentative].join(', '));
    }
  });
  check('no person mixes two different definitely-labelled people', contamination === 0, String(contamination));

  // merges: for each Phase 0 person (T=0.50) whose groups span 2 videos,
  // all app groups transferred to its groups must share one app person.
  const p0PersonOfGroup = new Map();
  sweep.forEach((p) => p.groupKeys.forEach((k) => p0PersonOfGroup.set(k, p.personId)));
  const crossVideo = sweep.filter((p) => new Set(p.groupKeys.map((k) => k.split('#')[0])).size > 1);
  console.log('  Phase 0 cross-video persons at T=0.50: ' + crossVideo.map((p) => p.personId).join(', '));
  crossVideo.forEach((p) => {
    const appPersons = new Set();
    personGroups.forEach((groups, personId) => {
      if (groups.some((g) => p.groupKeys.includes(g.p0Key) && g.label)) {
        appPersons.add(personId);
      }
    });
    check('Phase 0 ' + p.personId + ' (' + p.groupKeys.join(', ') + ') merged into one app person',
      appPersons.size === 1, [...appPersons].join(','));
  });

  // ---------------------------------------------------------------
  console.log('[stability] rename + recluster keeps IDs and assignment');
  const assignmentBefore = new Map();
  Object.values(index.videos).forEach((v) => v.faces.forEach((f) => assignmentBefore.set(f.faceId, f.personId)));
  // the person holding the most Davies-labelled groups gets a name
  let daviesPerson = '';
  let daviesCount = 0;
  personGroups.forEach((groups, personId) => {
    const n = groups.filter((g) => g.label === 'Greg Davies').length;
    if (n > daviesCount) {
      daviesCount = n;
      daviesPerson = personId;
    }
  });
  check('a Davies person exists', daviesPerson !== '', daviesPerson);
  store2.renamePerson(daviesPerson, 'Greg Davies');
  const result2 = clusterizeIndex(store2.getIndex(), (idx) => store2.readEmbedding(idx), T);
  store2.applyClustering(result2, T);
  const store3 = FaceStore.load(FACES_DIR, MODEL_ID, T);
  const renamed = store3.getIndex().persons.filter((p) => p.personId === daviesPerson)[0];
  check('renamed person keeps ID and name after re-clustering', !!renamed && renamed.name === 'Greg Davies',
    renamed ? renamed.name : 'person gone');
  let assignmentSame = true;
  Object.values(store3.getIndex().videos).forEach((v) => v.faces.forEach((f) => {
    if (assignmentBefore.get(f.faceId) !== f.personId) {
      assignmentSame = false;
    }
  }));
  check('face->person assignment identical after re-clustering', assignmentSame);
  const result3 = clusterizeIndex(store3.getIndex(), (idx) => store3.readEmbedding(idx), T);
  check('clusterize is deterministic (same persons out)', JSON.stringify(result3.persons.map((p) => p.personId)) ===
    JSON.stringify(store3.getIndex().persons.map((p) => p.personId)));

  // ---------------------------------------------------------------
  console.log('[attributes] genderage aggregates via re-extracted frames');
  const ga = await GenderAge.create(GA);
  const index3 = store3.getIndex();
  const frameCache = new Map();
  const attrResults = [];
  for (const person of index3.persons) {
    const samples = [];
    for (const hash of person.videoHashes) {
      const entry = index3.videos[hash];
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
      attrResults.push({ personId: person.personId, attributes });
    }
  }
  console.log('  aggregates computed for ' + attrResults.length + ' persons');
  const p0Attrs = JSON.parse(fs.readFileSync(path.join(CT, 'attrs_T0.50.json'), 'utf8'));
  const p0AttrByPerson = new Map(p0Attrs.map((a) => [a.personId, a]));
  // Phase 0 person -> its definite label (for like-for-like age comparison)
  const p0PersonLabel = new Map();
  sweep.forEach((p) => {
    const ls = p.groupKeys.map((k) => (labels[k] || {}).person).filter((l) => l && KNOWN_SEX[l]);
    if (ls.length > 0 && ls.every((l) => l === ls[0])) {
      p0PersonLabel.set(p.personId, ls[0]);
    }
  });
  let sexChecked = 0;
  let sexOk = 0;
  attrResults.forEach(({ personId, attributes }) => {
    const groups = personGroups.get(personId) || [];
    const labelCounts = {};
    groups.forEach((g) => {
      if (g.label && KNOWN_SEX[g.label]) {
        labelCounts[g.label] = (labelCounts[g.label] || 0) + 1;
      }
    });
    const label = Object.keys(labelCounts).sort((a, b) => labelCounts[b] - labelCounts[a])[0];
    if (!label) {
      return;
    }
    sexChecked++;
    const okSex = attributes.sex === KNOWN_SEX[label];
    if (okSex) {
      sexOk++;
    }
    // Age comparison only for label-pure persons whose Phase 0 counterpart
    // person carries the same label; otherwise the two aggregates do not
    // describe the same population and a diff is meaningless.
    const distinctLabels = Object.keys(labelCounts);
    const p0Counts = {};
    groups.forEach((g) => {
      const pp = p0PersonOfGroup.get(g.p0Key);
      if (pp) {
        p0Counts[pp] = (p0Counts[pp] || 0) + 1;
      }
    });
    const p0Id = Object.keys(p0Counts).sort((a, b) => p0Counts[b] - p0Counts[a])[0];
    const comparable = distinctLabels.length === 1 && p0Id && p0PersonLabel.get(p0Id) === label &&
      p0AttrByPerson.get(p0Id);
    const p0Age = comparable ? p0AttrByPerson.get(p0Id).ageMedian : null;
    console.log('    ' + personId + ' [' + label + '] sex=' + attributes.sex + ' (known ' + KNOWN_SEX[label] + ')' +
      ' ageMedian=' + attributes.ageMedian + (p0Age !== null ? ' (Phase 0: ' + p0Age + ', diff ' + Math.abs(attributes.ageMedian - p0Age) + ')' : ' (no like-for-like Phase 0 aggregate)') +
      ' samples=' + attributes.sampleCount + ' agreement=' + attributes.sexAgreement);
    check('sex matches for ' + label + ' (' + personId + ')', okSex, attributes.sex);
    if (p0Age !== null) {
      check('age median within 15y of Phase 0 for ' + label, Math.abs(attributes.ageMedian - p0Age) <= 15,
        attributes.ageMedian + ' vs ' + p0Age);
    }
  });
  check('sex checked for at least 5 known persons', sexChecked >= 5, sexChecked + ' persons, ' + sexOk + ' ok');

  store3.setPersonsAttributes(attrResults);
  const rawIndex = fs.readFileSync(path.join(FACES_DIR, 'index.json'), 'utf8');
  check('no per-face "attr" persisted in index.json', !rawIndex.includes('"attr"'));
  const parsed = JSON.parse(rawIndex);
  const anyFaceAttr = Object.values(parsed.videos).some((v) => v.faces.some((f) => 'attr' in f || 'age' in f || 'sex' in f));
  check('no face record carries attribute fields', !anyFaceAttr);
  const personsWithAttrs = parsed.persons.filter((p) => p.attributes).length;
  check('person-level aggregates persisted', personsWithAttrs === attrResults.length,
    personsWithAttrs + '/' + attrResults.length);

  // ---------------------------------------------------------------
  console.log('[cache] personsByVideo consistency');
  const byVideo = store3.personsByVideo();
  parsed.persons.forEach((person) => {
    person.videoHashes.forEach((hash) => {
      check('personsByVideo[' + hash + '] contains ' + person.personId,
        (byVideo[hash] || []).includes(person.personId));
    });
  });
  const davies = parsed.persons.filter((p) => p.personId === daviesPerson)[0];
  check('Davies person spans both Davies videos',
    !!davies && davies.videoHashes.includes('yuXGpUR7fXA') && davies.videoHashes.includes('V9DqzY1jPhY'),
    davies ? davies.videoHashes.join(',') : 'none');

  console.log('[disabled] clearing aggregates removes them');
  store3.clearPersonsAttributes();
  const parsed2 = JSON.parse(fs.readFileSync(path.join(FACES_DIR, 'index.json'), 'utf8'));
  check('no person keeps attributes after clear', parsed2.persons.every((p) => !('attributes' in p)));

  console.log(failures === 0 ? 'HARNESS: ALL CHECKS PASSED' : 'HARNESS: ' + failures + ' CHECK(S) FAILED');
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('HARNESS ERROR', err);
  process.exit(2);
});
