// LVFace vs ArcFace measured comparison on the Phase 0 BBC cluster set.
//
// All recognizers consume the *same* saved 112x112 aligned JPEGs (the
// Phase 0 SCRFD detections), the same face qualities and the same
// cluster.ts implementation:
//   §5.5 groupWithinVideo(T_in = 0.50), in the Phase 0 video order
//   §5.6 clusterGlobal() for T_global in 0.35..0.60
// Ground truth is inherited per face from the Phase 0 video-group labels
// (cluster-test/labels.json via cluster-test/groups.json).
//
// Usage:
//   npx tsx src/lvface-spike.ts [arcface] [t] [s] [b]
// With no arguments every model whose file exists is measured.
// Results are written to out/lvface-comparison.json (gitignored).

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { ArcFace } from "./arcface.js";
import { LVFace } from "./lvface.js";
import { decodeImage, type RawImage } from "./image-ops.js";
import { clusterGlobal, groupWithinVideo } from "./cluster.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CT = path.join(ROOT, "cluster-test");
const ALIGNED = path.join(CT, "faces", "aligned");
const T_IN = 0.5;
const THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
// Phase 0 processing order in cluster-test.ts. clusterGlobal() is a
// sequential leader-follower method, so this order is part of the method.
const VIDEOS = [
  "yuXGpUR7fXA",
  "dmbpagijVkk",
  "V9DqzY1jPhY",
  "8cPxeyZ6-g0",
  "z0B4KEXVaqo",
  "Gfa88SeNohY",
];
const REAL_PERSONS = [
  "Greg Davies",
  "Russell Crowe",
  "Ryan Gosling",
  "Jodie Whittaker",
  "Lee Mack",
  "John Cleese",
  "Unknown guest A (dmbp)",
  "Unknown guest B (dmbp)",
  "Colin Farrell",
  "Graham Norton",
  "David Attenborough",
];
const KEY_PERSONS = ["Greg Davies", "Ryan Gosling", "David Attenborough"];
const DET_MS_PER_FRAME_PHASE0 = 294.1; // det_10g median, Phase 0 RESULTS
const FACES_PER_VIDEO_PHASE0 = 30; // Phase 0 conversion: 30 frames x ~1 face

interface FaceMeta {
  faceId: string;
  videoId: string;
  quality: number;
  embedding: number[]; // Phase 0 stored ArcFace embedding (pre-JPEG)
}
interface OrigGroup {
  groupKey: string;
  videoId: string;
  memberFaceIds: string[];
  repFaceId: string;
  size: number;
}
interface LabelInfo {
  person?: string;
  tentative?: boolean;
  note?: string;
}
interface FaceLabel {
  person: string;
  tentative: boolean;
  isReal: boolean;
}
interface Recognizer {
  embedding(aligned: RawImage): Promise<Float32Array>;
}
interface ModelDef {
  key: string;
  label: string;
  file: string;
  create: () => Promise<Recognizer>;
  passes: number;
}

const MODEL_DEFS: ModelDef[] = [
  {
    key: "arcface",
    label: "ArcFace w600k_r50",
    file: path.join(ROOT, "models", "buffalo_l", "w600k_r50.onnx"),
    create: () => ArcFace.create(path.join(ROOT, "models", "buffalo_l", "w600k_r50.onnx")),
    passes: 3,
  },
  {
    key: "t",
    label: "LVFace-T (Glint360K)",
    file: path.join(ROOT, "models", "lvface", "LVFace-T_Glint360K.onnx"),
    create: () => LVFace.create(path.join(ROOT, "models", "lvface", "LVFace-T_Glint360K.onnx")),
    passes: 3,
  },
  {
    key: "s",
    label: "LVFace-S (Glint360K)",
    file: path.join(ROOT, "models", "lvface", "LVFace-S_Glint360K.onnx"),
    create: () => LVFace.create(path.join(ROOT, "models", "lvface", "LVFace-S_Glint360K.onnx")),
    passes: 2,
  },
  {
    key: "b",
    label: "LVFace-B (Glint360K)",
    file: path.join(ROOT, "models", "lvface", "LVFace-B_Glint360K.onnx"),
    create: () => LVFace.create(path.join(ROOT, "models", "lvface", "LVFace-B_Glint360K.onnx")),
    passes: 2,
  },
];

function sha256(file: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash("sha256");
    const s = fs.createReadStream(file);
    s.on("data", (d) => h.update(d));
    s.on("error", reject);
    s.on("end", () => resolve(h.digest("hex")));
  });
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)));
  return sorted[i];
}

function distStats(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const median = s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  return {
    n: xs.length,
    min: +s[0].toFixed(4),
    p05: +quantile(s, 0.05).toFixed(4),
    median: +median.toFixed(4),
    p95: +quantile(s, 0.95).toFixed(4),
    max: +s[s.length - 1].toFixed(4),
    fracGe050: +(xs.filter((x) => x >= 0.5).length / xs.length).toFixed(4),
  };
}

function cosine(a: Float32Array | number[], b: Float32Array | number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function loadGroundTruth() {
  const data = JSON.parse(fs.readFileSync(path.join(CT, "data.json"), "utf8")) as {
    faces: FaceMeta[];
  };
  const groups = JSON.parse(fs.readFileSync(path.join(CT, "groups.json"), "utf8")) as OrigGroup[];
  const labels = JSON.parse(fs.readFileSync(path.join(CT, "labels.json"), "utf8")) as {
    groups: Record<string, LabelInfo>;
  };
  const faceLabel = new Map<string, FaceLabel>();
  for (const g of groups) {
    const info = labels.groups[g.groupKey];
    if (!info?.person) throw new Error(`missing label for ${g.groupKey}`);
    for (const fid of g.memberFaceIds) {
      faceLabel.set(fid, {
        person: info.person,
        tentative: !!info.tentative,
        isReal: REAL_PERSONS.includes(info.person),
      });
    }
  }
  if (faceLabel.size !== data.faces.length) {
    throw new Error(`labelled ${faceLabel.size} faces, data has ${data.faces.length}`);
  }
  return { faces: data.faces, groups, faceLabel };
}

function pairSimilarities(
  emb: Map<string, Float32Array>,
  origGroups: OrigGroup[],
  stored: Map<string, number[]>,
) {
  const rep = (key: string) => origGroups.find((g) => g.groupKey === key)!.repFaceId;
  const pair = (aKey: string, bKey: string) => {
    const a = rep(aKey);
    const b = rep(bKey);
    return {
      aGroup: aKey,
      bGroup: bKey,
      aFace: a,
      bFace: b,
      similarity: +cosine(emb.get(a)!, emb.get(b)!).toFixed(4),
    };
  };
  const storedPair = (aKey: string, bKey: string) => {
    const a = rep(aKey);
    const b = rep(bKey);
    return +cosine(stored.get(a)!, stored.get(b)!).toFixed(4);
  };
  return {
    crossVideo: {
      davies: pair("yuXGpUR7fXA#0", "V9DqzY1jPhY#0"),
      goslingNormal: pair("yuXGpUR7fXA#2", "V9DqzY1jPhY#1"),
      attenboroughOlder: pair("z0B4KEXVaqo#0", "Gfa88SeNohY#0"),
    },
    attenboroughYoung: {
      youngBwVsSameVideoOlder: pair("z0B4KEXVaqo#1", "z0B4KEXVaqo#0"),
      youngBwVsOtherVideoOlder: pair("z0B4KEXVaqo#1", "Gfa88SeNohY#0"),
      youngBlurredTentativeVsSameVideoOlder: pair("z0B4KEXVaqo#2", "z0B4KEXVaqo#0"),
      youngBlurredTentativeVsOtherVideoOlder: pair("z0B4KEXVaqo#2", "Gfa88SeNohY#0"),
    },
    phase0StoredArcFaceReference: {
      davies: storedPair("yuXGpUR7fXA#0", "V9DqzY1jPhY#0"),
      goslingNormal: storedPair("yuXGpUR7fXA#2", "V9DqzY1jPhY#1"),
      attenboroughOlder: storedPair("z0B4KEXVaqo#0", "Gfa88SeNohY#0"),
      youngBwVsSameVideoOlder: storedPair("z0B4KEXVaqo#1", "z0B4KEXVaqo#0"),
      youngBwVsOtherVideoOlder: storedPair("z0B4KEXVaqo#1", "Gfa88SeNohY#0"),
      youngBlurredTentativeVsSameVideoOlder: storedPair("z0B4KEXVaqo#2", "z0B4KEXVaqo#0"),
      youngBlurredTentativeVsOtherVideoOlder: storedPair("z0B4KEXVaqo#2", "Gfa88SeNohY#0"),
    },
  };
}

function evaluateClustering(
  faces: FaceMeta[],
  emb: Map<string, Float32Array>,
  faceLabel: Map<string, FaceLabel>,
) {
  // §5.5 within-video grouping, Phase 0 video order.
  const groups: {
    videoId: string;
    groupIndex: number;
    memberFaceIds: string[];
    repFaceId: string;
    repEmbedding: number[];
    size: number;
  }[] = [];
  const groupCountsByVideo: Record<string, number> = {};
  for (const vid of VIDEOS) {
    const items = faces
      .filter((f) => f.videoId === vid)
      .map((f) => ({
        faceId: f.faceId,
        embedding: Array.from(emb.get(f.faceId)!),
        quality: f.quality,
      }));
    const gs = groupWithinVideo(items, T_IN);
    groupCountsByVideo[vid] = gs.length;
    gs.forEach((g, i) => {
      groups.push({
        videoId: vid,
        groupIndex: i,
        memberFaceIds: g.members.map((m) => m.faceId),
        repFaceId: g.representative.faceId,
        repEmbedding: g.representative.embedding,
        size: g.members.length,
      });
    });
  }

  const sweep: Record<string, unknown> = {};
  for (const t of THRESHOLDS) {
    const persons = clusterGlobal(
      groups.map((g) => ({ group: g, repEmbedding: g.repEmbedding, weight: g.size })),
      t,
    );
    const clusterRealLabels: string[][] = [];
    const contaminated: { personId: string; realLabels: string[]; faceCount: number }[] = [];
    const nonpersonJoinedReal: { personId: string; labels: string[] }[] = [];
    const span = new Map<string, Set<string>>();
    for (const p of persons) {
      const faceIds = p.groups.flatMap((g) => g.memberFaceIds);
      const real = new Set<string>();
      const nonperson = new Set<string>();
      for (const fid of faceIds) {
        const l = faceLabel.get(fid)!;
        if (l.isReal) real.add(l.person);
        else nonperson.add(l.person);
      }
      clusterRealLabels.push([...real]);
      if (real.size > 1) {
        contaminated.push({ personId: p.personId, realLabels: [...real], faceCount: faceIds.length });
      }
      if (real.size > 0 && nonperson.size > 0) {
        nonpersonJoinedReal.push({ personId: p.personId, labels: [...real, ...nonperson] });
      }
      for (const person of real) {
        if (!span.has(person)) span.set(person, new Set());
        span.get(person)!.add(p.personId);
      }
    }
    const splitPersons = REAL_PERSONS.filter((p) => (span.get(p)?.size ?? 0) > 1);
    const keySpan: Record<string, number> = {};
    for (const k of KEY_PERSONS) keySpan[k] = span.get(k)?.size ?? 0;
    sweep[t.toFixed(2)] = {
      estimatedPersons: persons.length,
      splitRealPersonCount: splitPersons.length,
      splitRealPersons: splitPersons,
      contaminatedClusterCount: contaminated.length,
      contaminatedClusters: contaminated,
      nonpersonJoinedRealClusterCount: nonpersonJoinedReal.length,
      nonpersonJoinedRealClusters: nonpersonJoinedReal,
      keyPersonSpan: keySpan,
      clusters: persons.map((p) => ({
        personId: p.personId,
        groupKeys: p.groups.map((g) => `${g.videoId}#${g.groupIndex}`),
        faceCount: p.groups.reduce((s, g) => s + g.size, 0),
      })),
      _clusterRealLabels: clusterRealLabels,
    };
  }
  return { groupCountsByVideo, withinVideoGroupCount: groups.length, groups, sweep };
}

async function main() {
  const requested = process.argv.slice(2).map((x) => x.toLowerCase());
  const defs = MODEL_DEFS.filter(
    (d) => (requested.length === 0 || requested.includes(d.key)) && fs.existsSync(d.file),
  );
  if (defs.length === 0) throw new Error("no requested model files exist");

  const { faces, groups: origGroups, faceLabel } = loadGroundTruth();
  const stored = new Map(faces.map((f) => [f.faceId, f.embedding]));

  // Decode every aligned crop once. Timing below therefore covers the
  // recognizer only (blob conversion + ONNX session + normalisation),
  // not JPEG decoding - the same boundary as the Phase 0 bench.ts.
  const rawByFace = new Map<string, RawImage>();
  for (const f of faces) {
    rawByFace.set(f.faceId, await decodeImage(path.join(ALIGNED, `${f.faceId}.jpg`)));
  }

  // Create every session up front. Speed is then measured interleaved
  // (face by face, model by model) so that background load drift on
  // this shared VM affects all models in the same passes; a first,
  // sequential run showed ~2x absolute swings between runs while the
  // within-run ordering stayed stable.
  const recognizers = new Map<string, Recognizer>();
  for (const def of defs) recognizers.set(def.key, await def.create());
  for (const def of defs) {
    const recognizer = recognizers.get(def.key)!;
    for (const f of faces.slice(0, 6)) await recognizer.embedding(rawByFace.get(f.faceId)!);
  }
  const timesByKey = new Map<string, number[]>(defs.map((d) => [d.key, []]));
  const embByKey = new Map<string, Map<string, Float32Array>>(
    defs.map((d) => [d.key, new Map()]),
  );
  const maxPasses = Math.max(...defs.map((d) => d.passes));
  for (let pass = 0; pass < maxPasses; pass++) {
    for (const f of faces) {
      const raw = rawByFace.get(f.faceId)!;
      for (const def of defs) {
        if (pass >= def.passes) continue;
        const t0 = process.hrtime.bigint();
        const e = await recognizers.get(def.key)!.embedding(raw);
        timesByKey.get(def.key)!.push(Number(process.hrtime.bigint() - t0) / 1e6);
        embByKey.get(def.key)!.set(f.faceId, e);
      }
    }
  }

  const results: Record<string, unknown> = {};
  for (const def of defs) {
    console.log(`\n=== ${def.label} ===`);
    const times = timesByKey.get(def.key)!;
    const emb = embByKey.get(def.key)!;
    const sortedTimes = [...times].sort((a, b) => a - b);
    const medianMs = sortedTimes[sortedTimes.length >> 1];
    const meanMs = times.reduce((s, x) => s + x, 0) / times.length;
    const dims = new Set([...emb.values()].map((e) => e.length));
    if (dims.size !== 1 || ![512].includes([...dims][0])) {
      throw new Error(`${def.label}: unexpected embedding dims ${[...dims]}`);
    }
    let maxNormErr = 0;
    for (const e of emb.values()) {
      let n = 0;
      for (const v of e) n += v * v;
      maxNormErr = Math.max(maxNormErr, Math.abs(Math.sqrt(n) - 1));
    }
    console.log(
      `speed: median ${medianMs.toFixed(1)} ms/face, mean ${meanMs.toFixed(1)} ` +
        `(n=${times.length}), dim=${[...dims][0]}, max |norm-1|=${maxNormErr.toExponential(1)}`,
    );

    // Pairwise separation smoke test on real-person labels.
    const pairStats = (includeTentative: boolean) => {
      const items = faces
        .map((f) => ({ f, l: faceLabel.get(f.faceId)! }))
        .filter(({ l }) => l.isReal && (includeTentative || !l.tentative));
      const same: number[] = [];
      const diff: number[] = [];
      for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
          const s = cosine(emb.get(items[i].f.faceId)!, emb.get(items[j].f.faceId)!);
          (items[i].l.person === items[j].l.person ? same : diff).push(s);
        }
      }
      return { same: distStats(same), different: distStats(diff) };
    };

    const clustering = evaluateClustering(faces, emb, faceLabel);
    console.log(
      `within-video groups: ${clustering.withinVideoGroupCount} ` +
        `(Phase 0 ArcFace: 21)`,
    );
    for (const t of THRESHOLDS) {
      const s = clustering.sweep[t.toFixed(2)] as {
        estimatedPersons: number;
        splitRealPersonCount: number;
        contaminatedClusterCount: number;
      };
      console.log(
        `T=${t.toFixed(2)} persons=${s.estimatedPersons} split=${s.splitRealPersonCount} contaminated=${s.contaminatedClusterCount}`,
      );
    }

    results[def.key] = {
      label: def.label,
      modelFile: path.relative(ROOT, def.file),
      modelBytes: fs.statSync(def.file).size,
      modelSha256: await sha256(def.file),
      outputDim: [...dims][0],
      maxNormError: +maxNormErr.toExponential(2),
      speed: {
        medianMsPerFace: +medianMs.toFixed(2),
        meanMsPerFace: +meanMs.toFixed(2),
        timedCalls: times.length,
        passes: def.passes,
        recognitionOnlyHoursPer1000Videos:
          +((medianMs * FACES_PER_VIDEO_PHASE0 * 1000) / 3_600_000).toFixed(3),
        totalWithPhase0DetectionHoursPer1000Videos:
          +(
            ((DET_MS_PER_FRAME_PHASE0 + medianMs) * FACES_PER_VIDEO_PHASE0 * 1000) /
            3_600_000
          ).toFixed(3),
      },
      pairSeparationDefinitiveRealFaces: pairStats(false),
      pairSeparationAllRealLabelledFaces: pairStats(true),
      pairSimilarities: pairSimilarities(emb, origGroups, stored),
      clustering: {
        groupCountsByVideo: clustering.groupCountsByVideo,
        withinVideoGroupCount: clustering.withinVideoGroupCount,
        groups: clustering.groups.map((g) => ({
          key: `${g.videoId}#${g.groupIndex}`,
          memberFaceIds: g.memberFaceIds,
          repFaceId: g.repFaceId,
          size: g.size,
        })),
        sweep: Object.fromEntries(
          Object.entries(clustering.sweep).map(([k, v]) => {
            const copy = { ...(v as Record<string, unknown>) };
            delete copy._clusterRealLabels;
            return [k, copy];
          }),
        ),
      },
    };
  }

  const out = {
    generatedAt: new Date().toISOString(),
    environment: {
      platform: `${process.platform}/${process.arch}`,
      node: process.version,
      cpuCount: (await import("node:os")).cpus().length,
      cpuModel: (await import("node:os")).cpus()[0]?.model,
    },
    methodology: {
      faces: faces.length,
      input: "Phase 0 saved aligned 112x112 JPEGs from common SCRFD detections",
      timingBoundary:
        "recognizer only on in-memory decoded crops (blob + ONNX run + L2 normalisation); JPEG decode excluded; models timed interleaved face-by-face in the same passes to cancel VM load drift",
      phase0ReferenceArcFaceMedianMsPerFace: 118.1,
      phase0DetectionMedianMsPerFrame: DET_MS_PER_FRAME_PHASE0,
      conversionFacesPerVideo: FACES_PER_VIDEO_PHASE0,
      thresholds: THRESHOLDS,
      tIn: T_IN,
      videoOrder: VIDEOS,
    },
    models: results,
  };
  fs.mkdirSync(path.join(ROOT, "out"), { recursive: true });
  fs.writeFileSync(path.join(ROOT, "out", "lvface-comparison.json"), JSON.stringify(out, null, 2));
  console.log("\nwrote out/lvface-comparison.json");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
