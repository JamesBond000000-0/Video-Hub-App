// Cluster validation driver (design doc §11 Phase 0, second half).
//
// Stages (argv[2]):
//   run    - sample frames per §5.1, detect (quality pack), embed, save
//            aligned + display crops, genderage per face -> data.json
//   sweep  - §5.5 within-video grouping (T_in=0.50) + §5.6 leader-follower
//            for T_global in 0.35..0.60 -> sweep.json (+ groups.json)
//   sheets - contact sheets: one for video-group representatives
//            (for ground-truth labelling), one per person at a given T
//   attrs  - §5.8 aggregation (median age / majority sex) per person at T
//
// All inputs/outputs live in ../cluster-test/ (gitignored except docs).

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { Scrfd, type DetectedFace } from "./scrfd.js";
import { ArcFace, GenderAge, normCrop } from "./arcface.js";
import { decodeImage, resizeBilinear, type RawImage } from "./image-ops.js";
import { clusterGlobal, groupWithinVideo } from "./cluster.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const CT = path.join(ROOT, "cluster-test");
const MODELS = path.join(ROOT, "models", "buffalo_l");

interface VideoSpec {
  id: string;
  mains: string[]; // ground-truth main persons expected on screen
}
const VIDEOS: VideoSpec[] = [
  { id: "yuXGpUR7fXA", mains: ["Graham Norton", "Ryan Gosling", "Greg Davies"] },
  { id: "dmbpagijVkk", mains: ["Graham Norton", "Lee Mack", "John Cleese"] },
  { id: "V9DqzY1jPhY", mains: ["Graham Norton", "Greg Davies"] },
  { id: "8cPxeyZ6-g0", mains: ["Graham Norton", "Colin Farrell"] },
  { id: "z0B4KEXVaqo", mains: ["David Attenborough"] },
  { id: "Gfa88SeNohY", mains: ["David Attenborough"] },
];

const MIN_FACE_PX = 40; // §5.2
const T_IN = 0.5; // §5.5

interface FaceRec {
  faceId: string;
  videoId: string;
  t: number;
  bbox: [number, number, number, number];
  score: number;
  quality: number;
  frameRescued320: boolean;
  embedding: number[];
  attr: { sex: "M" | "F"; age: number } | null;
}

function probeDuration(file: string): number {
  const out = execFileSync("ffprobe", [
    "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file,
  ]).toString();
  return parseFloat(out.trim());
}

/** §5.1: clamp(ceil(duration/30), 10, 60) frames, evenly spaced in [2%, 98%]. */
export function sampleTimes(duration: number): number[] {
  const n = Math.min(60, Math.max(10, Math.ceil(duration / 30)));
  const lo = duration * 0.02;
  const hi = duration * 0.98;
  const times: number[] = [];
  for (let i = 0; i < n; i++) times.push(lo + ((hi - lo) * i) / (n - 1));
  return times;
}

async function encodeJpeg(img: RawImage, file: string): Promise<void> {
  await sharp(Buffer.from(img.data), {
    raw: { width: img.width, height: img.height, channels: 3 },
  })
    .jpeg({ quality: 90 })
    .toFile(file);
}

/** §5.3 display crop: bbox expanded to a square of side max(w,h)*1.35. */
function displayCrop(img: RawImage, face: DetectedFace, outSize = 256): RawImage {
  const [x1, y1, x2, y2] = face.bbox;
  const side = Math.max(x2 - x1, y2 - y1) * 1.35;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const sx = Math.round(cx - side / 2);
  const sy = Math.round(cy - side / 2);
  const s = Math.round(side);
  const crop: RawImage = { data: new Uint8Array(s * s * 3), width: s, height: s };
  for (let y = 0; y < s; y++) {
    const yy = sy + y;
    if (yy < 0 || yy >= img.height) continue;
    for (let x = 0; x < s; x++) {
      const xx = sx + x;
      if (xx < 0 || xx >= img.width) continue;
      const si = (yy * img.width + xx) * 3;
      const di = (y * s + x) * 3;
      crop.data[di] = img.data[si];
      crop.data[di + 1] = img.data[si + 1];
      crop.data[di + 2] = img.data[si + 2];
    }
  }
  return resizeBilinear(crop, outSize, outSize);
}

async function stageRun(): Promise<void> {
  const scrfd = await Scrfd.create(path.join(MODELS, "det_10g.onnx"));
  const arcface = await ArcFace.create(path.join(MODELS, "w600k_r50.onnx"));
  const genderage = await GenderAge.create(path.join(MODELS, "genderage.onnx"));
  const faces: FaceRec[] = [];
  const stats: Record<string, unknown>[] = [];

  for (const v of VIDEOS) {
    const file = path.join(CT, "videos", `${v.id}.mp4`);
    const duration = probeDuration(file);
    const times = sampleTimes(duration);
    const frameDir = path.join(CT, "frames", v.id);
    fs.mkdirSync(frameDir, { recursive: true });
    let rawFaces = 0;
    let keptFaces = 0;
    let zeroFrames = 0;
    let rescuedFrames = 0;
    let rescuedFaces = 0;
    for (let fi = 0; fi < times.length; fi++) {
      const t = times[fi];
      const framePath = path.join(frameDir, `f${String(fi).padStart(3, "0")}.jpg`);
      execFileSync("ffmpeg", [
        "-v", "error", "-ss", t.toFixed(3), "-i", file,
        "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", "-q:v", "2", framePath,
      ]);
      const img = await decodeImage(framePath);
      let dets = await scrfd.detect(img, [640, 640]);
      let rescued = false;
      if (dets.length === 0) {
        const retry = await scrfd.detect(img, [320, 320]);
        if (retry.length > 0) {
          rescued = true;
          rescuedFrames++;
          rescuedFaces += retry.length;
          dets = retry;
        }
      }
      if (dets.length === 0) zeroFrames++;
      rawFaces += dets.length;
      for (let di = 0; di < dets.length; di++) {
        const d = dets[di];
        const w = d.bbox[2] - d.bbox[0];
        const h = d.bbox[3] - d.bbox[1];
        if (Math.min(w, h) < MIN_FACE_PX) continue;
        const faceId = `${v.id}-${fi}-${di}`;
        const aligned = normCrop(img, d);
        const emb = await arcface.embedding(aligned);
        const quality =
          d.score * Math.sqrt((w * h) / (img.width * img.height));
        const attr = await genderage.get(img, d);
        const aDir = path.join(CT, "faces", "aligned");
        const cDir = path.join(CT, "faces", "crops");
        fs.mkdirSync(aDir, { recursive: true });
        fs.mkdirSync(cDir, { recursive: true });
        await encodeJpeg(aligned, path.join(aDir, `${faceId}.jpg`));
        await encodeJpeg(displayCrop(img, d), path.join(cDir, `${faceId}.jpg`));
        faces.push({
          faceId,
          videoId: v.id,
          t: Math.round(t * 10) / 10,
          bbox: d.bbox.map((x) => Math.round(x * 10) / 10) as [number, number, number, number],
          score: Math.round(d.score * 1000) / 1000,
          quality: Math.round(quality * 10000) / 10000,
          frameRescued320: rescued,
          embedding: Array.from(emb).map((x) => Math.round(x * 1e6) / 1e6),
          attr,
        });
        keptFaces++;
      }
    }
    const st = {
      videoId: v.id,
      duration: Math.round(duration * 10) / 10,
      frames: times.length,
      rawFaces,
      keptFaces,
      zeroFrames,
      rescuedFrames,
      rescuedFaces,
    };
    stats.push(st);
    console.log(JSON.stringify(st));
  }
  fs.writeFileSync(path.join(CT, "data.json"), JSON.stringify({ stats, faces }, null, 1));
  console.log(`total faces kept: ${faces.length}`);
}

interface GroupRec {
  groupKey: string; // `${videoId}#${index}`
  videoId: string;
  memberFaceIds: string[];
  repFaceId: string;
  repEmbedding: number[];
  size: number;
}

function loadGroups(): { groups: GroupRec[]; faces: FaceRec[] } {
  const data = JSON.parse(fs.readFileSync(path.join(CT, "data.json"), "utf8"));
  const faces: FaceRec[] = data.faces;
  const groups: GroupRec[] = [];
  for (const v of VIDEOS) {
    const vf = faces.filter((f) => f.videoId === v.id);
    const gs = groupWithinVideo(vf, T_IN);
    gs.forEach((g, i) => {
      groups.push({
        groupKey: `${v.id}#${i}`,
        videoId: v.id,
        memberFaceIds: g.members.map((m) => m.faceId),
        repFaceId: g.representative.faceId,
        repEmbedding: g.representative.embedding,
        size: g.members.length,
      });
    });
  }
  return { groups, faces };
}

function stageSweep(): void {
  const { groups } = loadGroups();
  fs.writeFileSync(path.join(CT, "groups.json"), JSON.stringify(groups, null, 1));
  const thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
  const sweep: Record<string, { personId: string; groupKeys: string[] }[]> = {};
  for (const t of thresholds) {
    const persons = clusterGlobal(
      groups.map((g) => ({ group: g, repEmbedding: g.repEmbedding, weight: g.size })),
      t,
    );
    sweep[t.toFixed(2)] = persons.map((p) => ({
      personId: p.personId,
      groupKeys: p.groups.map((g) => g.groupKey),
    }));
    console.log(`T=${t.toFixed(2)} persons=${persons.length}`);
  }
  fs.writeFileSync(path.join(CT, "sweep.json"), JSON.stringify(sweep, null, 1));
  console.log(`groups: ${groups.length}`);
}

async function tileSheet(
  tiles: { file: string; label: string }[],
  outFile: string,
  cols: number,
  tile = 180,
): Promise<void> {
  const labelH = 22;
  const rows = Math.ceil(tiles.length / cols);
  const W = cols * tile;
  const H = rows * (tile + labelH);
  const comps: sharp.OverlayOptions[] = [];
  for (let i = 0; i < tiles.length; i++) {
    const x = (i % cols) * tile;
    const y = Math.floor(i / cols) * (tile + labelH);
    const imgBuf = await sharp(tiles[i].file)
      .resize(tile, tile, { fit: "cover" })
      .jpeg()
      .toBuffer();
    comps.push({ input: imgBuf, left: x, top: y + labelH });
    const svg = `<svg width="${tile}" height="${labelH}"><rect width="100%" height="100%" fill="black"/><text x="4" y="15" font-size="12" fill="white" font-family="monospace">${tiles[i].label}</text></svg>`;
    comps.push({ input: Buffer.from(svg), left: x, top: y });
  }
  await sharp({
    create: { width: W, height: H, channels: 3, background: { r: 30, g: 30, b: 30 } },
  })
    .composite(comps)
    .jpeg({ quality: 88 })
    .toFile(outFile);
  console.log(`sheet: ${outFile} (${tiles.length} tiles)`);
}

async function stageSheets(): Promise<void> {
  const { groups } = loadGroups();
  const crops = (fid: string) => path.join(CT, "faces", "crops", `${fid}.jpg`);
  // group representative sheet for ground-truth labelling
  await tileSheet(
    groups.map((g, i) => ({
      file: crops(g.repFaceId),
      label: `G${i} ${g.groupKey} n=${g.size}`,
    })),
    path.join(CT, "sheet_groups.jpg"),
    6,
  );
  // per-person sheets at T=0.45 (default; re-run with argv for another T)
  const t = process.argv[3] ?? "0.45";
  const sweep = JSON.parse(fs.readFileSync(path.join(CT, "sweep.json"), "utf8"));
  const persons = sweep[t];
  const byKey = new Map(groups.map((g) => [g.groupKey, g]));
  const data = JSON.parse(fs.readFileSync(path.join(CT, "data.json"), "utf8"));
  const faceById = new Map<string, FaceRec>(data.faces.map((f: FaceRec) => [f.faceId, f]));
  for (const p of persons) {
    const tiles: { file: string; label: string }[] = [];
    for (const gk of p.groupKeys) {
      const g = byKey.get(gk)!;
      for (const fid of g.memberFaceIds.slice(0, 8)) {
        const f = faceById.get(fid)!;
        tiles.push({ file: crops(fid), label: `${f.videoId.slice(0, 4)} t=${f.t}` });
      }
    }
    await tileSheet(
      tiles.slice(0, 24),
      path.join(CT, `sheet_${p.personId}_T${t}.jpg`),
      6,
      150,
    );
  }
}

function stageAttrs(): void {
  const t = process.argv[3] ?? "0.45";
  const { groups, faces } = loadGroups();
  const faceById = new Map(faces.map((f) => [f.faceId, f]));
  const sweep = JSON.parse(fs.readFileSync(path.join(CT, "sweep.json"), "utf8"));
  const persons = sweep[t];
  const byKey = new Map(groups.map((g) => [g.groupKey, g]));
  const out: Record<string, unknown>[] = [];
  for (const p of persons) {
    // §5.8: per (video, group) take top-5 quality faces as samples
    const samples: { sex: "M" | "F"; age: number }[] = [];
    for (const gk of p.groupKeys) {
      const g = byKey.get(gk)!;
      const members = g.memberFaceIds
        .map((fid) => faceById.get(fid)!)
        .sort((a, b) => b.quality - a.quality)
        .slice(0, 5);
      for (const m of members) if (m.attr) samples.push(m.attr);
    }
    if (samples.length === 0) continue;
    const ages = samples.map((s) => s.age).sort((a, b) => a - b);
    const median = ages[Math.floor(ages.length / 2)];
    const fCount = samples.filter((s) => s.sex === "F").length;
    const sex = fCount * 2 >= samples.length ? "F" : "M";
    const agreement =
      Math.max(fCount, samples.length - fCount) / samples.length;
    out.push({
      personId: p.personId,
      groups: p.groupKeys,
      sex,
      sexAgreement: Math.round(agreement * 100) / 100,
      ageMedian: median,
      sampleCount: samples.length,
    });
  }
  fs.writeFileSync(path.join(CT, `attrs_T${t}.json`), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out, null, 1));
}

const stage = process.argv[2];
if (stage === "run") await stageRun();
else if (stage === "sweep") stageSweep();
else if (stage === "sheets") await stageSheets();
else if (stage === "attrs") stageAttrs();
else {
  console.error("usage: cluster-test.ts run|sweep|sheets [T]|attrs [T]");
  process.exit(1);
}
