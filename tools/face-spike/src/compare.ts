// Compare the TypeScript port against the Python golden output.
// For every golden image: run TS detection with the recorded det_size,
// match faces greedily by bbox IoU, and report per-face embedding cosine
// similarity, bbox IoU, landmark distance, and sex/age agreement.
// Design doc section 10 thresholds: cosine >= 0.999, IoU >= 0.9,
// sex equal, age within +/-2 years.
//
// Usage: npx tsx src/compare.ts [buffalo_l|buffalo_sc|all]

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodeImage } from "./image-ops.js";
import { Scrfd, type DetectedFace } from "./scrfd.js";
import { ArcFace, GenderAge, normCrop } from "./arcface.js";

const SPIKE = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

interface GoldenFace {
  bbox: number[];
  detScore: number;
  kps: [number, number][];
  embedding: number[];
  sex: string | null;
  age: number | null;
}
interface GoldenImage {
  file: string;
  detSize: [number, number];
  width: number;
  height: number;
  faces: GoldenFace[];
}
interface Golden {
  pack: string;
  images: GoldenImage[];
}

function iou(a: number[], b: [number, number, number, number]): number {
  const xx1 = Math.max(a[0], b[0]);
  const yy1 = Math.max(a[1], b[1]);
  const xx2 = Math.min(a[2], b[2]);
  const yy2 = Math.min(a[3], b[3]);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

async function runPack(pack: string): Promise<void> {
  const modelsDir = path.join(SPIKE, "models", pack);
  const detFile = pack === "buffalo_l" ? "det_10g.onnx" : "det_500m.onnx";
  const recFile = pack === "buffalo_l" ? "w600k_r50.onnx" : "w600k_mbf.onnx";
  const det = await Scrfd.create(path.join(modelsDir, detFile));
  const rec = await ArcFace.create(path.join(modelsDir, recFile));
  const ga = await GenderAge.create(path.join(modelsDir, "genderage.onnx"));
  const golden: Golden = JSON.parse(
    await readFile(path.join(SPIKE, "golden", `golden_${pack}.json`), "utf8"),
  );
  console.log(`\n=== ${pack} (det=${detFile}, rec=${recFile}) ===`);
  const rows: Record<string, unknown>[] = [];
  let allCos = 1;
  let allIou = 1;
  let sexOk = true;
  let ageMaxDiff = 0;
  for (const gimg of golden.images) {
    const img = await decodeImage(path.join(SPIKE, "samples", gimg.file));
    const faces: DetectedFace[] = await det.detect(img, gimg.detSize);
    console.log(
      `${gimg.file} det_size=${gimg.detSize.join("x")}: golden=${gimg.faces.length} ts=${faces.length}`,
    );
    // greedy matching by IoU
    const pairs: { gi: number; ti: number; v: number }[] = [];
    gimg.faces.forEach((gf, gi) => {
      faces.forEach((tf, ti) => pairs.push({ gi, ti, v: iou(gf.bbox, tf.bbox) }));
    });
    pairs.sort((p, q) => q.v - p.v);
    const usedG = new Set<number>();
    const usedT = new Set<number>();
    for (const p of pairs) {
      if (p.v < 0.3 || usedG.has(p.gi) || usedT.has(p.ti)) continue;
      usedG.add(p.gi);
      usedT.add(p.ti);
      const gf = gimg.faces[p.gi];
      const tf = faces[p.ti];
      const aligned = normCrop(img, tf);
      const emb = await rec.embedding(aligned);
      let cos = 0;
      for (let i = 0; i < 512; i++) cos += emb[i] * gf.embedding[i];
      let kpsErr = 0;
      gf.kps.forEach((kp, k) => {
        kpsErr = Math.max(kpsErr, Math.hypot(kp[0] - tf.kps[k][0], kp[1] - tf.kps[k][1]));
      });
      const attr = await ga.get(img, tf);
      const ageDiff = gf.age !== null ? Math.abs(attr.age - gf.age) : -1;
      const sexMatch = gf.sex === attr.sex;
      allCos = Math.min(allCos, cos);
      allIou = Math.min(allIou, p.v);
      sexOk = sexOk && sexMatch;
      if (ageDiff >= 0) ageMaxDiff = Math.max(ageMaxDiff, ageDiff);
      const row = {
        image: gimg.file,
        goldenIdx: p.gi,
        tsIdx: p.ti,
        iou: +p.v.toFixed(5),
        cosine: +cos.toFixed(6),
        kpsMaxErrPx: +kpsErr.toFixed(2),
        detScoreGolden: gf.detScore,
        detScoreTs: +tf.score.toFixed(5),
        sexGolden: gf.sex,
        sexTs: attr.sex,
        sexMatch,
        ageGolden: gf.age,
        ageTs: attr.age,
        ageDiff,
      };
      rows.push(row);
      console.log(
        `  face#${p.gi}: IoU=${row.iou} cos=${row.cosine} kpsErr=${row.kpsMaxErrPx}px ` +
          `sex ${gf.sex}/${attr.sex}${sexMatch ? "" : " MISMATCH"} age ${gf.age}/${attr.age} (diff ${ageDiff})`,
      );
    }
    const missed = gimg.faces.length - usedG.size;
    const extra = faces.length - usedT.size;
    if (missed || extra) console.log(`  unmatched: missed=${missed} extra=${extra}`);
  }
  console.log(
    `SUMMARY ${pack}: minCosine=${allCos.toFixed(6)} minIoU=${allIou.toFixed(5)} ` +
      `sexAllMatch=${sexOk} ageMaxDiff=${ageMaxDiff}`,
  );
  await writeFile(
    path.join(SPIKE, "out", `compare_${pack}.json`),
    JSON.stringify({ pack, minCosine: allCos, minIoU: allIou, sexAllMatch: sexOk, ageMaxDiff, rows }, null, 1),
  );
}

const arg = process.argv[2] ?? "all";
const packs = arg === "all" ? ["buffalo_l", "buffalo_sc"] : [arg];
for (const p of packs) await runPack(p);
