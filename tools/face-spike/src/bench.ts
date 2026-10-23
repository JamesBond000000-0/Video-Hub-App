// CPU speed measurement for both model configurations.
// Measures end-to-end detect() per frame (preprocess + inference +
// postprocess, as the app pipeline would pay it) and per-face
// recognition / attribute inference on pre-aligned crops.
// Reports median and mean. Environment is whatever machine runs this;
// record it alongside the numbers (see RESULTS.md).
//
// Usage: npx tsx src/bench.ts

import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodeImage } from "./image-ops.js";
import { Scrfd } from "./scrfd.js";
import { ArcFace, GenderAge, normCrop } from "./arcface.js";

const SPIKE = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function stats(samples: number[]): { median: number; mean: number } {
  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
  return { median: +median.toFixed(1), mean: +mean.toFixed(1) };
}

async function timeIt(fn: () => Promise<void>, warmup: number, runs: number): Promise<number[]> {
  for (let i = 0; i < warmup; i++) await fn();
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await fn();
    samples.push(performance.now() - t0);
  }
  return samples;
}

async function benchPack(pack: string, detFile: string, recFile: string): Promise<void> {
  const modelsDir = path.join(SPIKE, "models", pack);
  const det = await Scrfd.create(path.join(modelsDir, detFile));
  const rec = await ArcFace.create(path.join(modelsDir, recFile));
  const ga = await GenderAge.create(path.join(modelsDir, "genderage.onnx"));
  const img = await decodeImage(path.join(SPIKE, "samples", "t1.jpg")); // 1280x886, 6 faces
  const faces = await det.detect(img, [640, 640]);
  const aligned = faces.map((f) => normCrop(img, f));

  const detSamples = await timeIt(async () => {
    await det.detect(img, [640, 640]);
  }, 3, 15);
  // recognition measured over all faces in the frame, reported per face
  const recSamples = await timeIt(async () => {
    for (const a of aligned) await rec.embedding(a);
  }, 2, 10);
  const gaSamples = await timeIt(async () => {
    for (const f of faces) await ga.get(img, f);
  }, 2, 10);
  const detS = stats(detSamples);
  const recS = stats(recSamples.map((v) => v / faces.length));
  const gaS = stats(gaSamples.map((v) => v / faces.length));
  console.log(`\n=== ${pack} (det=${detFile}, rec=${recFile}) ===`);
  console.log(`faces in bench frame: ${faces.length}`);
  console.log(`detection   ms/frame: median=${detS.median} mean=${detS.mean}`);
  console.log(`recognition ms/face : median=${recS.median} mean=${recS.mean}`);
  console.log(`genderage   ms/face : median=${gaS.median} mean=${gaS.mean}`);
  // per-video estimate under the design-doc sampling policy:
  // 30 frames per video, on average 1 recognised face per frame
  const secPerVideo = (detS.median * 30 + recS.median * 30) / 1000;
  console.log(
    `estimate: ${secPerVideo.toFixed(2)} s/video (30 frames, 1 face/frame) -> ` +
      `${(secPerVideo * 1000 / 3600).toFixed(2)} h per 1000 videos`,
  );
}

console.log(`node ${process.version}, platform ${process.platform}/${process.arch}`);
await benchPack("buffalo_l", "det_10g.onnx", "w600k_r50.onnx");
await benchPack("buffalo_sc", "det_500m.onnx", "w600k_mbf.onnx");
