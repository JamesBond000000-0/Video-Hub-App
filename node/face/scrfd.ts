// SCRFD face detector - TypeScript port of the Python reference
// (insightface/model_zoo/scrfd.py: SCRFD.forward / _detect_candidates /
// detect / nms). Only the fmc=3, 9-output variant (SCRFD-500M / 10G with
// 5-point landmarks) is supported, which covers buffalo_sc / buffalo_l.
//
// Reference semantics that are reproduced exactly:
//  - input blob: aspect-preserving resize into the det_size canvas,
//    pasted at the top-left of a zero canvas; (px - 127.5) / 128, RGB
//  - strides [8, 16, 32], 2 anchors per cell, anchor centre = (x, y) * stride
//  - bbox/kps decoded with distance2bbox / distance2kps, divided by det_scale
//  - candidates sorted by score descending (stable), then NMS with
//    IoU threshold 0.4 and the "+1 pixel" area convention of the original

import * as ort from "onnxruntime-node";
import { padToCanvas, resizeBilinear, toBlob, type RawImage } from "./image-ops";

export interface DetectedFace {
  bbox: [number, number, number, number]; // x1, y1, x2, y2 in original image px
  score: number;
  kps: [number, number][]; // 5 landmarks in original image px
}

const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;
const NMS_THRESH = 0.4;

export class Scrfd {
  private session!: ort.InferenceSession;
  private inputName = "";
  private outputNames: string[] = [];
  private centerCache = new Map<string, Float32Array>();

  static async create(modelPath: string): Promise<Scrfd> {
    const d = new Scrfd();
    d.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
    });
    d.inputName = d.session.inputNames[0];
    d.outputNames = [...d.session.outputNames];
    if (d.outputNames.length !== 9) {
      throw new Error(`expected 9 SCRFD outputs, got ${d.outputNames.length}`);
    }
    return d;
  }

  private anchorCenters(height: number, width: number, stride: number): Float32Array {
    const key = `${height},${width},${stride}`;
    let centers = this.centerCache.get(key);
    if (centers) return centers;
    centers = new Float32Array(height * width * NUM_ANCHORS * 2);
    let i = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let a = 0; a < NUM_ANCHORS; a++) {
          centers[i++] = x * stride;
          centers[i++] = y * stride;
        }
      }
    }
    if (this.centerCache.size < 100) this.centerCache.set(key, centers);
    return centers;
  }

  async detect(
    img: RawImage,
    inputSize: [number, number], // [width, height], e.g. [640, 640]
    threshold = 0.5,
  ): Promise<DetectedFace[]> {
    const [inW, inH] = inputSize;
    // --- _detect_candidates: resize + pad ---
    const imRatio = img.height / img.width;
    const modelRatio = inH / inW;
    let newWidth: number;
    let newHeight: number;
    if (imRatio > modelRatio) {
      newHeight = inH;
      newWidth = Math.trunc(newHeight / imRatio);
    } else {
      newWidth = inW;
      newHeight = Math.trunc(newWidth * imRatio);
    }
    const detScale = newHeight / img.height;
    const resized = resizeBilinear(img, newWidth, newHeight);
    const canvas = padToCanvas(resized, inW, inH);
    const blob = toBlob(canvas, 127.5, 128.0);
    const tensor = new ort.Tensor("float32", blob, [1, 3, inH, inW]);
    const outs = await this.session.run({ [this.inputName]: tensor });

    // --- forward: decode per stride ---
    interface Cand {
      score: number;
      bbox: [number, number, number, number];
      kps: [number, number][];
    }
    const cands: Cand[] = [];
    for (let idx = 0; idx < STRIDES.length; idx++) {
      const stride = STRIDES[idx];
      const scores = outs[this.outputNames[idx]].data as Float32Array;
      const bboxPred = outs[this.outputNames[idx + 3]].data as Float32Array;
      const kpsPred = outs[this.outputNames[idx + 6]].data as Float32Array;
      const h = inH / stride;
      const w = inW / stride;
      const centers = this.anchorCenters(h, w, stride);
      const total = h * w * NUM_ANCHORS;
      for (let i = 0; i < total; i++) {
        const score = scores[i];
        if (score < threshold) continue;
        const cx = centers[i * 2];
        const cy = centers[i * 2 + 1];
        const bo = i * 4;
        const bbox: [number, number, number, number] = [
          (cx - bboxPred[bo] * stride) / detScale,
          (cy - bboxPred[bo + 1] * stride) / detScale,
          (cx + bboxPred[bo + 2] * stride) / detScale,
          (cy + bboxPred[bo + 3] * stride) / detScale,
        ];
        const kps: [number, number][] = [];
        const ko = i * 10;
        for (let k = 0; k < 5; k++) {
          kps.push([
            (cx + kpsPred[ko + k * 2] * stride) / detScale,
            (cy + kpsPred[ko + k * 2 + 1] * stride) / detScale,
          ]);
        }
        cands.push({ score, bbox, kps });
      }
    }
    // --- detect(): sort desc (Array.sort is stable in JS) + NMS ---
    cands.sort((p, q) => q.score - p.score);
    const keep: Cand[] = [];
    const suppressed = new Array<boolean>(cands.length).fill(false);
    for (let i = 0; i < cands.length; i++) {
      if (suppressed[i]) continue;
      keep.push(cands[i]);
      const [ax1, ay1, ax2, ay2] = cands[i].bbox;
      const areaA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1);
      for (let j = i + 1; j < cands.length; j++) {
        if (suppressed[j]) continue;
        const [bx1, by1, bx2, by2] = cands[j].bbox;
        const xx1 = Math.max(ax1, bx1);
        const yy1 = Math.max(ay1, by1);
        const xx2 = Math.min(ax2, bx2);
        const yy2 = Math.min(ay2, by2);
        const wgt = Math.max(0, xx2 - xx1 + 1);
        const hgt = Math.max(0, yy2 - yy1 + 1);
        const inter = wgt * hgt;
        const areaB = (bx2 - bx1 + 1) * (by2 - by1 + 1);
        const ovr = inter / (areaA + areaB - inter);
        if (ovr > NMS_THRESH) suppressed[j] = true;
      }
    }
    return keep.map((c) => ({ bbox: c.bbox, score: c.score, kps: c.kps }));
  }
}
