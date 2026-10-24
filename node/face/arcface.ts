// ArcFace recognition - TypeScript port of
// insightface/model_zoo/arcface_onnx.py and
// insightface/utils/face_align.py (estimate_norm / norm_crop / transform).
// Ported from the Phase 0 spike (tools/face-spike/src/arcface.ts), where
// equivalence with the Python reference was verified.
// (The GenderAge attribute model is Phase 2 and intentionally not ported.)
//
// The similarity transform is solved with the closed-form least-squares
// solution for 2D similarity (equivalent to skimage's Umeyama estimate
// when no reflection is involved, which always holds for face landmarks
// in normal orientation).

import * as ort from "onnxruntime-node";
import { toBlob, warpAffine, type RawImage } from "./image-ops";
import { createOrtSession, type OrtSessionFactory } from "./ort-provider";
import type { DetectedFace } from "./scrfd";

// ArcFace alignment template for 112x112 (design doc appendix B).
export const ARCFACE_DST: [number, number][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

export type Affine = [number, number, number, number, number, number];

/** Least-squares similarity transform mapping src points onto dst points. */
export function estimateNorm(
  lmk: [number, number][],
  imageSize = 112,
): Affine {
  const ratio = imageSize / 112;
  const dst = ARCFACE_DST.map(([x, y]) => [x * ratio, y * ratio] as [number, number]);
  const n = lmk.length;
  let msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) {
    msx += lmk[i][0]; msy += lmk[i][1];
    mdx += dst[i][0]; mdy += dst[i][1];
  }
  msx /= n; msy /= n; mdx /= n; mdy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sx = lmk[i][0] - msx;
    const sy = lmk[i][1] - msy;
    const dx = dst[i][0] - mdx;
    const dy = dst[i][1] - mdy;
    num1 += sx * dx + sy * dy; // -> a
    num2 += sx * dy - sy * dx; // -> b
    den += sx * sx + sy * sy;
  }
  const a = num1 / den;
  const b = num2 / den;
  const tx = mdx - (a * msx - b * msy);
  const ty = mdy - (b * msx + a * msy);
  return [a, -b, tx, b, a, ty];
}

/** face_align.norm_crop: warp the image so landmarks land on the template. */
export function normCrop(img: RawImage, face: DetectedFace, imageSize = 112): RawImage {
  const m = estimateNorm(face.kps, imageSize);
  return warpAffine(img, m, imageSize, imageSize);
}

export class ArcFace {
  private session!: ort.InferenceSession;
  private inputName = "";

  static async create(modelPath: string, sessions?: OrtSessionFactory): Promise<ArcFace> {
    const f = new ArcFace();
    f.session = await createOrtSession(modelPath, sessions);
    f.inputName = f.session.inputNames[0];
    return f;
  }

  /** Returns the L2-normalized 512-d embedding of an aligned 112x112 face. */
  async embedding(aligned: RawImage): Promise<Float32Array> {
    const blob = toBlob(aligned, 127.5, 127.5);
    const tensor = new ort.Tensor("float32", blob, [1, 3, 112, 112]);
    const outs = await this.session.run({ [this.inputName]: tensor });
    const raw = outs[this.session.outputNames[0]].data as Float32Array;
    const emb = new Float32Array(raw);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < emb.length; i++) emb[i] /= norm;
    return emb;
  }
}

