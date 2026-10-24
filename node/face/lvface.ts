// LVFace recognition (ByteDance, ICCV 2025) - port of the official ONNX
// inference path (bytedance/LVFace, inference_onnx.py LVFaceONNXInferencer),
// via the Phase 0 follow-up spike (tools/face-spike/src/lvface.ts), where
// the preprocessing was verified against the official code (commit 80bd20c)
// and the model was measured against ArcFace on the BBC sample set
// (tools/face-spike/LVFACE_COMPARISON.md).
//
// Preprocessing equivalence with the official inferencer:
// - it loads with OpenCV (BGR), converts BGR -> RGB, transposes to NCHW and
//   applies ((px / 255) - 0.5) / 0.5. The RawImage here is already RGB, so
//   feeding RGB order through toBlob(aligned, 127.5, 127.5) is identical
//   (the normalization formulas are algebraically the same).
// - alignment uses the same 5-point template as ArcFace (the official IJB
//   script's SRC with x += 8 equals ARCFACE_DST), so the aligned crop from
//   arcface.normCrop is the correct input unchanged.
// - no horizontal-flip TTA (the official single-image inferencer does not
//   flip either, and the app pipeline is single-view).
// The official inferencer does NOT normalize its output; embeddings are
// L2-normalized here, like ArcFace, because clustering uses cosine
// similarity throughout.

import * as ort from "onnxruntime-node";
import { toBlob, type RawImage } from "./image-ops";
import { createOrtSession, type OrtSessionFactory } from "./ort-provider";

export class LVFace {
  private session!: ort.InferenceSession;
  private inputName = "";
  private outputName = "";

  static async create(modelPath: string, sessions?: OrtSessionFactory): Promise<LVFace> {
    const f = new LVFace();
    f.session = await createOrtSession(modelPath, sessions);
    f.inputName = f.session.inputNames[0];
    f.outputName = f.session.outputNames[0];
    return f;
  }

  /** Returns the L2-normalized 512-d embedding of an aligned 112x112 face. */
  async embedding(aligned: RawImage): Promise<Float32Array> {
    const blob = toBlob(aligned, 127.5, 127.5);
    const tensor = new ort.Tensor("float32", blob, [1, 3, 112, 112]);
    const outs = await this.session.run({ [this.inputName]: tensor });
    const raw = outs[this.outputName].data as Float32Array;
    const emb = new Float32Array(raw);
    let norm = 0;
    for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
    norm = Math.sqrt(norm);
    if (!Number.isFinite(norm) || norm === 0) {
      throw new Error(`invalid LVFace embedding norm: ${norm}`);
    }
    for (let i = 0; i < emb.length; i++) emb[i] /= norm;
    return emb;
  }
}
