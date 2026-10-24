// LVFace recognition - TypeScript port of the official ONNX inference
// path (bytedance/LVFace, inference_onnx.py LVFaceONNXInferencer).
//
// Preprocessing evidence (official repo, commit 80bd20c):
// - inference_onnx.py loads with OpenCV (BGR), resizes to 112x112,
//   converts BGR -> RGB, transposes to NCHW and applies
//   ((px / 255) - 0.5) / 0.5. RawImage in this spike is already RGB, so
//   preserving RGB channel order through toBlob() is the equivalent
//   input. (px - 127.5) / 127.5 is algebraically identical.
// - The official IJB evaluation (onnx_ijbc.py) aligns five landmarks to
//   SRC = [[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],
//   [33.5493,92.3655],[62.7299,92.2041]] with SRC[:,0] += 8, which is
//   exactly the ArcFace 112x112 template used to make the Phase 0
//   aligned crops. The comparison therefore reuses those crops.
// - No horizontal-flip test-time augmentation is used here: the
//   official single-image ONNX inferencer does not flip either, and the
//   app pipeline is single-view. (The IJB scripts optionally use flip
//   TTA; that is an evaluation protocol difference, not preprocessing.)

import * as ort from "onnxruntime-node";
import { toBlob, type RawImage } from "./image-ops.js";

export class LVFace {
  private session!: ort.InferenceSession;
  private inputName = "";
  private outputName = "";
  outputDim = 0;

  static async create(modelPath: string): Promise<LVFace> {
    const f = new LVFace();
    f.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
    });
    f.inputName = f.session.inputNames[0];
    f.outputName = f.session.outputNames[0];
    return f;
  }

  get inputNames(): readonly string[] {
    return this.session.inputNames;
  }

  get outputNames(): readonly string[] {
    return this.session.outputNames;
  }

  /** Returns the L2-normalized embedding of an aligned 112x112 face. */
  async embedding(aligned: RawImage): Promise<Float32Array> {
    const blob = toBlob(aligned, 127.5, 127.5);
    const tensor = new ort.Tensor("float32", blob, [1, 3, 112, 112]);
    const outs = await this.session.run({ [this.inputName]: tensor });
    const raw = outs[this.outputName].data as Float32Array;
    const emb = new Float32Array(raw);
    if (this.outputDim === 0) this.outputDim = emb.length;
    let norm = 0;
    for (const v of emb) norm += v * v;
    norm = Math.sqrt(norm);
    if (!Number.isFinite(norm) || norm === 0) {
      throw new Error(`invalid LVFace embedding norm: ${norm}`);
    }
    for (let i = 0; i < emb.length; i++) emb[i] /= norm;
    return emb;
  }
}
