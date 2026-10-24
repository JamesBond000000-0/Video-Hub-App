// GenderAge attribute model (design doc §5.8) - TypeScript port of
// insightface/model_zoo/attribute.py, from the Phase 0 spike
// (tools/face-spike/src/arcface.ts), where equivalence with the Python
// reference was verified (sex match, age within ±2 years on the samples).
//
// Used only inside the face worker, and only when the user opted in via
// FaceSettings.attributesEnabled. Estimates are computed per sample face
// and aggregated per person (face-types.aggregateAttributes); per-face
// values are never persisted anywhere.

import * as ort from 'onnxruntime-node';
import type { Affine } from './arcface';
import { toBlob, warpAffine, type RawImage } from './image-ops';
import { createOrtSession, type OrtSessionFactory } from './ort-provider';
import type { DetectedFace } from './scrfd';
import type { AttributeSample } from './face-types';

export class GenderAge {
  private session!: ort.InferenceSession;
  private inputName = '';
  static readonly INPUT_SIZE = 96;

  static async create(modelPath: string, sessions?: OrtSessionFactory): Promise<GenderAge> {
    const g = new GenderAge();
    g.session = await createOrtSession(modelPath, sessions);
    g.inputName = g.session.inputNames[0];
    return g;
  }

  /**
   * Port of Attribute.get: the crop is NOT the landmark-aligned one.
   * The reference centres a similarity transform on the bbox centre with
   * scale = 96 / (max(w, h) * 1.5), rotation 0; with rotation 0 the
   * skimage transform composition collapses to the affine below.
   * The model's own graph starts with Sub/Mul scalars, so the Python
   * code feeds raw pixels (mean 0, std 1).
   *
   * `img` must be the frame the face was detected in and `face.bbox`
   * in that frame's pixel coordinates.
   */
  async get(img: RawImage, face: DetectedFace): Promise<AttributeSample> {
    const size = GenderAge.INPUT_SIZE;
    const [x1, y1, x2, y2] = face.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const s = size / (Math.max(w, h) * 1.5);
    const m: Affine = [s, 0, size / 2 - s * cx, 0, s, size / 2 - s * cy];
    const crop = warpAffine(img, m, size, size);
    const blob = toBlob(crop, 0, 1);
    const tensor = new ort.Tensor('float32', blob, [1, 3, size, size]);
    const outs = await this.session.run({ [this.inputName]: tensor });
    const pred = outs[this.session.outputNames[0]].data as Float32Array;
    const gender = pred[1] > pred[0] ? 1 : 0; // argmax of first two logits
    return { sex: gender === 1 ? 'M' : 'F', age: Math.round(pred[2] * 100) };
  }
}
