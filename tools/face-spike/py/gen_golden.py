#!/usr/bin/env python3
"""Generate golden (reference) results with the official Python InsightFace.

Uses the locally downloaded buffalo_l / buffalo_sc ONNX files under
tools/face-spike/models/ (NOT the auto-downloaded packs), runs
FaceAnalysis on the sample images, and writes bbox / landmarks /
512-d embedding / sex / age per face to golden/golden_<pack>.json.

Sample images are the ones bundled with the official `insightface`
pip package (insightface/data/images/), copied into ../samples/:
  - t1.jpg               (multi-face photo, 1280x886)
  - Tom_Hanks_54745.png  (single aligned face, 112x112)
  - mask_black.jpg       (single face with mask)
They are test fixtures shipped by the library itself; no person is
identified beyond the numeric outputs the library returns.
"""
import json
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.dirname(HERE)
# Per-image detection size. det_size must be recorded because SCRFD's
# behaviour depends on it: the 112x112 single-face sample is only
# detected at det_size 320 (at 640 the face covers >~40% of the input,
# a known SCRFD weakness for very large faces). mask_black.jpg stays
# in the set as a documented negative (masked face, 0 detections).
SAMPLES = [
    ("t1.jpg", (640, 640)),
    ("Tom_Hanks_54745.png", (320, 320)),
    ("mask_black.jpg", (640, 640)),
]


def r6(x):
    return round(float(x), 6)


def main(pack: str) -> None:
    app = FaceAnalysis(name=pack, root=SPIKE, providers=["CPUExecutionProvider"])
    out = {"pack": pack, "images": []}
    for name, det_size in SAMPLES:
        app.prepare(ctx_id=-1, det_size=det_size)
        path = os.path.join(SPIKE, "samples", name)
        img = cv2.imread(path)
        if img is None:
            print(f"skip (unreadable): {name}")
            continue
        faces = app.get(img)
        entry = {
            "file": name,
            "detSize": list(det_size),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "faces": [],
        }
        for f in faces:
            emb = f.normed_embedding if f.normed_embedding is not None else f.embedding
            sex = getattr(f, "sex", None)
            if sex is None and getattr(f, "gender", None) is not None:
                sex = "M" if int(f.gender) == 1 else "F"
            entry["faces"].append(
                {
                    "bbox": [r6(v) for v in f.bbox.tolist()],
                    "detScore": r6(f.det_score),
                    "kps": [[r6(x), r6(y)] for x, y in f.kps.tolist()],
                    "embedding": [r6(v) for v in np.asarray(emb).tolist()],
                    "sex": sex,
                    "age": (int(f.age) if getattr(f, "age", None) is not None else None),
                }
            )
        out["images"].append(entry)
        print(f"{pack} {name}: {len(entry['faces'])} face(s)")
    dest = os.path.join(SPIKE, "golden", f"golden_{pack}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "buffalo_l")
