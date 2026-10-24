// Image primitives for face recognition (ported from the Phase 0 spike,
// tools/face-spike/src/image-ops.ts).
//
// Design note: sharp is used ONLY to decode JPEG/PNG into raw RGB bytes.
// Every resampling step (resize, affine warp) is implemented here in
// plain TypeScript to mirror OpenCV semantics (INTER_LINEAR resize and
// warpAffine with BORDER_CONSTANT 0), which is what the Python
// InsightFace reference uses via cv2. This keeps the numbers in
// RESULTS.md comparable with the Python golden output and keeps the
// future app-side image-ops swappable.

const sharp = require('sharp');

export interface RawImage {
  data: Uint8Array; // RGB, row-major, width*height*3
  width: number;
  height: number;
}

export async function decodeImage(path: string): Promise<RawImage> {
  const { data, info } = await sharp(path)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) {
    throw new Error(`expected 3 channels, got ${info.channels} for ${path}`);
  }
  return { data: new Uint8Array(data), width: info.width, height: info.height };
}

function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
  }
  return i;
}

/** Bilinear resize with the same coordinate mapping as cv2.resize INTER_LINEAR. */
export function resizeBilinear(src: RawImage, newWidth: number, newHeight: number): RawImage {
  const { data, width: sw, height: sh } = src;
  const out = new Uint8Array(newWidth * newHeight * 3);
  const scaleX = sw / newWidth;
  const scaleY = sh / newHeight;
  for (let y = 0; y < newHeight; y++) {
    const fy = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    const ya = reflect101(y0, sh);
    const yb = reflect101(y0 + 1, sh);
    for (let x = 0; x < newWidth; x++) {
      const fx = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.floor(fx);
      const wx = fx - x0;
      const xa = reflect101(x0, sw);
      const xb = reflect101(x0 + 1, sw);
      const o = (y * newWidth + x) * 3;
      const i00 = (ya * sw + xa) * 3;
      const i01 = (ya * sw + xb) * 3;
      const i10 = (yb * sw + xa) * 3;
      const i11 = (yb * sw + xb) * 3;
      for (let c = 0; c < 3; c++) {
        const v =
          data[i00 + c] * (1 - wx) * (1 - wy) +
          data[i01 + c] * wx * (1 - wy) +
          data[i10 + c] * (1 - wx) * wy +
          data[i11 + c] * wx * wy;
        out[o + c] = Math.round(v);
      }
    }
  }
  return { data: out, width: newWidth, height: newHeight };
}

/**
 * Affine warp with the same semantics as cv2.warpAffine(M, borderValue=0):
 * M maps source -> destination; sampling is bilinear; samples outside the
 * source contribute 0 (BORDER_CONSTANT).
 */
export function warpAffine(
  src: RawImage,
  m: [number, number, number, number, number, number], // [a,b,tx, c,d,ty]
  outWidth: number,
  outHeight: number,
): RawImage {
  const [a, b, tx, c, d, ty] = m;
  const det = a * d - b * c;
  if (det === 0) throw new Error("singular affine matrix");
  // inverse mapping dst -> src
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const itx = -(ia * tx + ib * ty);
  const ity = -(ic * tx + id * ty);
  const { data, width: sw, height: sh } = src;
  const out = new Uint8Array(outWidth * outHeight * 3);
  const sample = (x: number, y: number, ch: number): number => {
    if (x < 0 || y < 0 || x >= sw || y >= sh) return 0;
    return data[(y * sw + x) * 3 + ch];
  };
  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      const sx = ia * x + ib * y + itx;
      const sy = ic * x + id * y + ity;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const wx = sx - x0;
      const wy = sy - y0;
      const o = (y * outWidth + x) * 3;
      for (let ch = 0; ch < 3; ch++) {
        const v =
          sample(x0, y0, ch) * (1 - wx) * (1 - wy) +
          sample(x0 + 1, y0, ch) * wx * (1 - wy) +
          sample(x0, y0 + 1, ch) * (1 - wx) * wy +
          sample(x0 + 1, y0 + 1, ch) * wx * wy;
        out[o + ch] = Math.round(v);
      }
    }
  }
  return out ? { data: out, width: outWidth, height: outHeight } : src;
}

/** Place src at the top-left of a zero-filled canvas (SCRFD det_img). */
export function padToCanvas(src: RawImage, width: number, height: number): RawImage {
  const out = new Uint8Array(width * height * 3);
  for (let y = 0; y < src.height; y++) {
    const sOff = y * src.width * 3;
    out.set(src.data.subarray(sOff, sOff + src.width * 3), y * width * 3);
  }
  return { data: out, width, height };
}

/** NCHW float blob: (v - mean) / std per channel, RGB order preserved. */
export function toBlob(img: RawImage, mean: number, std: number): Float32Array {
  const { data, width, height } = img;
  const n = width * height;
  const blob = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      blob[c * n + i] = (data[i * 3 + c] - mean) / std;
    }
  }
  return blob;
}

// Encode a raw RGB image as a JPEG file (sharp is used for encoding only;
// all resampling above stays in TypeScript to mirror OpenCV semantics).
export async function encodeJpeg(img: RawImage, outPath: string, quality = 90): Promise<void> {
  const buf = Buffer.alloc(img.width * img.height * 3);
  for (let i = 0; i < img.data.length; i++) {
    buf[i] = Math.max(0, Math.min(255, Math.round(img.data[i])));
  }
  await sharp(buf, { raw: { width: img.width, height: img.height, channels: 3 } })
    .jpeg({ quality })
    .toFile(outPath);
}
