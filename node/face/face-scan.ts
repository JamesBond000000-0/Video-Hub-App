import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

import { generateFaceFrameArgs } from '../main-extract';

import { ArcFace, normCrop } from './arcface';
import { groupWithinVideo } from './cluster';
import { decodeImage, encodeJpeg, resizeBilinear, type RawImage } from './image-ops';
import { Scrfd, type DetectedFace } from './scrfd';
import { WITHIN_VIDEO_THRESHOLD, type FaceScanJob, type FaceScanResult, type ScannedFace } from './face-types';
import { FaceStore } from './face-store';

const ffmpegPath = require('ffmpeg-static').replace('app.asar', 'app.asar.unpacked');

// ============================================================================
// Per-video face scan pipeline (design doc §5.1-§5.5):
// sample frames -> SCRFD detect -> filter -> align -> ArcFace embedding ->
// thumbnails -> within-video grouping. Pure Node (no Electron imports) so it
// can run inside the utility-process worker and in a plain Node harness.
// ============================================================================

interface KeptFace {
  scanned: ScannedFace;
  framePath: string;      // kept until grouping so representative crops can be cut
  det: DetectedFace;
  aligned: RawImage;
}

function sampleTimes(duration: number, intervalSec: number, maxFrames: number): number[] {
  const count = Math.max(10, Math.min(maxFrames, Math.ceil(duration / intervalSec)));
  const times: number[] = [];
  for (let i = 0; i < count; i++) {
    times.push(Math.min(duration - 0.1, ((i + 0.5) * duration) / count));
  }
  return times;
}

/**
 * Extract and decode the frame at `t` seconds of a video (same ffmpeg
 * arguments as the scan pipeline). Used by the worker's attribute job
 * (§5.8), which re-extracts the handful of sample frames instead of
 * keeping every scanned frame on disk. Returns null on failure.
 */
export async function grabFrameImage(videoPath: string, t: number, outPath: string): Promise<RawImage | null> {
  try {
    await runFfmpeg(generateFaceFrameArgs(videoPath, t, 1280, outPath));
  } catch (err) {
    return null;
  }
  if (!fs.existsSync(outPath)) {
    return null;
  }
  try {
    return await decodeImage(outPath);
  } catch (err) {
    return null;
  }
}

function runFfmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const ff = spawn(ffmpegPath, args);
    let stderr = '';
    const timer = setTimeout(() => {
      try {
        ff.kill('SIGKILL');
      } catch (e) { /* already gone */ }
      reject(new Error('ffmpeg timed out'));
    }, 60000);
    ff.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    ff.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    ff.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve();
      } else {
        reject(new Error('ffmpeg exited ' + code + ': ' + stderr.slice(-300)));
      }
    });
  });
}

// Display crop (§5.3): square around the face centre, side = max(w, h) * 1.35,
// clamped to the frame, resized to 256x256.
function displayCrop(img: RawImage, face: DetectedFace): RawImage {
  const [x1, y1, x2, y2] = face.bbox;
  const side = Math.max(x2 - x1, y2 - y1) * 1.35;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const sx = Math.max(0, Math.min(img.width - side, cx - side / 2));
  const sy = Math.max(0, Math.min(img.height - side, cy - side / 2));
  const sw = Math.min(side, img.width - sx);
  const sh = Math.min(side, img.height - sy);
  const crop: RawImage = { data: new Uint8Array(sw * sh * 3), width: sw, height: sh };
  const x0 = Math.round(sx);
  const y0 = Math.round(sy);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const si = ((y0 + y) * img.width + (x0 + x)) * 3;
      const di = (y * sw + x) * 3;
      crop.data[di] = img.data[si];
      crop.data[di + 1] = img.data[si + 1];
      crop.data[di + 2] = img.data[si + 2];
    }
  }
  return resizeBilinear(crop, 256, 256);
}

export class FaceEngine {
  private detector!: Scrfd;
  private recognizer!: ArcFace;

  static async create(detModelPath: string, recModelPath: string): Promise<FaceEngine> {
    const engine = new FaceEngine();
    engine.detector = await Scrfd.create(detModelPath);
    engine.recognizer = await ArcFace.create(recModelPath);
    return engine;
  }

  async scanVideo(job: FaceScanJob): Promise<FaceScanResult> {
    if (!job.duration || job.duration <= 0) {
      return { status: 'error', frameCount: 0, faces: [], error: 'video duration unknown' };
    }
    const times = sampleTimes(job.duration, job.sampleIntervalSec, job.maxFramesPerVideo);
    const tmpDir = path.join(job.facesDir, 'tmp', job.hash);
    fs.mkdirSync(tmpDir, { recursive: true });
    fs.mkdirSync(FaceStore.alignedDir(job.facesDir), { recursive: true });
    fs.mkdirSync(FaceStore.cropsDir(job.facesDir), { recursive: true });

    const kept: KeptFace[] = [];
    let framesRead = 0;
    {
      for (let fi = 0; fi < times.length; fi++) {
        const t = times[fi];
        const framePath = path.join(tmpDir, 'frame-' + String(fi).padStart(3, '0') + '.jpg');
        try {
          await runFfmpeg(generateFaceFrameArgs(job.videoPath, t, 1280, framePath));
        } catch (err) {
          continue; // a single un-grabbable frame must not fail the whole video
        }
        if (!fs.existsSync(framePath)) {
          continue;
        }
        framesRead++;
        const img = await decodeImage(framePath);
        let faces = await this.detector.detect(img, [640, 640]);
        if (faces.length === 0) {
          // §5.2: SCRFD at 640 can miss faces that fill most of the frame
          // (Phase 0 finding); retry once at 320.
          faces = await this.detector.detect(img, [320, 320]);
        }
        for (let di = 0; di < faces.length; di++) {
          const face = faces[di];
          // §5.2: discard when the shorter side is below the minimum
          // (raw float size, matching the Phase 0 spike semantics)
          const w = face.bbox[2] - face.bbox[0];
          const h = face.bbox[3] - face.bbox[1];
          if (Math.min(w, h) < job.minFaceSizePx) {
            continue;
          }
          const aligned = normCrop(img, face);
          const embedding = await this.recognizer.embedding(aligned);
          const faceAreaRatio = (w * h) / (img.width * img.height);
          const faceId = job.hash + '-' + String(kept.length + 1).padStart(4, '0');
          kept.push({
            scanned: {
              faceId,
              t: Math.round(t * 10) / 10,
              bbox: [
                Math.round((face.bbox[0] / img.width) * 10000) / 10000,
                Math.round((face.bbox[1] / img.height) * 10000) / 10000,
                Math.round((w / img.width) * 10000) / 10000,
                Math.round((h / img.height) * 10000) / 10000,
              ],
              score: Math.round(face.score * 1000) / 1000,
              quality: Math.round(face.score * Math.sqrt(faceAreaRatio) * 10000) / 10000,
              groupId: -1,
              representative: false,
              embedding,
            },
            framePath,
            det: face,
            aligned,
          });
          await encodeJpeg(aligned, path.join(FaceStore.alignedDir(job.facesDir), faceId + '.jpg'), 90);
        }
      }
    }

    if (framesRead === 0) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
      return { status: 'error', frameCount: times.length, faces: [], error: 'frame extraction failed for all sampled times' };
    }

    // §5.5 within-video grouping; groups ordered by representative quality
    const groups = groupWithinVideo(kept.map((k) => k.scanned), WITHIN_VIDEO_THRESHOLD);
    groups.sort((a, b) => b.representative.quality - a.representative.quality);
    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      group.members.forEach((member) => {
        member.groupId = gi;
        member.representative = member === group.representative;
      });
      const repKept = kept.filter((k) => k.scanned === group.representative)[0];
      if (repKept) {
        const frameImg = await decodeImage(repKept.framePath);
        const crop = displayCrop(frameImg, repKept.det);
        await encodeJpeg(crop, path.join(FaceStore.cropsDir(job.facesDir), repKept.scanned.faceId + '.jpg'), 90);
      }
    }

    fs.rmSync(tmpDir, { recursive: true, force: true });
    return {
      status: kept.length > 0 ? 'done' : 'none',
      frameCount: times.length,
      faces: kept.map((k) => k.scanned),
    };
  }
}
