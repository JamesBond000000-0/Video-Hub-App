import { Injectable } from '@angular/core';

import type { FaceSettings, ImageElement } from '../../../interfaces/final-object.interface';
import type { FaceIndexFile, FaceIndexPayload, FaceModelStatus, FaceStats } from '../../../node/face/face-types';

// Angular-side defaults for FaceSettings ( mirror of node/main-globals.ts;
// kept in sync with the design doc §7 )
export const DEFAULT_FACE_SETTINGS: FaceSettings = {
  enabled: false,
  modelPack: 'quality',
  maxFramesPerVideo: 60,
  sampleIntervalSec: 30,
  minFaceSizePx: 40,
  clusterThreshold: 0.5,
  autoScanNew: true,
  attributesEnabled: false,
  attributeSamplesPerGroup: 5,
};

export interface FaceGroupForDisplay {
  groupId: number;
  representativeFaceId: string; // file name stem of the display crop in faces/crops/
  times: number[];              // appearance times (seconds), ascending
}

/**
 * Renderer-side window onto face recognition (design doc §9.1):
 * holds the model placement status and the face index received from the
 * main process, and sends face requests back over IPC.
 */
@Injectable({ providedIn: 'root' })
export class FaceService {

  modelStatus: FaceModelStatus = null;
  faceIndex: FaceIndexFile = null;
  modelMismatch = false;
  stats: FaceStats = { scannedVideos: 0, videosWithFaces: 0, totalFaces: 0, totalGroups: 0, totalPersons: 0 };

  private initialized = false;

  initialize(): void {
    if (this.initialized) {
      return;
    }
    this.initialized = true;

    (window as any).myElectron.receiveFromMain('face-model-status-returning', (status: FaceModelStatus) => {
      this.modelStatus = status;
    });

    (window as any).myElectron.receiveFromMain('face-index-returning', (payload: FaceIndexPayload) => {
      this.faceIndex = payload ? payload.index : null;
      this.modelMismatch = payload ? payload.modelMismatch : false;
      if (payload && payload.stats) {
        this.stats = payload.stats;
      }
    });

    (window as any).myElectron.receiveFromMain('face-index-updated', () => {
      this.requestIndex();
    });
  }

  requestModelStatus(): void {
    (window as any).myElectron.sendToMain('face-get-model-status');
  }

  requestIndex(): void {
    (window as any).myElectron.sendToMain('face-get-index');
  }

  requestScanAll(imageElements: ImageElement[]): void {
    (window as any).myElectron.sendToMain('face-scan-request', imageElements);
  }

  sendSettings(settings: FaceSettings): void {
    (window as any).myElectron.sendToMain('face-settings-updated', settings);
  }

  /**
   * Faces of one video, grouped by their within-video group (§5.5),
   * ordered by representative quality (the index already stores them
   * in that order). Empty when the video has no face data.
   */
  faceGroupsForVideo(hash: string): FaceGroupForDisplay[] {
    if (!this.faceIndex || !this.faceIndex.videos) {
      return [];
    }
    const entry = this.faceIndex.videos[hash];
    if (!entry || !entry.faces || entry.faces.length === 0) {
      return [];
    }
    const groups = new Map<number, FaceGroupForDisplay>();
    entry.faces.forEach((face) => {
      if (!groups.has(face.groupId)) {
        groups.set(face.groupId, { groupId: face.groupId, representativeFaceId: '', times: [] });
      }
      const group = groups.get(face.groupId);
      group.times.push(face.t);
      if (face.representative) {
        group.representativeFaceId = face.faceId;
      }
    });
    const result: FaceGroupForDisplay[] = [];
    groups.forEach((group) => {
      group.times.sort((a, b) => a - b);
      if (group.representativeFaceId) {
        result.push(group);
      }
    });
    return result;
  }

}
