import { Injectable } from '@angular/core';

import { ImageElementService } from './image-element.service';

import type { FaceSettings, ImageElement } from '../../../interfaces/final-object.interface';
import type { FaceIndexFile, FaceIndexPayload, FaceModelStatus, FaceStats, PersonRecord } from '../../../node/face/face-types';

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
  personId: string;             // person this group was clustered into ('' when unclustered)
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
  personsByVideo: Record<string, string[]> = {};
  personsVersion = 0; // bumped on every index receipt; used as a pipe trigger

  private initialized = false;

  constructor(
    private imageElementService: ImageElementService
  ) { }

  /** All clustered persons (empty until the first re-clustering). */
  get persons(): PersonRecord[] {
    return (this.faceIndex && this.faceIndex.persons) || [];
  }

  personById(personId: string): PersonRecord {
    return this.persons.filter((p) => p.personId === personId)[0];
  }

  /** Numeric part of a person ID, for the "Person N" fallback label. */
  personNumber(person: PersonRecord): number {
    const n = parseInt(person.personId.substring(1), 10);
    return isNaN(n) ? 0 : n;
  }

  /** Sidebar order (§8.1): most videos first; hidden persons excluded. */
  sidebarPersons(limit: number): PersonRecord[] {
    return this.persons
      .filter((p) => !p.hidden)
      .sort((a, b) => (b.videoHashes.length - a.videoHashes.length) || (b.faceCount - a.faceCount))
      .slice(0, limit);
  }

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
      this.personsByVideo = (payload && payload.personsByVideo) || {};
      this.personsVersion++;
      this.applyPersonsCache();
    });

    (window as any).myElectron.receiveFromMain('face-index-updated', () => {
      this.requestIndex();
    });
  }

  /**
   * Write the index-derived person IDs into the in-memory image elements
   * (the .vha2 `persons` cache, §6.5). The index is the source of truth:
   * videos absent from it lose their cached IDs (e.g. after the faces
   * folder was deleted). Marks the hub dirty when anything changed; the
   * file itself is written by the app's normal save flow.
   */
  applyPersonsCache(): void {
    let changed = false;
    this.imageElementService.imageElements.forEach((element) => {
      const next: string[] = this.personsByVideo[element.hash] || [];
      const current: string[] = element.persons || [];
      const same = next.length === current.length && next.every((id, i) => id === current[i]);
      if (!same) {
        element.persons = next.length > 0 ? [...next] : undefined;
        changed = true;
      }
    });
    if (changed) {
      this.imageElementService.finalArrayNeedsSaving = true;
    }
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

  requestRecluster(imageElements: ImageElement[]): void {
    (window as any).myElectron.sendToMain('face-recluster', imageElements);
  }

  requestAttributes(imageElements: ImageElement[]): void {
    (window as any).myElectron.sendToMain('face-attributes-request', imageElements);
  }

  renamePerson(personId: string, name: string): void {
    (window as any).myElectron.sendToMain('face-rename-person', personId, name);
  }

  setPersonHidden(personId: string, hidden: boolean): void {
    (window as any).myElectron.sendToMain('face-set-person-hidden', personId, hidden);
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
        groups.set(face.groupId, { groupId: face.groupId, representativeFaceId: '', times: [], personId: '' });
      }
      const group = groups.get(face.groupId);
      group.times.push(face.t);
      if (face.personId) {
        group.personId = face.personId;
      }
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
