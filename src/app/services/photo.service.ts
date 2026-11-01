import { Injectable } from '@angular/core';

import type { PhotoSettings } from '../../../interfaces/final-object.interface';

// Angular-side defaults for PhotoSettings
// (docs/PHOTO_SUPPORT_DESIGN.md P2: opt-in, default off)
export const DEFAULT_PHOTO_SETTINGS: PhotoSettings = {
  enabled: false,
};

@Injectable({ providedIn: 'root' })
export class PhotoService {

  /** Sync the hub's photo settings to the main process / Tauri backend */
  sendSettings(settings: PhotoSettings): void {
    (window as any).myElectron.sendToMain('photo-settings-updated', settings);
  }

}
