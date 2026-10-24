export type StarRating = 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5;

// must be heights from true `16:9` resolutions AND divisible by 8
export type AllowedScreenshotHeight = 144 | 216 | 288 | 360 | 432 | 504;

export type AllowedScreenshotHeightString = '144' | '216' | '288' | '360' | '432' | '504';

export type ResolutionString = '' | 'SD' | '720' | '720+' | '1080' | '1080+' | '4K' | '4K+';

export interface SourceFolder {
  path: string;
  watch: boolean;
}

export type InputSources = Record<number, SourceFolder>;

export interface FinalObject {
  addTags: string[];             // tags to add
  hubName: string;               // the name of the hub -- for recently-opened
  images: ImageElement[];
  // inputDir: string;           // became `inputDirs` in VHA3 - gets upgraded to `inputDirs` on first open in VHA3
  inputDirs: InputSources;       // map the `inputSource` number to input directory (replaces `inputDir`)
  numOfFolders: number;          // number of folders - is always re-counted when app starts
  removeTags: string[];          // tags to remove
  screenshotSettings: ScreenshotSettings;
  faceSettings?: FaceSettings; // optional so pre-existing .vha2 files still parse; defaults are filled in on open
  tagColors?: Record<string, string>; // lookup table for tag name to color mapping
  version: number;               // version of this vha file
}

export interface ImageElement {
  birthtime: number;             // file creation time
  cleanName: string;             // file name cleaned of dots, underscores,and file extension; for searching. Can be *FOLDER* sometimes
  duration: number;              // number of seconds - duration of film
  fileName: string;              // full file name with extension - for opening the file
  fileSize: number;              // file size in bytes
  bitrate: number;               // bitrate of the displayed video file - (fileSize/duration)*1024
  fps: number;                   // base frame rate of the video in fps
  hash: string;                  // used for detecting changed files and as a screenshot identifier
  height: number;                // height of the video (px)
  inputSource: number;           // corresponding to `inputDirs`
  mtime: number;                 // file modification time
  partialPath: string;           // for opening the file, just prepend the `inputDir` (starts with "/", is "/fldr1/fldr2", or can be "")
  screens: number;               // number of screenshots for this file
  stars: StarRating;             // star rating 0 = n/a, otherwise 1, 2, 3
  timesPlayed: number;           // number of times the file has been launched by VHA
  lastPlayed: number;            // timestamp of last time video was played; for last-played sorting order
  width: number;                 // width of the video (px)
  // ========================================================================
  // OPTIONAL
  // ------------------------------------------------------------------------
  defaultScreen?: number;        // index of default screenshot to show
  notes?: string;                // any free-form notes a user may want to add to any video
  persons?: string[];            // face-recognition person IDs in this video (cache; source of truth is the hub's faces/index.json)
  playlist?: number;             // timestamp of when user clicked `add to playlist`
  tags?: string[];               // tags associated with this particular file
  year?: number;                 // optional tag to track the year of the video
  // ========================================================================
  // Stripped out and not saved in the VHA file
  // ------------------------------------------------------------------------
  deleted?: boolean;             // toggled after a successful delete of file; ImageElement removed before saving .vha2 file
  durationDisplay: string;       // displayed duration in X:XX:XX format
  fileSizeDisplay: string;       // displayed as XXXmb or X.Xgb -- also co-opted for showing number of files in a *FOLDER*
  index: number;                 // for the `default` sort order
  resBucket: number;             // the resolution category the video falls into (for faster sorting)
  resolution: ResolutionString;  // e.g. `720`, `1080`, `SD`, `HD`, etc
  selected?: boolean;            // for batch-tagging of videos
  uuid: string                   // to help track inside @for in home.component.html
}

export interface ImageElementPlus extends ImageElement {
  fullPath: string;              // the full path to video file -- used only in node for extracting meta & thumbnails
}

// Use this to create a new ImageElement if needed
export function NewImageElement(): ImageElement {
  return {
    birthtime: 0,
    bitrate: 0,
    cleanName: '',
    duration: 0,
    durationDisplay: '',
    fileName: '',
    fileSize: 0,
    fileSizeDisplay: '',
    fps: 0,
    hash: '',
    height: 144,
    index: 0,
    inputSource: 0,
    lastPlayed: 0,
    mtime: 0,
    partialPath: '',
    resBucket: 0,
    resolution: '',
    screens: 0,
    stars: 0.5,
    timesPlayed: 0,
    uuid: '',
    width: 0,
  };
}

export interface ScreenshotSettings {
  clipHeight: AllowedScreenshotHeight;
  clipSnippetLength: number;
  clipSnippets: number;        // the number of video snippets in every clip; 0 == no clip extracted
  fixed: boolean;
  height: AllowedScreenshotHeight;
  n: number;
}

export type FaceModelPack = 'quality' | 'light' | 'lvface';

// Execution provider for face inference (design doc decision D9).
// 'auto' walks a per-platform fallback ladder (win32: CUDA -> DirectML ->
// CPU; linux: CUDA -> CPU; darwin: CoreML -> CPU); the other values pin
// the first candidate, still with CPU as the last-resort fallback.
export type FaceProviderSetting = 'auto' | 'cpu' | 'cuda' | 'directml' | 'coreml';

// Face recognition settings (design doc: docs/FACE_RECOGNITION_DESIGN.md §7).
// Saved inside the .vha2 file; face data itself lives in the hub's faces/ folder.
export interface FaceSettings {
  enabled: boolean;                // opt-in: face scanning (default off)
  modelPack: FaceModelPack;        // 'quality' = buffalo_l (det_10g + w600k_r50); 'light' = buffalo_sc (det_500m + w600k_mbf); 'lvface' = det_10g + LVFace-T
  maxFramesPerVideo: number;       // cap for the §5.1 frame sampling
  sampleIntervalSec: number;       // base interval for the §5.1 frame sampling
  minFaceSizePx: number;           // faces smaller than this are discarded (§5.2)
  clusterThreshold: number;        // global clustering threshold T_global (§5.6, used from Phase 2)
  autoScanNew: boolean;            // scan faces automatically after thumbnail extraction
  attributesEnabled: boolean;      // age/gender estimation (§5.8, used from Phase 2)
  attributeSamplesPerGroup: number; // samples per (video, group) for attributes (§5.8, Phase 2)
  provider: FaceProviderSetting;   // ONNX execution provider (decision D9); absent in older files = 'auto'
}
