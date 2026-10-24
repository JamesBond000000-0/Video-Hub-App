import { invoke, isTauri } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { openPath, openUrl, revealItemInDir } from '@tauri-apps/plugin-opener';

/**
 * Electron preserves Map objects over IPC, while Tauri serializes command
 * arguments as JSON. Tag Maps so the compatibility layer can restore them.
 */
const IPC_MAP_KEY = '__video_hub_app_ipc_map__';

type IpcCallback = (...args: any[]) => void;

/** Values pending registration must be awaited before sending an IPC message. */
const pendingListeners = new Set<Promise<unknown>>();
const subscribers = new Map<string, Set<IpcCallback>>();
const registeredChannels = new Set<string>();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object') {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/** Convert values which JSON cannot represent but the existing IPC API uses. */
export function encodeIpcValue(value: unknown): unknown {
  if (value instanceof Map) {
    return {
      [IPC_MAP_KEY]: Array.from(value.entries(), ([key, entryValue]) => [
        encodeIpcValue(key),
        encodeIpcValue(entryValue),
      ]),
    };
  }

  if (Array.isArray(value)) {
    return value.map(encodeIpcValue);
  }

  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entryValue]) => [key, encodeIpcValue(entryValue)])
    );
  }

  return value;
}

/** Restore values tagged by encodeIpcValue after they cross the Tauri boundary. */
export function decodeIpcValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(decodeIpcValue);
  }

  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === IPC_MAP_KEY && Array.isArray(value[IPC_MAP_KEY])) {
      return new Map(
        value[IPC_MAP_KEY].map((entry: unknown) => {
          if (!Array.isArray(entry) || entry.length !== 2) {
            throw new Error('Invalid Map entry received over Tauri IPC');
          }
          return [decodeIpcValue(entry[0]), decodeIpcValue(entry[1])];
        })
      );
    }

    return Object.fromEntries(
      Object.entries(value).map(([key, entryValue]) => [key, decodeIpcValue(entryValue)])
    );
  }

  return value;
}

async function waitForListeners(): Promise<void> {
  // receiveFromMain is intentionally synchronous from the caller's perspective.
  // Await its in-flight Tauri registrations before sending a message which could
  // otherwise receive a response before anybody is listening (notably startup).
  await Promise.all(Array.from(pendingListeners));
}

function reportIpcError(channel: string, error: unknown): void {
  console.error(`[Tauri IPC] "${channel}" failed:`, error);
}

function dispatchToAngular(channel: string, payload: unknown): void {
  const decoded = decodeIpcValue(payload);
  const args = Array.isArray(decoded) ? decoded : [decoded];
  subscribers.get(channel)?.forEach((callback) => callback(...args));
}

function invokeMain(channel: string, args: unknown[]): Promise<void> {
  return invoke<void>('ipc_send', {
    channel,
    args: encodeIpcValue(args),
  });
}

/** Handle channels implemented by Tauri's official dialog and opener plugins. */
async function handlePluginChannel(channel: string, args: unknown[]): Promise<boolean> {
  switch (channel) {
    case 'choose-input': {
      const selected = await openDialog({ directory: true, multiple: false, recursive: true });
      if (typeof selected === 'string') {
        dispatchToAngular('input-folder-chosen', [selected]);
      }
      return true;
    }

    case 'choose-output': {
      const selected = await openDialog({ directory: true, multiple: false, recursive: true });
      if (typeof selected === 'string') {
        dispatchToAngular('output-folder-chosen', [selected]);
      }
      return true;
    }

    case 'reconnect-this-folder': {
      const selected = await openDialog({ directory: true, multiple: false, recursive: true });
      if (typeof selected === 'string') {
        dispatchToAngular('old-folder-reconnected', [args[0], selected]);
        // Electron keeps this path in its global input-source state. Update the
        // equivalent Tauri state so a later rescan does not use the old folder.
        await invokeMain('update-source-folder', [args[0], selected, false]);
      }
      return true;
    }

    case 'select-default-video-player': {
      const selected = await openDialog({
        title: 'Select default video player',
        multiple: false,
        filters: [
          { name: 'Executable', extensions: ['exe', 'app'] },
          { name: 'All files', extensions: ['*'] },
        ],
      });
      if (typeof selected === 'string') {
        dispatchToAngular('preferred-video-player-returning', [selected]);
      }
      return true;
    }

    case 'system-open-file-through-modal': {
      const selected = await openDialog({
        title: 'Select a Video Hub App library',
        multiple: false,
        filters: [{ name: 'Video Hub App files', extensions: ['vha2'] }],
      });
      if (typeof selected === 'string') {
        await invokeMain('load-this-vha-file', [selected, null]);
      }
      return true;
    }

    case 'open-media-file': {
      try {
        await openPath(String(args[0]));
      } catch {
        dispatchToAngular('file-not-found', []);
      }
      return true;
    }

    case 'open-in-explorer':
      await revealItemInDir(String(args[0]));
      return true;

    case 'please-open-url':
      await openUrl(String(args[0]));
      return true;

    default:
      return false;
  }
}

/**
 * Install the Tauri implementation of the API exposed by node/preload.js.
 * Electron keeps using its preload implementation because isTauri() is false.
 */
export function installTauriBridge(): boolean {
  if (!isTauri()) {
    return false;
  }

  const webview = getCurrentWebview();

  window.myElectron = {
    sendToMain: (channel: string, ...args: unknown[]): void => {
      void waitForListeners()
        .then(async () => {
          if (!(await handlePluginChannel(channel, args))) {
            await invokeMain(channel, args);
          }
        })
        .catch((error: unknown) => reportIpcError(channel, error));
    },

    receiveFromMain: (channel: string, callback: IpcCallback): void => {
      if (!subscribers.has(channel)) {
        subscribers.set(channel, new Set());
      }
      subscribers.get(channel)!.add(callback);

      if (registeredChannels.has(channel)) {
        return;
      }
      registeredChannels.add(channel);

      const registration = listen<unknown>(channel, (event) => {
        dispatchToAngular(channel, event.payload);
      }).catch((error: unknown) => {
        registeredChannels.delete(channel);
        reportIpcError(channel, error);
        return () => undefined;
      });

      pendingListeners.add(registration);
      void registration.then(() => pendingListeners.delete(registration));
    },

    setZoomFactor: (zoomFactor: number): void => {
      void webview.setZoom(zoomFactor).catch((error: unknown) => {
        reportIpcError('set-zoom-factor', error);
      });
    },

    clearCache: (): void => {
      // Tauri currently exposes cache clearing together with the other browsing
      // data. VHA keeps its settings on disk rather than in web storage.
      void webview.clearAllBrowsingData().catch((error: unknown) => {
        reportIpcError('clear-cache', error);
      });
    },
  };

  window.myAPI = {
    getPathForFile: (file: File): string => {
      // Some WebViews annotate dropped File objects with a native path. Keep
      // supporting that case, but never mistake a bare file name for a path.
      const nativePath = (file as File & { path?: string }).path;
      if (nativePath) {
        return nativePath;
      }

      throw new Error(
        'Native file paths from drag and drop have not been ported to Tauri yet. '
        + 'Use the file picker instead.'
      );
    },
  };

  return true;
}
