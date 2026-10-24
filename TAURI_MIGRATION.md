# Tauri migration

This repository contains an **experimental Tauri 2 runtime alongside Electron**. Electron is still the supported runtime and `npm start` / `npm run electron` are unchanged.

> **Migration status: existing hubs can be restored and saved, and new hubs can be scanned and imported with external FFmpeg/FFprobe executables, but the runtime is not yet feature-complete.** Continuous folder watchers and several integrations still require Electron. Test Tauri with copies of important hubs until native cross-platform testing is complete.

## Run the Tauri target

Install the JavaScript dependencies as usual:

```sh
npm install --legacy-peer-deps
```

Install Rust 1.90 or newer and the [Tauri platform prerequisites](https://v2.tauri.app/start/prerequisites/), then run:

```sh
npm run tauri:dev
```

Create a production frontend and platform bundle with:

```sh
npm run tauri:build
```

Useful additional commands:

```sh
npm run build:renderer       # Angular frontend only
npm run build:renderer-prod  # optimized Angular frontend only
npx tauri info               # inspect the local Rust/WebView toolchain
cargo test --manifest-path src-tauri/Cargo.toml
```

## Architecture

The Angular application continues to use the API previously exposed by `node/preload.js`:

```ts
window.myElectron.sendToMain(channel, ...args);
window.myElectron.receiveFromMain(channel, callback);
```

`src/main.ts` installs `src/desktop/tauri-bridge.ts` only when it detects a Tauri WebView. Electron therefore continues to use its existing preload bridge.

The Tauri message flow is:

1. `sendToMain` calls the Rust `ipc_send` command with a channel and an argument array, or routes through an official Tauri plugin.
2. `src-tauri/src/ipc.rs` routes backend channels.
3. Rust emits frontend events using the original channel name and an array payload.
4. `receiveFromMain` spreads that payload into the existing Angular callback.

The bridge also tags and restores `Map` values because Electron IPC preserves them but Tauri uses JSON serialization. Event registration is awaited before outgoing messages to avoid losing immediate responses during startup. Multiple Angular subscribers share one Tauri event subscription per channel.

### Rust modules

- `state.rs`: compatible settings location, current hub, localized messages, and shutdown state
- `settings.rs`: settings loading and saving
- `hub.rs`: VHA2-to-VHA3 upgrade, display metadata, source connectivity, and hub serialization
- `media.rs`: recursive scanning, FFprobe metadata, compatible hashing, and FFmpeg extraction jobs
- `storage.rs`: tagged-Map conversion, backups, and atomic file replacement
- `ipc.rs`: Electron-compatible channel routing and user-facing error events

## Implemented

### Shell and IPC

- Tauri 2 project metadata, application icon, and Windows/macOS/Linux bundling configuration
- Existing borderless Angular UI, including native Tauri title-bar dragging
- Minimize, maximize, restore, zoom, and WebView browsing-data clearing
- Explicit errors for IPC channels whose backend has not been ported
- Electron and Tauri dependencies and build paths kept separate

### Settings and hubs

- Restores the existing Electron `settings.json`, including portable-mode placement
- Restores the current hub at startup and opens `.vha2` files through a native picker
- Upgrades version-2 hubs and computes the temporary fields expected by Angular
- Saves changed hub metadata and settings on close, or before opening another hub
- Removes deleted/temporary entries, sorts and deduplicates images, and updates folder counts on save
- Creates `<filename>.bak` before replacing existing settings or hub files
- Uses atomic replacement for new data and backups
- Keeps the application open and shows an error when a close-time save fails
- Converts the shortcut `Map` back to the plain JSON object used by Electron

### Directory scanning and media import

- Creates a version-3 `.vha2` and its `vha-<hubName>` asset directories before importing
- Rejects duplicate hub names and uses the same atomic/backed-up persistence as existing hubs
- Recursively scans all supported video extensions plus the user's additional extensions
- Skips directories beginning with `vha-` so generated clips are not imported as source videos
- Emits the existing scan, metadata, and import-progress events expected by Angular
- Marks files missing during a rescan through the existing `all-files-found-in-dir` Map event
- Extracts duration, dimensions, frame rate, timestamps, and file size with FFprobe
- Reuses Electron's sampled MD5 algorithm, so existing generated-media hashes remain compatible
- Generates `<hash>.jpg` thumbnails and filmstrips, plus optional `<hash>.mp4` and `<hash>.jpg` clips
- Generates missing media on demand and skips files which are already present
- Runs metadata and extraction sequentially, with timeouts and cancellation of the active child process
- Invalidates old jobs when another hub is opened so late events cannot populate the wrong hub

Tauri currently locates `ffmpeg` and `ffprobe` on `PATH`. Set `VHA_FFMPEG_PATH` and `VHA_FFPROBE_PATH` to override either executable. Automatic use of Electron's bundled binaries and Tauri bundle packaging are not implemented yet.

A source whose `watch` flag is enabled receives an initial scan when its hub opens, and manual rescans are supported. **Continuous filesystem watching is not implemented yet.**

### Native integration and media

- Native input, output, reconnect, executable, and hub file pickers
- Opens videos with the operating system default application
- Opens external URLs and reveals files in the system file explorer
- Converts generated thumbnails, filmstrips, and clips to Tauri `asset:` URLs
- Starts with an empty asset scope and grants only user-selected paths and each opened hub's `vha-<hubName>` media directory; it does not enable a blanket filesystem scope
- Native close requests use Angular's save flow
- Initial and subsequent OS theme changes use the existing dark-mode event

## Still to migrate

- Automatic FFmpeg/FFprobe bundle discovery and platform-specific packaging
- Continuous folder watchers and native sleep prevention during long imports
- File deletion/trash, rename, playlist creation, and thumbnail replacement/cleanup
- Preferred-player command lines and timestamp arguments
- Native drag-and-drop paths and dragging videos into external editors
- Express/WebSocket remote-control server
- Recent documents, single-instance/file-open behavior, shutdown notifications, window-state persistence, and Touch Bar
- `.vha2` file associations and portable bundle variants
- A restrictive CSP after WebView behavior has been tested with the asset protocol
- Windows, macOS, and Linux feature-parity testing before removing Electron

## Verification status

The implemented source has **not yet passed native compilation or runtime testing**. The checks already performed are limited to the following:

- `npm run check:tsc` completed successfully
- `npm run build:renderer-prod` completed successfully, with the existing CommonJS optimization warnings
- Rust sources were parsed for syntax ERROR nodes, but this is not a substitute for `cargo check`
- JSON/TOML manifests, extension-list parity, whitespace, and Electron-compatible hash fixtures were checked

The following remain unverified:

- Rust compilation, unit tests, and Tauri command/event type checking with `cargo check` and `cargo test`
- Native startup, window close/save, native dialogs, OS theme events, and `asset:` media display
- Real FFprobe metadata and FFmpeg thumbnail/filmstrip/clip extraction
- Cancellation and process cleanup with real child processes
- Real-hub upgrades, rescanning, deletion detection, and save/backup behavior
- Windows, macOS, and Linux behavior, including bundled applications and portable mode

The current migration environment lacks Rust/Cargo, the Linux WebView dependencies (`webkit2gtk-4.1` and `rsvg2`), and FFmpeg/FFprobe executables. Consequently, the Rust tests included in the source have not been run, and no native media-import result is claimed.

## Manual test checklist

These are **acceptance tests to perform, not tests which have already passed**. Use disposable copies of hubs and source videos. Record the OS, FFmpeg/FFprobe versions, test data, and result for each run.

### Build and startup

- [ ] Run `cargo check --manifest-path src-tauri/Cargo.toml` and `cargo test --manifest-path src-tauri/Cargo.toml` without errors.
- [ ] Start `npm run tauri:dev`, open a copy of an existing hub, and confirm that its thumbnails and filmstrips display through scoped `asset:` URLs.
- [ ] Verify that FFmpeg and FFprobe are found through `PATH`, then repeat with `VHA_FFMPEG_PATH` and `VHA_FFPROBE_PATH` overrides.
- [ ] Set an invalid executable path and confirm that a visible error appears, the progress state returns to `done`, and the application remains usable.
- [ ] Confirm that the existing Electron development and production paths still start and build independently of Tauri.

### New hub creation and directory scanning

- [ ] Create a new hub and confirm that a version-3 `.vha2`, `vha-<hubName>/thumbnails`, `filmstrips`, and `clips` exist before media extraction finishes.
- [ ] Attempt to create another hub with the same output path and name; confirm that no files are overwritten and the name-correction event is shown.
- [ ] Import nested folders containing lowercase and uppercase video extensions, non-video files, and a custom additional extension.
- [ ] Place a video inside a directory beginning with `vha-` and confirm that it is not imported.
- [ ] Verify root and nested `partialPath` values, source indexes, and paths containing spaces and non-ASCII characters.
- [ ] Confirm event order: scan start, complete file Map, per-video metadata, screenshot progress, then `done`, without zero-division or stale progress indicators.
- [ ] Add a source, manually rescan it, reconnect a disconnected source, and confirm that subsequent scans use the new path.
- [ ] Confirm that `watch=true` performs the documented initial scan; do not treat continuous watching as implemented.

### Metadata and generated media

- [ ] Compare duration, dimensions, FPS, timestamps, file size, resolution labels, and screenshot counts with Electron for the same videos.
- [ ] Compare generated hashes with Electron for files below, at, and above the 128 KiB sampling threshold.
- [ ] Confirm names and locations: `thumbnails/<hash>.jpg`, `filmstrips/<hash>.jpg`, `clips/<hash>.mp4`, and `clips/<hash>.jpg`.
- [ ] Reuse a hub with Electron-generated assets and confirm that matching files are not regenerated.
- [ ] Test short videos, long videos, SD/HD/4K sources, variable frame rates, multiple video streams, and videos with missing stream duration metadata.
- [ ] Import with clips disabled and enabled; verify snippet count, snippet duration, clip scaling, and clip poster generation.
- [ ] Delete selected generated files, run missing-thumbnail generation, and confirm that only missing outputs are recreated.
- [ ] Test corrupt videos and videos without audio; confirm that one failure does not stall the queue or prevent later videos from being processed.

### Cancellation, failures, and data safety

- [ ] Cancel during FFprobe, thumbnail, filmstrip, and clip extraction; confirm that the active child exits and no later events advance the canceled job.
- [ ] Open another hub during import and confirm that old metadata, progress, and generated files cannot enter the new hub.
- [ ] Close the application during extraction and confirm that no FFmpeg/FFprobe process is left running.
- [ ] Make a source directory unreadable or disconnect it during scanning; confirm that other sources continue and existing videos are not marked deleted because of an incomplete scan.
- [ ] Remove a video, perform a complete successful rescan, and confirm that only the genuinely missing entry is marked deleted.
- [ ] Reject hub names containing separators or traversal segments and confirm that no output is written outside the selected directory.
- [ ] Simulate a close-time save failure and confirm that the application stays open, reports the error, and preserves the previous hub and `.bak` files.

### Persistence and platform coverage

- [ ] Open, upgrade, modify, save, and reopen version-2 and version-3 hub copies; compare their persisted fields with Electron.
- [ ] Save settings containing shortcut Maps and confirm that they remain plain JSON objects compatible with Electron.
- [ ] Verify portable settings placement, native close-to-save, save-before-open, and restart restoration.
- [ ] Repeat the core import and save tests on Windows, macOS, and Linux, including non-ASCII and long paths.
- [ ] Build platform bundles and verify the documented executable discovery, generated-media loading, file pickers, opening videos with the operating system default application, and application shutdown from each packaged app.

## Recommended next stage

Validate the scanner and media pipeline on Windows, macOS, and Linux with real videos, then port continuous directory watching. Keep the Electron runtime until all data-writing paths and platform-specific media behavior have regression coverage. Run the Rust unit tests and exercise cancellation, corrupt media, missing executables, and save failures with copies of real hubs before using Tauri as the primary runtime.
