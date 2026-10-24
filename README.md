# Video Hub App 3

[_Video Hub App 3_](https://videohubapp.com/) is the fastest way to browse, search, and organize videos on your computer. Think of it like YouTube for videos on your computer. Works on **Windows**, **Mac**, and **Linux**!

## Download Now

This software is available for $10.00 through [videohubapp.com](https://videohubapp.com/en/download)

Over $16,000 donated from the sale of this software to the [_cost-effective_](https://www.givewell.org/charities/top-charities) charity [Against Malaria Foundation](https://www.againstmalaria.com/). See [donation history](https://www.againstmalaria.com/Fundraiser.aspx?FundraiserID=8035)

![video-hub-app](https://github.com/user-attachments/assets/9f13fc21-63ab-46ae-8409-ba2d9b5f289a)

## About

_Video Hub App_ was created by [Boris Yakubchik](https://videohubapp.com/en/about). It uses the frameworks _Angular_ and _Electron_.

## License

This software was built on top of [`angular-electron`](https://github.com/maximegris/angular-electron) by [Belnadris](https://github.com/belnadris). It carries an _MIT_ license (see the _LICENSE_ file). While the license is permissive, I ask that you do not distribute free copies of this software unless you have significantly changed it.

## Contributing

I would love to see the improvements you make to this app and am happy to accept pull requests. See [issues](https://github.com/whyboris/Video-Hub-App/issues) for ideas for improving this app or to add your suggestions.

Please consider improving any of the translations, or [add a new translation](https://github.com/whyboris/Video-Hub-App/tree/master/i18n)!

Please consider improving or adding an icon to the app. It's a simple process, just [follow the instructions](https://github.com/whyboris/Video-Hub-App/tree/master/src/app/components/icon)!

## Upcoming features

See [issues](https://github.com/whyboris/Video-Hub-App/issues) for what is in progress.

## Development

⚠ The repository is usually ahead of the publicly [released version](https://github.com/whyboris/Video-Hub-App/releases) - it is 🚧 WIP and may have bugs in it.

How to start:

- `npm install --legacy-peer-deps` to install
- `npm start` to develop
- `npm run electron` to build

An experimental Tauri 2 runtime is also available while the backend is being migrated:

- `npm run tauri:dev` to develop with Tauri
- `npm run tauri:build` to build the Tauri target

The Tauri target can restore hubs and scan/import new ones when `ffmpeg` and `ffprobe` are available on `PATH` (or through `VHA_FFMPEG_PATH` and `VHA_FFPROBE_PATH`). Continuous folder watching, bundled-media discovery, and several integrations are not yet ported.

The Tauri target is not yet feature-complete; Electron remains the supported runtime. See [TAURI_MIGRATION.md](./TAURI_MIGRATION.md) for the current status, unverified items, manual test checklist, and migration plan.

### Building from source (per platform)

Common prerequisites for every platform:

- **Node.js** 24 (see `.nvmrc`) and npm
- **Git**
- For running the app (both runtimes): `ffmpeg` and `ffprobe` available on `PATH`, or pointed to with the `VHA_FFMPEG_PATH` / `VHA_FFPROBE_PATH` environment variables — needed for thumbnails and media import
- For the Tauri runtime only: **Rust** stable, version 1.90 or newer, installed via [rustup](https://rustup.rs/)

Install the JavaScript dependencies first (all platforms):

```bash
npm install --legacy-peer-deps
```

#### Electron (supported runtime)

```bash
npm start          # develop (Angular + Electron)
npm run electron   # build and package with electron-builder
```

Installers are written under `dist/`.

#### Tauri — Windows

Additional prerequisites:

- Visual Studio 2022 Build Tools with the **Desktop development with C++** workload (MSVC + Windows SDK)
- The **WebView2 Runtime** (preinstalled on Windows 11 and recent Windows 10)

```powershell
npm run tauri:dev                                   # develop
npm run tauri:build -- --bundles nsis,msi           # release build
```

Output: `src-tauri/target/release/bundle/nsis/` (`.exe` installer) and `src-tauri/target/release/bundle/msi/` (`.msi`).

#### Tauri — macOS

Additional prerequisites:

- Xcode Command Line Tools: `xcode-select --install`

```bash
npm run tauri:dev    # develop (builds for your machine's architecture)

# Universal binary (Apple Silicon + Intel):
rustup target add aarch64-apple-darwin x86_64-apple-darwin
npm run tauri:build -- --target universal-apple-darwin
```

Output (universal build): `src-tauri/target/universal-apple-darwin/release/bundle/dmg/` (`.dmg`) and `.../bundle/macos/` (`.app`). Builds are unsigned, so on first launch Gatekeeper may block the app or report it as damaged; right-click the app and choose **Open**, or run `xattr -cr "Video Hub App 3.app"` to clear the quarantine attribute.

#### Tauri — Linux (Ubuntu / Debian)

Additional system packages (see also the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)):

```bash
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev \
  libxdo-dev libssl-dev patchelf build-essential
# add `rpm libfuse2` as well if you want the .rpm and .AppImage bundles
```

```bash
npm run tauri:dev                                     # develop
npm run tauri:build -- --bundles deb,rpm,appimage    # release build
```

Output: `src-tauri/target/release/bundle/` under `deb/`, `rpm/`, and `appimage/`.

#### Checks

```bash
npm run check:tsc              # TypeScript type check
cargo test --manifest-path src-tauri/Cargo.toml   # Rust unit tests
```

#### Prebuilt test releases (GitHub Actions)

The workflow [`.github/workflows/tauri-release.yml`](./.github/workflows/tauri-release.yml) builds all three platforms (Windows NSIS/MSI, macOS universal DMG, Linux deb/rpm/AppImage) and publishes them as a prerelease. Push a tag matching `tauri-v*` — for example `tauri-v3.3.0-test.2` — or run the workflow manually from the **Actions** tab and provide the tag.

Main dependencies in use:

| Library          | Version | Date      | Comment |
| ---------------- | ------- | --------- | ------- |
| Angular          | v20     | Oct 2025  |         |
| Angular-CLI      | v20     | Oct 2025  |         |
| Electron         | v42     | Jun 2026  | (internally uses Node `v24.16.0` and Chromium 148) |
| Electron Builder | v26     | Oct 2025  |         |

- **Node**: It may be best to use the same version as _Electron_ uses internally (`v24.16.0`)

- **Angular CLI**: not required but may be useful: [Angular CLI](https://cli.angular.io).

- **Mac**: We now have `M1` and `Intel` builds, but you may need to manually update the _FFmpeg_ and _FFprobe_ executables if you're building for a different architecture than you are on.

To help debug a production build of VHA you can use [Debugtron](https://github.com/bytedance/debugtron)

## Remote

👩‍🚀 a new feature in VHA is the option for the user to turn on a _server_ after the app starts. This will let the user open a simpler version of the VHA user interface on their phone or tablet (if both PC and device are on the same WiFi) and use it as remote control to play videos 🚀

For details, see [instructions](https://github.com/whyboris/Video-Hub-App/blob/master/remote/README.md).

## Thank you

This software would not be possible without the tremendous work by other people:

- [Angular](https://github.com/angular/angular)
- [Electron](https://github.com/electron/electron)
- [FFmpeg](https://www.ffmpeg.org/)
- [angular-electron](https://github.com/maximegris/angular-electron)
- [ngx-virtual-scroller](https://github.com/rintoj/ngx-virtual-scroller)
- [fdir](https://github.com/thecodrr/fdir)
- [async](https://github.com/caolan/async)
- [chokidar](https://github.com/paulmillr/chokidar)
- [trash](https://github.com/sindresorhus/trash)
- [Fuse.js](https://github.com/krisk/Fuse)
- [electron-window-state](https://github.com/mawie81/electron-window-state)

This software uses libraries from the FFmpeg project under the LGPLv2.1 with binaries from [here](https://github.com/kribblo/node-ffmpeg-installer#the-binaries)

Since becoming open source, this software was made better with the awesome contributions by [cal2195](https://github.com/cal2195)
