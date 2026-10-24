# Video Hub App 3 ＋ 顔認識（フォーク）

> このリポジトリは [Video Hub App 3](https://github.com/whyboris/Video-Hub-App)（whyboris 作）のフォークに、**顔認識による人物ごとの自動分類**を追加したものです。個人利用のためのフォークで、本家とは関係ありません。
>
> _Fork of Video Hub App 3 with local, on-device face recognition: videos are automatically grouped by the people appearing in them._

## このフォークで追加した機能（ブランチ `feature/face-recognition`）

- **顔検出と顔サムネイル**: 動画から顔のある場面を自動で抽出し、顔サムネイルを生成します。検出は SCRFD、認識はモデルパックで ArcFace（InsightFace）または LVFace-T（ByteDance）を選択でき、推論はすべてローカルで動きます（CPU のほか、後述の GPU 実行プロバイダも選択可）
- **人物ごとの自動分類**: 顔の特徴量をもとに人物ごとに自動でグループ化。サイドバーの人物欄や人物フィルタで「あの人が出ている動画」だけを表示できます
- **名前付けと手直し**: 人物への名前付け、間違った分類の結合・分割・付け替え、代表顔の変更に対応。手直しの内容は再分類しても保持されます
- **年齢・性別の推定（任意機能、既定はオフ）**: 人物単位の集約値（年齢は中央値、性別は多数決）だけを保存・表示します。年齢は ±10 歳級の目安です

顔データは各ハブのフォルダ内の `faces/` に保存されます。元の動画ファイルには触れず、本家のデータ形式（`.vha2`）への追加は人物 ID のキャッシュのみに留めています。

## 使い方（顔認識）

1. 設定の「顔認識」セクションで顔認識を有効化する
2. 使うモデルパックのファイルを配置先フォルダに置く（ファイル・配置先・ダウンロード先の一覧は下の「必要なファイルの配置場所とダウンロード先」を参照）
   - モデルはアプリに同梱されません。buffalo / LVFace いずれのモデルもライセンスは**非商用の研究目的のみ**ですのでご注意ください
3. ハブを開くと、通常のサムネイル抽出に続いて顔スキャンが走り、人物欄に結果が表示されます

### 必要なファイルの配置場所とダウンロード先

モデルファイルはアプリに同梱されないため、各自でダウンロードして指定フォルダに配置してください。配置先フォルダは、設定の「顔認識」セクションにも表示されます（選択中のパックのフォルダがそのまま出ます）。

配置先の基準フォルダ（`<userData>/face-models/`）は OS ごとに次の場所です。その下にパック名のフォルダ（`quality` / `light` / `lvface`）を作ってファイルを置きます。

- Windows: `%APPDATA%\Video Hub App 3\face-models\`（例: `C:\Users\<ユーザー名>\AppData\Roaming\Video Hub App 3\face-models\`）
- macOS: `~/Library/Application Support/Video Hub App 3/face-models/`
- Linux: `~/.config/Video Hub App 3/face-models/`

| パック | ファイル | 配置先フォルダ | ダウンロード元 |
|---|---|---|---|
| quality（既定） | `det_10g.onnx`（検出） | `face-models/quality/` | buffalo_l.zip（下記） |
| quality | `w600k_r50.onnx`（認識） | 同上 | 同上 |
| quality | `genderage.onnx`（任意、年齢・性別推定用） | 同上 | 同上 |
| light | `det_500m.onnx`（検出） | `face-models/light/` | buffalo_sc.zip（下記） |
| light | `w600k_mbf.onnx`（認識） | 同上 | 同上 |
| lvface（高速） | `det_10g.onnx`（検出、quality と同じファイル） | `face-models/lvface/` | buffalo_l.zip（quality フォルダからコピーでも可） |
| lvface | `LVFace-T_Glint360K.onnx`（認識） | 同上 | HuggingFace（下記） |

ダウンロード元（いずれも配布元の公式リンク）:

- **buffalo_l.zip**（約289MB、InsightFace 公式 GitHub リリース）: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  zip の中にファイルがそのまま入っています。quality の3ファイルはこちらから取り出してください（残りの2ファイルは使いません）
- **buffalo_sc.zip**（約15MB、同上）: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
  中身は `det_500m.onnx` と `w600k_mbf.onnx` の2ファイルだけです
- **LVFace-T**（約77MB、単体ファイルの直接ダウンロード）: https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-T_Glint360K/LVFace-T_Glint360K.onnx
  リポジトリのトップは https://huggingface.co/bytedance-research/LVFace です。ダウンロードしたファイル名（`LVFace-T_Glint360K.onnx`）のまま置いてください

補足:

- `genderage.onnx` は buffalo_l にのみ含まれ、buffalo_sc には入っていません。light / lvface パックで年齢・性別推定を使う場合は、quality フォルダに置いた `genderage.onnx` が自動で共用されます（使うパックのフォルダにコピーしても構いません）
- lvface パックは認識が ArcFace の約 3 分の 1 の時間で済みます（実測比較は `tools/face-spike/LVFACE_COMPARISON.md`）
- パックを切り替えると埋め込みを作り直すため、再スキャンが必要です

### GPU で速くする（任意）

設定の「実行プロバイダ」で推論に使う計算デバイスを選べます。既定の **Auto** は、使える GPU があれば自動で使い、無ければ CPU で動きます（Windows: CUDA → DirectML → CPU、Linux: CUDA → CPU、macOS: CoreML → CPU）。実際に使われているプロバイダは設定画面に表示されます。手動で CUDA / DirectML / CoreML / CPU を固定することもできます（失敗した場合は自動で次の候補に落ちます）。

- **DirectML（Windows）**: 追加の準備は不要です。DirectX 12 対応 GPU（NVIDIA / AMD / Intel）で動きます
- **CUDA（NVIDIA GPU）**: 以下がすべて必要です。CUDA ランタイムはアプリに同梱されないため、各自でインストールしてください
  1. 最新の NVIDIA ドライバ
  2. **CUDA Toolkit 13.x**（cudart / cublas などのランタイムライブラリ）: https://developer.nvidia.com/cuda-downloads （13.x が最新でない場合は https://developer.nvidia.com/cuda-toolkit-archive から 13.x を選んでください）
  3. **cuDNN 9.x**（CUDA 13 用。ライブラリを PATH の通った場所へ）: https://developer.nvidia.com/cudnn-downloads （ダウンロードには NVIDIA アカウントが必要です）
  4. ONNX Runtime の CUDA プロバイダライブラリ 2 ファイルを、ONNX Runtime 本体と同じフォルダへ配置する（**バージョンは必ず 1.30.0 に合わせてください**）
     - ファイル: Windows は `onnxruntime_providers_cuda.dll` と `onnxruntime_providers_shared.dll`、Linux は `libonnxruntime_providers_cuda.so` と `libonnxruntime_providers_shared.so`
     - 配置先（設定画面にも表示されます）: Windows は `<アプリのインストールフォルダ>\resources\app.asar.unpacked\node_modules\onnxruntime-node\bin\napi-v6\win32\x64\`（`DirectML.dll` があるフォルダ）、Linux は `<アプリのインストールフォルダ>/resources/app.asar.unpacked/node_modules/onnxruntime-node/bin/napi-v6/linux/x64/`
     - 入手元（公式 ONNX Runtime 1.30.0 の GPU パッケージ。どちらからでも可）:
       - NuGet パッケージ（.nupkg は zip 形式です）: Windows 用 https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/1.30.0 （約153MB。2ファイルは中の `runtimes/win-x64/native/` にあります）/ Linux 用 https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Linux/1.30.0 （約236MB。中の `runtimes/linux-x64/native/` にあります）
       - GitHub リリース: https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-win-x64-gpu_cuda13-1.30.0.zip （約294MB。2ファイルは展開したフォルダの `lib/` にあります）。Linux 版は同じリリースの `onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz` です
  - 配置先フォルダはアプリのインストールフォルダ内です。書き込めない場合は管理者権限でコピーしてください。ポータブル版ならそのままコピーできます。AppImage 版は読み取り専用のイメージのため、この配置はできません
  - CUDA が使えない環境では、理由が設定画面に表示されたうえで DirectML / CPU に自動で切り替わります

設計と検証の記録: [設計書](docs/FACE_RECOGNITION_DESIGN.md) / 検証結果は `tools/face-spike/` の `RESULTS.md`、`CLUSTER_RESULTS.md`、`PHASE1_RESULTS.md`〜`PHASE3_RESULTS.md`、`LVFACE_COMPARISON.md`、`LVFACE_PACK_RESULTS.md`、`GPU_PROVIDER_RESULTS.md`

## ライセンス上の注意点とクレジット

このフォークのコード自体は本家と同じ MIT ライセンスですが、顔認識のために使う**学習済みモデルには別の条件**が付きます。

- **顔認識モデル（InsightFace / buffalo パック）**: 顔検出（SCRFD）、顔認識（ArcFace）、年齢・性別推定（genderage）のモデルは [InsightFace](https://github.com/deepinsight/insightface)（DeepInsight）が公開している buffalo_l / buffalo_sc パックのものです。InsightFace のコードは MIT ライセンスですが、**公開されている学習済みモデルの重みは「非商用の研究目的のみ」**とされており、商用利用はできません。このフォークは個人による非商用利用を前提としています。モデルファイルはこのリポジトリにもアプリの配布物にも含まれておらず、利用する各自が InsightFace の配布元から取得してください
  - 論文: ArcFace — Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019 / SCRFD — Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", 2021
- **顔認識モデル（LVFace、lvface パック使用時）**: 認識モデル LVFace-T は [LVFace](https://github.com/bytedance/LVFace)（ByteDance）の公開重みで、HuggingFace（[bytedance-research/LVFace](https://huggingface.co/bytedance-research/LVFace)）から各自で取得してください。LVFace のコードは MIT ライセンスですが、**学習済みモデルの重みは非商用の研究目的のみ**とされており、buffalo パックと同条件です。モデルファイルはこのリポジトリにもアプリの配布物にも含まれていません。検出には quality と同じ `det_10g.onnx`（SCRFD）を使います
  - 論文: LVFace — You et al., ICCV 2025（詳細は公式リポジトリ参照）
- **追加した依存ライブラリ**: [onnxruntime-node](https://github.com/microsoft/onnxruntime)（MIT License、Microsoft）/ [sharp](https://github.com/lovell/sharp)（Apache License 2.0。画像処理に使う libvips は LGPL 3.0 以降のライセンスです）
- **本家**: Video Hub App 3 は Boris Yakubchik（[whyboris](https://github.com/whyboris)）作で、MIT ライセンスです（詳細は下記の本家 README の License 節と `LICENSE` ファイルを参照）。このフォークは本家とは無関係の非公式なものです
- **データの取り扱いについて**: 顔の特徴量（埋め込みベクトル）は生体情報にあたるデータです。処理はすべて自分の PC 内で完結し、外部に送信されることはありませんが、各ハブの `faces/` フォルダ（特徴量と顔サムネイルを保存）には個人を識別しうる情報が含まれるため、取り扱いと共有には十分注意してください

---

以下は本家 Video Hub App 3 の README です。

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
