# Video Hub App 3 ＋ 顔認識（フォーク）

> このリポジトリは [Video Hub App 3](https://github.com/whyboris/Video-Hub-App)（whyboris 作）のフォークに、**顔認識による人物ごとの自動分類**を追加したものです。個人利用のためのフォークで、本家とは関係ありません。
>
> _Fork of Video Hub App 3 with local, on-device face recognition: videos are automatically grouped by the people appearing in them._

## このフォークで追加した機能（ブランチ `feature/face-recognition`）

- **顔検出と顔サムネイル**: 動画から顔のある場面を自動で抽出し、顔サムネイルを生成します。検出は SCRFD、認識は ArcFace（InsightFace のモデル）で、推論はすべてローカル・CPU で動きます
- **人物ごとの自動分類**: 顔の特徴量をもとに人物ごとに自動でグループ化。サイドバーの人物欄や人物フィルタで「あの人が出ている動画」だけを表示できます
- **名前付けと手直し**: 人物への名前付け、間違った分類の結合・分割・付け替え、代表顔の変更に対応。手直しの内容は再分類しても保持されます
- **年齢・性別の推定（任意機能、既定はオフ）**: 人物単位の集約値（年齢は中央値、性別は多数決）だけを保存・表示します。年齢は ±10 歳級の目安です

顔データは各ハブのフォルダ内の `faces/` に保存されます。元の動画ファイルには触れず、本家のデータ形式（`.vha2`）への追加は人物 ID のキャッシュのみに留めています。

## 使い方（顔認識）

1. 設定の「顔認識」セクションで顔認識を有効化する
2. モデルファイルを、設定画面に表示される配置先（`<userData>/face-models/quality/`）に置く
   - quality（既定）: `det_10g.onnx` と `w600k_r50.onnx`（InsightFace の buffalo_l パックから取得、計 約190MB）
   - light: `det_500m.onnx` と `w600k_mbf.onnx`（buffalo_sc パック、計 約16MB）
   - モデルはアプリに同梱されません。buffalo モデルのライセンスは**非商用の研究目的のみ**ですのでご注意ください
3. ハブを開くと、通常のサムネイル抽出に続いて顔スキャンが走り、人物欄に結果が表示されます

設計と検証の記録: [設計書](docs/FACE_RECOGNITION_DESIGN.md) / 検証結果は `tools/face-spike/` の `RESULTS.md`、`CLUSTER_RESULTS.md`、`PHASE1_RESULTS.md`〜`PHASE3_RESULTS.md`

## ライセンス上の注意点とクレジット

このフォークのコード自体は本家と同じ MIT ライセンスですが、顔認識のために使う**学習済みモデルには別の条件**が付きます。

- **顔認識モデル（InsightFace / buffalo パック）**: 顔検出（SCRFD）、顔認識（ArcFace）、年齢・性別推定（genderage）のモデルは [InsightFace](https://github.com/deepinsight/insightface)（DeepInsight）が公開している buffalo_l / buffalo_sc パックのものです。InsightFace のコードは MIT ライセンスですが、**公開されている学習済みモデルの重みは「非商用の研究目的のみ」**とされており、商用利用はできません。このフォークは個人による非商用利用を前提としています。モデルファイルはこのリポジトリにもアプリの配布物にも含まれておらず、利用する各自が InsightFace の配布元から取得してください
  - 論文: ArcFace — Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019 / SCRFD — Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", 2021
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
