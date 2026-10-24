# Tauri 顔認識バックエンド移植 設計書

対象ブランチ: `integration/tauri-face-recognition`
前提: 顔認識の仕様は `docs/FACE_RECOGNITION_DESIGN.md`（以下「顔認識設計書」）に従う。
本書は Electron main（`node/face/*` + `node/main-ipc.ts`）にあるバックエンドを
Tauri/Rust（`src-tauri/`）へ移植するための差分設計である。レンダラ（Angular）は
原則無改造とする（統合時に済んだ `file-path.service.ts` の Tauri 対応を除く）。

## 1. 現状の実測（2026-10-24）

### Electron 側の IPC 契約（レンダラ → main）

| チャンネル | 引数 | 応答（main → レンダラ） | 実装 |
|---|---|---|---|
| `face-get-model-status` | なし | `face-model-status-returning` (FaceModelStatus) | main-face.ts |
| `face-get-index` | なし | `face-index-returning` (FaceIndexPayload) | main-face.ts |
| `face-settings-updated` | FaceSettings | `face-model-status-returning` | main-face.ts |
| `face-scan-request` | ImageElement[] | 完了後に `face-index-updated` | main-ipc.ts → main-face.ts |
| `face-recluster` | ImageElement[]? | `face-index-returning` | main-face.ts |
| `face-attributes-request` | ImageElement[]? | `face-index-returning` | main-face.ts |
| `face-rename-person` | personId, name | `face-index-returning`（変更時のみ） | main-face.ts |
| `face-set-person-hidden` | personId, hidden | 同上 | main-face.ts |
| `face-merge-persons` | personIds[] | `face-index-returning` | main-face.ts |
| `face-split-person` | personId, groupKeys[] | `face-index-returning` | main-face.ts |
| `face-move-groups` | groupKeys[], targetPersonId\|null | `face-index-returning` | main-face.ts |
| `face-set-person-representative` | personId, faceId | `face-index-returning` | main-face.ts |

### 永続データ（形式は言語中立）

- `<output>/vha-<hubName>/faces/index.json` — バージョン1の JSON（動画別 faces、
  persons、nextPersonSeq 等）。書き込みは tmp + rename のアトミック方式。
- `<output>/vha-<hubName>/faces/embeddings.bin` — Float32LE × 512次元の
  append-only バイナリ。index 側の `embeddingIdx` がレコード番号。
- `faces/crops/`（表示用切り抜き）、`faces/aligned/`（属性推定用）。
- モデル配置: `<userData>/face-models/<pack>/` に det/rec の ONNX 2ファイル
  (+ 任意で `genderage.onnx`)。pack は quality / light / lvface の3種
  （`FACE_MODEL_PACKS`、modelId は index に記録され不一致は mismatch 扱い）。
- `faceSettings` の保存先は .vha2 の `FinalObject.faceSettings`（ハブ単位）。
  Electron main が hub を開くときに `GLOBALS.faceSettings` へ読み込む。

### 推論まわり

- 推論は Electron utilityProcess（`face-worker.ts`）で onnxruntime-node を実行。
  パイプライン: ffmpeg でフレーム抽出 → SCRFD 検出 → 5点アライメント →
  ArcFace / LVFace 認識 → 動画内グルーピング → crops 保存 → embeddings 追記 →
  全体クラスタリング（cluster.ts、純粋ロジック）→ 属性集計（genderage、任意）。
- プロバイダ梯子（D9）: auto / cpu / cuda / directml / coreml、失敗理由つき。

### Tauri 側の受け皿

- `src-tauri/src/ipc.rs` の `ipc_send` がチャンネル名で分岐し、未移植は
  明示エラー（`unsupported_channel`）。応答は `emit_to_angular` で
  Electron の `event.sender.send` と同じ引数配列形式を emit する。
- レンダラ側は `src/desktop/tauri-bridge.ts` が `window.myElectron` シムを提供し、
  `sendToMain` / `receiveFromMain` の契約をそのまま使える。
- ハブを開くと `VhaState.current_hub` に `asset_folder`
  （`<output>/vha-<hubName>`）が入る。faces ディレクトリは
  `asset_folder/faces` で Electron の `hubFacesDir()` と一致する。
- `LoadedHub.final_object` から `faceSettings` を取り出せる（hub.rs）。

## 2. 決定事項

- **T1: Rust ネイティブ移植とする。** Node サイドカー方式は採らない。
  Tauri 移行の目的自体が Electron/Node ランタイムの排除であり、サイドカーは
  node + onnxruntime-node（約200MB級）を同梱し続け二重保守になる。
  永続形式が JSON + Float32 バイナリで言語中立なため、Rust 移植でも
  既存ハブの faces データをそのまま読み書きできる。
- **T2: IPC 契約は無変更。** チャンネル名・ペイロード形は Electron と同一にし、
  レンダラを変更しない。未移植チャンネルは従来どおり明示エラーのまま残す。
- **T3: モデル配置場所は Electron と同一に解決する。**
  face-models ルート = OS の config ディレクトリ直下の `Video Hub App 3`
  （Win: `%APPDATA%`、mac: `~/Library/Application Support`、Linux: `~/.config`）
  配下の `face-models`。Electron の userData と一致し、既存の配置が流用できる。
  ※ Tauri 側のアプリ設定ディレクトリ（`video-hub-app-2`）とは別物である点に注意。
- **T4: faceSettings の保持は Electron の GLOBALS を VhaState に写す。**
  hub を開くときに FinalObject から読み、既定値（main-globals.ts と同一）で
  補完する。`face-settings-updated` は部分マージして保持する。
- **T5: 永続形式は変更しない。** index.json v1、embeddings.bin、
  crops/aligned の配置・命名をそのまま踏襲する（Electron で作ったハブを
  Tauri で開いても faces が読めることが受け入れ条件）。
- **T6: 推論エンジンは Rust の `ort` クレート（ONNX Runtime）を Phase C で導入する。**
  数値互換は `tools/face-spike/golden/*.json` のゴールデンデータで検証し、
  同一 aligned 顔に対する埋め込みの cosine が node 実装と実用上一致
  （目安 0.999 以上）することを Phase C の開始条件とする。一致しない場合は
  既存 embeddings との混在が壊れるため、Phase C の進め方を再検討する。

## 3. フェーズ計画

### Phase A: 読み取り系 IPC（本書と同時に実装）

- `face.rs` 新設: パック表、モデル状態（配置チェック + genderage の
  quality からの借用解決）、index ペイロード生成（load / stats /
  personsByVideo / mismatch 判定）。いずれも face-types.ts / face-store.ts /
  main-face.ts の純粋部分の移植。
- `VhaState` に faceSettings を追加し、hub オープン時に FinalObject から初期化。
- チャンネル: `face-get-model-status`、`face-get-index`、`face-settings-updated`。
- providerInfo は「未解決」として返す（requested のみ設定、effective=null、
  attempts=[]、cudaLibsPresent=false、ortBinDir=""）。実解決は Phase C。
- 検証: Rust 単体テスト（モデル配置・genderage 借用・stats・personsByVideo・
  mismatch・空ディレクトリ）。tsc / cargo test の既存ライン維持。

### Phase B: 純粋ロジック（クラスタリング + ストア編集）（実装済み）

- cluster.ts（全体クラスタリング、ピン保持、index 編集ヘルパ群）と
  face-store.ts の編集系を移植: `face-recluster`、`face-rename-person`、
  `face-set-person-hidden`、`face-merge-persons`、`face-split-person`、
  `face-move-groups`、`face-set-person-representative`。
  実装は `face_cluster.rs`（純関数群）+ `face.rs` の FaceStore。
  数値の再現のため、ベクトル平均は Float32Array の格納丸めまで模倣し、
  類似度計算は f32 値に対する f64 演算とした。
- embeddings.bin の読み出しのみで推論は不要。検証は TS 実装との
  クロスチェック（同一 index + embeddings に対する結果一致）で行う。
  → 実施済み: 合成ハブ（3動画7顔）で再クラスタリング + 名前/非表示/
  マージ/分割/移動/代表顔の一連の操作を両実装で実行し、保存後の
  index.json が値として完全一致した。同条件の単体テストも同梱。
- 属性の再計算フック（編集後の computeFaceAttributes）は Phase C まで
  無効（属性が有効なハブでは次回スキャン/属性リクエスト時に再計算される、
  という Electron 側の元々のフォールバックと同じ挙動）。

### Phase C: 推論 + スキャン（実装済み）

- `ort` クレートで SCRFD / ArcFace / LVFace / genderage のセッション管理と
  プロバイダ梯子（D9 の意味論を移植: 候補順、失敗理由の記録、CPU フォールバック）。
  実装は `face_infer.rs`。ort は **2.0.0-rc.15（ONNX Runtime 1.31.0 世代）+
  load-dynamic** とし、ライブラリは ORT_DYLIB_PATH → 実行ファイル隣 →
  Tauri リソース dir → システムローダの順に解決する。GPU プロバイダの
  登録コードは Phase D でプラットフォーム別にコンパイルされ、
  プロバイダライブラリが無い環境では失敗が記録されて CPU に落ちる。
- 画像処理: アライメント（5点類似変換 + ワープ）、リサイズ、JPEG 保存
  （crops / aligned）を `image` クレート等で実装。image-ops.ts と出力が
  近似一致することを確認する（切り抜きは表示用、aligned は属性用）。
  → リサイズ/ワープは TS と同じ OpenCV 意味論の自前実装に置換済み。
  JPEG のデコード/エンコードのみ `image` クレート。
- スキャン: フレーム抽出（generateFaceFrameArgs と同一引数、ffmpeg は
  VHA_FFMPEG_PATH → `ffmpeg`）→ 検出 → 認識 → ストア追記 →
  再クラスタリング。`face-scan-request`（未スキャン分を背景スキャン）、
  `face-attributes-request`、`face-index-updated` 通知を有効化。
  自動スキャン（autoScanNew）は media のジョブ完了フックから同じパスを
  回す（Electron のスクリーンショットキュー→顔キューの等価物）。
- 数値互換ゲート（T6）を最初に通すこと。
  → **合格（2026-10-24 実測）**: golden 全7顔で埋め込みコサイン
  **0.99964〜0.99986**、bbox 差 ≤0.19px、kps 差 ≤0.05px、性別全一致、
  年齢差 ≤2歳。同条件のゲートテスト（環境変数ゲート付き）を
  `face_infer.rs` に同梱。あわせて E2E スキャンテスト（静止画ループ
  動画→60顔/6グループ/6人物、aligned 60・crops 6・embeddings.bin
  60レコード）も合格。

### Phase D: パッケージング + 実機確認（実装済み・VM 検証済み、実機確認は残）

- ORT バンドル: 公式リリース 1.31.0 の共有ライブラリを Tauri リソース
  として同梱する。`src-tauri/fetch-ort.sh` がプラットフォーム別に
  取得し（リリース CI でも実行）、`tauri.<platform>.conf.json` の
  resources マップでリソースルート直下に配置される。実行時の解決順は
  ORT_DYLIB_PATH → 実行ファイル隣 (+lib/) → Tauri リソース dir
  （lib.rs の setup が `face_infer::set_resource_dir_hint` で登録、
  +lib/・+resources/ も探索）→ システムローダ。
- GPU EP: Cargo.toml の target 別依存で ort 機能を有効化
  （Windows: cuda + directml、macOS: coreml、Linux: cuda）し、
  face_infer の梯子が同じ cfg で登録コードをコンパイルする。
  プロバイダ用ライブラリが無い環境では試行が失敗として記録され
  CPU に落ちる（Electron 版と同じ意味論）。同梱するのは CPU パッケージ
  のライブラリで、CUDA / DirectML のプロバイダライブラリは Electron 版
  と同様に別途配置（ORT の DirectML/CUDA パッケージは GitHub リリース
  には無く NuGet / CUDA 配布側にあるため同梱しない）。
- macOS の制約: ORT 1.31.0 の macOS パッケージは arm64 のみのため、
  Intel Mac では ORT がロードできず顔認識だけ無効になる
  （アプリ本体はユニバーサルで動作する）。
- VM 検証: Linux で CUDA 機能入りビルドの cargo test 17/17、
  golden ゲートに auto 梯子の確認を追加（CUDA を最初に試行・記録し
  CPU 解決）。deb ビルドで `usr/lib/<app>/` 直下への
  libonnxruntime.so（29.3MB）の配置と、抽出ツリーからの xvfb 起動を
  確認。実機確認（Win/Mac のインストール・実 GPU・Electron 製ハブとの
  相互運用）はユーザー環境で実施。

## 4. 未決事項

- Q1: ~~`ort` クレートのバージョンと ONNX Runtime のバージョン対応
  （onnxruntime-node 1.30 世代との数値差）。Phase C 冒頭で実測する。~~
  → 解決: ort 2.0.0-rc.15 / ORT 1.31.0 でゲート合格（Phase C 参照）。
- Q2: ポータブル版（PORTABLE_EXECUTABLE_DIR）での face-models ルート。
  Tauri 側にポータブル配布形態が無く、当面は OS 標準の config_dir で
  Electron 版（非ポータブル）と一致する。ポータブル配布を作る場合は
  モデルルートの上書き手段が別途必要（未対応として記録）。
- Q3: ~~hub 保存時に Rust が faceSettings を FinalObject へ書き戻す
  必要があるか~~ → 不要と確定（実測）: レンダラの保存フロー
  （home.component.ts）が `faceSettings: this.currentFaceSettings` を
  FinalObject に含めて送るため、.vha2 には常に入る。Rust 側の
  VhaState ミラーは開封時に FinalObject から初期化される。
