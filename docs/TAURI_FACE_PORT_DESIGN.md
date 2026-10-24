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

### Phase B: 純粋ロジック（クラスタリング + ストア編集）

- cluster.ts（全体クラスタリング、ピン保持、index 編集ヘルパ群）と
  face-store.ts の編集系を移植: `face-recluster`、`face-rename-person`、
  `face-set-person-hidden`、`face-merge-persons`、`face-split-person`、
  `face-move-groups`、`face-set-person-representative`。
- embeddings.bin の読み出しのみで推論は不要。検証は TS 実装との
  クロスチェック（同一 index + embeddings に対する結果一致）で行う。
- 属性の再計算フック（編集後の computeFaceAttributes）は Phase C まで
  無効（属性が有効なハブでは次回スキャン/属性リクエスト時に再計算される、
  という Electron 側の元々のフォールバックと同じ挙動）。

### Phase C: 推論 + スキャン

- `ort` クレートで SCRFD / ArcFace / LVFace / genderage のセッション管理と
  プロバイダ梯子（D9 の意味論を移植: 候補順、失敗理由の記録、CPU フォールバック）。
- 画像処理: アライメント（5点類似変換 + ワープ）、リサイズ、JPEG 保存
  （crops / aligned）を `image` クレート等で実装。image-ops.ts と出力が
  近似一致することを確認する（切り抜きは表示用、aligned は属性用）。
- スキャン: media.rs の ffmpeg 実行パターンを再利用してフレーム抽出 →
  検出 → 認識 → ストア追記 → 再クラスタリング。進捗・キャンセルは
  既存の background generation の仕組みに乗せる。`face-scan-request`、
  `face-attributes-request`、`face-index-updated` 通知を有効化。
- 数値互換ゲート（T6）を最初に通すこと。

### Phase D: パッケージング + 実機確認

- ONNX Runtime バイナリの同梱方法（ort の download-binaries / load-dynamic、
  プラットフォーム別 EP: DirectML / CoreML / CUDA の扱い）と
  Tauri バンドル設定。CUDA は node 版と同様「ユーザー配置」方式の可否を検討。
- 実機確認: モデル配置済み環境でのスキャン、人物表示・編集、Electron で
  作ったハブとの相互運用（同じ faces を両ランタイムで開く）。

## 4. 未決事項

- Q1: `ort` クレートのバージョンと ONNX Runtime のバージョン対応
  （onnxruntime-node 1.30 世代との数値差）。Phase C 冒頭で実測する。
- Q2: ポータブル版（PORTABLE_EXECUTABLE_DIR）での face-models ルート。
  Electron ポータブル時の userData 解決に合わせる必要があれば Phase D で対応。
- Q3: hub 保存時に Rust が faceSettings を FinalObject へ書き戻す必要があるか。
  Electron は main の GLOBALS から書き戻す。Tauri ではレンダラが送る
  FinalObject に faceSettings が含まれていれば不要 — Phase A/B の間に
  保存フローを実測して確定する。
