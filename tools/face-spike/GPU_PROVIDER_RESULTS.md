# GPU 実行プロバイダ（CUDA / DirectML）調査・実装・検証結果

対象: 決定 D9（設計書 §14）。実施日: 2026-10-24。環境: この VM（Linux x64、GPU なし）。
結論を先に: **CUDA 経路は存在するが、プロバイダライブラリは同梱せずユーザー配置とする**
（実測サイズが win 178.3MiB / linux 259.5MiB と大きく、CUDA ランタイムはユーザー側
インストールが方針のため）。DirectML は前提条件なしでそのまま使える構成だった。

---

## 1. ステップ1: CUDA バイナリの入手経路（実測）

### 1.1 onnxruntime-node 1.30.0 のインストール機構

- postinstall スクリプト（`script/install.js` + `install-metadata.js`）は、npm パッケージに
  同梱されていないファイルを NuGet フィードから取得する仕組みを持つ。CUDA EP バイナリは
  「大きすぎて npm レジストリに入らない」ためこの経路で配る設計（スクリプト冒頭コメント）。
- ただし 1.30.0 のメタデータで CUDA マニフェストが定義されているのは **linux/x64 のみ**:
  - 取得元: NuGet `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0
  - 配置ファイル: `libonnxruntime_providers_cuda.so` / `libonnxruntime_providers_shared.so` /
    `libonnxruntime_providers_tensorrt.so`（→ `bin/napi-v6/linux/x64/`）
  - linux/x64 では既定で必須（requirements に `cuda12`）。このリポジトリの開発環境では
    過去の NuGet 通信失敗の回避策 `ONNXRUNTIME_NODE_INSTALL=skip` を使ったため未配置。
- **win32/x64 にはマニフェスト定義がない**（packages 表に `Microsoft.ML.OnnxRuntime.Gpu.Windows`
  の名前だけある）。`--onnxruntime-node-install=cuda12` を Windows で指定すると
  "Invalid installation: cuda12 for platform: win32/x64" で落ちる。つまり公式インストーラ経路では
  Windows に CUDA バイナリを入れられず、**手動配置が唯一の経路**。
- npm tarball 自体には CUDA プロバイダは常に入らない（全プラットフォーム）。

### 1.2 公式 GPU パッケージの中身とサイズ（実測、2026-10-24 取得）

NuGet パッケージ（ダウンロードサイズ）:
- `Microsoft.ML.OnnxRuntime.Gpu.Windows` 1.30.0: 153,100,313 B
- `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0: 236,037,232 B

GitHub リリース v1.30.0 にも同等物がある（ユーザー向けの入手元として併記）:
- `onnxruntime-win-x64-gpu_cuda13-1.30.0.zip`: 294,386,823 B
- `onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz`: 235,697,517 B

`runtimes/<platform>/native/` の実ファイルサイズ（展開後）:

| ファイル | Windows | Linux | 備考 |
|---|---|---|---|
| `onnxruntime_providers_cuda.*` | 186,986,848 | 272,054,000 | **必須** |
| `onnxruntime_providers_shared.*` | 21,856 | 14,632 | **必須**（ブリッジ本体） |
| `onnxruntime_providers_tensorrt.*` | 891,704 | 1,084,896 | 不要（TensorRT は使わない） |
| `onnxruntime.*`（GPU ビルド本体） | 16,921,400 | 32,515,904 | 不要（下記 1.3） |
| **配置が必要な合計** | **187,008,704（178.3MiB）** | **272,068,632（259.5MiB）** | |

同梱判断: プラットフォームごとにインストーラが 178〜260MiB 増えるのは非常識と判断し、
**同梱しない**。モデル配置（D2）と同じ「所定フォルダへのユーザー配置」方式を採る。

### 1.3 バンドル済み本体ライブラリとの組み合わせ（実測で証明）

GPU ビルドの本体ライブラリに差し替える必要があるかが焦点だったが、不要だった:

- アプリ同梱の CPU 版 `libonnxruntime.so.1` + binding のコピーに対し、セッション生成で
  `executionProviders: ['cuda']` を要求すると、ORT の共有プロバイダブリッジ
  （`provider_bridge_ort.cc`）が **ORT バイナリと同じフォルダ**から
  `libonnxruntime_providers_shared.so` を読みに行く（不在時はそのパスで
  "Failed to load shared library" エラーになることを確認）。
- そこへ 1.2 の 2 ファイルを配置すると読み込みが進み、次の失敗が
  `libcublasLt.so.13: cannot open shared object file` になった。つまり
  **本体は CPU 版のままで、プロバイダ 2 ファイルの配置だけで CUDA 経路が開く**
  （残る不足はユーザー側 CUDA ランタイムだけ）。
- Windows でもブリッジは同一ソースの仕組みであり、配置先は
  `bin/napi-v6/win32/x64/`（`DirectML.dll` と同じフォルダ）。ただし Windows 実機での
  実行確認はできていない（未確認事項参照）。

### 1.4 ユーザー側の前提条件（バイナリの依存関係から実測）

`libonnxruntime_providers_cuda.so` の DT_NEEDED / DLL の import・動的ロード先:

- 直接依存: `libcudart.so.13` / `libcublas.so.13` / `libcublasLt.so.13` / `libcurand.so.10`
  （Windows: `cublas64_13.dll` / `cublasLt64_13.dll` を import、`cudart64_13.dll` 等を動的ロード）
  → **CUDA Toolkit 13.x** が必要（12.x では soname が合わない）
- 動的ロード: `libcudnn.so.9` / `cudnn64_9.dll`（cuDNN API はプロバイダが dlopen する）
  → **cuDNN 9.x** が必要
- `libcuda.so.1` / ドライバ側スタブ → 最新の **NVIDIA ドライバ**
- TensorRT ライブラリは不要（TensorRT EP は要求しない）

まとめ（CUDA を使うユーザー）: NVIDIA ドライバ + CUDA Toolkit 13.x + cuDNN 9.x を
インストールし、公式 GPU パッケージ（1.30.0）からプロバイダ 2 ファイルを ORT 本体と
同じフォルダに配置する。DirectML はこの一切が不要（Windows パッケージに
`DirectML.dll` 18,527,584 B が同梱済みで、DX12 対応 GPU なら動く構成）。

---

## 2. 実装内容（D9）

- **設定**: `FaceSettings.provider: 'auto' | 'cpu' | 'cuda' | 'directml' | 'coreml'`
  （既定 `'auto'`、旧 .vha2 に項目が無ければ `'auto'` 扱い）。
- **候補梯子**（`node/face/ort-provider.ts` の `OrtSessionFactory`）:
  auto は win32: CUDA→DirectML→CPU、linux: CUDA→CPU、darwin: CoreML→CPU。
  手動指定はそのプロバイダ→CPU。セッション生成に失敗した候補は例外にせず記録して次へ進む。
  最初の成功が「実効プロバイダ」となり、以降のセッションはまずそれを使う
  （モデル側の都合で失敗したら残り候補を再走査）。
- **適用範囲**: 検出（SCRFD）・認識（ArcFace / LVFace）・genderage の全セッションを
  ワーカー共有の 1 つのファクトリに統一。ジョブ（scan / attributes）と init に
  provider を載せ、パック変更と同様にプロバイダ変更でもワーカーを再起動する。
- **可視化**: ワーカーは `ready` / 各結果に `providerInfo`（requested / effective /
  attempts[失敗理由つき]）を返す。main は要約をログに出し、設定画面は
  `FaceModelStatus.providerInfo` で「実効プロバイダ（要求値）」「失敗した候補と理由」
  「CUDA ライブラリ配置の有無と配置先フォルダ」を表示する。
- **パッケージング**: CUDA ライブラリは同梱しないため electron-builder 設定は変更なし。
  `onnxruntime-node` は既に全プラットフォームで asarUnpack 済みで、ユーザーが配置した
  ファイルは `app.asar.unpacked` 側に置かれる（配置先として設定画面に表示するパスも
  unpacked 側に解決している）。
- **modelId**: プロバイダでは分けない（数値差は誤差レベル想定。実機で大きな乖離が
  観測されたら分離する、と設計書 D9 に明記）。

## 3. 検証結果（GPU なし環境で可能な範囲）

新規ハーネス `tools/face-spike/gpu-provider-harness.cjs` — **全チェック合格**:

- [candidates] 設定×プラットフォームの梯子順序（純関数）7 件 + 正規化 2 件
- [ortbin] bin フォルダ解決が実インストールを指すこと、この環境で CUDA ライブラリが
  未配置と判定されること、OS 別ファイル名
- [fallback] auto / cuda / directml / coreml / cpu の全設定で `FaceEngine.create` が
  例外なく CPU に着地し、attempts が梯子どおりに記録されること（失敗理由の文字列つき）。
  LVFace 認識でも同様。実測した失敗理由:
  - cuda: `OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library`
  - directml: `no available backend found. ERR: [dml] backend not found.`
  - coreml: `no available backend found. ERR: [coreml] backend not found.`
- [scan] `cuda` 要求のまま BBC 動画 1 本をエンドツーエンドでスキャン完了（22 顔、512 次元）
- [genderage] `directml` ファクトリ経由で CPU に落ちて属性サンプル生成まで動作
- [worker] 実物の `face-worker.js` を worker_threads の parentPort シムで駆動し、
  init（cuda / directml）→ ready の providerInfo、scan 結果、attributes 結果の
  いずれにも `effective = 'cpu'` と失敗試行が載ることを確認

回帰: 既存ハーネス Phase 1 / Phase 2 / Phase 3 / LVFace パック — **すべて合格**。
tsc 3 プロジェクト合格、Angular ビルド成功（警告 2 件は既存）。
Linux パッケージング: `electron-builder --linux --dir` 成功し、unpacked 側に
onnxruntime-node の bin が配置されることを確認（初回は作業ツリー内の未追跡ファイル
`tools/face-spike/.venv` の /usr/bin シンボリックリンクを builder が拒否して失敗したが、
CI チェックアウトには存在しないローカル artifact が原因で、本変更とは無関係。
退避すると成功した）。

## 4. 実機でしか確認できない項目（未確認）

1. **Windows + NVIDIA GPU での CUDA 実行**: プロバイダ配置後に `effective = 'cuda'` に
   なること、その速度（CPU 比）。ブリッジの仕組みは Linux で実測済みだが、
   Windows 版バインディングでの読み込みは未実行。
2. **Windows での DirectML 実行**: `DirectML.dll` 同梱は確認済みだが、実 GPU での
   セッション生成・速度・SCRFD の全演算子が DirectML に載るか（部分的に CPU に
   残る演算子があるか）は未確認。
3. **macOS での CoreML 実行**: darwin 用 dylib に CoreML が含まれる前提で候補に入れて
   あるが、実機確認なし。
4. **GPU / CPU 間の埋め込み数値差**: 同一顔の埋め込みコサインが閾値運用に影響しない
   範囲か（modelId を分けない判断の検証）。GPU で作った index と CPU で作った index の
   混在時のクラスタリング安定性も含む。
5. **CUDA 配置フォルダの書き込み権限**: Program Files 配下へのインストール時は
   管理者権限が要る想定（ポータブル版は回避可）。インストーラ種別ごとの実際の可否。
6. Linux の AppImage 形態ではアプリフォルダが読み取り専用マウントになり、配置方式の
   CUDA は使えない見込み（deb / 展開形なら可）。リポジトリから直接動かす場合は
   `ONNXRUNTIME_NODE_INSTALL=cuda12 npm install`（linux の公式経路）で配置できる。
