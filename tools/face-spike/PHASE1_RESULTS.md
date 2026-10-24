# Phase 1 結果記録: 検出 + 顔サムネイルのアプリ統合

- 実施日: 2026-10-23
- 対象: 設計書 `docs/FACE_RECOGNITION_DESIGN.md` §11 Phase 1（§5.1〜§5.5、§6、§7、§9、§10 の Phase 1 分）
- ブランチ: `feature/face-recognition`（ローカルコミットのみ。push は親レビュー後に Git Data API で実施）

## 1. 実装内容

### エンジン（`node/face/`、設計書 §9.1 の構成）

Phase 0 スパイク（`tools/face-spike/src/`）で Python 版と一致検証済みのコードを移植:

| ファイル | 内容 |
| --- | --- |
| `scrfd.ts` / `arcface.ts` / `image-ops.ts` | 検出・認識・画像処理。genderage は Phase 2 のため移植しない。アプリの es5 ターゲットに合わせ arcface の反復をインデックスループ化、sharp は既存 node コードの流儀で `require` に変更（esModuleInterop を tsconfig に足さないため） |
| `cluster.ts` | 動画内統合（§5.5）のみ。全体クラスタリング（§5.6）は入れていない |
| `face-scan.ts` | §5.1〜§5.5 パイプライン。フレーム抽出 → 検出（顔ゼロ時は det_size 320 で1回再試行）→ 最小顔 40px フィルタ（短辺・生 float 判定、スパイクと同一セマンティクス）→ 整列サムネイル（全顔）+ 表示 crop（グループ代表のみ）→ 埋め込み → 動画内統合。Electron 非依存で Node から直接駆動可能 |
| `face-store.ts` | `faces/index.json`（tmp+rename の原子的書込）+ `embeddings.bin`（追記専用）。modelId 不一致は無効扱いで書込を拒否 |
| `face-worker.ts` | Electron utility process の入口（init / scan プロトコル） |
| `main-face.ts` | ワーカー調停、モデル配置状態（D2）、FaceStore のキャッシュ、IPC 登録 |
| `face-types.ts` | 索引スキーマ、パック定義（quality=`buffalo_l-r50`、light=`buffalo_sc-mbf`）、統計計算 |

index スキーマの §6.2 からの追加: 各顔に `groupId`（動画内グループ）と `representative`（代表フラグ）を記録する。§5.5 の統合結果であり、Phase 2 の全体クラスタリングで再計算不要にするため。`personId` は付けていない、`persons` は空配列。

### キュー / IPC / 進捗

- `node/main-extract-async.ts` に `faceQueue`（4本目、並列度1）を追加。サムネイル抽出の完了後に enqueue される。最終 `'done'` 進捗と `allowSleep()` は thumb / face 両キューの drain が揃った時点で一度だけ送るよう調整（face が空なら従来どおり thumb drain で完結）。`enabled=false`（既定）または `autoScanNew=false` のとき enqueue 自体が起きず完全無動作。既存3キューの順序・進捗・powerSaveBlocker の扱いは不変。
- 手動スキャン用に `extractAnyMissingFaces(fullArray)` を export。
- IPC（`node/main-ipc.ts` + `main-face.ts`）: `face-get-model-status` → `face-model-status-returning`、`face-get-index` → `face-index-returning`、`face-settings-updated`（GLOBALS 同期 + パック変更時はワーカー停止）、`face-scan-request`、通知 `face-index-updated`。
- 進捗は既存の `sendCurrentProgress` に段階 `'scanningFaces'` を追加。レンダラ側は既存の `progressString` 方式で `scanning faces x/y - NN%` を表示する。

### 設定・データ

- `interfaces/final-object.interface.ts` に `FaceSettings` を追加、`FinalObject.faceSettings` は任意（旧 .vha2 がそのまま読める）。GLOBALS に既定値を追加し、ハブを開くときに既定値とマージ。
- **D1 決定: 既定パックは quality**（Phase 0 実測 3.44 時間/1000 動画が §11 の基準 8 時間以内）。
- **D2 決定: モデルは同梱せず** `<userData>/face-models/<パック名>/` に手動配置。設定画面に配置先パス・必要ファイル名・配置状態を表示。自動ダウンロードなし。
- パック切替時: 埋め込みはモデル間で互換性がないため（§4.3）、スキャン要求時に modelId 不一致を検出したら hub の顔データをリセットして再構築する。

### Angular UI（最小）

- `src/app/services/face.service.ts`: レンダラ側の IPC 窓口（モデル状態・索引・統計の保持、動画内グループの表示用整形）。
- 設定画面（その他タブ）に顔セクション: 有効化トグル、自動スキャントグル、パック選択、モデル配置状態と配置先の案内、統計（処理済み動画数・顔数）、手動スキャンボタン（無効時・モデル未配置時は disabled）。
- 動画詳細の顔ストリップ: 詳細ビューは仮想スクロールでアイテム高さが固定のため、既存構造を崩さない形としてメタ欄（app-meta-item）の末尾に代表顔の display crop（48px）と出現時刻を表示する形にした。顔データがない動画では何も描画しない。人物サイドバー・人物フィルタは Phase 2 のため作っていない。
- i18n は `en.json` のみ追加（他言語は順次、§8）。

### 判断記録

- **属性推定の型は先行して入れた**: `FaceSettings.attributesEnabled`（既定 false）と `attributeSamplesPerGroup`（既定 5）を §7 どおり型と既定値だけ用意した。Phase 1 のコードは一切参照しない。Phase 2 で .vha2 の移行なしに有効化できる。
- §7 の項目名「`framesPerVideo` 上限」はフィールド名を `maxFramesPerVideo` とした（上限であることを名前で明示）。

## 2. 検証結果

### npm install / 型チェック / ビルド

| 項目 | 結果 |
| --- | --- |
| `npm install --legacy-peer-deps`（onnxruntime-node 追加時） | 成功。postinstall は Phase 0 の知見どおり `ONNXRUNTIME_NODE_INSTALL=skip` を使用 |
| ベースライン（変更前 603c3d9）`npm run check` | `check:tsc` 合格。`check:lint` は `tslint: not found` で失敗（exit 127） |
| 変更後 `npm run check:tsc`（tsconfig / tsconfig-serve / tsconfig.worker の3プロジェクト） | **合格（エラーなし）** |
| 変更後 `npm run check` 全体 | tsc は合格、lint はベースラインと同じ理由で失敗（後述） |
| Angular ビルド `npm run build` | **成功**。警告2件はいずれも変更前からあるもの（home.component.ts の `path` 依存、ng-qrcode の `qrcode` 依存）。顔関連の新規警告なし |

lint について: この repo では lint 経路が2つとも**変更前から壊れている**（ベースライン 603c3d9 で確認）。`check:lint` が呼ぶ `tslint` は依存に存在せず、`npm run lint`（ng lint / eslint）も `eslint` パッケージ自体が依存にないため "Unable to find ESLint" で起動できない。Phase 1 の変更起因ではないため、検証の実効基準は `check:tsc` 合格 + Angular AOT ビルド成功とした。lint 基盤の修復は Phase 1 の範囲外（親への申し送り事項）。

### ハーネス実走（`tools/face-spike/phase1-harness.cjs`）

コンパイル済みのアプリ側エンジン（`node/face/*.js`）を Node から直接駆動し、Phase 0 の BBC テスト動画6本で「フレーム抽出 → 検出 → サムネイル生成 → index/embeddings 保存 → 再読込」を検証。**全24チェック合格**。

顔数（アプリ §5.1 のサンプリング vs Phase 0）:

| 動画 | フレーム数 | Phase 1 | Phase 0 | 差 |
| --- | ---: | ---: | ---: | ---: |
| 8cPxeyZ6-g0 | 10 | 12 | 9 | +3 |
| Gfa88SeNohY | 10 | 7 | 7 | 0 |
| V9DqzY1jPhY | 10 | 7 | 9 | -2 |
| dmbpagijVkk | 10 | 13 | 13 | 0 |
| yuXGpUR7fXA | 10 | 22 | 18 | +4 |
| z0B4KEXVaqo | 11 | 11 | 7 | +4 |
| **合計** | **61** | **72** | **63** | **+9（+14%）** |

フレーム数は完全一致（61）。顔数の差はサンプリング位置の違いによるもの: 設計書 §5.1 は「[0, duration] を n 等分した各区間の中点」であり Phase 1 はこれに従うが、Phase 0 のクラスタテストは [2%, 98%] の等間隔だった。トーク番組はフレームごとに映る顔が変わるため、この差で数個前後する。

エンジン自体の同等性は同一フレームでの厳密比較で別途確認した（ハーネスの [parity] セクション）: yuXGpUR7fXA を Phase 0 と同じ [2%, 98%] の時刻で抽出すると保持顔 **19 vs Phase 0 の 18**（±2 以内）。残差1は duration の 0.1 秒丸めでフレームが僅かにずれることによる境界差。

保存・再読込まわり（全て合格）:

- aligned サムネイル数 = 顔総数（72）、`embeddings.bin` = 72 × 2048 バイトちょうど
- display crop 数（29）= 動画内グループ総数（29）と一致
- 再読込後の統計が保存前と一致、埋め込みの往復コサイン = 1.000000、代表フラグ保持
- modelId 不一致（light で読込）→ mismatch 判定が立ち、`setVideo` は書込を拒否
- 一時フレーム（faces/tmp）が残らない

### enabled=false の無動作確認

ハーネスの [gate] セクションで実測: GLOBALS の既定値（enabled=false）のまま `scanVideoFaces` を呼んでも、出力フォルダも `faces/` フォルダも一切作られない（3チェック合格）。キュー側も enqueue 自体が発生しない実装で、設定画面の手動スキャンボタンも無効時は disabled。

### 所要時間

ハーネスのスキャン部は6本合計 138 秒（この VM、他セッションと競合した状態での実測。Phase 0 の単体計測とは条件が違うため速度の結論には使わない）。

## 3. D6: linux の asar 扱い（確認結果）

electron-builder 26.15.3 で確認:

1. **linux でも asar の既定は true**（スキーマ `LinuxConfiguration.asar` の default）。未指定でも app.asar は生成されていた。
2. ネイティブモジュールの自動展開（smartUnpack、既定 ON）: `.node` / `.so` / `.dll` / `.dylib` / 拡張子なしバイナリを含むモジュールのルートを自動で app.asar.unpacked へ展開する。ffmpeg-static / ffprobe は名前特例もある。
3. **既存の `"../node_modules/..."` 形式の asarUnpack エントリは、この repo のレイアウト（appDir = ルート）では相対パスにマッチせず無効**だった。従来 ffmpeg-static 等が展開されていたのは smartUnpack の効果。Phase 1 で追加するエントリは実効形式 `"node_modules/..."` で指定した。既存2エントリの書き換えは Phase 1 の範囲外として現状維持（申し送り事項）。
4. **実害のあったケース**: `@img/sharp-libvips-linux-x64` は smartUnpack の対象から漏れて asar 内に残り、sharp の `.node` が `libvips-cpp.so.8.17.3` を開けずロード失敗した（`--dir` パッケージングで実測）。対策として `node_modules/@img` を asarUnpack に追加し解消。

パッケージング試行（`npx electron-builder --linux --dir`）:

- 初回: `tools/face-spike/.venv`（Phase 0 の Python venv、作業コピー内に物理的に存在）内のシンボリックリンクが `files: **/*` に紛れて失敗。venv を同一ファイルシステム内へ一時退避して再試行し**成功**（クリーンなチェックアウトでは発生しない、ローカル artifacts 起因の問題）。
- 成功後のレイアウトから実測: `onnxruntime-node`（InferenceSession 生成 OK）、`sharp`（JPEG 生成パイプライン OK、libvips 8.17.3 ロード確認）、`ffmpeg-static`、`@img/sharp-libvips-linux-x64` が app.asar.unpacked に揃い、パッケージ済み配置のままネイティブモジュールが動作することを確認した。
- 対象は `--dir`（展開形式）のみ。AppImage の生成・起動確認はしていない。

## 4. 未確認事項

- **GUI の見た目・操作感**: この VM では確認不可。設定セクションと顔ストリップの実表示は実機（Windows 等）での確認が必要。特に顔ストリップはメタ欄末尾への配置のため、タグが多い動画では表示位置が下がる。
- **Electron 本体での utility process 疎通**: face-worker のプロトコルは型とコンパイルでは検証済みだが、実アプリ起動下での fork → init → scan の往復は未実施（ハーネスはエンジン直駆動）。Phase 2 着手前か実機確認時に要検証。
- **win / mac のパッケージング**: asarUnpack の指定は linux と同じ形式で入れたが、ビルド実測は linux のみ。
- **大規模ライブラリでの実運用性能**: 1000 本規模のキュー動作・進捗表示の追随性は未実測（Phase 0 の推定 3.44 時間/1000 動画が前提）。
- モデル配置後の初回スキャンでのみ分かる問題（userData パスの実値、権限など）は未確認。

## 5. Phase 2 への引き継ぎ

- 埋め込みは全顔分 `embeddings.bin` に保存済みで、全体クラスタリング（§5.6）に再スキャンは不要。`FaceStore.getIndex()` / `payload()` から索引と埋め込み件数が取れる。
- `FaceSettings.clusterThreshold`（= T_global、決定済み 0.50）と属性2項目は型・既定値・保存まで配線済み。genderage の移植元はスパイクの `arcface.ts`（Phase 1 では移植しなかった部分）。
- Phase 0 の知見: 低い閾値では重心ドリフトで混入が起きる（CLUSTER_RESULTS.md）。全体クラスタリングの実装時に代表同士の類似度チェック等の対策を検討すること。
- 申し送り: ①lint 基盤（tslint / eslint 本体が依存にない）②electron-builder.json の既存 `"../node_modules/..."` エントリ2件が無効形式であること（smartUnpack が代替しているため実害なし）。

## 6. コミット一覧（ブランチ `feature/face-recognition`、push は未実施）

| コミット | 内容 |
| --- | --- |
| `1493944` | build: onnxruntime-node 1.30.0 と sharp 0.34.5 を依存に追加 |
| `74f93d7` | feat: FaceSettings 型と既定値を追加 |
| `8e41979` | feat: 顔認識エンジンを node/face/ に配置 |
| `cfdd925` | feat: faceQueue（4本目）と顔 IPC を追加 |
| `191e442` | feat: Angular UI（設定セクションと詳細の顔ストリップ） |
| `605c3aa` | build: win/mac の asarUnpack に onnxruntime-node と sharp を追加 |
| `d4c86b2` | build: linux にも asar と asarUnpack を明示（D6 対応） |
| `b9b1a55` | fix: 最小顔フィルタを生 float の短辺判定に統一 |
| `ec418d5` | fix: asarUnpack に @img を追加（libvips が asar 内に残る問題） |
| `dc86326` | fix: asarUnpack の追加分を実効形式（node_modules/...）に修正 |
| （このコミット） | docs: PHASE1_RESULTS.md と設計書ステータス更新、検証ハーネス |

## 7. 作業メモ（環境起因の事故と復旧）

- /tmp は 512MB の tmpfs で、`.venv`（約850MB）を /tmp へ退避しようとして ENOSPC で mv が途中失敗し、venv が一時的に分割状態になった。上書きなしコピーで復元し、`yt_dlp` / `insightface` / `onnxruntime` の import と yt-dlp 実行で健全性を確認済み。リポジトリの追跡ファイルへの影響はない（.venv は gitignore 対象のローカル artifacts）。以降の退避は同一ファイルシステム（ホーム配下）で行った。
