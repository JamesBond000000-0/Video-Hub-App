# LVFace パック実装の検証結果（第 3 パック `lvface`、決定 D8）

実施: 2026-10-24。ユーザー決定（D8）に基づき、LVFace-T を第 3 のモデルパックとして本体に実装した記録。実測比較の前提は `LVFACE_COMPARISON.md`、設計側の記録は `docs/FACE_RECOGNITION_DESIGN.md` の D8 / §3.2 / §5.4 / 付録 A。

検証は `tools/face-spike/lvface-pack-harness.cjs`（コンパイル済みのアプリ本体コード `node/face/*.js` を実走）＋ Phase 1〜3 ハーネスの再実行。

## 1. 実装内容

| 項目 | 内容 |
|---|---|
| パック id | `lvface`（`FaceModelPack` に追加） |
| 構成 | 検出 `det_10g.onnx`（quality と同一の SCRFD）＋ 認識 `LVFace-T_Glint360K.onnx` |
| modelId | `lvface-t-glint360k`（quality の `buffalo_l-r50`、light の `buffalo_sc-mbf` と別。既存の modelId 不一致の仕組みで切替時は再スキャンになる） |
| 認識ファイル名 | HuggingFace（bytedance-research/LVFace）の配布ファイル名そのまま。ダウンロードしたファイルを改名なしで `<userData>/face-models/lvface/` に置ける |
| 認識器の切替 | `FaceModelPackInfo.recKind`（`'arcface' \| 'lvface'`）を新設し、FaceScanJob / worker 初期化メッセージ経由で `FaceEngine` に渡す。LVFace 推論は `node/face/lvface.ts`（スパイクで公式コードと照合済みの前処理を移植: RGB 順のまま `(px − 127.5) / 127.5`、出力は L2 正規化、位置合わせは ArcFace テンプレートと同一なので既存の aligned crop がそのまま入力になる） |
| 閾値 | T_global の既定 0.50 は変更なし（LVFace-T でも実測で妥当だったため、パック別の既定値は持たない） |
| 属性推定 | **無変更で動作する**（下記 §3.5）。genderage はフレーム crop を入力とし認識埋め込みに依存しないため、lvface でも quality フォルダの `genderage.onnx` を共用する既存の解決がそのまま効く |
| 設定画面 | パック選択に「LVFace-T (fast, ~94 MB)」を追加。モデル状態表示は既存の動的一覧（ファイル名・配置先）がそのまま対応。配置案内文（en）を 3 パック分に更新（LVFace は HF から取得、`det_10g.onnx` は quality フォルダからコピーでよい旨） |
| バージョン | 本体 `package.json` の version は変更していない |

## 2. ビルド

- `npm run check:tsc`（tsconfig / tsconfig-serve / tsconfig.worker の 3 プロジェクト）: 合格
- Angular ビルド（`ng build`）: 成功（警告 2 件は既存の CommonJS 依存に関するもの）

## 3. ハーネス結果（BBC 6 本、同一素材・同一サンプリング）

### 3.1 パック配線

`FACE_MODEL_PACKS.lvface` の detFile / recFile / recKind / modelId と、スパイク側モデルファイルが期待名で存在することを確認。全項目合格。

### 3.2 スキャンと検出の一致

検出が quality と同一（det_10g）であるため、顔検出は両パックで**顔単位で完全一致**した（faceId・時刻・bbox・スコアの列が全動画で同一）。

| 動画 | 顔（両パック共通） | グループ quality | グループ lvface |
|---|---|---|---|
| yuXGpUR7fXA | 22 | 8 | 8 |
| dmbpagijVkk | 13 | 7 | 7 |
| V9DqzY1jPhY | 7 | 2 | 2 |
| 8cPxeyZ6-g0 | 12 | 4 | 4 |
| z0B4KEXVaqo | 11 | 7 | **8** |
| Gfa88SeNohY | 7 | 1 | 1 |
| **合計** | **72** | **29** | **30** |

総顔数 72 は Phase 1/2 の quality 基準値と一致。グループ数の差は z0B4KEXVaqo の 1 件のみ（§3.4）。

### 3.3 modelId の混在防止

- lvface の index を quality の modelId で開く → `modelMismatch = true`、書き込み（setVideo）は例外で拒否
- quality の index を lvface の modelId で開く → 同様に不一致
- 自分の modelId では正常にロードされる

### 3.4 全体クラスタリング（T_global = 0.50）

| | グループ数 | 統合（joins） | 人物数 |
|---|---|---|---|
| quality | 29 | 5 | **24** |
| lvface | 30 | 4 | **26** |

人物数 = グループ数 − 統合件数で、差 +2 の内訳は次の 2 件。**いずれも分割側の差で、混入（確定ラベルで異なる 2 名が 1 人物に混ざる）は両パックとも 0 件**。

1. **Attenborough のアーカイブ顔 1 件**（z0B4KEXVaqo、t=73.8）: quality では主グループに含まれ本隊の人物に統合されたが、LVFace では動画内で単独グループ化し（z0B4 のグループ +1 の正体）、全体でも統合されず単独人物になった。年齢推定の中央値は 31 で、若年寄りのアーカイブ顔が本隊（高齢期）と統合されにくい LVFace-T の傾向（`LVFACE_COMPARISON.md` の若年期 Attenborough の結果）と整合する
2. **Norton の周辺グループ 1 件**（ラベル転移でラベルが付かないグループ）: quality では Norton の人物に 4 グループが統合されたが、LVFace では 3 グループまでで、残り 1 グループが単独人物になった

動画間統合の確認（Phase 2 と同じ意味論、quality を対照として実施）:

- **Davies**: 両パックとも 2 動画の全グループが 1 人物に統合（ゲートとして合格）
- **Attenborough（高齢期の本隊）**: LVFace でも 2 動画にまたがって 1 人物に統合（ラベル付き 11 顔）。上記 1. の断片だけが別人物。若年期アーカイブの断片（z0B4、t=162.4 の 1 顔）は quality でも LVFace でも単独人物のままで、両パック共通の挙動
- **Gosling**: アプリのサンプリングでは、もう一方の動画（V9DqzY1jPhY）の Gosling 相当グループにラベル転移が付かず（転移類似度 0.095）、**両パックともラベル経由では判定不能**だった（Phase 2 ハーネスにも同じ盲点がある）。Phase 0 素材そのものでの LVFace-T の Gosling 統合は `LVFACE_COMPARISON.md` で確認済み（代表類似度 0.736 ≥ 0.50 で統合）

### 3.5 属性推定（genderage）

LVFace パックの人物に対しても、参照経路（フレーム再抽出 → genderage → 人物単位の集約）が無変更で動作した。

- 集約は 26/26 人物で生成された
- ラベルで判定可能な 11 人物について性別は **11/11 一致**（全員 agreement 1.00）。年齢中央値は quality の Phase 2 記録と同水準（Davies 50、Attenborough 本隊 75、Whittaker 28 など）
- index.json に顔単位の属性値は保存されていない（人物単位の集約のみ）ことを確認

### 3.6 回帰

- Phase 1 ハーネス: 全 24 チェック合格
- Phase 2 ハーネス: 全 67 チェック合格（quality で 72 顔・24 人物を再現）
- Phase 3 ハーネス: 全 56 チェック合格
- LVFace パックハーネス: 全チェック合格

## 4. 未確認事項

- 実機での GUI 確認（設定画面の表示とパック切替の操作、人物 UI の見た目）は未実施。utility process の IPC 往復と `.vha2` への persons キャッシュ保存も、従来分と合わせて実機確認待ち
- light パックの実スキャンは今回未実施（recKind は quality と同じ `'arcface'` で、配線チェックのみ。エンジン経路は quality の回帰で担保）
- Phase 3 の手直し（結合・分割・ピン留め）の実走は quality の埋め込みでのみ確認済み。クラスタリング以降は認識器に依存しない共通コードだが、LVFace 埋め込みでの編集→再分類の実走は未実施
- 検証素材は BBC 6 本・72 顔の 1 セットのみ（`LVFACE_COMPARISON.md` の留保と同じ）

## 5. 結論

LVFace-T は第 3 パックとして成立する。検出は quality と完全一致し、クラスタリングの差は分割側の 2 件のみで混入はなく、属性推定も無変更で使える。認識速度の優位（ArcFace の約 0.34 倍、モデル 76.7MB）は `LVFACE_COMPARISON.md` の実測どおりで、閾値も 0.50 のまま流用できる。
