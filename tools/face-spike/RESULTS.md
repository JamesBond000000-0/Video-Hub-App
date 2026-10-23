# Face Spike RESULTS（Phase 0: 移植の正しさ＋速度計測）

実施: 2026-10-23。範囲は設計書 §11 Phase 0 のうち「モデル取得と台帳」「SCRFD/ArcFace 移植と Python 版との一致検証」「速度計測」まで。動画サンプルでのクラスタリング検証と閾値スイープは次ステップ（§6）。

## 0. 結論

- **移植は合格**。設計書 §10 の条件（埋め込みコサイン類似度 0.999 以上、bbox IoU 0.9 以上、性別一致、年齢差 ±2 歳以内）を **quality / light の両構成で**満たした
- **速度は合格**。この VM（2 vCPU）でも quality 構成で 1000 動画あたり推定 3.44 時間と、設計書の基準（8 時間以内）内。light は 0.72 時間
- 次ステップに進める状態。詰まりらしい詰まりは onnxruntime-node の postinstall 通信 1 件のみ（回避策あり、§5）

## 1. 環境

| 項目 | 値 |
|---|---|
| OS / CPU | Linux 7.0.0-39-generic (x64)、AMD EPYC 9D25（この VM は **2 vCPU**、RAM 7.7GB） |
| Node | v24.22.1（onnxruntime-node 1.30.0、sharp 0.34.5 / libvips 8.17.3、tsx 4.23.15） |
| Python（正解データ用） | 3.12.3 venv、insightface 2.1（pip)、onnxruntime 1.31.0、opencv-python-headless 5.0.0.93、numpy 2.5.4 |
| 実行プロバイダ | 両側とも CPUExecutionProvider、スレッド数は両側とも既定（=コア数） |

※ デスクトップ実機はコア数が多く、実測より速くなる見込み。数値はこの VM での相対比較として読むこと。

## 2. モデル台帳

取得元: InsightFace 公式 GitHub リリース v0.7 の `buffalo_l.zip`（288,621,354 bytes）/ `buffalo_sc.zip`（14,969,382 bytes)、2026-10-23 取得。詳細な SHA256 は設計書の付録 A に記録済み（コミット `1a13768`）。

要点のみ再掲:

| ファイル | サイズ (bytes) |
|---|---|
| det_10g.onnx | 16,923,827 |
| w600k_r50.onnx | 174,383,860 |
| det_500m.onnx | 2,524,817 |
| w600k_mbf.onnx | 13,616,099 |
| genderage.onnx | 1,322,532 |

判明事項: **`genderage.onnx` は buffalo_sc.zip に同梱されていない**（buffalo_sc は det + rec の 2 ファイルだけ）。スパイクでは buffalo_l 側の genderage（SHA256 は付録 A）を両構成で共用した。light + 属性推定を本実装で使う場合も同じ共用になる。

## 3. 一致検証（TS 移植 vs Python 正解）

### 3.1 サンプル画像と出自

いずれも公式 `insightface` pip パッケージ同梱のテスト画像（`insightface/data/images/`）で、出自が明確なものだけを使った。画像自体はライセンスと容量の都合でリポジトリに入れず、`samples/` は gitignore 対象。再現時は pip パッケージから同名ファイルをコピーする。

| ファイル | 内容 | det_size | 備考 |
|---|---|---|---|
| t1.jpg (1280×886) | 複数人の集合写真 | 640×640 | 両構成とも 6 顔検出 |
| Tom_Hanks_54745.png (112×112) | 単一の位置合わせ済み顔 | 320×320 | 640 では検出 0 件（§5 の既知の弱さ）。320 で 1 顔 |
| mask_black.jpg (440×231) | マスク着用の顔 | 640×640 | 両側とも検出 0 件の**負例**（一致として扱う） |

### 3.2 結果（`npx tsx src/compare.ts all`、詳細は `out/compare_*.json` ※gitignore 対象）

| 構成 | 画像 | 顔数 (正解/TS) | 最小 IoU | 最小コサイン類似度 | ランドマーク最大誤差 | 性別 | 年齢最大差 |
|---|---|---|---|---|---|---|---|
| buffalo_l (quality) | t1.jpg | 6/6 | 1.00000 | 1.000000 | 0.00 px | 全一致 | 0 歳 |
| buffalo_l (quality) | Tom_Hanks_54745.png | 1/1 | 0.99650 | 0.999860 | 0.07 px | 一致 | 2 歳 |
| buffalo_sc (light) | t1.jpg | 6/6 | 1.00000 | 0.999999 | 0.00 px | 全一致 | 0 歳 |
| buffalo_sc (light) | Tom_Hanks_54745.png | 1/1 | 0.99861 | 0.999819 | 0.08 px | 一致 | 0 歳 |
| 両構成 | mask_black.jpg | 0/0 | — | — | — | — | — |

- t1.jpg では bbox・埋め込みとも小数 6 桁まで完全一致した。JPEG デコード（sharp）と自前実装の OpenCV 互換リサンプリングで十分な精度が出ている
- 112px 画像でのみ僅かな差（類似度 0.99982〜0.99986、IoU 0.9965〜0.9986）が出たが、基準（0.999 / 0.9）は両構成とも上回る
- genderage の年齢差 2 歳（buffalo_l、34 vs 32）は §10 基準「±2 歳以内」の境界上にある。入力が 112px の極端に小さい画像であることが要因とみられ、通常サイズの顔（t1）では 0 歳差だった

### 3.3 移植で踏襲した参照実装の要点

- SCRFD: `_detect_candidates` のアスペクト維持リサイズ＋左上ゼロ詰め、`(px-127.5)/128`、ストライド [8,16,32]・各セル 2 アンカー、`distance2bbox/distance2kps`、スコア降順＋NMS (IoU 0.4、面積の +1 規約まで再現）
- 位置合わせ: skimage の SimilarityTransform.estimate（Umeyama）と等価な閉形式の最小二乗で相似変換を求め、`warpAffine` (BORDER_CONSTANT 0、双線形）を自前実装
- ArcFace: aligned 112×112 に `(px-127.5)/127.5`、出力 512 次元を L2 正規化
- genderage: **位置合わせ顔ではなく bbox 中心の相似変換クロップ**（96×96、scale = 96/(max(w,h)×1.5)) が参照実装の実態。モデルのグラフ先頭に Sub/Mul があるため入力は正規化なしの生ピクセル（mean 0 / std 1)。性別は `argmax(pred[:2])` で 1→M、年齢は `round(pred[2]×100)`
- 画像デコードだけ sharp を使い、リサイズとワープは OpenCV 互換の自前実装にした（§5 の判断）

## 4. 速度計測（`npx tsx src/bench.ts`、中央値）

測定対象: 検出は t1.jpg (1280×886) を det_size 640 で end-to-end（前処理＋推論＋後処理）。認識・属性は同画像の 6 顔について顔あたりに換算。

| 構成 | 検出 ms/frame（中央値 / 平均） | 認識 ms/face（中央値 / 平均） | genderage ms/face（中央値） |
|---|---|---|---|
| quality (det_10g + w600k_r50) | 294.1 / 339.9 | 118.1 / 133.7 | 1.7 |
| light (det_500m + w600k_mbf) | 49.4 / 51.7 | 37.1 / 48.2 | 1.6 |

動画換算（設計書 §5.1 のサンプリング前提 = 30 フレーム/動画、平均 1 顔/フレームとして中央値から換算）:

| 構成 | 秒 / 動画 | 1000 動画換算 |
|---|---|---|
| quality | 12.37 | **3.44 時間**（基準 8 時間以内を満たす） |
| light | 2.60 | **0.72 時間** |

- genderage は 1.7ms/face と桁違いに軽く、属性推定の追加コストは無視できる
- 検出が支配的。quality の検出は light の約 6 倍、認識は約 3.2 倍
- D1（既定パック）の判断材料: この VM でも quality が基準内なので、設計書の推奨どおり quality 既定を支持する結果。ただし実ライブラリのフレームは顔が無いフレームも多く、認識コストはこの換算より下がる方向

## 5. 詰まった点と採った退避策

1. **onnxruntime-node 1.30.0 の postinstall が失敗**（NuGet フィードへの通信で socket hang up を繰り返す）。npm パッケージ自体に CPU 用バイナリ（`bin/napi-v6/linux/x64/onnxruntime_binding.node` 等）が同梱されていることを確認し、公式の回避策 `ONNXRUNTIME_NODE_INSTALL=skip` を付けて導入した。本実装（Electron 配布）でも、制限のあるネットワークでは同じ問題が起こりうる
2. **SCRFD は「画像の大半を占める大きな顔」を検出できない**既知の弱さを確認: 112px の単一顔は det_size 640 で検出 0 件だったが、det_size 320 では検出できた（Python / TS 両側で同じ挙動）。動画のクローズアップフレームでも同じ取りこぼしが起こりうる。本実装では対策候補として「det_size の二段実行（640＋320）」か「フレームへの余白追加」を Phase 1 で検討すること（コストは検出時間の増加と要相談）
3. **マスク顔は検出できない**: mask_black.jpg は両側・両構成とも 0 件。SCRFD の素の実力として記録するに留める
4. det_size 320 実行時、onnxruntime-node が "VerifyOutputSizes（モデル内の形状注釈と実形状の不一致）" 警告を出す。モデルが 640 前提の形状注釈を持つための表示だけで、推論結果は Python 側と一致したため無視してよい
5. Python 側は pip の insightface がそのまま wheel で入ったため、想定していた退避策（ソース直接実行）は不要だった

## 6. 次ステップへの引き継ぎ

残りは設計書 §11 Phase 0 の後半（動画サンプルでのクラスタリング検証）:

1. ユーザーの実ライブラリから、正解人物が分かる動画を数本選ぶ（対象の指定待ち）
2. §5.1 のサンプリングでフレームを抜き、本スパイクの TS 実装で埋め込みまで通す（`src/` の Scrfd / ArcFace はそのまま再利用できる構造にしてある）
3. 動画内統合（T_in）と全体クラスタリングを実装し、`T_global` 0.35〜0.60 のスイープを付録 C の表に記録する
4. あわせて検討: 大きな顔の取りこぼし対策（§5-2）を動画サンプルで必要性判定する

本実装（Phase 1）へ持ち込む資産: `src/scrfd.ts`、`src/arcface.ts`（GenderAge 含む）、`src/image-ops.ts`。配置先は設計書 §9 の `node/face/`。
