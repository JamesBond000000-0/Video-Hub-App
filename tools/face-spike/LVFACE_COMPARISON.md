# LVFace vs ArcFace 実測比較（Phase 0 と同一素材・同一手順）

実施: 2026-10-24。ユーザー指示による、現行 ArcFace（w600k_r50）から LVFace（ByteDance、ICCV 2025）へ認識モデルだけを差し替えた場合の実測比較。検出（SCRFD det_10g）と素材は Phase 0 と共通で、認識器のみを差し替える。
実行は `tools/face-spike/src/lvface-spike.ts`（`npm run lvface-compare`、引数で対象モデルを絞れる）。生結果は `out/lvface-comparison.json`（gitignore 対象）。

## 0. 結論

- **採用候補は LVFace-T のみ**。同一実行内での速度は ArcFace の約 0.34 倍（約 3 倍速）、モデルサイズは 76.7MB（ArcFace 174.4MB の 0.44 倍）。精度は本素材では ArcFace とほぼ互角で、動画間統合 3 組はすべて代表類似度 0.61〜0.79 と余裕を持って成立し、推奨閾値も ArcFace と同じ **T_global = 0.50** がそのまま使える。代償は分割が 1 件増えること（後述の Davies の難顔 1 件が単独グループ化）のみ
- **LVFace-B は見送り**。456MB・ArcFace の約 1.56 倍の遅さにもかかわらず本素材では T より悪い（Gosling の動画間類似度が 0.498 と閾値境界、T=0.50 で推定人物 20・分割 5）。IJB-C の公称値（B 97.70 > T 96.67）の順序は、この用途（動画フレームの aligned crop＋重心クラスタリング）では再現しなかった
- **LVFace-S は現行閾値体系では不適**。類似度のスケールが圧縮され（同一人物ペア中央値 0.462）、T_in=0.50 の動画内グループ化が 29 件に爆発する。混入は全閾値でゼロだが過分割が支配的で、使うなら T_in から全面的な再調整が前提になる
- **若年期 Attenborough の年齢不変性は全バリアントで改善なし**。高齢期との類似度は ArcFace と同じく実質 0（-0.06〜+0.10）で、約 60 歳差は LVFace でも別人として扱われる
- 採用判断の留保: 本比較は 63 顔の小サンプル 1 セットのみ。一般論としてのモデル優劣ではなく「このアプリのパイプライン条件での実測」として読むこと。採用する場合は埋め込み空間が変わるため全再スキャンが必要で、属性推定（genderage 相当）は LVFace 側に存在しない

## 1. 取得したモデル

取得元は公式 HuggingFace `bytedance-research/LVFace`（repo revision `b12702ab1f5c721748e054a66dc90e1edd1f0724`）、認証なしで取得できた。配置先は `tools/face-spike/models/lvface/`（gitignore 対象）。L（1GB 超）は今回対象外。

| バリアント | ファイル | サイズ (bytes) | SHA256 |
|---|---|---|---|
| LVFace-T | `LVFace-T_Glint360K/LVFace-T_Glint360K.onnx` | 76,653,813 | `bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3` |
| LVFace-S | `LVFace-S_Glint360K/LVFace-S_Glint360K.onnx` | 304,196,926 | `cd09f27c82ce0a3633fb8b1966d779a7171b23aa4f14ca0de6edf9677573d119` |
| LVFace-B | `LVFace-B_Glint360K/LVFace-B_Glint360K.onnx` | 455,533,594 | `9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382` |
| （基準）ArcFace | `buffalo_l/w600k_r50.onnx` | 174,383,860 | Phase 0 台帳（設計書付録 A）参照 |

取得 URL の形式: `https://huggingface.co/bytedance-research/LVFace/resolve/main/<フォルダ>/<ファイル>`。B の SHA256 は HF 側の Xet ハッシュ（`X-Linked-ETag`）とも一致した。
全バリアントとも出力は 512 次元で、L2 正規化後のノルム誤差は最大でも 5.5e-9（実装: `src/lvface.ts`）。

## 2. 前処理の根拠（公式コードで確認）

公式リポジトリ `bytedance/LVFace`（commit `80bd20c8bb78e56eb708941180d800a27c396a6b`）の推論コードで確認した:

- `inference_onnx.py` の `LVFaceONNXInferencer`: OpenCV で読む（BGR）→ 112×112 リサイズ → **BGR→RGB 変換** → NCHW 転置 → `((px/255) − 0.5) / 0.5` で正規化。本スパイクの `RawImage` は sharp 由来で最初から RGB のため、RGB 順のまま `toBlob((px−127.5)/127.5)` を渡すことが公式と等価になる（正規化式も代数的に同一）。出力埋め込みは公式側では正規化されないため、類似度計算前にこちらで L2 正規化した
- `onnx_ijbc.py` の位置合わせ: 5 点テンプレート `SRC` に x 方向 +8.0 を加えた値が、Phase 0 の aligned crop を作った ArcFace テンプレート（[38.2946, 51.6963] 等）と**完全一致**する。よって Phase 0 の aligned 112×112 JPEG（63 顔）をそのまま全認識器の共通入力にできる
- 水平反転の flip test は使っていない。公式の単一画像 inferencer も反転を使わず（IJB 評価スクリプトの評価プロトコル上の選択）、アプリのパイプラインも単一視点のため、比較条件として単一視点で揃えた

## 3. 方法

- 入力: Phase 0 の SCRFD 検出から作った aligned JPEG 63 顔（`cluster-test/faces/aligned/`）を全モデル共通で使用。JPEG デコードは計時対象外に置
...[truncated 6719 chars]