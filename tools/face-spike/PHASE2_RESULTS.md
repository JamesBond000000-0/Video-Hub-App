# Phase 2 検証結果（全体クラスタリング＋人物 UI＋属性集約）

実施日: 2026-10-23
対象: `feature/face-recognition`（Phase 1 完了時点 = 88a6c97 の上に実装）
設計書: `docs/FACE_RECOGNITION_DESIGN.md` §5.6 / §5.8 / §6 / §8.2 / §11 Phase 2

## 実装内容（コミット単位）

1. **型と基盤**（`7230939`）: `PersonRecord` / `PersonAttributes` を face-types に定義、`FaceIndexFile.persons` を型付け、`FaceRecord.personId`、`nextPersonSeq` を追加。`FaceIndexPayload.personsByVideo`（.vha2 キャッシュの導出元）を追加し、payload 生成に配線。属性の純関数ヘルパー（サンプル選択 `selectGroupSamples`、集約 `aggregateAttributes`）と worker ジョブ型もここ。`ImageElement.persons?: string[]` を final-object に追加。`FaceModelStatus` に genderage の解決結果を追加（**genderage.onnx は buffalo_l のみに同梱**のため、light パック時は quality パック側のファイルを共用する解決にした。設計書 §3 の「両パックに同梱」は誤りで、§5.8 に訂正を記録）
2. **全体クラスタリング**（`a0af553`）: `cluster.ts` に `collectGroupItems` / `clusterGlobalGroups` / `clusterizeIndex` を追加（方式は次節）
3. **genderage.ts**（`e12b529`）: スパイクで検証済みの GenderAge を worker 専用モジュールとして移植
4. **実行経路**（`6b3b4f3`）: face-store に `readEmbedding` / `applyClustering` / `renamePerson` / `setPersonHidden` / 属性の保存・消去を追加。face-worker に `attributes` ジョブを追加。main-face に `reclusterFaces` / `computeFaceAttributes` / `reclusterAfterScan` と IPC（`face-recluster`、`face-attributes-request`、`face-rename-person`、`face-set-person-hidden`）。faceQueue の drain から自動再分類が走る
5. **Angular UI**（`c555ec6`）: 後述
6. **バグ修正 2 件**（`ca5d5f6`、`5e83969`）: 検証中に発見した問題。後述

### 全体クラスタリングの方式（§5.6 の実装確定）

- 単位は (動画, 動画内グループ)。代表埋め込みの品質降順に leader-follower で割り当てる
- 割当て条件: 対象人物の**代表埋め込みと重心の両方**に類似度 ≥ T_global（= 0.50、D5）が必要で、スコア = min(代表類似度, 重心類似度) が最大の人物に割り当てる。重心は (動画,グループ) の顔数で重み付けした代表埋め込みの平均（正規化）。片方だけの条件では Phase 0 で実測した重心ドリフト混入（T=0.35〜0.45 で Norton が Unknown B のクラスタに吸収）が起きるため、両方要求が対策になる
- 重心の更新は割当てのたびに行う

### 人物 ID の安定化方式

- 再分類のたびに、新クラスタと旧 PersonRecord を**代表埋め込み同士の類似度**で対応付ける（類似度 ≥ T_global のペアを降順に貪欲に 1 対 1 マッチ）。設計書の「代表 faceId が一致すれば引き継ぐ」は類似度 1.0 のケースとして包含される（代表顔が再スキャンで差し替わっても、埋め込みが近ければ引き継げる）
- 引き継ぐもの: personId、名前（name）、非表示フラグ（hidden）。属性集約は**グループ集合が完全一致する場合のみ**引き継ぎ、変わっていれば属性パスで再計算されるまで付けない（古い集約を別構成の人物に表示しない）
- 新規人物の ID は単調増加の連番（`nextPersonSeq`、index に保存）で採番し、**消えた人物の ID は再利用しない**。古い .vha2 キャッシュが別人を指す事故を防ぐため

### 属性推定の入力経路（実測で決定）

- 設計の基準経路（元フレーム＋bbox 切り抜き）と、保存済み aligned 112×112 を入力にする経路を Phase 0 の 63 顔で比較実測した:
  - aligned を 96×96 にリサイズして入力: 性別一致 **55/63**、年齢差の中央値 **9 年**（最大 30）
  - aligned 上に名目 bbox を置いて基準の切り抜き処理を適用: 性別一致 **55/63**、年齢差の中央値 **7 年**（最大 23）
  - いずれも基準経路（フレーム再抽出）に対して明確に劣化するため不採用
- 採用方式: 属性ジョブ（worker）がサンプル顔のフレームを**その場で再抽出**し、基準経路で推論する。サンプルは (動画,グループ) ごとに品質上位 5 顔（`attributeSamplesPerGroup`）までなので、再抽出は最大でも数十フレーム/回で収まる
- 推論は再分類時に実行（自動再分類＝スキャン drain 後、手動再分類、属性オンへの切替時）。オフ時は集約処理自体が走らず、既存の集約値も再分類時に消去する
- 顔ごとの推定値は worker のメモリ内でのみ存在し、**index.json 等に一切保存しない**（設計書 §6.2 の任意 `attr` フィールドは実装しないことに確定し、設計書側も訂正済み）
- 動画ファイルのパスが判明している動画だけが属性の対象（main プロセスはスキャンした動画と renderer から渡された要素のパスを保持する。パス不明の (動画,グループ) はサンプルから外れる）

### .vha2 キャッシュの書戻し経路（D4 決定済み）

- 正は faces/index.json。main が payload に `personsByVideo`（hash → personId 配列）を載せ、renderer の FaceService が受信のたびに全 ImageElement の `persons` と照合して差分を書き込む。差分があれば `finalArrayNeedsSaving = true` にし、**保存自体は既存の保存フロー**（ハブを閉じる/切り替える際の getFinalObjectForSaving）に乗る（顔機能専用の保存経路は作らない）
- faces/ を消した場合は index が無効になり personsByVideo が空で返るため、全要素のキャッシュが除去され dirty 化する（再スキャンで是正される設計どおり）。モデル不一致時も同様に空になる

### Angular UI（§8.2）

- サイドバー人物欄: 代表顔の丸サムネイル＋名前＋登場動画数、上位 10 件、クリックで人物フィルタ（再クリックで解除）、「すべて見る」で人物一覧モーダル。表示トグルはグローバル設定ボタン `showPersonsSidebar`（§7 の指定どおり、設定の顔セクションから切替）
- 人物フィルタ: `person-filter.pipe` をギャラリーの pipe チェーンに追加（既存フィルタと AND）。フィルタは要素の `persons` キャッシュで判定し、index 更新のたびに FaceService の版数カウンタで再評価する
- 人物一覧ビュー（`app-persons` コンポーネント、モーダル）: 人物カードのグリッド（代表顔・名前・動画数/顔数・属性集約・非表示切替）。名前はインライン編集（クリック→入力→Enter/blur で IPC 保存）。再分類ボタンと属性再計算ボタン、空状態（「まだ顔スキャンされていません」相当＋スキャンボタン）を持つ。**結合・分割・顔の付け替え・代表顔の変更は Phase 3 のため実装していない**
- 動画詳細の顔ストリップに人物名チップを追加（未命名は「人物 N」）。クリックで人物フィルタ（meta → details / 下部トレイ → home へ伝播）
- 設定の顔セクション追加分: 属性トグル＋注記（集約のみ保存・±10 年級の推定である旨）、genderage 配置状態の表示、全体クラスタリング閾値の数値入力、再分類ボタン、統計への人物数追加。閾値変更・属性切替時は home が自動で再分類を要求する
- i18n は en.json のみ追加（他言語は後からでよいという設計 §8 の注記どおり）
- 人物の並び順は §8.1 どおり「登場動画数の多い順」

## 検証中に発見・修正したバグ（Phase 1 由来を含む）

1. **display crop が真っ黒**（`5e83969`、Phase 1 のバグ）: `displayCrop` の切り抜き幅・高さが float のまま画素インデックス計算に使われ、非整数インデックスへの TypedArray 書込みが全て無視され、crop が全面ゼロになっていた。実アプリでも代表顔画像が全て黒く表示される状態だった。Phase 1 のハーネスは枚数のみ確認しており見逃した。ジオメトリを整数化して修正し、Phase 2 ハーネスに「crop が黒でないこと」の検査を追加（修正後は 29/29 件が正常な内容）
2. **grabFrameImage の上書きプロンプト停止**（`ca5d5f6`）: ffmpeg 引数（generateFaceFrameArgs）に `-y` がなく、出力先に同名ファイルがあると上書き確認で停止する。属性パスが同じ出力パスを再利用する使い方で顕在化した。抽出前に出力先を消去し、失敗時に旧ファイルを誤ってデコードしないよう修正

## 検証結果

ハーネス: `tools/face-spike/phase2-harness.cjs`（Phase 1 ハーネスと同様、コンパイル済みのアプリ本体コード `node/face/*.js` を実走）。BBC 6 本・人物16 名相当の構成は Phase 1 と同一。

### ビルド

- `npm run check:tsc`（renderer / main / worker の 3 プロジェクト）: 合格
- `npm run build`（Angular 本番ビルド）: 成功。**警告は既存の 2 件のみ**（home.component の `path` 依存、ng-qrcode の `qrcode` 依存。Phase 1 のベースラインと同一）

### スキャン（Phase 1 と同一条件）

- 顔 72 件・動画内グループ 29 件（Phase 1 と完全一致 = 回帰なし）
- display crop 29 件すべてに実内容があることを画素レベルで確認（上記バグ 1 の回帰検査）

### 全体クラスタリング（T = 0.50）

- 人物数: **24**（Phase 0 の T=0.50 は 18）
- **動画をまたぐ 3 組の統合を確認**: Davies（yuXG + V9Dq の 2 グループが同一人物に）、Gosling（同）、Attenborough 高齢期（z0B4 + Gfa88 の 2 グループが同一人物に）。いずれも Phase 0 の T=0.50 クラスタと対応
- **混入なし**: Phase 0 の確定ラベル（tentative を除く）で、異なる 2 名が 1 人物に混ざったケースは 0 件。Phase 0 で T=0.35〜0.45 にあった重心ドリフト混入（Norton ↔ Unknown B）も再現しなかった
  - 補足: アプリの Norton 人物には「Unknown B（tentative ラベル）の Phase 0 グループに最も近いアプリグループ」が 1 件含まれるが、目視確認ではそのグループの代表顔は **2016 年放送回の髭のない Norton 本人**だった（Phase 0 の dmbp には Norton のグループ自体が存在しないため、転移先が最も近い別人になった＝ラベル転移側の人工物）。この人物の 4 グループ間の代表類似度は最小でも 0.517 で内的にも整合し、属性の年齢中央値 53 も 2016 年当時の Nort
...[truncated 1685 chars]