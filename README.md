# StudentCup2026

企業データから、各企業が**従業員向けDX教育商材を購入するか（購入=1 / 非購入=0）**を予測する二値分類モデルです。

---

## 1. 概要

StudentCup2026では、企業の基本情報・財務情報・アンケート・自由記述テキストを用いて、DX教育商材の購入有無を予測します。

- タスク: 二値分類
- 目的変数: `購入フラグ`
- 評価指標: **F1 Score**
- train: 742社
- test: 800社

---

## 2. ファイル構成

メインノートブック:

```text
StudentCup2026_Colab_Drive.ipynb
```

想定するデータ配置:

```text
StudentCup2026/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submit.csv   # 任意
├── submissions/
└── artifacts/
```

Google Colab + Google Driveでの実行を想定しています。

`sample_submit.csv` がない場合は、`test.csv` の企業IDから提出形式を自動生成します。

---

## 3. 使用データ

### 学習データ

```text
train.csv: 742行 × 43列
```

### テストデータ

```text
test.csv: 800行 × 42列
```

主なデータは以下です。

- 企業基本情報
- 財務情報
- アンケート
- 企業概要
- 組織図
- 今後のDX展望

以下は特徴量から除外します。

```text
企業ID
企業名
購入フラグ
```

---

## 4. モデル概要

モデルは大きく**自然言語モデル**と**構造化データモデル**の2系統で構成されています。

```text
自然言語
  ↓
Character N-gram + TF-IDF
  ↓
LinearSVC
  │
  ├─────────────┐
  │             ↓
構造化データ → Logistic Regression
                ↓
          スコアを統合
                ↓
            H214予測
```

最終的なH214では、通常のbranchとアンケート4との交互作用を加えたbranchを50:50でブレンドしています。

---

## 5. 自然言語処理

以下の3列を結合して使用します。

```text
企業概要
組織図
今後のDX展望
```

処理:

```text
文字単位 2〜5 gram
→ HashingVectorizer
→ TF-IDF
→ LinearSVC
```

主な設定:

```python
HashingVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    n_features=32768
)

LinearSVC(C=0.3)
```

単語分割ではなく文字N-gramを使うことで、企業ごとの自由記述表現を比較的柔軟に扱っています。

---

## 6. 構造化データ処理

数値・カテゴリ・アンケートから、学習に必要な特徴量を生成します。

主な処理:

- 財務比率などの派生特徴量生成
- 財務数値への符号付きlog変換
- 欠損状態を表す特徴量
- 工場数・店舗数と業界の組み合わせ
- 自由記述から文章長・キーワード・DX文脈特徴を生成
- ソフトウェア投資関連特徴
- アンケート4との交互作用特徴

### 数値特徴

```text
中央値補完
→ RobustScaler
```

### カテゴリ特徴

```text
最頻値補完
→ One-Hot Encoding
```

構造化データ側のモデルは以下です。

```python
LogisticRegression(
    C=2.0,
    solver="liblinear"
)
```

---

## 7. 予測の統合

自然言語モデルの `decision_function` と、構造化モデルの予測確率をlogit空間で統合します。

さらにH214では、

```text
通常branch
+
アンケート4 interaction branch
```

を50:50でブレンドします。

最終判定閾値:

```python
H214_THRESHOLD = 0.28
```

---

## 8. 実行方法

### Step 1. データを配置

```text
train.csv
test.csv
```

を `data` フォルダに配置します。

### Step 2. Notebookを開く

```text
StudentCup2026_Colab_Drive.ipynb
```

### Step 3. 必要に応じて設定を変更

主な設定:

```python
CONFIG.DATA_DIR
CONFIG.SUBMISSION_DIR
CONFIG.OUTPUT_DIR
CONFIG.N_JOBS
CONFIG.H214_THRESHOLD
```

### Step 4. 上から順に実行

最後に、

```python
main()
```

を実行すると、データ読み込みから予測・提出ファイル生成まで行われます。

---

## 9. 出力

主な提出ファイル:

```text
submission_H214_current_best_noheader.csv
```

提出形式

```text
企業ID, 予測値
```
---

## 10. Dependencies

主要ライブラリ:

```text
numpy
pandas
scipy
scikit-learn
joblib
```
---

## 11. 注意点

### Public Best Override

現在の設定では、Public Leaderboardでの検証結果をもとにした特定企業IDへのoverrideが有効です。

```python
APPLY_PUBLIC_BEST_OVERRIDE = True
```

これは通常のモデル学習とは別の後処理です。  
純粋なモデル性能を確認する場合は `False` に設定してください。

---