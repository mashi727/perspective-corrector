# Perspective Corrector

プレゼンテーション写真の台形歪みを補正するデスクトップアプリケーション。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)

## 機能

- **4隅自動検出**: Cannyエッジ検出による四角形領域の自動認識
- **手動座標指定**: クリックで4隅を指定、ドラッグで微調整
- **拡大鏡表示**: マウス位置周辺を拡大表示し、精密な座標指定が可能
- **一括処理**: 複数画像の台形補正を一括実行
- **HEIC対応**: iPhone撮影画像（HEIC/HEIF形式）に対応

## デモ

https://github.com/user-attachments/assets/ca2963dc-b7ba-4e17-aa18-5a1c793dcb91

## インストール

### pipでインストール（推奨）

```bash
# GitHubから直接インストール
pip install git+https://github.com/mashi727/perspective-corrector.git
```

### 開発用インストール

```bash
git clone https://github.com/mashi727/perspective-corrector.git
cd perspective-corrector
pip install -e .
```

### 手動インストール

```bash
pip install PySide6 opencv-python numpy pillow pillow-heif
```

## 実行

```bash
# pipインストール後
perspective-corrector

# または直接実行
python perspective_corrector.py

# ディレクトリを指定して起動
perspective-corrector /path/to/image/directory
```

## 使い方

1. メニュー「ファイル」→「フォルダを開く」(Cmd+O / Ctrl+O) で作業フォルダを選択
2. 左側のファイル一覧から画像を選択
3. 「自動認識」ボタンで4隅を自動検出（または手動でクリック指定）
4. 必要に応じて4隅をドラッグで微調整
5. 「一括処理」で台形補正を実行

### その他の機能

- **最近使用したフォルダ**: メニューから最大10件の履歴にアクセス可能
- **ドラッグ&ドロップ起動**: フォルダまたはファイルをアプリにドロップして起動

### 自動認識設定

「認識設定」ボタンでパラメータを調整可能:

| パラメータ | 説明 |
|-----------|------|
| Canny低閾値 | エッジ検出の感度（低） |
| Canny高閾値 | エッジ検出の感度（高） |
| ぼかしサイズ | ノイズ除去の強度 |
| 近似精度 | 輪郭近似の精度 |
| 最小面積比率 | 検出する四角形の最小サイズ |

## Windows用EXE作成

詳細は [BUILD_WINDOWS.md](BUILD_WINDOWS.md) を参照。

```bash
pip install pyinstaller
pyinstaller perspective_corrector.spec
```

## 出力

- 出力サイズ: 1920x1080
- 出力形式: PNG
- 出力ファイル名: `[出力名]_corrected.png`
- 出力先: 元画像と同じディレクトリ

## ライセンス

MIT License
