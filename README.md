# Perspective Corrector

プレゼンテーション写真の台形歪みを補正するデスクトップアプリケーション。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-Qt6-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%7C%205.x-red.svg)
![uv](https://img.shields.io/badge/managed%20by-uv-blueviolet.svg)
[![Release](https://img.shields.io/github/v/release/mashi727/perspective-corrector)](https://github.com/mashi727/perspective-corrector/releases/latest)

## ダウンロード

### Windows版（インストール不要）

**[PerspectiveCorrector.exe をダウンロード](https://github.com/mashi727/perspective-corrector/releases/latest/download/PerspectiveCorrector.exe)**

ダウンロードしたEXEファイルをダブルクリックで起動できます。

> 初回起動時はWindowsのSmartScreenが警告を表示する場合があります。「詳細情報」→「実行」で起動できます。

### macOS版（インストール不要）

**[PerspectiveCorrector.dmg をダウンロード](https://github.com/mashi727/perspective-corrector/releases/latest/download/PerspectiveCorrector.dmg)**

DMGファイルを開き、アプリケーションをApplicationsフォルダにドラッグしてください。

> 初回起動時は「開発元を確認できません」と表示される場合があります。「システム設定」→「プライバシーとセキュリティ」→「このまま開く」で起動できます。

## 機能

### 基本機能
- **4隅自動検出**: Cannyエッジ検出による四角形領域の自動認識
- **手動座標指定**: クリックで4隅を指定、ドラッグで微調整
- **拡大鏡表示**: マウス位置周辺を拡大表示し、精密な座標指定が可能
- **一括処理**: 複数画像の台形補正を一括実行
- **HEIC対応**: iPhone撮影画像（HEIC/HEIF形式）に対応

### 出力機能
- **高品質PNG出力**: 1920x1080の可逆圧縮PNG（品質劣化なし）
- **高品質PDF出力**: 300dpi A4横サイズのマルチページPDF（印刷に最適）
- **自動色補正**: プロジェクター投影写真の色かぶりを自動補正

## デモ

https://github.com/user-attachments/assets/fc22470a-5e27-4404-bf19-a1fdc936953e

## インストール

パッケージ管理には [uv](https://docs.astral.sh/uv/) を使用します。未導入の場合は先にインストールしてください。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Homebrew
brew install uv
```

### ツールとしてインストール（推奨）

```bash
# GitHubから直接インストール（隔離環境に導入され、PATHにコマンドが追加される）
uv tool install git+https://github.com/mashi727/perspective-corrector.git
```

インストールせずに一度だけ実行する場合:

```bash
uvx --from git+https://github.com/mashi727/perspective-corrector.git perspective-corrector
```

### 開発用セットアップ

```bash
git clone https://github.com/mashi727/perspective-corrector.git
cd perspective-corrector
uv sync --all-groups   # .venv を作成し、uv.lock の内容で依存を固定インストール
```

`uv sync` は `.python-version`（3.11）に従って Python 処理系を自動取得するため、事前の Python インストールや venv の手動作成は不要です。

### pipを使う場合

uv を使わない場合も従来どおりインストールできます。

```bash
pip install git+https://github.com/mashi727/perspective-corrector.git
```

## 実行

```bash
# uv tool install 後
perspective-corrector

# 開発用セットアップ後（.venv を明示的に有効化せずに実行）
uv run perspective-corrector
uv run python perspective_corrector.py

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

## 実行ファイルのビルド

詳細は [BUILD_WINDOWS.md](BUILD_WINDOWS.md) を参照。

```bash
uv sync --all-groups                                # PyInstaller は dev グループに含まれる
uv run pyinstaller perspective_corrector.spec --noconfirm
```

Windows では `dist/PerspectiveCorrector.exe`、macOS では `dist/PerspectiveCorrector.app` が生成されます。

## 出力

### 個別PNG出力
- **出力サイズ**: 1920×1080 pixels
- **出力形式**: PNG（可逆圧縮、品質劣化なし）
- **圧縮レベル**: 3（速度とファイルサイズのバランス）
- **出力ファイル名**: `[出力名]_corrected.png`
- **出力先**: 元画像と同じディレクトリ

### PDF出力
- **解像度**: 300dpi（高品質印刷対応）
- **用紙サイズ**: A4横（3508×2480 pixels）
- **マルチページ**: 複数画像を1つのPDFにまとめて出力
- **レイアウト**: 画像はアスペクト比を維持して最大サイズで中央配置
- **保存先**: ダイアログで任意の場所を選択

### 画質について
- PNG出力は可逆圧縮のため、圧縮レベルに関わらず画質劣化なし
- PDF出力は300dpi（従来の72dpiから大幅改善）で印刷に最適
- HEIC→JPEG変換時は品質95%で視覚的劣化を最小化

## 色補正機能

プロジェクター投影写真に特有の色かぶりを自動補正:

| 設定項目 | 説明 |
|---------|------|
| ホワイトバランス | 自動/手動（色温度調整） |
| コントラスト | 画像のコントラスト強調 |
| 明るさ | 全体の明るさ調整 |
| 彩度 | 色の鮮やかさ調整 |

## ライセンス

MIT License
