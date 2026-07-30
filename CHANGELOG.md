# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- パッケージ管理をpip/venvからuvへ移行
  - `uv.lock`による依存関係の固定（CI・開発環境で同一構成を再現）
  - `.python-version`でPython 3.11を固定
  - PyInstallerを`[dependency-groups]`のdevグループへ移動
  - GitHub Actionsを`astral-sh/setup-uv` + `uv sync --frozen` + `uv run`に変更
- `perspective_corrector.spec`の`CFBundleVersion`を`pyproject.toml`から自動読み込み（バージョンの二重管理を解消）
- README.md / BUILD_WINDOWS.md の手順をuvベースに刷新

### Fixed
- `pyproject.toml`のバージョンが1.2.0のまま更新されていなかった問題を修正（1.3.2へ同期）
- PDF出力処理のdocstringが72dpiと記載されていた誤りを修正（実装は300dpi）
- `DEVELOPMENT_LOG.md`のファイル構成が古いワークフロー名（`build-windows.yml`）のままだった点を修正

### Removed
- `requirements.txt`（依存関係は`pyproject.toml`と`uv.lock`に一本化）
- `order_corners()`内の未使用コード（重心・角度ソート）

## [1.3.2] - 2025-12-17

### Fixed
- PyInstallerのモジュール除外設定が原因で起動に失敗する問題を修正
- macOS互換性のためopencv-python-headlessからopencv-pythonへ戻した

## [1.3.1] - 2025-12-17

### Changed
- 未使用モジュールを除外してビルドサイズを削減

## [1.3.0] - 2025-12-17

### Added
- アプリケーションアイコン（Windows: `icon.ico` / macOS: `icon.icns`）

### Changed
- ウィンドウサイズを1500×800に固定し、OSによるタイリング/最大化を無効化

### Fixed
- specファイル内のアイコンパス解決を修正（spec配置ディレクトリ基準に変更）
- Windowsビルドでのアイコン検証ステップをbashシェルで実行するよう修正

## [1.2.0] - 2025-12-16

### Added
- macOS版ビルド対応（DMG形式で配布）
- GitHub ActionsでWindows/macOS同時ビルド

### Changed
- ワークフローを`build.yml`に統合

## [1.1.1] - 2025-12-16

### Changed
- 拡大鏡のサイズを10%縮小（400px → 360px）

## [1.1.0] - 2025-12-16

### Added
- 色補正機能: ホワイトバランス、コントラスト、明るさ、彩度の調整
- 詳細なコードコメント: 各関数・処理の「なぜ」を説明するドキュメント

### Changed
- PDF出力の解像度を72dpiから300dpiに向上（印刷品質の大幅改善）
- PDF出力のA4サイズを841×595ピクセルから3508×2480ピクセルに変更
- PNG出力の圧縮設定にコメント追加（可逆圧縮で品質劣化なし）

### Fixed
- PDF出力時の画質劣化問題を修正

## [1.0.0] - 2025-12-15

### Added
- 4隅自動検出機能（Cannyエッジ検出ベース）
- 手動座標指定とドラッグ調整
- 拡大鏡表示機能
- 一括処理機能
- HEIC/HEIF形式対応
- PDF出力機能（複数画像をマルチページPDFに）
- 最近使用したフォルダ履歴
- ドラッグ&ドロップ対応
- Windows EXE版（GitHub Actions自動ビルド）

## [0.1.0] - 2025-12-14

### Added
- 初期リリース
- 基本的な台形補正機能
- PySide6 GUIアプリケーション
