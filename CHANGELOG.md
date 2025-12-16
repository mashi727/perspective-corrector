# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
