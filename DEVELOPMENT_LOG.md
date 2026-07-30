# プレゼン写真 台形補正ツール - 開発記録

## 概要

プレゼンテーション写真の台形歪みを補正するデスクトップアプリケーション。
PySide6 (Qt) + OpenCVで構築。

## 現在の仕様 (2026-07-30)

### 基本情報

| 項目 | 値 |
|------|-----|
| ファイル名 | `perspective_corrector.py` |
| フレームワーク | PySide6 (Qt6) |
| 画像処理 | OpenCV, NumPy |
| 対応画像形式 | JPEG, PNG, HEIC/HEIF |
| ウィンドウサイズ | 固定 1500x800（`setFixedSize`） |
| スプリッター比率 | 350:1050 (ファイル一覧:画像表示) |
| パッケージ管理 | uv（`pyproject.toml` + `uv.lock`） |
| Pythonバージョン | 3.11（`.python-version`で固定、動作要件は3.9以上） |

### 機能一覧

1. **ファイル一覧表示**
   - 左側パネルにディレクトリ内の画像ファイルを表示
   - 出力ファイル名の編集が可能
   - 2カラム構成（元ファイル名 | 出力名）

2. **4隅座標指定**
   - 手動クリック: 左上→右上→右下→左下の順で指定
   - 自動認識: Cannyエッジ検出による四角形検出
   - ドラッグによる座標調整（精密モード）

3. **拡大鏡表示**
   - マウス位置周辺を拡大表示
   - 象限に応じたオフセット表示（見やすさ向上）
   - 選択領域を半透明マゼンタでオーバーレイ

4. **自動認識設定**
   - リアルタイムプレビュー付き設定ダイアログ
   - パラメータ調整可能

5. **一括処理**
   - 設定済み画像を一括で台形補正
   - 出力サイズ: 1920x1080
   - PNG個別出力またはPDF一括出力を選択可能

6. **PDF出力**
   - 複数画像を1つのPDFにまとめて出力
   - A4横サイズ（アスペクト比維持、中央配置）
   - 保存先をダイアログで選択

### UI構成

```
+------------------+----------------------------------------+
|                  |  操作説明                              |
|  ファイル一覧    +----------------------------------------+
|  (350px)         |                                        |
|                  |           画像キャンバス               |
|  - 元ファイル    |           (拡大鏡表示あり)             |
|  - 出力名        |                                        |
|                  +----------------------------------------+
|  設定済み: n件   | 座標表示 | [座標クリア][自動認識]      |
|                  |          | [認識設定][一括処理][終了]   |
+------------------+----------------------------------------+
```

### ボタン配色

| ボタン | 色 | カラーコード | 用途 |
|--------|-----|-------------|------|
| 座標クリア | オレンジ | #E67E22 | リセット/警告 |
| 自動認識 | 青 | #3498DB | ツール/アクション |
| 認識設定 | 紫 | #9B59B6 | 設定/調整 |
| 一括処理 | 緑 | #27AE60 | 実行/確定 |
| 終了 | グレー | #7F8C8D | 終了 |

### 拡大鏡設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `magnifier_base_size` | 400px | 拡大鏡の基準サイズ（長辺） |
| `magnifier_zoom` | 1.75 | 拡大率 |
| `drag_precision` | 0.15 | ドラッグ時の減衰率（精密モード） |
| `drag_threshold` | 15px | ドラッグ判定の距離 |

### 自動認識パラメータ

| パラメータ | デフォルト値 | 範囲 | 説明 |
|-----------|-------------|------|------|
| `canny_low` | 50 | 0-255 | Cannyエッジ検出の低閾値 |
| `canny_high` | 150 | 0-255 | Cannyエッジ検出の高閾値 |
| `blur_size` | 5 | 1-21 | ガウシアンブラーのカーネルサイズ |
| `approx_epsilon` | 0.02 | 0.001-0.1 | 輪郭近似の精度係数 |
| `min_area_ratio` | 0.05 | 0.01-0.5 | 最小面積比率 |

### コーナー表示色

| コーナー | 色 |
|---------|-----|
| 左上 | 赤 (255, 0, 0) |
| 右上 | 緑 (0, 255, 0) |
| 右下 | 青 (0, 0, 255) |
| 左下 | 黄 (255, 255, 0) |

### 設定ファイル

- ファイル名: `perspective_config.json`
- 保存場所: 作業ディレクトリ
- 内容: 各画像の座標情報、出力ファイル名、自動認識設定（`_detection_settings`）、色調補正設定（`_color_settings`）
- 最近使用したフォルダ: `~/.perspective_corrector_recent.json`（最大10件）

---

## 開発履歴

### Phase 1: 基本機能実装

- PySide6によるGUI構築
- OpenCVによる透視変換（台形補正）機能
- ファイル一覧表示と選択
- 4隅座標のクリック指定
- JSON設定ファイルによる座標保存

### Phase 2: 拡大鏡機能

- マウス位置周辺の拡大表示機能追加
- 半透明オーバーレイ（選択領域表示）
  - 赤 → 緑 → 黄 → シアン → **マゼンタ**（最終採用）
- 象限ベースのオフセット表示
  - マウスが画像の左半分にあれば拡大鏡を右にオフセット
  - マウスが画像の上半分にあれば拡大鏡を下にオフセット
  - オフセット量: 25%

### Phase 3: 精密ドラッグ

- コーナーのドラッグ移動機能
- 精密モード実装（0.15倍減衰）
- クリック時のコーナー位置同期

### Phase 4: 自動認識機能

- Cannyエッジ検出による四角形自動検出
- 「自動認識」ボタン追加

### Phase 5: 認識設定ダイアログ

- パラメータ調整ダイアログ実装
- リアルタイムプレビュー機能
  - ダイアログ内で即座に結果確認
  - OKで適用、キャンセルで破棄
- 設定のJSON保存対応

### Phase 6: UI調整

- ボタン配色の意味付け
- スプリッター比率調整 (350:1050)
- ウィンドウ最小サイズ調整 (1500px)
- 拡大鏡サイズ: 280 → 308 → 339 → **400px**
- クロスプラットフォームフォント対応
  - `QFont("Sans")` → `QFont()` (システムデフォルト)

### Phase 7: 通知の最適化

- 正常完了時のメッセージ表示を削除
  - 自動認識成功時: ステータスバーメッセージを削除
  - 一括処理成功時: 完了ダイアログを削除
- エラー発生時のみ警告ダイアログを表示

### Phase 8: パッケージ化・GitHub公開

- `pyproject.toml`追加によるpipインストール対応
- GitHubリポジトリ作成・公開
- README.md、LICENSE、.gitignore整備

### Phase 9: フォルダ操作機能

- **メニューバー追加**
  - ファイル(F) メニュー
  - フォルダを開く... (Cmd+O / Ctrl+O)
  - 最近使用したフォルダ（最大10件、履歴クリア機能）
  - 終了 (Cmd+Q / Ctrl+Q)
- **ウィンドウタイトル**: 現在のフォルダ名を表示
- **ドラッグ&ドロップ起動**: フォルダ/ファイルをexeにドロップで起動
- **ステータスバー改善**: 選択ファイルのフルパス or 作業フォルダを表示
- 最近使用したフォルダは `~/.perspective_corrector_recent.json` に保存

### Phase 10: ダイアログ中央配置

- **全ダイアログをメインウィンドウ中央に配置**
  - `center_dialog`ヘルパーメソッド追加
  - `show_message_box`メソッド追加（中央配置メッセージボックス）
- **対象ダイアログ**:
  - フォルダ選択ダイアログ (700x500)
  - 認識設定ダイアログ (800x600)
  - プログレスダイアログ (400x100)
  - 全メッセージボックス（警告、確認等）
- 非ネイティブダイアログ使用で位置制御を確実化
- ウィンドウ移動後も移動先の中心に表示
- **認識設定ダイアログの修正**
  - `showEvent`オーバーライドで表示時に中央配置
  - レイアウト確定後のサイズで正確に配置

### Phase 11: Windows HEIC対応強化

- **PyInstaller設定の改善**
  - `collect_all('pillow_heif')`で依存関係を自動収集
  - バイナリ、データ、hiddenimportsを自動バンドル
- WindowsでのHEIC/HEIF画像読み込みをサポート
- BUILD_WINDOWS.mdにHEIC対応の説明を追加

### Phase 12: PDF出力機能

- **出力形式選択ダイアログ**
  - 一括処理時にPNG/PDF出力を選択可能
  - `show_output_format_dialog()`メソッド追加
- **PDF出力機能**
  - 複数画像を1つのPDFファイルにまとめて出力
  - A4横サイズ (841.89 x 595.28 pt / 297mm x 210mm)
  - アスペクト比を維持して最大サイズで中央配置
  - 保存先をダイアログで選択
  - Pillowによるマルチページ PDF生成
- **実装詳細**
  - `run_batch_process_png()`: 従来のPNG個別出力
  - `run_batch_process_pdf()`: 新規PDF出力処理

### Phase 13: 色調補正プレビュー改善・パフォーマンス最適化

- **メイン画面での色調補正プレビュー**
  - オリジナル画角＋四隅マーカー＋色調補正（ON/OFF連動）を表示
  - 透視変換プレビューではなく、元画像に色調補正を適用して表示
  - `get_display_image()`メソッド追加（色調補正済み画像を返す）
- **色調補正設定ダイアログの簡素化**
  - 有効/無効チェックボックスをダイアログから削除
  - メイン画面のトグルボタンのみで色調補正ON/OFFを制御
- **パフォーマンス最適化**
  - 色調補正済み画像のキャッシュ機能追加
    - `color_corrected_pixmap`: 色調補正済みフルサイズ画像
    - `color_correction_cache_valid`: キャッシュ有効フラグ
  - 画像変更・設定変更時のみキャッシュを無効化
- **ドラッグ操作の軽量化**
  - `scaled_image_cache`: スケール済み背景画像のキャッシュ
  - `update_corners_only()`: コーナーマーカーのみ再描画
  - ドラッグ中は背景を再計算せず、マーカーのみ更新
  - マウスポインタと拡大鏡の十字カーソルの連動が滑らかに
- **一括処理のフィルタリング改善**
  - 4隅座標が設定されたファイルのみ出力対象に
  - 内部キー（`_`で始まるエントリ）をスキップ
- **ファイル一覧の視覚的フィードバック**
  - 緑色: 4隅設定済み（処理対象）
  - オレンジ色: 1〜3隅のみ設定（処理対象外）
  - 灰色: 未設定

### Phase 14: Windows HEIC対応強化

- **pillow-heif初期化の改善**
  - 初期化を一度だけ行うようにキャッシュ機能追加
  - `_heif_registered`, `_heif_available`フラグで状態管理
  - `_init_heif_support()`関数で初期化を一元化
- **エラーハンドリング強化**
  - `ImportError`だけでなく全ての例外をキャッチ
  - エラーメッセージをコンソールに出力（デバッグ用）
- **EXIF情報の保持**
  - HEIC→JPEG変換時にEXIF情報を維持
- **ImageMagickフォールバック追加**
  - pillow-heifが動作しない場合のWindows用代替手段
  - `magick`コマンド（ImageMagick 7）を優先
  - `convert`コマンド（ImageMagick 6）もサポート
- **ドキュメント整備**
  - BUILD_WINDOWS.mdにHEICトラブルシューティングガイド追加
  - デバッグ用コンソール付きビルド方法を記載

### Phase 15: GitHub Actions CI/CD

- **Windows EXE自動ビルド**
  - `.github/workflows/build-windows.yml`追加
  - タグプッシュ時（`v*`）に自動ビルド
  - 手動実行（workflow_dispatch）もサポート
- **ビルド環境**
  - Windows Server (windows-latest)
  - Python 3.11
  - PyInstallerによるEXE生成
- **アーティファクト管理**
  - ビルドしたEXEを30日間保持
  - タグ時はGitHub Releaseを自動作成
  - リリースにEXEファイルを添付
- **権限設定**
  - `permissions: contents: write`でリリース作成権限を付与
- **README更新**
  - リリースバージョンバッジ追加
  - ダウンロードセクション追加（EXE直接ダウンロードリンク）
  - SmartScreen警告についての注意書き

### Phase 16: コードドキュメント・コメント整備

- **モジュールドキュメント**
  - ファイル先頭に包括的なdocstring追加
  - 主要機能、アーキテクチャ、設定ファイルの説明
- **インポートの整理**
  - 標準ライブラリ、サードパーティ、Qtの3セクションに分類
  - 各セクションにコメントヘッダー追加
- **関数ドキュメント**
  - `auto_color_correction()`: Gray WorldとCLAHEの詳細説明
  - `perspective_transform_cv()`: 透視変換アルゴリズムの説明
  - `auto_detect_corners()`: エッジ検出パイプラインの説明
  - `order_corners()`: 4点ソートアルゴリズムの説明
- **クラスドキュメント**
  - `DetectionSettingsDialog`: 自動認識パラメータの使用シーン説明
  - `ColorCorrectionSettingsDialog`: 補正処理フローの説明
  - `ImageCanvas`: 座標系、拡大鏡、ドラッグ操作、パフォーマンス最適化の説明
  - `PerspectiveCorrectorApp`: 責務、ワークフロー、設定ファイルの説明
- **複雑なロジックへの「なぜ」コメント**
  - ImageCanvas初期化: 各変数の役割とパフォーマンス最適化の理由
  - 拡大鏡の象限オフセット: なぜオフセットが必要かの説明
  - PDF出力: A4サイズ計算とフィット処理の説明
  - 遅延色調補正: 体感レスポンス向上のための設計説明

### Phase 17: macOS対応・アイコン整備（v1.3.0 - v1.3.2）

- **macOSビルド対応**
  - specファイルでプラットフォーム判定（macOS: one-dir + `.app`バンドル / Windows: one-file `.exe`）
  - GitHub Actionsを`build.yml`に統合し、Windows/macOSを同時ビルド
  - DMG作成ステップ（`hdiutil`）を追加
- **アプリケーションアイコン**
  - `icon.svg`を原本とし、`icon.png` → `icon.icns`（macOS）/ `icon.ico`（Windows）を生成
  - specファイルのアイコンパスをspec配置ディレクトリ基準に修正
- **ウィンドウ管理**
  - ウィンドウサイズを1500×800に固定し、OSのタイリング/最大化を無効化
- **ビルドサイズ削減**
  - 未使用のPySide6モジュール（QtQml, Qt3D, QtWebEngine等）を除外
  - 除外しすぎて起動に失敗したため、必要なモジュールを再度含めるよう修正（v1.3.2）
  - opencv-python-headlessはmacOSで問題があったためopencv-pythonへ差し戻し

---

### Phase 18: uvへの移行・リポジトリ整合

- **パッケージ管理をuvへ移行**
  - `uv.lock`をコミットし、CI・開発環境で同一の依存構成を再現
  - `.python-version`（3.11）でPython処理系を固定（uvが自動取得するため事前インストール不要）
  - PyInstallerを`[dependency-groups]`のdevグループへ分離（配布物には含まれない）
  - `requirements.txt`を廃止し、依存定義を`pyproject.toml`へ一本化
  - GitHub Actionsを`astral-sh/setup-uv` + `uv sync --frozen` + `uv run`に変更
- **バージョン情報の一元化**
  - `pyproject.toml`のバージョンを1.2.0 → 1.3.2へ同期
  - specファイルの`CFBundleVersion`を`pyproject.toml`から読み込むよう変更（二重管理を解消）
  - CHANGELOG.mdにv1.3.0 - v1.3.2の記録を追加
- **ドキュメント・コードの整合**
  - README.md / BUILD_WINDOWS.md をuvベースの手順に刷新
  - PDF出力のdocstringの誤記（72dpi → 実装どおり300dpi）を修正
  - `order_corners()`内の未使用コード（重心・角度ソート）を削除

---

## ファイル構成

```
perspective-corrector/
├── .github/
│   └── workflows/
│       └── build.yml           # GitHub Actions設定（Windows/macOS同時ビルド）
├── perspective_corrector.py    # メインアプリケーション
├── pyproject.toml              # パッケージ設定・依存定義（バージョンの単一情報源）
├── uv.lock                     # 依存関係のロックファイル
├── .python-version             # 開発・CIで使用するPythonバージョン
├── perspective_corrector.spec  # PyInstallerビルド設定
├── icon.svg / icon.png         # アイコン原本
├── icon.ico / icon.icns        # プラットフォーム別アイコン
├── README.md                   # プロジェクト説明
├── BUILD_WINDOWS.md            # ビルド手順（Windows/macOS）
├── CHANGELOG.md                # 変更履歴
├── DEVELOPMENT_LOG.md          # 本ファイル
└── LICENSE                     # MITライセンス
```

---

## インストール・実行

```bash
# GitHubからインストール（uv）
uv tool install git+https://github.com/mashi727/perspective-corrector.git

# 実行
perspective-corrector

# ディレクトリ指定
perspective-corrector /path/to/image/directory

# 開発用セットアップ
git clone https://github.com/mashi727/perspective-corrector.git
cd perspective-corrector
uv sync --all-groups
uv run perspective-corrector
```
