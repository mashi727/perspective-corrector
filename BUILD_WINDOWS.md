# Windows用EXEファイルのビルド手順

## 概要

`perspective_corrector.py`をWindows用の単一実行ファイル（.exe）にビルドする手順です。

## 必要な環境

- Windows 10/11
- [uv](https://docs.astral.sh/uv/)（Pythonは uv が自動取得するため、事前インストール不要）

uvのインストール（PowerShell）:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 手順

### 1. リポジトリを取得

```bash
git clone https://github.com/mashi727/perspective-corrector.git
cd perspective-corrector
```

ビルドには最低限、以下のファイルが必要です:

- `perspective_corrector.py`
- `perspective_corrector.spec`
- `pyproject.toml` / `uv.lock` / `.python-version`
- `icon.ico`（Windows用アイコン）

### 2. 依存パッケージのインストール

```bash
uv sync --all-groups
```

`uv.lock` に固定されたバージョンで、実行時依存（PySide6, OpenCV, NumPy, Pillow, pillow-heif）と
ビルド用依存（PyInstaller）が `.venv` にインストールされます。
CIと完全に同じ依存構成を再現したい場合は `--frozen` を付けてください。

### 3. EXEファイルのビルド

```bash
uv run pyinstaller perspective_corrector.spec --noconfirm
```

specファイルはプラットフォームを判定し、Windowsではone-fileモードの`.exe`、
macOSではone-dirモードの`.app`バンドルを生成します。

### 4. 生成されたEXEファイルの場所

ビルドが成功すると、以下の場所にEXEファイルが生成されます:

```
dist/PerspectiveCorrector.exe
```

## macOS版のビルド

同一のspecファイルでビルドできます。

```bash
uv sync --all-groups
uv run pyinstaller perspective_corrector.spec --noconfirm
# dist/PerspectiveCorrector.app が生成される
```

DMGを作成する場合:

```bash
mkdir -p dmg_contents && cp -r dist/PerspectiveCorrector.app dmg_contents/
hdiutil create -volname "PerspectiveCorrector" -srcfolder dmg_contents -ov -format UDZO dist/PerspectiveCorrector.dmg
```

## アイコンについて

specファイルはプラットフォームに応じて`icon.ico`（Windows）/`icon.icns`（macOS）を自動選択します。
差し替える場合はリポジトリ直下の同名ファイルを置き換えてください（specの編集は不要）。

`.app`バンドルの`CFBundleVersion`は`pyproject.toml`の`version`から自動的に読み込まれるため、
バージョン更新は`pyproject.toml`のみを編集してください。

## 注意事項

- 初回起動時は、EXEファイルの展開処理のため数秒かかることがあります
- ウイルス対策ソフトが誤検知する場合は、除外設定を行ってください
- ビルド時に`build/`と`dist/`ディレクトリが作成されます
- タグ`v*`をpushすると、GitHub Actions（`.github/workflows/build.yml`）がWindows/macOS版を自動ビルドしてReleaseに添付します

## HEIC/HEIF対応について

WindowsでHEIC/HEIF画像（iPhoneで撮影した写真など）を読み込むには、`pillow-heif`パッケージが必要です。

- `pillow-heif`は`pyproject.toml`の依存に含まれているため、`uv sync`で自動的に導入されます
- `perspective_corrector.spec`には`pillow-heif`の依存関係を自動収集する設定が含まれています
- ビルド時に`collect_all('pillow_heif')`により必要なバイナリが自動的にバンドルされます

## トラブルシューティング

### DLLが見つからないエラー

Visual C++ 再頒布可能パッケージをインストールしてください:
https://learn.microsoft.com/ja-jp/cpp/windows/latest-supported-vc-redist

### モジュールが見つからないエラー

`perspective_corrector.spec`の`hiddenimports`に不足しているモジュールを追加してください。

### HEIC/HEIFファイルが表示できない

HEIC対応は以下の優先順位で試行されます:

1. **pillow-heif** (推奨)
2. **ImageMagick** (フォールバック)

#### pillow-heifが動作しない場合

コマンドプロンプトで以下を実行して確認:

```bash
uv run python -c "import pillow_heif; pillow_heif.register_heif_opener(); print('OK')"
```

エラーが出る場合は、キャッシュを使わずに環境を作り直す:

```bash
uv cache clean pillow-heif
uv sync --all-groups --reinstall-package pillow-heif
```

#### ImageMagickをフォールバックとして使用

pillow-heifが動作しない場合、ImageMagickをインストールすることでHEIC対応が可能です:

1. [ImageMagick公式サイト](https://imagemagick.org/script/download.php#windows)からWindows版をダウンロード
2. インストール時に「Add application directory to your system path」にチェック
3. HEIC対応のために「Install HEIC/HEIF」オプションにチェック

インストール確認:

```bash
magick -version
```

#### デバッグ用ビルド

HEIC関連のエラーを確認するには、コンソール付きでビルド:

```bash
uv run pyinstaller --onefile --console --name PerspectiveCorrector_debug perspective_corrector.py
```

コンソールにエラーメッセージが表示されます。
