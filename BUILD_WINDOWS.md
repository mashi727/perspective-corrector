# Windows用EXEファイルのビルド手順

## 概要

`perspective_corrector.py`をWindows用の単一実行ファイル（.exe）にビルドする手順です。

## 必要な環境

- Windows 10/11
- Python 3.9以上

## 手順

### 1. 必要なファイルをWindowsにコピー

以下のファイルをWindows環境にコピーしてください:

- `perspective_corrector.py`
- `perspective_corrector.spec`

### 2. 依存パッケージのインストール

コマンドプロンプトまたはPowerShellで以下を実行:

```bash
pip install pyinstaller PySide6 opencv-python numpy pillow pillow-heif
```

### 3. EXEファイルのビルド

#### 方法A: specファイルを使用（推奨）

```bash
pyinstaller perspective_corrector.spec
```

#### 方法B: コマンドラインオプションで直接ビルド

```bash
pyinstaller --onefile --noconsole --name PerspectiveCorrector perspective_corrector.py
```

### 4. 生成されたEXEファイルの場所

ビルドが成功すると、以下の場所にEXEファイルが生成されます:

```
dist/PerspectiveCorrector.exe
```

## オプション

### アイコンを設定する場合

1. `.ico`形式のアイコンファイルを用意
2. 以下のコマンドでビルド:

```bash
pyinstaller --onefile --noconsole --icon=app.ico --name PerspectiveCorrector perspective_corrector.py
```

または`perspective_corrector.spec`の`icon=None`を`icon='app.ico'`に変更してビルド。

## 注意事項

- 初回起動時は、EXEファイルの展開処理のため数秒かかることがあります
- ウイルス対策ソフトが誤検知する場合は、除外設定を行ってください
- ビルド時に`build/`と`dist/`ディレクトリが作成されます

## HEIC/HEIF対応について

WindowsでHEIC/HEIF画像（iPhoneで撮影した写真など）を読み込むには、`pillow-heif`パッケージが必要です。

- `pillow-heif`は依存パッケージに含まれているため、上記手順でインストールすれば自動的に対応されます
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
python -c "import pillow_heif; pillow_heif.register_heif_opener(); print('OK')"
```

エラーが出る場合は、pillow-heifを再インストール:

```bash
pip uninstall pillow-heif
pip install pillow-heif --no-cache-dir
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
pyinstaller --onefile --console --name PerspectiveCorrector_debug perspective_corrector.py
```

コンソールにエラーメッセージが表示されます。
