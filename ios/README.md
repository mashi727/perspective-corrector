# Perspective Corrector for iOS / iPadOS

デスクトップ版（PySide6 + OpenCV）を、iPhone / iPad で動くネイティブアプリとして
再実装した SwiftUI プロジェクトです。台形補正・色補正をすべて端末内（Vision /
Core Image）で処理し、画像を外部に送信しません。

## 機能

| デスクトップ版 | iOS 版の対応 |
|---|---|
| 4隅自動検出（Canny） | Vision `VNDetectRectanglesRequest` |
| 手動4隅指定＋ドラッグ微調整 | ドラッグ可能なハンドル |
| 拡大鏡表示 | ドラッグ中に表示されるルーペ（十字線つき） |
| 台形補正 | Core Image `CIPerspectiveCorrection` |
| 1920×1080 PNG 出力 | 写真ライブラリへ PNG 保存 |
| 300dpi A4横 PDF 出力 | A4横・複数ページ PDF を共有シートで書き出し |
| 色補正（WB / コントラスト / 明るさ / 彩度） | Core Image フィルタ |
| HEIC 対応 | iOS ネイティブで自動対応 |
| 一括処理 | 複数枚をまとめて保存 / PDF 化 |

## 必要なもの

- **Mac**（Apple Silicon / Intel）
- **Xcode 16 以降**（このプロジェクトは新しい "synchronized groups" 形式を使用）
- iPhone / iPad（iOS / iPadOS **17.0 以降**）
- 実機で動かす場合：無料の Apple ID（App Store 配布には有料の Apple Developer
  Program が必要）

## ビルドして実機で動かす手順

1. この `ios/` フォルダの `PerspectiveCorrector.xcodeproj` を Xcode で開く。
2. プロジェクト設定 → **Signing & Capabilities** で自分の Apple ID（Team）を選択。
   - `PRODUCT_BUNDLE_IDENTIFIER`（既定 `com.mashi727.PerspectiveCorrector`）が
     他と衝突する場合は、ユニークな値に変更してください。
3. iPhone / iPad を Mac に接続し、上部のデバイス選択で実機を選ぶ。
4. **⌘R** で実行。初回は端末側で
   「設定 → 一般 → VPN とデバイス管理」から開発元を信頼する必要があります。

シミュレータでも動作しますが、カメラ撮影は実機のみです。

## 使い方

1. 右上の「写真を追加」またはカメラで画像を取り込む。
2. 一覧から画像をタップしてエディタを開く。
3. 「自動認識」で4隅を検出、またはハンドルをドラッグして手動調整
   （ドラッグ中はルーペで精密に合わせられます）。
4. 「プレビュー」タブで補正結果を確認。必要なら右上のスライダーで色補正。
5. 一覧画面の共有ボタンから
   - **補正PNGを写真に保存**、または
   - **PDFを書き出し**（共有シートで保存 / 送信）。

## 構成

```
ios/
├── PerspectiveCorrector.xcodeproj/      # Xcode プロジェクト
└── PerspectiveCorrector/
    ├── PerspectiveCorrectorApp.swift    # アプリのエントリポイント
    ├── Models.swift                     # データモデル (AppModel / EditableImage 等)
    ├── ImageProcessor.swift             # 台形補正・色補正・4隅検出
    ├── Exporter.swift                   # PNG 保存 / PDF 生成
    ├── ContentView.swift                # 一覧・取り込み・書き出し
    ├── EditorView.swift                 # 4隅編集 + ルーペ + 色補正パネル
    ├── CameraPicker.swift               # カメラ / 共有シート
    └── Assets.xcassets/                 # アクセントカラー・アプリアイコン
```

## アプリアイコンについて

`Assets.xcassets/AppIcon.appiconset` は中身が空のプレースホルダです。
リポジトリ直下の `icon.svg` / `icon.png` を 1024×1024 の PNG に書き出し、
Xcode の AppIcon にドラッグすると設定できます。未設定でもビルドは可能です。

## メモ

- 出力解像度は `AppModel.outputSize`（既定 1920×1080）で変更できます。
- オートホワイトバランスはグレーワールド仮説による簡易実装です。
- 古い Xcode（15 以前）で `project.pbxproj` を開けない場合は、新規 iOS App
  プロジェクトを作成し、`PerspectiveCorrector/` 内の `.swift` と
  `Assets.xcassets` をドラッグして追加すれば同じ構成になります。
