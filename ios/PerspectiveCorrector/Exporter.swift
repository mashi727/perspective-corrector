import UIKit
import Photos

/// 補正済み画像の書き出し（PNG を写真へ保存 / PDF 生成）。
enum Exporter {

    enum ExportError: LocalizedError {
        case processingFailed
        case photoPermissionDenied

        var errorDescription: String? {
            switch self {
            case .processingFailed: return "画像の補正処理に失敗しました。"
            case .photoPermissionDenied: return "写真ライブラリへの保存が許可されていません。設定アプリで許可してください。"
            }
        }
    }

    // MARK: - PNG を写真ライブラリへ保存

    static func saveCorrectedToPhotos(
        _ items: [EditableImage],
        outputSize: CGSize
    ) async throws {
        let status = await requestAddPermission()
        guard status == .authorized || status == .limited else {
            throw ExportError.photoPermissionDenied
        }

        var pngs: [Data] = []
        for item in items {
            guard
                let corrected = ImageProcessor.correctedImage(for: item, outputSize: outputSize),
                let data = corrected.pngData()
            else {
                throw ExportError.processingFailed
            }
            pngs.append(data)
        }

        try await PHPhotoLibrary.shared().performChanges {
            for data in pngs {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: data, options: nil)
            }
        }
    }

    private static func requestAddPermission() async -> PHAuthorizationStatus {
        await withCheckedContinuation { continuation in
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                continuation.resume(returning: status)
            }
        }
    }

    // MARK: - PDF 生成 (A4 横 / 複数ページ)

    /// 補正済み画像を A4横の複数ページ PDF にまとめ、一時ファイルの URL を返す。
    static func makePDF(
        _ items: [EditableImage],
        outputSize: CGSize
    ) throws -> URL {
        // A4 横 (ポイント単位 / 72dpi)。埋め込む画像自体は高解像度なので印刷品質を保てる。
        let pageRect = CGRect(x: 0, y: 0, width: 842, height: 595)
        let renderer = UIGraphicsPDFRenderer(bounds: pageRect)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("PerspectiveCorrector.pdf")

        var corrected: [UIImage] = []
        for item in items {
            if let img = ImageProcessor.correctedImage(for: item, outputSize: outputSize) {
                corrected.append(img)
            }
        }
        guard !corrected.isEmpty else { throw ExportError.processingFailed }

        try renderer.writePDF(to: url) { ctx in
            let inset = pageRect.insetBy(dx: 24, dy: 24)
            for img in corrected {
                ctx.beginPage()
                img.draw(in: aspectFitRect(imageSize: img.size, in: inset))
            }
        }
        return url
    }

    /// アスペクト比を保ったまま矩形内に最大配置する矩形を返す。
    private static func aspectFitRect(imageSize: CGSize, in bounds: CGRect) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return bounds }
        let imageAspect = imageSize.width / imageSize.height
        let boundsAspect = bounds.width / bounds.height
        if imageAspect > boundsAspect {
            let h = bounds.width / imageAspect
            return CGRect(x: bounds.minX, y: bounds.midY - h / 2, width: bounds.width, height: h)
        } else {
            let w = bounds.height * imageAspect
            return CGRect(x: bounds.midX - w / 2, y: bounds.minY, width: w, height: bounds.height)
        }
    }
}
