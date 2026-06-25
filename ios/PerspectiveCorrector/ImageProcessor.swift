import UIKit
import CoreImage
import CoreImage.CIFilterBuiltins
import Vision

/// 台形補正・色補正・4隅自動検出を担う処理エンジン。
enum ImageProcessor {

    /// CIContext は生成コストが高いので使い回す。
    static let context = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - 4隅の自動検出 (Vision)

    /// Cannyベースのデスクトップ版に相当する四角形検出。
    /// 戻り値は正規化座標 [0,1]・左上原点の `Corners`。検出できなければ nil。
    static func detectRectangle(in image: UIImage) -> Corners? {
        guard let cg = image.cgImage else { return nil }

        let request = VNDetectRectanglesRequest()
        request.minimumAspectRatio = 0.2
        request.maximumAspectRatio = 1.0
        request.minimumSize = 0.2
        request.minimumConfidence = 0.5
        request.quadratureTolerance = 30
        request.maximumObservations = 1

        let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let obs = request.results?.first else { return nil }

        // Vision の座標は正規化・左下原点なので、左上原点へ y を反転する。
        func flip(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: 1.0 - p.y) }

        return Corners(
            topLeft: flip(obs.topLeft),
            topRight: flip(obs.topRight),
            bottomRight: flip(obs.bottomRight),
            bottomLeft: flip(obs.bottomLeft)
        )
    }

    // MARK: - 台形補正 + 色補正

    /// 指定した4隅で台形補正し、色補正を適用して `outputSize` ちょうどの画像を返す。
    static func correctedImage(
        for item: EditableImage,
        outputSize: CGSize
    ) -> UIImage? {
        correctPerspective(
            image: item.source,
            corners: item.corners,
            adjustments: item.adjustments,
            outputSize: outputSize
        )
    }

    static func correctPerspective(
        image: UIImage,
        corners: Corners,
        adjustments: ColorAdjustments,
        outputSize: CGSize
    ) -> UIImage? {
        guard let cg = image.cgImage else { return nil }
        let ci = CIImage(cgImage: cg)
        let w = ci.extent.width
        let h = ci.extent.height

        // 正規化・左上原点 → CoreImage のピクセル・左下原点へ。
        func toCI(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * w, y: (1.0 - p.y) * h)
        }

        let filter = CIFilter.perspectiveCorrection()
        filter.inputImage = ci
        filter.topLeft = toCI(corners.topLeft)
        filter.topRight = toCI(corners.topRight)
        filter.bottomRight = toCI(corners.bottomRight)
        filter.bottomLeft = toCI(corners.bottomLeft)

        guard var out = filter.outputImage else { return nil }

        // 色補正。
        out = applyColorAdjustments(out, adjustments)

        // extent の原点を 0 に寄せてから outputSize ちょうどへスケール。
        out = out.transformed(by: CGAffineTransform(
            translationX: -out.extent.origin.x,
            y: -out.extent.origin.y
        ))
        guard out.extent.width > 0, out.extent.height > 0 else { return nil }

        let sx = outputSize.width / out.extent.width
        let sy = outputSize.height / out.extent.height
        out = out.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        let rect = CGRect(origin: .zero, size: outputSize)
        guard let cgOut = context.createCGImage(out, from: rect) else { return nil }
        return UIImage(cgImage: cgOut)
    }

    // MARK: - 色補正

    static func applyColorAdjustments(_ image: CIImage, _ adj: ColorAdjustments) -> CIImage {
        var img = image

        if adj.autoWhiteBalance {
            img = autoWhiteBalanced(img)
        }

        if adj.brightness != 0 || adj.contrast != 1 || adj.saturation != 1 {
            let cc = CIFilter.colorControls()
            cc.inputImage = img
            cc.brightness = Float(adj.brightness)
            cc.contrast = Float(adj.contrast)
            cc.saturation = Float(adj.saturation)
            img = cc.outputImage ?? img
        }

        if adj.temperature != 0 {
            let tt = CIFilter.temperatureAndTint()
            tt.inputImage = img
            // 6500K を基準に ±3000K 程度動かす。
            tt.neutral = CIVector(x: 6500, y: 0)
            tt.targetNeutral = CIVector(x: 6500 + adj.temperature * 3000, y: 0)
            img = tt.outputImage ?? img
        }

        return img
    }

    /// グレーワールド仮説による簡易オートホワイトバランス。
    private static func autoWhiteBalanced(_ image: CIImage) -> CIImage {
        let avg = CIFilter.areaAverage()
        avg.inputImage = image
        avg.extent = image.extent
        guard let avgOut = avg.outputImage else { return image }

        var bitmap = [UInt8](repeating: 0, count: 4)
        context.render(
            avgOut,
            toBitmap: &bitmap,
            rowBytes: 4,
            bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
            format: .RGBA8,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        let r = Double(bitmap[0]), g = Double(bitmap[1]), b = Double(bitmap[2])
        guard r > 0, g > 0, b > 0 else { return image }

        let gray = (r + g + b) / 3.0
        let m = CIFilter.colorMatrix()
        m.inputImage = image
        m.rVector = CIVector(x: CGFloat(gray / r), y: 0, z: 0, w: 0)
        m.gVector = CIVector(x: 0, y: CGFloat(gray / g), z: 0, w: 0)
        m.bVector = CIVector(x: 0, y: 0, z: CGFloat(gray / b), w: 0)
        m.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
        return m.outputImage ?? image
    }
}

// MARK: - UIImage の向き正規化

extension UIImage {
    /// imageOrientation を .up に焼き込んだ画像を返す（座標計算を単純化するため）。
    func normalizedUp() -> UIImage {
        if imageOrientation == .up { return self }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
