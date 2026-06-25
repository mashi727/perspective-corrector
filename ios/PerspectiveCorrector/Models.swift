import SwiftUI
import UIKit
import Observation

/// 台形補正の4隅。値は正規化座標 [0,1]、原点は左上 (SwiftUI / UIKit と同じ向き)。
struct Corners: Equatable {
    var topLeft: CGPoint
    var topRight: CGPoint
    var bottomRight: CGPoint
    var bottomLeft: CGPoint

    /// 画像中央に内接する初期枠。
    static let `default` = Corners(
        topLeft: CGPoint(x: 0.10, y: 0.10),
        topRight: CGPoint(x: 0.90, y: 0.10),
        bottomRight: CGPoint(x: 0.90, y: 0.90),
        bottomLeft: CGPoint(x: 0.10, y: 0.90)
    )

    enum Handle: CaseIterable {
        case topLeft, topRight, bottomRight, bottomLeft
    }

    subscript(_ handle: Handle) -> CGPoint {
        get {
            switch handle {
            case .topLeft: return topLeft
            case .topRight: return topRight
            case .bottomRight: return bottomRight
            case .bottomLeft: return bottomLeft
            }
        }
        set {
            switch handle {
            case .topLeft: topLeft = newValue
            case .topRight: topRight = newValue
            case .bottomRight: bottomRight = newValue
            case .bottomLeft: bottomLeft = newValue
            }
        }
    }
}

/// 色補正パラメータ。デスクトップ版の色補正機能に対応。
struct ColorAdjustments: Equatable {
    var autoWhiteBalance: Bool = false
    var brightness: Double = 0.0    // -1.0 ... 1.0  (0 = 変化なし)
    var contrast: Double = 1.0      //  0.5 ... 1.5  (1 = 変化なし)
    var saturation: Double = 1.0    //  0.0 ... 2.0  (1 = 変化なし)
    var temperature: Double = 0.0   // -1.0 ... 1.0  (- 寒色 / + 暖色, 0 = 変化なし)

    static let none = ColorAdjustments()

    var isIdentity: Bool { self == .none }
}

/// 1枚の編集対象画像。隅と色補正を保持する参照型。
@Observable
final class EditableImage: Identifiable {
    let id = UUID()
    let source: UIImage          // 向きを .up に正規化済みの元画像
    var name: String
    var corners: Corners = .default
    var adjustments: ColorAdjustments = .none
    var hasAutoDetected = false

    init(source: UIImage, name: String) {
        self.source = source.normalizedUp()
        self.name = name
    }
}

/// アプリ全体の状態。
@Observable
final class AppModel {
    var items: [EditableImage] = []

    /// 出力解像度（デスクトップ版の既定と同じ 1920x1080）。
    var outputSize = CGSize(width: 1920, height: 1080)

    func add(_ images: [(UIImage, String)]) {
        for (img, name) in images {
            items.append(EditableImage(source: img, name: name))
        }
    }

    func remove(_ item: EditableImage) {
        items.removeAll { $0.id == item.id }
    }
}
