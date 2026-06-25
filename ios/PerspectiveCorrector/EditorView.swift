import SwiftUI
import UIKit

struct EditorView: View {
    @Bindable var item: EditableImage
    @Environment(AppModel.self) private var model

    enum Mode: String, CaseIterable, Identifiable {
        case edit = "編集"
        case preview = "プレビュー"
        var id: String { rawValue }
    }

    @State private var mode: Mode = .edit
    @State private var activeHandle: Corners.Handle?
    @State private var fingerLocation: CGPoint = .zero
    @State private var showColorPanel = false
    @State private var previewImage: UIImage?
    @State private var isDetecting = false

    private let editorSpace = "editor"

    var body: some View {
        VStack(spacing: 0) {
            Picker("表示", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding()

            GeometryReader { geo in
                if mode == .edit {
                    editingCanvas(in: geo.size)
                } else {
                    previewCanvas
                }
            }

            controls
        }
        .navigationTitle(item.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showColorPanel = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showColorPanel) {
            ColorAdjustPanel(adjustments: $item.adjustments)
                .presentationDetents([.medium, .large])
        }
        .task(id: previewToken) {
            if mode == .preview { await updatePreview() }
        }
        .onChange(of: mode) { _, newMode in
            if newMode == .preview { Task { await updatePreview() } }
        }
    }

    // MARK: - 編集モード（4隅 + ルーペ）

    private func editingCanvas(in size: CGSize) -> some View {
        let rect = fittedRect(imageSize: item.source.size, in: size)
        return ZStack {
            Image(uiImage: item.source)
                .resizable()
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            // 4隅を結ぶ枠線
            quadPath(in: rect)
                .stroke(Color.accentColor, lineWidth: 2)

            // ハンドル
            ForEach(Corners.Handle.allCases, id: \.self) { handle in
                handleView(handle, in: rect)
            }

            // ルーペ
            if let active = activeHandle {
                magnifier(for: active, rect: rect, container: size)
            }
        }
        .frame(width: size.width, height: size.height)
        .coordinateSpace(.named(editorSpace))
        .clipped()
    }

    private func quadPath(in rect: CGRect) -> Path {
        let c = item.corners
        let pts = [c.topLeft, c.topRight, c.bottomRight, c.bottomLeft].map { point(for: $0, in: rect) }
        var path = Path()
        path.move(to: pts[0])
        for p in pts.dropFirst() { path.addLine(to: p) }
        path.closeSubpath()
        return path
    }

    private func handleView(_ handle: Corners.Handle, in rect: CGRect) -> some View {
        let pos = point(for: item.corners[handle], in: rect)
        let isActive = activeHandle == handle
        return Circle()
            .fill(Color.accentColor.opacity(0.25))
            .frame(width: 28, height: 28)
            .overlay(Circle().stroke(Color.accentColor, lineWidth: 2))
            .overlay(Circle().fill(Color.white).frame(width: 6, height: 6))
            .scaleEffect(isActive ? 1.3 : 1.0)
            .position(pos)
            .gesture(
                DragGesture(coordinateSpace: .named(editorSpace))
                    .onChanged { value in
                        activeHandle = handle
                        fingerLocation = value.location
                        let nx = ((value.location.x - rect.minX) / rect.width).clamped(to: 0...1)
                        let ny = ((value.location.y - rect.minY) / rect.height).clamped(to: 0...1)
                        item.corners[handle] = CGPoint(x: nx, y: ny)
                    }
                    .onEnded { _ in activeHandle = nil }
            )
    }

    private func magnifier(for handle: Corners.Handle, rect: CGRect, container: CGSize) -> some View {
        let diameter: CGFloat = 130
        let corner = item.corners[handle]
        let dispScale = rect.width / item.source.size.width
        // 指の上（上端付近では下）に表示し、横方向は画面内に収める。
        let y = fingerLocation.y < 160 ? fingerLocation.y + 100 : fingerLocation.y - 100
        let upper = max(diameter / 2, container.width - diameter / 2)
        let x = fingerLocation.x.clamped(to: (diameter / 2)...upper)
        return MagnifierView(image: item.source, corner: corner, dispScale: dispScale, diameter: diameter)
            .position(x: x, y: y)
            .allowsHitTesting(false)
    }

    // MARK: - プレビューモード

    private var previewCanvas: some View {
        ZStack {
            Color(.systemGroupedBackground)
            if let previewImage {
                Image(uiImage: previewImage)
                    .resizable()
                    .scaledToFit()
                    .padding()
            } else {
                ProgressView()
            }
        }
    }

    // MARK: - 下部コントロール

    private var controls: some View {
        HStack {
            Button {
                detect()
            } label: {
                Label("自動認識", systemImage: "viewfinder")
            }
            .buttonStyle(.borderedProminent)
            .disabled(isDetecting)

            Spacer()

            Button {
                item.corners = .default
            } label: {
                Label("枠をリセット", systemImage: "arrow.counterclockwise")
            }
        }
        .padding()
    }

    // MARK: - ロジック

    private func detect() {
        isDetecting = true
        DispatchQueue.global(qos: .userInitiated).async {
            let detected = ImageProcessor.detectRectangle(in: item.source)
            DispatchQueue.main.async {
                if let detected {
                    item.corners = detected
                    item.hasAutoDetected = true
                }
                isDetecting = false
            }
        }
    }

    private func updatePreview() async {
        let source = item.source
        let corners = item.corners
        let adjustments = item.adjustments
        let size = model.outputSize
        let img = await Task.detached(priority: .userInitiated) {
            ImageProcessor.correctPerspective(
                image: source,
                corners: corners,
                adjustments: adjustments,
                outputSize: size
            )
        }.value
        previewImage = img
    }

    /// プレビュー再計算のトリガー。
    private var previewToken: String {
        let c = item.corners
        let a = item.adjustments
        return "\(c.topLeft)\(c.topRight)\(c.bottomRight)\(c.bottomLeft)\(a.autoWhiteBalance)\(a.brightness)\(a.contrast)\(a.saturation)\(a.temperature)"
    }

    // MARK: - 座標変換

    private func fittedRect(imageSize: CGSize, in container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(container.width / imageSize.width, container.height / imageSize.height)
        let w = imageSize.width * scale
        let h = imageSize.height * scale
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2, width: w, height: h)
    }

    private func point(for normalized: CGPoint, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + normalized.x * rect.width, y: rect.minY + normalized.y * rect.height)
    }
}

// MARK: - ルーペ

struct MagnifierView: View {
    let image: UIImage
    let corner: CGPoint      // 正規化座標 [0,1]
    let dispScale: CGFloat   // 表示スケール (表示幅 / 画像幅)
    var magnify: CGFloat = 2.5
    var diameter: CGFloat = 130

    var body: some View {
        let imgSize = image.size
        let cornerPixel = CGPoint(x: corner.x * imgSize.width, y: corner.y * imgSize.height)
        let s = dispScale * magnify

        ZStack {
            Canvas { ctx, size in
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                ctx.translateBy(x: center.x, y: center.y)
                ctx.scaleBy(x: s, y: s)
                ctx.translateBy(x: -cornerPixel.x, y: -cornerPixel.y)
                let resolved = ctx.resolve(Image(uiImage: image))
                ctx.draw(resolved, in: CGRect(origin: .zero, size: imgSize))
            }
            // 十字線
            Path { p in
                let c = diameter / 2
                p.move(to: CGPoint(x: c - 14, y: c)); p.addLine(to: CGPoint(x: c + 14, y: c))
                p.move(to: CGPoint(x: c, y: c - 14)); p.addLine(to: CGPoint(x: c, y: c + 14))
            }
            .stroke(Color.red.opacity(0.9), lineWidth: 1)
            Circle().fill(Color.clear).frame(width: 6, height: 6)
                .overlay(Circle().stroke(Color.red, lineWidth: 1))
        }
        .frame(width: diameter, height: diameter)
        .background(Color(.systemBackground))
        .clipShape(Circle())
        .overlay(Circle().stroke(Color.accentColor, lineWidth: 3))
        .shadow(radius: 6)
    }
}

// MARK: - 色補正パネル

struct ColorAdjustPanel: View {
    @Binding var adjustments: ColorAdjustments
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("ホワイトバランス") {
                    Toggle("自動補正", isOn: $adjustments.autoWhiteBalance)
                    sliderRow("色温度", value: $adjustments.temperature, range: -1...1, neutral: 0)
                }
                Section("画質") {
                    sliderRow("明るさ", value: $adjustments.brightness, range: -0.5...0.5, neutral: 0)
                    sliderRow("コントラスト", value: $adjustments.contrast, range: 0.5...1.5, neutral: 1)
                    sliderRow("彩度", value: $adjustments.saturation, range: 0...2, neutral: 1)
                }
                Section {
                    Button("リセット", role: .destructive) {
                        adjustments = .none
                    }
                }
            }
            .navigationTitle("色補正")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完了") { dismiss() }
                }
            }
        }
    }

    private func sliderRow(_ title: String, value: Binding<Double>, range: ClosedRange<Double>, neutral: Double) -> some View {
        VStack(alignment: .leading) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: value, in: range) {
                Text(title)
            } minimumValueLabel: {
                Image(systemName: "minus")
            } maximumValueLabel: {
                Image(systemName: "plus")
            }
        }
    }
}

// MARK: - ユーティリティ

extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
