import SwiftUI
import UIKit
import PhotosUI

struct ContentView: View {
    @Environment(AppModel.self) private var model

    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var showCamera = false
    @State private var isWorking = false
    @State private var statusMessage: String?
    @State private var alert: AlertItem?
    @State private var pdfURL: URL?
    @State private var showShare = false

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 12)]

    var body: some View {
        NavigationStack {
            Group {
                if model.items.isEmpty {
                    emptyState
                } else {
                    gallery
                }
            }
            .navigationTitle("台形補正")
            .toolbar { toolbarContent }
            .onChange(of: pickerItems) { _, newItems in
                Task { await loadPicked(newItems) }
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraPicker { image in
                    model.add([(image, "Camera \(model.items.count + 1)")])
                }
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showShare) {
                if let pdfURL {
                    ShareSheet(items: [pdfURL])
                }
            }
            .overlay { if isWorking { workingOverlay } }
            .alert(item: $alert) { item in
                Alert(title: Text(item.title), message: Text(item.message), dismissButton: .default(Text("OK")))
            }
        }
    }

    // MARK: - サブビュー

    private var emptyState: some View {
        ContentUnavailableView {
            Label("画像がありません", systemImage: "photo.on.rectangle.angled")
        } description: {
            Text("写真を追加して、台形歪みを補正しましょう。")
        } actions: {
            HStack {
                addPhotosButton
                cameraButton
            }
        }
    }

    private var gallery: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(model.items) { item in
                    NavigationLink {
                        EditorView(item: item)
                    } label: {
                        ThumbnailCell(item: item)
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button(role: .destructive) {
                            model.remove(item)
                        } label: {
                            Label("削除", systemImage: "trash")
                        }
                    }
                }
            }
            .padding()
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarTrailing) {
            addPhotosButton
            cameraButton
            if !model.items.isEmpty {
                exportMenu
            }
        }
    }

    private var addPhotosButton: some View {
        PhotosPicker(
            selection: $pickerItems,
            maxSelectionCount: 20,
            matching: .images
        ) {
            Image(systemName: "photo.badge.plus")
        }
    }

    private var cameraButton: some View {
        Button {
            showCamera = true
        } label: {
            Image(systemName: "camera")
        }
        .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
    }

    private var exportMenu: some View {
        Menu {
            Button {
                Task { await savePNGs() }
            } label: {
                Label("補正PNGを写真に保存", systemImage: "square.and.arrow.down")
            }
            Button {
                Task { await exportPDF() }
            } label: {
                Label("PDFを書き出し", systemImage: "doc.richtext")
            }
        } label: {
            Image(systemName: "square.and.arrow.up")
        }
    }

    private var workingOverlay: some View {
        ZStack {
            Color.black.opacity(0.3).ignoresSafeArea()
            VStack(spacing: 12) {
                ProgressView()
                if let statusMessage {
                    Text(statusMessage).font(.callout)
                }
            }
            .padding(24)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }

    // MARK: - アクション

    private func loadPicked(_ items: [PhotosPickerItem]) async {
        guard !items.isEmpty else { return }
        isWorking = true
        statusMessage = "画像を読み込み中…"
        defer {
            isWorking = false
            pickerItems = []
        }

        var loaded: [(UIImage, String)] = []
        for item in items {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                loaded.append((image, "Image \(model.items.count + loaded.count + 1)"))
            }
        }
        model.add(loaded)
    }

    private func savePNGs() async {
        isWorking = true
        statusMessage = "写真に保存中…"
        defer { isWorking = false }
        do {
            try await Exporter.saveCorrectedToPhotos(model.items, outputSize: model.outputSize)
            alert = AlertItem(title: "保存完了", message: "\(model.items.count)枚の補正画像を写真に保存しました。")
        } catch {
            alert = AlertItem(title: "保存に失敗しました", message: error.localizedDescription)
        }
    }

    private func exportPDF() async {
        isWorking = true
        statusMessage = "PDFを作成中…"
        defer { isWorking = false }
        do {
            let url = try Exporter.makePDF(model.items, outputSize: model.outputSize)
            pdfURL = url
            showShare = true
        } catch {
            alert = AlertItem(title: "PDF作成に失敗しました", message: error.localizedDescription)
        }
    }
}

// MARK: - サムネイル

struct ThumbnailCell: View {
    let item: EditableImage

    var body: some View {
        Image(uiImage: item.source)
            .resizable()
            .scaledToFill()
            .frame(height: 150)
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(alignment: .bottomLeading) {
                Text(item.name)
                    .font(.caption2)
                    .padding(4)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(6)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 12).stroke(.quaternary)
            }
    }
}

struct AlertItem: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

#Preview {
    ContentView()
        .environment(AppModel())
}
