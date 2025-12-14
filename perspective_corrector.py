#!/usr/bin/env python3
"""
プレゼン写真 台形補正アプリケーション

機能:
- 左側TreeViewでディレクトリ内の画像ファイルを選択
- 右側で4隅をクリックして座標指定
- 座標はJSONファイルに自動保存
- 一括処理ボタンでOpenCVによる台形補正を実行
"""

import sys
import json
import subprocess
import tempfile
import platform
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QProgressDialog, QSplitter, QStatusBar, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QSlider, QFileDialog, QMenuBar, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QBrush, QAction, QKeySequence


def convert_heic_to_temp_jpeg(heic_path: str) -> str:
    """HEICファイルを一時的なJPEGに変換"""
    # pillow-heifが使える場合はそれを使用
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        img = Image.open(heic_path)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG', quality=95)
        return temp_file.name
    except ImportError:
        pass

    # macOSの場合はsipsコマンドを使用
    if platform.system() == 'Darwin':
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        try:
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', heic_path, '--out', temp_path],
                check=True, capture_output=True
            )
            return temp_path
        except subprocess.CalledProcessError:
            pass

    return None


def perspective_transform_cv(input_path: str, corners: list, output_path: str,
                              output_size: tuple = (1920, 1080)) -> bool:
    """
    OpenCVによる台形補正

    Args:
        input_path: 入力画像パス
        corners: 4隅の座標 [(x,y), ...] 左上、右上、右下、左下の順
        output_path: 出力画像パス
        output_size: 出力サイズ (width, height)

    Returns:
        成功時True
    """
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False

        # 入力座標（左上、右上、右下、左下）
        src = np.float32(corners)

        # 出力座標（矩形）
        w, h = output_size
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 変換行列を計算
        matrix = cv2.getPerspectiveTransform(src, dst)

        # 台形補正を適用
        result = cv2.warpPerspective(img, matrix, output_size)

        # 拡張子に応じて保存
        output_path_str = str(output_path)
        if output_path_str.lower().endswith('.png'):
            cv2.imwrite(output_path_str, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(output_path_str, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"Perspective transform error: {e}")
        return False


def auto_detect_corners(image_path: str, settings: dict = None) -> list:
    """
    画像からプレゼンテーション領域の4隅を自動検出

    Args:
        image_path: 画像ファイルパス
        settings: 検出パラメータ辞書
            - canny_low: Cannyエッジ検出の低閾値 (デフォルト: 50)
            - canny_high: Cannyエッジ検出の高閾値 (デフォルト: 150)
            - blur_size: ガウシアンブラーのカーネルサイズ (デフォルト: 5)
            - approx_epsilon: 輪郭近似の精度係数 (デフォルト: 0.02)
            - min_area_ratio: 最小面積比率 (デフォルト: 0.05)

    Returns:
        [(x, y), ...] 左上、右上、右下、左下の順。検出失敗時はNone
    """
    # デフォルト設定
    if settings is None:
        settings = {}
    canny_low = settings.get('canny_low', 50)
    canny_high = settings.get('canny_high', 150)
    blur_size = settings.get('blur_size', 5)
    approx_epsilon = settings.get('approx_epsilon', 0.02)
    min_area_ratio = settings.get('min_area_ratio', 0.05)

    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    # 処理用にリサイズ（大きい画像の処理を高速化）
    max_dim = 1000
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    if scale < 1.0:
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img
        scale = 1.0

    # グレースケール変換
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ノイズ除去（カーネルサイズは奇数である必要がある）
    blur_k = blur_size if blur_size % 2 == 1 else blur_size + 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # エッジ検出
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # モルフォロジー処理でエッジを強化
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 面積でソート（大きい順）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 四角形を探す
    for contour in contours[:10]:  # 上位10個をチェック
        # 輪郭の周囲長
        peri = cv2.arcLength(contour, True)
        # 輪郭を近似
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)

        # 4点の多角形で、十分な面積がある場合
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = img_resized.shape[0] * img_resized.shape[1]

            # 指定比率以上の面積がある四角形
            if area > img_area * min_area_ratio:
                # 4点を取得
                pts = approx.reshape(4, 2)

                # 4点を左上、右上、右下、左下の順に並び替え
                ordered = order_corners(pts)

                # 元のスケールに戻す
                ordered = [(int(x / scale), int(y / scale)) for x, y in ordered]

                return ordered

    return None


def order_corners(pts):
    """
    4点を左上、右上、右下、左下の順に並び替え
    """
    # 重心を計算
    center = pts.mean(axis=0)

    # 各点を角度でソート
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    # 左上から時計回りに並び替え
    # 最も左上の点（x+yが最小）を見つける
    sums = sorted_pts.sum(axis=1)
    top_left_idx = np.argmin(sums)

    # 左上から始まるように回転
    ordered = np.roll(sorted_pts, -top_left_idx, axis=0)

    # y座標が小さい2点が上、大きい2点が下
    # x座標が小さい方が左
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]      # 左上: x+yが最小
    rect[2] = pts[np.argmax(s)]      # 右下: x+yが最大
    rect[1] = pts[np.argmin(diff)]   # 右上: x-yが最小
    rect[3] = pts[np.argmax(diff)]   # 左下: x-yが最大

    return [(int(x), int(y)) for x, y in rect]


class DetectionSettingsDialog(QDialog):
    """自動認識パラメータ設定ダイアログ（ダイアログ内プレビュー）"""

    def __init__(self, parent=None, settings=None, image_path=None):
        super().__init__(parent)
        self.setWindowTitle("自動認識設定")
        self.setMinimumSize(700, 500)
        self.main_app = parent
        self.image_path = image_path
        self.detected_corners = None  # 検出されたコーナー

        # 現在の設定（デフォルト値）
        self.settings = settings or {}

        layout = QHBoxLayout(self)

        # === 左側: パラメータ設定 ===
        left_panel = QVBoxLayout()

        # フォームレイアウト
        form = QFormLayout()

        # Canny低閾値
        self.canny_low = QSpinBox()
        self.canny_low.setRange(1, 255)
        self.canny_low.setValue(self.settings.get('canny_low', 50))
        self.canny_low.setToolTip("低いほど弱いエッジも検出（推奨: 30-80）")
        self.canny_low.valueChanged.connect(self.on_value_changed)
        form.addRow("Canny低閾値:", self.canny_low)

        # Canny高閾値
        self.canny_high = QSpinBox()
        self.canny_high.setRange(1, 255)
        self.canny_high.setValue(self.settings.get('canny_high', 150))
        self.canny_high.setToolTip("低いほど多くのエッジを検出（推奨: 100-200）")
        self.canny_high.valueChanged.connect(self.on_value_changed)
        form.addRow("Canny高閾値:", self.canny_high)

        # ブラーサイズ
        self.blur_size = QSpinBox()
        self.blur_size.setRange(1, 31)
        self.blur_size.setSingleStep(2)
        self.blur_size.setValue(self.settings.get('blur_size', 5))
        self.blur_size.setToolTip("大きいほどノイズ除去が強い（奇数、推奨: 3-9）")
        self.blur_size.valueChanged.connect(self.on_value_changed)
        form.addRow("ブラー強度:", self.blur_size)

        # 近似精度
        self.approx_epsilon = QDoubleSpinBox()
        self.approx_epsilon.setRange(0.001, 0.1)
        self.approx_epsilon.setSingleStep(0.005)
        self.approx_epsilon.setDecimals(3)
        self.approx_epsilon.setValue(self.settings.get('approx_epsilon', 0.02))
        self.approx_epsilon.setToolTip("大きいほど歪んだ四角形も許容（推奨: 0.01-0.04）")
        self.approx_epsilon.valueChanged.connect(self.on_value_changed)
        form.addRow("輪郭近似精度:", self.approx_epsilon)

        # 最小面積比率
        self.min_area_ratio = QDoubleSpinBox()
        self.min_area_ratio.setRange(0.01, 0.5)
        self.min_area_ratio.setSingleStep(0.01)
        self.min_area_ratio.setDecimals(2)
        self.min_area_ratio.setValue(self.settings.get('min_area_ratio', 0.05))
        self.min_area_ratio.setToolTip("検出する四角形の最小サイズ（画像面積比、推奨: 0.03-0.10）")
        self.min_area_ratio.valueChanged.connect(self.on_value_changed)
        form.addRow("最小面積比率:", self.min_area_ratio)

        left_panel.addLayout(form)

        # ステータスラベル
        self.status_label = QLabel("※ パラメータ変更でプレビュー更新")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 10px;")
        left_panel.addWidget(self.status_label)

        left_panel.addStretch()

        # ボタン
        self.reset_btn = QPushButton("デフォルトに戻す")
        self.reset_btn.clicked.connect(self.reset_to_default)
        left_panel.addWidget(self.reset_btn)

        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet("background-color: #27AE60; color: white;")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        left_panel.addLayout(button_layout)

        layout.addLayout(left_panel)

        # === 右側: プレビュー ===
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        layout.addWidget(self.preview_label, 1)

        # 初期プレビュー
        self.load_preview_image()
        self.on_value_changed()

    def load_preview_image(self):
        """プレビュー用画像を読み込み"""
        if not self.image_path:
            self.preview_label.setText("画像が選択されていません")
            return

        self.original_pixmap = QPixmap(self.image_path)
        if self.original_pixmap.isNull():
            self.preview_label.setText("画像を読み込めません")
            self.original_pixmap = None

    def update_preview(self):
        """プレビュー画像を更新"""
        if not hasattr(self, 'original_pixmap') or not self.original_pixmap:
            return

        # プレビューサイズに合わせてスケール
        preview_w = self.preview_label.width() - 10
        preview_h = self.preview_label.height() - 10
        scaled = self.original_pixmap.scaled(
            preview_w, preview_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # コーナーを描画
        if self.detected_corners:
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint

            result = scaled.copy()
            painter = QPainter(result)

            # スケール計算
            scale_x = scaled.width() / self.original_pixmap.width()
            scale_y = scaled.height() / self.original_pixmap.height()

            # ポリゴン描画
            points = []
            for (x, y) in self.detected_corners:
                px = int(x * scale_x)
                py = int(y * scale_y)
                points.append(QPoint(px, py))

            polygon = QPolygon(points)
            painter.setBrush(QColor(255, 0, 255, 60))
            painter.setPen(QPen(QColor(255, 0, 255), 2))
            painter.drawPolygon(polygon)

            # コーナーポイント
            for i, (x, y) in enumerate(self.detected_corners):
                px = int(x * scale_x)
                py = int(y * scale_y)
                colors = [QColor(255, 0, 0), QColor(0, 255, 0),
                          QColor(0, 0, 255), QColor(255, 255, 0)]
                painter.setBrush(colors[i])
                painter.setPen(QPen(Qt.white, 1))
                painter.drawEllipse(px - 6, py - 6, 12, 12)

            painter.end()
            self.preview_label.setPixmap(result)
        else:
            self.preview_label.setPixmap(scaled)

    def on_value_changed(self):
        """パラメータ変更時にプレビュー更新"""
        if not self.image_path:
            self.status_label.setText("※ 画像が選択されていません")
            self.status_label.setStyleSheet("color: #E74C3C; font-size: 11px;")
            return

        # 現在の設定で検出を実行
        self.detected_corners = auto_detect_corners(self.image_path, self.get_settings())
        if self.detected_corners:
            self.status_label.setText("✓ 検出成功")
            self.status_label.setStyleSheet("color: #27AE60; font-size: 11px;")
        else:
            self.status_label.setText("✗ 検出失敗")
            self.status_label.setStyleSheet("color: #E74C3C; font-size: 11px;")

        self.update_preview()

    def reset_to_default(self):
        """デフォルト値にリセット"""
        self.canny_low.blockSignals(True)
        self.canny_high.blockSignals(True)
        self.blur_size.blockSignals(True)
        self.approx_epsilon.blockSignals(True)
        self.min_area_ratio.blockSignals(True)

        self.canny_low.setValue(50)
        self.canny_high.setValue(150)
        self.blur_size.setValue(5)
        self.approx_epsilon.setValue(0.02)
        self.min_area_ratio.setValue(0.05)

        self.canny_low.blockSignals(False)
        self.canny_high.blockSignals(False)
        self.blur_size.blockSignals(False)
        self.approx_epsilon.blockSignals(False)
        self.min_area_ratio.blockSignals(False)

        self.on_value_changed()

    def resizeEvent(self, event):
        """リサイズ時にプレビュー更新"""
        super().resizeEvent(event)
        self.update_preview()

    def get_settings(self):
        """現在の設定を辞書で返す"""
        return {
            'canny_low': self.canny_low.value(),
            'canny_high': self.canny_high.value(),
            'blur_size': self.blur_size.value(),
            'approx_epsilon': self.approx_epsilon.value(),
            'min_area_ratio': self.min_area_ratio.value()
        }

    def get_detected_corners(self):
        """検出されたコーナーを返す"""
        return self.detected_corners


class ImageCanvas(QLabel):
    """画像表示と4隅座標選択用のキャンバス"""

    coordinates_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2d2d2d; border: 1px solid #555;")
        self.setMouseTracking(True)  # マウス移動を追跡
        self.setCursor(Qt.CrossCursor)  # 精密選択用クロスヘアカーソル
        self.setAttribute(Qt.WA_Hover, True)  # ホバーイベントを有効化

        self.original_pixmap = None
        self.display_pixmap = None
        self.base_pixmap = None  # 拡大鏡なしのベース画像
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.image_path = None
        self.temp_file = None  # HEIC変換用一時ファイル
        self.original_size = (0, 0)

        # 拡大鏡設定
        self.magnifier_base_size = 400  # 拡大鏡の基準サイズ（長辺）
        self.magnifier_zoom = 1.75  # 拡大率
        self.mouse_pos = None  # 現在のマウス位置

        # ドラッグ操作用
        self.dragging_corner = None  # ドラッグ中のコーナーインデックス
        self.drag_threshold = 15  # ドラッグ判定の距離（ピクセル）
        self.drag_start_mouse = None  # ドラッグ開始時のマウス位置
        self.drag_start_corner = None  # ドラッグ開始時のコーナー位置
        self.drag_precision = 0.15  # 精密モードの減衰率（小さいほど精密）

        # ガイドメッセージ表示
        self.show_guide_message = False
        self.guide_message = ""

        # 4隅の座標（表示座標系）
        self.corners = []  # [(x, y), ...]
        self.corner_labels = ["左上", "右上", "右下", "左下"]
        self.corner_colors = [
            QColor(255, 0, 0),      # 赤
            QColor(0, 255, 0),      # 緑
            QColor(0, 0, 255),      # 青
            QColor(255, 255, 0),    # 黄
        ]

    def load_image(self, path: str):
        """画像を読み込み"""
        # 以前の一時ファイルを削除
        if self.temp_file and Path(self.temp_file).exists():
            try:
                Path(self.temp_file).unlink()
            except:
                pass
            self.temp_file = None

        self.image_path = path
        load_path = path

        # HEICファイルの場合は変換
        if path.lower().endswith(('.heic', '.heif')):
            temp_path = convert_heic_to_temp_jpeg(path)
            if temp_path:
                self.temp_file = temp_path
                load_path = temp_path
            else:
                self.setText("HEICファイルを変換できません")
                return False

        self.original_pixmap = QPixmap(load_path)
        if self.original_pixmap.isNull():
            self.setText("画像を読み込めません")
            return False

        self.original_size = (self.original_pixmap.width(), self.original_pixmap.height())
        self.corners = []
        self.update_display()
        return True

    def set_corners(self, corners_orig: list):
        """元画像座標系の座標を設定"""
        if not self.original_pixmap:
            return
        self.corners = []
        for (ox, oy) in corners_orig:
            dx = ox * self.scale + self.offset_x
            dy = oy * self.scale + self.offset_y
            self.corners.append((dx, dy))
        self.update_display()

    def get_corners_original(self) -> list:
        """元画像座標系での座標を取得"""
        result = []
        for (dx, dy) in self.corners:
            ox = int((dx - self.offset_x) / self.scale)
            oy = int((dy - self.offset_y) / self.scale)
            result.append((ox, oy))
        return result

    def update_display(self):
        """表示を更新"""
        if not self.original_pixmap:
            return

        # ウィジェットサイズに合わせてスケール計算
        w = self.width() - 20
        h = self.height() - 20

        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()

        scale_w = w / img_w
        scale_h = h / img_h
        target_scale = min(scale_w, scale_h)

        new_w = int(img_w * target_scale)
        new_h = int(img_h * target_scale)

        # スケーリング（IgnoreAspectRatioで正確なサイズを指定）
        scaled = self.original_pixmap.scaled(
            new_w, new_h,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )

        # 実際のスケールを計算（誤差を防ぐ）
        self.scale = scaled.width() / img_w

        # オフセット計算（中央配置）
        self.offset_x = (self.width() - scaled.width()) // 2
        self.offset_y = (self.height() - scaled.height()) // 2

        # 描画用ピクスマップ作成
        self.display_pixmap = QPixmap(self.size())
        self.display_pixmap.fill(QColor("#2d2d2d"))

        painter = QPainter(self.display_pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, scaled)

        # 座標点を描画
        if self.corners:
            # 線を描画
            pen = QPen(QColor(0, 255, 255), 2)
            painter.setPen(pen)
            for i in range(len(self.corners)):
                if i > 0:
                    painter.drawLine(
                        int(self.corners[i-1][0]), int(self.corners[i-1][1]),
                        int(self.corners[i][0]), int(self.corners[i][1])
                    )
            if len(self.corners) == 4:
                painter.drawLine(
                    int(self.corners[3][0]), int(self.corners[3][1]),
                    int(self.corners[0][0]), int(self.corners[0][1])
                )

            # 点とラベルを描画
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            for i, (x, y) in enumerate(self.corners):
                color = self.corner_colors[i]
                painter.setBrush(color)
                painter.setPen(QPen(Qt.white, 2))
                painter.drawEllipse(int(x) - 8, int(y) - 8, 16, 16)

                painter.setPen(color)
                painter.drawText(int(x) + 12, int(y) - 5, f"{i+1}:{self.corner_labels[i]}")

        # ガイドメッセージ表示（コーナーが未設定の場合）
        if self.show_guide_message and not self.corners and self.guide_message:
            # 半透明の背景
            msg_font = QFont()
            msg_font.setPointSize(14)
            msg_font.setBold(True)
            painter.setFont(msg_font)
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(self.guide_message)
            text_height = fm.height()

            # メッセージボックスの位置（上部中央）
            box_padding = 20
            box_width = text_width + box_padding * 2
            box_height = text_height + box_padding * 2
            box_x = (self.width() - box_width) // 2
            box_y = 50  # 上部に固定

            # 背景描画
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawRoundedRect(box_x, box_y, box_width, box_height, 10, 10)

            # テキスト描画
            painter.setPen(QColor(255, 200, 0))
            painter.drawText(box_x + box_padding, box_y + box_padding + fm.ascent(),
                           self.guide_message)

        painter.end()
        self.base_pixmap = self.display_pixmap.copy()  # ベース画像を保存
        self.draw_magnifier()

    def draw_magnifier(self):
        """拡大鏡を描画"""
        if not self.base_pixmap or not self.mouse_pos or not self.original_pixmap:
            if self.base_pixmap:
                self.setPixmap(self.base_pixmap)
            return

        mx, my = self.mouse_pos

        # 画像範囲内かチェック
        img_w = self.original_pixmap.width() * self.scale
        img_h = self.original_pixmap.height() * self.scale

        if not (self.offset_x <= mx <= self.offset_x + img_w and
                self.offset_y <= my <= self.offset_y + img_h):
            self.setPixmap(self.base_pixmap)
            return

        # 拡大鏡用のピクスマップを作成
        result = self.base_pixmap.copy()
        painter = QPainter(result)

        # 元画像での座標を計算（コーナーの実際の位置）
        corner_orig_x = (mx - self.offset_x) / self.scale
        corner_orig_y = (my - self.offset_y) / self.scale

        # 画像の縦横比を計算して拡大鏡サイズを決定
        orig_w = self.original_pixmap.width()
        orig_h = self.original_pixmap.height()
        aspect_ratio = orig_w / orig_h

        if aspect_ratio >= 1:  # 横長
            mag_width = self.magnifier_base_size
            mag_height = int(self.magnifier_base_size / aspect_ratio)
        else:  # 縦長
            mag_height = self.magnifier_base_size
            mag_width = int(self.magnifier_base_size * aspect_ratio)

        # 拡大する領域のサイズ（元画像座標系）
        src_width = mag_width / self.magnifier_zoom
        src_height = mag_height / self.magnifier_zoom

        # マウス位置が画像のどの象限にあるかでオフセットを決定
        # 画像の中心を基準に4分割
        image_center_x = orig_w / 2
        image_center_y = orig_h / 2

        # マウスが左半分か右半分か、上半分か下半分かを判定
        is_left = corner_orig_x < image_center_x
        is_top = corner_orig_y < image_center_y

        # 象限に応じたオフセット（マウス位置の反対側を多く表示）
        # 左上象限: 右下にオフセット → マウス位置が左上に表示
        # 右上象限: 左下にオフセット → マウス位置が右上に表示
        # 右下象限: 左上にオフセット → マウス位置が右下に表示
        # 左下象限: 右上にオフセット → マウス位置が左下に表示
        offset_ratio = 0.25
        view_offset_x = src_width * offset_ratio if is_left else -src_width * offset_ratio
        view_offset_y = src_height * offset_ratio if is_top else -src_height * offset_ratio

        # ビューの中心位置（オフセット適用）
        view_center_x = corner_orig_x + view_offset_x
        view_center_y = corner_orig_y + view_offset_y
        src_x = view_center_x - src_width / 2
        src_y = view_center_y - src_height / 2

        # 元画像から切り出し（境界処理）
        src_rect_x = int(src_x)
        src_rect_y = int(src_y)
        src_rect_w = int(src_width)
        src_rect_h = int(src_height)

        # クリッピングによるオフセットを記録
        clip_offset_x = 0
        clip_offset_y = 0

        if src_rect_x < 0:
            clip_offset_x = -src_rect_x
            src_rect_w += src_rect_x
            src_rect_x = 0
        if src_rect_y < 0:
            clip_offset_y = -src_rect_y
            src_rect_h += src_rect_y
            src_rect_y = 0
        if src_rect_x + src_rect_w > orig_w:
            src_rect_w = orig_w - src_rect_x
        if src_rect_y + src_rect_h > orig_h:
            src_rect_h = orig_h - src_rect_y

        if src_rect_w <= 0 or src_rect_h <= 0:
            self.setPixmap(self.base_pixmap)
            return

        # 切り出して拡大
        cropped = self.original_pixmap.copy(src_rect_x, src_rect_y, src_rect_w, src_rect_h)

        # 拡大鏡の表示位置（画面中央に固定）
        box_width = mag_width + 10   # 左右5pxずつの余白
        box_height = mag_height + 35  # 下部にテキスト領域

        box_x = (self.width() - box_width) // 2
        box_y = (self.height() - box_height) // 2
        mag_x = box_x + 5
        mag_y = box_y + 5

        # 背景（半透明の黒）
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawRect(box_x, box_y, box_width, box_height)

        # 拡大画像を描画（クリッピング分のオフセットを考慮）
        magnified = cropped.scaled(
            int(src_rect_w * self.magnifier_zoom),
            int(src_rect_h * self.magnifier_zoom),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        draw_x = mag_x + int(clip_offset_x * self.magnifier_zoom)
        draw_y = mag_y + int(clip_offset_y * self.magnifier_zoom)
        painter.drawPixmap(draw_x, draw_y, magnified)

        # 選択領域を半透明の緑で表示（4点指定時）
        if len(self.corners) == 4:
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint, QRect

            # 拡大鏡の画像領域にクリッピング
            painter.setClipRect(QRect(mag_x, mag_y, mag_width, mag_height))

            # 元画像座標系でのコーナー座標を取得
            orig_corners = self.get_corners_original()

            # 拡大鏡内の座標に変換（クリッピング前のsrc_x/yを基準に）
            mag_points = []
            for (cx, cy) in orig_corners:
                # 元画像座標から拡大鏡内座標に変換
                mag_cx = mag_x + int((cx - src_x) * self.magnifier_zoom)
                mag_cy = mag_y + int((cy - src_y) * self.magnifier_zoom)
                mag_points.append(QPoint(mag_cx, mag_cy))

            # ポリゴンを描画
            polygon = QPolygon(mag_points)
            painter.setBrush(QColor(255, 0, 255, 80))  # 半透明のマゼンタ
            painter.setPen(QPen(QColor(255, 0, 255, 200), 2))  # 輪郭線
            painter.drawPolygon(polygon)

            # クリッピング解除
            painter.setClipping(False)

        # クロスヘア位置（コーナーの実際の位置を表示）
        cross_x = mag_x + int((corner_orig_x - src_x) * self.magnifier_zoom)
        cross_y = mag_y + int((corner_orig_y - src_y) * self.magnifier_zoom)
        painter.setPen(QPen(QColor(255, 0, 0), 1))
        painter.drawLine(cross_x - 15, cross_y, cross_x + 15, cross_y)
        painter.drawLine(cross_x, cross_y - 15, cross_x, cross_y + 15)

        # 座標表示
        orig_x_int = int((mx - self.offset_x) / self.scale)
        orig_y_int = int((my - self.offset_y) / self.scale)
        painter.setPen(QColor(255, 255, 255))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(mag_x, mag_y + mag_height + 18, f"({orig_x_int}, {orig_y_int})")

        # 次に選択する点の情報
        next_idx = len(self.corners)
        if next_idx < 4:
            painter.setPen(self.corner_colors[next_idx])
            painter.drawText(mag_x + 90, mag_y + mag_height + 18,
                           f"→ {self.corner_labels[next_idx]}")

        painter.end()
        self.setPixmap(result)
        self.update()  # 明示的に再描画を要求

    def mouseMoveEvent(self, event):
        """マウス移動時に拡大鏡を更新、またはドラッグ処理"""
        x, y = event.position().x(), event.position().y()
        self.mouse_pos = (x, y)

        # ドラッグ中の場合（精密モード）
        if self.dragging_corner is not None and self.drag_start_mouse and self.drag_start_corner:
            # マウス移動量を計算
            dx = x - self.drag_start_mouse[0]
            dy = y - self.drag_start_mouse[1]

            # 減衰率を適用して精密移動
            new_x = self.drag_start_corner[0] + dx * self.drag_precision
            new_y = self.drag_start_corner[1] + dy * self.drag_precision

            # 画像範囲内に制限
            img_w = self.original_pixmap.width() * self.scale
            img_h = self.original_pixmap.height() * self.scale
            new_x = max(self.offset_x, min(new_x, self.offset_x + img_w))
            new_y = max(self.offset_y, min(new_y, self.offset_y + img_h))

            self.corners[self.dragging_corner] = (new_x, new_y)
            # 拡大鏡もコーナー位置に追従させる
            self.mouse_pos = (new_x, new_y)
            self.update_display()
            self.draw_magnifier()
            self.coordinates_changed.emit()
        else:
            # コーナー付近ではカーソルを変更
            if self.find_corner_at(x, y) is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)

            if self.base_pixmap:
                self.draw_magnifier()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ドラッグ終了"""
        if event.button() == Qt.LeftButton and self.dragging_corner is not None:
            self.dragging_corner = None
            self.drag_start_mouse = None
            self.drag_start_corner = None
            self.setCursor(Qt.CrossCursor)
            # ドラッグ終了時に保存
            self.coordinates_changed.emit()
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        """マウスがウィジェットに入った時"""
        super().enterEvent(event)

    def leaveEvent(self, event):
        """マウスがウィジェットから離れた時"""
        self.mouse_pos = None
        if self.base_pixmap:
            self.setPixmap(self.base_pixmap)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        """リサイズ時に再描画"""
        super().resizeEvent(event)
        if self.original_pixmap:
            # 座標を元画像座標系で保持してから再計算
            orig_corners = self.get_corners_original() if self.corners else []
            self.update_display()
            if orig_corners:
                self.set_corners(orig_corners)

    def find_corner_at(self, x, y):
        """指定座標付近のコーナーを検索"""
        for i, (cx, cy) in enumerate(self.corners):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= self.drag_threshold:
                return i
        return None

    def mousePressEvent(self, event: QMouseEvent):
        """クリックで座標追加またはドラッグ開始"""
        if not self.original_pixmap:
            return

        if event.button() == Qt.LeftButton:
            x, y = event.position().x(), event.position().y()

            # 既存のコーナー付近かチェック
            corner_idx = self.find_corner_at(x, y)
            if corner_idx is not None:
                # ドラッグ開始（コーナーの中心位置を基準にする）
                self.dragging_corner = corner_idx
                corner_pos = self.corners[corner_idx]
                # マウス開始位置をコーナーの中心に合わせる（ずれを補正）
                self.drag_start_mouse = corner_pos
                self.drag_start_corner = corner_pos
                # 拡大鏡もコーナー位置に合わせる
                self.mouse_pos = corner_pos
                self.draw_magnifier()
                self.setCursor(Qt.ClosedHandCursor)
                return

            # 画像範囲内かチェック
            img_w = self.original_pixmap.width() * self.scale
            img_h = self.original_pixmap.height() * self.scale

            if (self.offset_x <= x <= self.offset_x + img_w and
                self.offset_y <= y <= self.offset_y + img_h):

                if len(self.corners) >= 4:
                    self.corners = []

                # ガイドメッセージを消す
                self.show_guide_message = False
                self.guide_message = ""

                self.corners.append((x, y))
                # クリック位置に拡大鏡を合わせる
                self.mouse_pos = (x, y)
                self.update_display()
                self.draw_magnifier()
                self.coordinates_changed.emit()

    def clear_corners(self):
        """座標をクリア"""
        self.corners = []
        self.update_display()
        self.coordinates_changed.emit()

    def clear_image(self):
        """画像をクリア"""
        self.original_pixmap = None
        self.base_pixmap = None
        self.corners = []
        self.mouse_pos = None
        self.clear()
        self.setText("画像を選択してください")


class PerspectiveCorrectorApp(QMainWindow):
    """メインアプリケーション"""

    CONFIG_FILE = "perspective_config.json"

    RECENT_FOLDERS_FILE = ".perspective_corrector_recent.json"  # ホームに保存

    def __init__(self, start_dir: str = None):
        super().__init__()
        self.setMinimumSize(1500, 800)

        self.start_dir = start_dir or str(Path.cwd())
        self.current_image = None
        self.config = {}
        self.detection_settings = {}  # 自動認識パラメータ
        self.recent_folders = []  # 最近使用したフォルダ

        self.load_recent_folders()
        self.load_config()
        self.setup_ui()
        self.update_window_title()

    def load_config(self):
        """設定ファイルを読み込み"""
        config_path = Path(self.start_dir) / self.CONFIG_FILE
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 検出設定を分離
                    self.detection_settings = data.pop('_detection_settings', {})
                    self.config = data
            except Exception as e:
                print(f"Config load error: {e}")
                self.config = {}
                self.detection_settings = {}

    def save_config(self):
        """設定ファイルに保存"""
        config_path = Path(self.start_dir) / self.CONFIG_FILE
        try:
            # 検出設定を含めて保存
            data = dict(self.config)
            if self.detection_settings:
                data['_detection_settings'] = self.detection_settings
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Config save error: {e}")

    def load_recent_folders(self):
        """最近使用したフォルダを読み込み"""
        recent_path = Path.home() / self.RECENT_FOLDERS_FILE
        if recent_path.exists():
            try:
                with open(recent_path, 'r', encoding='utf-8') as f:
                    self.recent_folders = json.load(f)
            except Exception:
                self.recent_folders = []

    def save_recent_folders(self):
        """最近使用したフォルダを保存"""
        recent_path = Path.home() / self.RECENT_FOLDERS_FILE
        try:
            with open(recent_path, 'w', encoding='utf-8') as f:
                json.dump(self.recent_folders, f, ensure_ascii=False)
        except Exception:
            pass

    def add_recent_folder(self, folder_path: str):
        """最近使用したフォルダに追加"""
        # 既存のエントリを削除
        if folder_path in self.recent_folders:
            self.recent_folders.remove(folder_path)
        # 先頭に追加
        self.recent_folders.insert(0, folder_path)
        # 最大10件に制限
        self.recent_folders = self.recent_folders[:10]
        self.save_recent_folders()
        self.update_recent_folders_menu()

    def update_window_title(self):
        """ウィンドウタイトルを更新"""
        folder_name = Path(self.start_dir).name
        self.setWindowTitle(f"台形補正ツール - {folder_name}")

    def setup_ui(self):
        """UIをセットアップ"""
        # メニューバー
        self.setup_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # スプリッター
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === 左側パネル: ファイルテーブル ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # ファイルテーブル（2カラム: 元ファイル名 | 出力名）
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["元ファイル", "出力名"])
        # カラム幅をウィジェット幅の半分に自動調整
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # フォントサイズ
        table_font = QFont()
        table_font.setPointSize(11)
        self.file_table.setFont(table_font)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.file_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_table.clicked.connect(self.on_table_clicked)
        self.file_table.currentCellChanged.connect(self.on_table_selection_changed)
        self.file_table.cellChanged.connect(self.on_output_name_changed)

        # ファイル一覧を読み込み
        self.image_files = []  # [(filepath, output_name), ...]
        self.load_image_files()

        left_layout.addWidget(QLabel("画像ファイル（出力名は編集可能）:"))
        left_layout.addWidget(self.file_table)

        # ステータス表示
        self.file_count_label = QLabel("設定済み: 0 ファイル")
        self.file_count_label.setStyleSheet("color: #888; padding: 5px;")
        left_layout.addWidget(self.file_count_label)

        splitter.addWidget(left_panel)

        # === 右側パネル: 画像表示 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 操作説明
        help_label = QLabel(
            "操作: 4隅を自動検出 → 点をドラッグで調整 | "
            "手動指定: 左上→右上→右下→左下の順にクリック | 5回目でリセット"
        )
        help_label.setStyleSheet("color: #888; padding: 5px;")
        right_layout.addWidget(help_label)

        # 画像キャンバス
        self.canvas = ImageCanvas()
        self.canvas.coordinates_changed.connect(self.on_coordinates_changed)
        right_layout.addWidget(self.canvas)

        # 下部パネル: 座標情報 + ボタン
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 5, 0, 0)

        # 座標情報（左側）
        self.coord_labels = []
        for i, name in enumerate(["左上", "右上", "右下", "左下"]):
            lbl = QLabel(f"{name}: ---")
            lbl.setMinimumWidth(120)
            self.coord_labels.append(lbl)
            bottom_layout.addWidget(lbl)

        # スペーサー（座標とボタンの間）
        bottom_layout.addStretch()

        # ボタン（右側に配置）
        # 座標クリア - オレンジ（警告/リセット）
        self.clear_btn = QPushButton("座標クリア")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setMinimumWidth(100)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #E67E22;
                color: white;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D35400;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_current_corners)
        bottom_layout.addWidget(self.clear_btn)

        # 自動認識 - 青（ツール/アクション）
        self.auto_detect_btn = QPushButton("自動認識")
        self.auto_detect_btn.setMinimumHeight(40)
        self.auto_detect_btn.setMinimumWidth(100)
        self.auto_detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #888;
            }
        """)
        self.auto_detect_btn.clicked.connect(self.run_auto_detect)
        bottom_layout.addWidget(self.auto_detect_btn)

        # 認識設定 - 紫（設定/調整）
        self.settings_btn = QPushButton("認識設定")
        self.settings_btn.setMinimumHeight(40)
        self.settings_btn.setMinimumWidth(90)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        self.settings_btn.clicked.connect(self.show_detection_settings)
        bottom_layout.addWidget(self.settings_btn)

        # 一括処理 - 緑（実行/確定）
        self.process_btn = QPushButton("一括処理")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setMinimumWidth(100)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E8449;
            }
            QPushButton:disabled {
                background-color: #888;
            }
        """)
        self.process_btn.clicked.connect(self.run_batch_process)
        bottom_layout.addWidget(self.process_btn)

        # 終了 - グレー（中立/終了）
        self.quit_btn = QPushButton("終了")
        self.quit_btn.setMinimumHeight(40)
        self.quit_btn.setMinimumWidth(80)
        self.quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #95A5A6;
            }
        """)
        self.quit_btn.clicked.connect(self.close)
        bottom_layout.addWidget(self.quit_btn)

        right_layout.addWidget(bottom_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])

        # ステータスバー
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.update_status()

    def load_image_files(self):
        """ディレクトリから画像ファイルを読み込みテーブルに表示"""
        self.file_table.blockSignals(True)
        self.file_table.setRowCount(0)
        self.image_files = []

        # 対応拡張子
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}

        # ファイル一覧取得
        start_path = Path(self.start_dir)
        for f in sorted(start_path.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                rel_path = f.name
                # 設定から出力名を取得、なければファイル名のstemを使用
                if rel_path in self.config:
                    output_name = self.config[rel_path].get("output_name", f.stem)
                else:
                    output_name = f.stem

                self.image_files.append((str(f), output_name))

                row = self.file_table.rowCount()
                self.file_table.insertRow(row)

                # 元ファイル名（読み取り専用・選択不可）
                orig_item = QTableWidgetItem(rel_path)
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)
                self.file_table.setItem(row, 0, orig_item)

                # 出力名（編集可能）
                output_item = QTableWidgetItem(output_name)
                self.file_table.setItem(row, 1, output_item)

                # 設定済みの場合は右カラムの背景色を変更
                if rel_path in self.config and "corners" in self.config[rel_path]:
                    if len(self.config[rel_path]["corners"]) == 4:
                        output_item.setBackground(QColor(144, 238, 144, 100))

        self.file_table.blockSignals(False)

    def on_table_clicked(self, index):
        """テーブルクリック時 - 右カラムを選択して画像を読み込み"""
        row = index.row()
        if 0 <= row < len(self.image_files):
            # 右カラム（出力名）を選択状態にする（シグナルをブロック）
            if index.column() != 1:
                self.file_table.blockSignals(True)
                self.file_table.setCurrentCell(row, 1)
                self.file_table.blockSignals(False)
            # 画像読み込み
            path = self.image_files[row][0]
            self.load_image(path)

    def on_table_selection_changed(self, current_row, current_col, prev_row, prev_col):
        """テーブル選択変更時 - 常に右カラムを選択"""
        if current_row >= 0 and current_row < len(self.image_files):
            # 右カラム（出力名）を選択状態にする（シグナルをブロック）
            if current_col != 1:
                self.file_table.blockSignals(True)
                self.file_table.setCurrentCell(current_row, 1)
                self.file_table.blockSignals(False)
            # 画像読み込み
            path = self.image_files[current_row][0]
            self.load_image(path)

    def on_output_name_changed(self, row, col):
        """出力名が変更された時"""
        if col != 1 or row >= len(self.image_files):
            return

        filepath = self.image_files[row][0]
        rel_path = Path(filepath).name
        new_output_name = self.file_table.item(row, 1).text()

        # image_filesを更新
        self.image_files[row] = (filepath, new_output_name)

        # configを更新
        if rel_path not in self.config:
            self.config[rel_path] = {}
        self.config[rel_path]["output_name"] = new_output_name
        self.save_config()

    def load_image(self, path: str):
        """画像を読み込み"""
        # 同じ画像は再読み込みしない
        if self.current_image == path:
            return

        # 現在の座標を保存
        self.save_current_corners()

        self.current_image = path
        if self.canvas.load_image(path):
            rel_path = Path(path).name

            # ガイドメッセージをリセット
            self.canvas.show_guide_message = False
            self.canvas.guide_message = ""

            # 保存済み座標があれば復元
            corners = self.config.get(rel_path, {}).get("corners", [])

            if corners and len(corners) == 4:
                # 4点の座標がある場合は復元
                self.canvas.set_corners(corners)
                self.statusBar.showMessage(f"読み込み: {Path(path).name} (保存済み座標を復元)")
            else:
                # 未設定または不完全な場合は自動検出を試行
                self.statusBar.showMessage(f"読み込み: {Path(path).name} - 自動検出中...")
                QApplication.processEvents()

                # HEICの場合は一時ファイルを使用
                detect_path = path
                if path.lower().endswith(('.heic', '.heif')) and self.canvas.temp_file:
                    detect_path = self.canvas.temp_file

                detected = auto_detect_corners(detect_path, self.detection_settings)
                if detected:
                    self.canvas.set_corners(detected)
                    self.statusBar.showMessage(f"読み込み: {Path(path).name} (4隅を自動検出)")
                    # 自動検出結果を保存
                    self.save_current_corners()
                else:
                    # 自動検出失敗 - ガイドメッセージを表示
                    self.canvas.show_guide_message = True
                    self.canvas.guide_message = "自動検出失敗 - 4隅を手動で指定してください"
                    self.canvas.update_display()
                    self.statusBar.showMessage(f"読み込み: {Path(path).name} (自動検出失敗)")

            self.update_coord_labels()

    def save_current_corners(self):
        """現在の座標を保存"""
        if not self.current_image:
            return

        corners = self.canvas.get_corners_original()
        rel_path = Path(self.current_image).name

        if len(corners) == 4:
            # 既存の出力名を保持
            existing_output_name = self.config.get(rel_path, {}).get("output_name", Path(self.current_image).stem)
            self.config[rel_path] = {
                "corners": corners,
                "original_size": self.canvas.original_size,
                "output_name": existing_output_name
            }
        elif rel_path in self.config and len(corners) == 0:
            # 座標がクリアされた場合はcornersのみ削除（output_nameは保持）
            if "corners" in self.config[rel_path]:
                del self.config[rel_path]["corners"]
            if "original_size" in self.config[rel_path]:
                del self.config[rel_path]["original_size"]

        self.save_config()
        self.update_status()
        # テーブルの背景色を更新
        self.update_table_highlighting()

    def update_table_highlighting(self):
        """テーブルの背景色を更新（右カラムのみ）"""
        for row in range(self.file_table.rowCount()):
            orig_item = self.file_table.item(row, 0)
            output_item = self.file_table.item(row, 1)
            if not orig_item or not output_item:
                continue

            rel_path = orig_item.text()
            if rel_path in self.config and "corners" in self.config[rel_path]:
                if len(self.config[rel_path]["corners"]) == 4:
                    output_item.setBackground(QColor(144, 238, 144, 100))
                    continue

            # 設定なしの場合は背景色をクリア
            output_item.setBackground(QColor(0, 0, 0, 0))

    def on_coordinates_changed(self):
        """座標変更時"""
        self.update_coord_labels()
        if len(self.canvas.corners) == 4:
            self.save_current_corners()

    def update_coord_labels(self):
        """座標表示を更新"""
        corners = self.canvas.get_corners_original()
        for i, lbl in enumerate(self.coord_labels):
            name = ["左上", "右上", "右下", "左下"][i]
            if i < len(corners):
                x, y = corners[i]
                lbl.setText(f"{name}: ({x}, {y})")
                lbl.setStyleSheet("color: green;")
            else:
                lbl.setText(f"{name}: ---")
                lbl.setStyleSheet("color: #888;")

    def clear_current_corners(self):
        """現在の画像の座標をクリア"""
        self.canvas.clear_corners()
        self.save_current_corners()

    def show_detection_settings(self):
        """自動認識設定ダイアログを表示"""
        # 画像パスを取得（HEIC対応）
        image_path = None
        if self.current_image:
            image_path = self.current_image
            if image_path.lower().endswith(('.heic', '.heif')) and self.canvas.temp_file:
                image_path = self.canvas.temp_file

        dialog = DetectionSettingsDialog(self, self.detection_settings, image_path)
        dialog.resize(800, 600)  # 明示的にサイズ設定
        self.center_dialog(dialog, 800, 600)
        if dialog.exec() == QDialog.Accepted:
            self.detection_settings = dialog.get_settings()
            self.save_config()

            # 検出されたコーナーをメインキャンバスに適用
            detected = dialog.get_detected_corners()
            if detected:
                self.canvas.set_corners(detected)
                self.save_current_corners()
                self.update_coord_labels()
                self.statusBar.showMessage("認識設定を保存し、検出結果を適用しました")
            else:
                self.statusBar.showMessage("認識設定を保存しました")

    def run_auto_detect(self):
        """現在の画像で自動認識を実行"""
        if not self.current_image:
            self.statusBar.showMessage("画像が選択されていません")
            return

        self.statusBar.showMessage("自動認識中...")
        QApplication.processEvents()

        # HEICの場合は一時ファイルを使用
        detect_path = self.current_image
        if self.current_image.lower().endswith(('.heic', '.heif')) and self.canvas.temp_file:
            detect_path = self.canvas.temp_file

        detected = auto_detect_corners(detect_path, self.detection_settings)
        if detected:
            self.canvas.set_corners(detected)
            self.save_current_corners()
            self.update_coord_labels()
        else:
            self.canvas.show_guide_message = True
            self.canvas.guide_message = "自動検出失敗 - 4隅を手動で指定してください"
            self.canvas.update_display()
            self.statusBar.showMessage("自動認識失敗 - 手動で指定してください")

    def update_status(self):
        """ステータス表示を更新"""
        # 座標が設定されているファイルのみカウント
        count = sum(
            1 for data in self.config.values()
            if "corners" in data and len(data.get("corners", [])) == 4
        )
        self.file_count_label.setText(f"設定済み: {count} ファイル")
        self.process_btn.setEnabled(count > 0)

        # ステータスバーにパス表示
        if self.current_image:
            self.statusBar.showMessage(self.current_image)
        else:
            self.statusBar.showMessage(f"フォルダ: {self.start_dir}")

    def run_batch_process(self):
        """一括処理実行"""
        # 座標が設定されているファイルのみカウント
        files_to_process = [
            (rel_path, data) for rel_path, data in self.config.items()
            if "corners" in data and len(data.get("corners", [])) == 4
        ]

        if not files_to_process:
            self.show_message_box(QMessageBox.Warning, "警告", "処理するファイルがありません")
            return

        # 現在の座標を保存
        self.save_current_corners()

        reply = self.show_message_box(
            QMessageBox.Question, "確認",
            f"{len(files_to_process)} ファイルの台形補正を実行しますか？\n"
            "出力ファイルは「[出力名]_corrected.png」として保存されます。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # プログレスダイアログ
        progress = QProgressDialog("処理中...", "キャンセル", 0, len(files_to_process), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setFixedSize(400, 100)
        self.center_dialog(progress, 400, 100)

        success_count = 0
        error_files = []

        for i, (rel_path, data) in enumerate(files_to_process):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"処理中: {rel_path}")
            QApplication.processEvents()

            full_path = Path(self.start_dir) / rel_path
            if not full_path.exists():
                error_files.append((rel_path, "ファイルが見つかりません"))
                continue

            corners = data.get("corners", [])
            if len(corners) != 4:
                error_files.append((rel_path, "座標が不完全です"))
                continue

            # 出力ファイル名（設定された出力名を使用、なければ元ファイル名）
            output_name = data.get("output_name", full_path.stem)
            output_path = full_path.parent / f"{output_name}_corrected.png"

            # HEICの場合は先に変換
            input_path = str(full_path)
            temp_heic = None
            if str(full_path).lower().endswith(('.heic', '.heif')):
                temp_heic = convert_heic_to_temp_jpeg(str(full_path))
                if temp_heic:
                    input_path = temp_heic
                else:
                    error_files.append((rel_path, "HEIC変換エラー"))
                    continue

            # OpenCVで台形補正
            import time
            start_time = time.time()
            try:
                if perspective_transform_cv(input_path, corners, str(output_path)):
                    elapsed = time.time() - start_time
                    print(f"{rel_path}: {elapsed:.3f}秒")
                    success_count += 1
                else:
                    error_files.append((rel_path, "変換に失敗しました"))
            except Exception as e:
                error_files.append((rel_path, str(e)))
            finally:
                # 一時ファイル削除
                if temp_heic and Path(temp_heic).exists():
                    try:
                        Path(temp_heic).unlink()
                    except:
                        pass

        progress.setValue(len(files_to_process))

        # エラーがあった場合のみ表示
        if error_files:
            msg = f"処理完了: {success_count}/{len(files_to_process)} ファイル\n\nエラー:\n"
            for path, err in error_files[:5]:
                msg += f"  {path}: {err}\n"
            if len(error_files) > 5:
                msg += f"  ... 他 {len(error_files) - 5} ファイル"
            self.show_message_box(QMessageBox.Warning, "エラー", msg)

    def setup_menu_bar(self):
        """メニューバーをセットアップ"""
        menubar = self.menuBar()

        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル(&F)")

        # フォルダを開く
        open_action = QAction("フォルダを開く...(&O)", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        # 最近使用したフォルダ
        self.recent_menu = file_menu.addMenu("最近使用したフォルダ(&R)")
        self.update_recent_folders_menu()

        file_menu.addSeparator()

        # 終了
        quit_action = QAction("終了(&Q)", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def update_recent_folders_menu(self):
        """最近使用したフォルダメニューを更新"""
        self.recent_menu.clear()

        if not self.recent_folders:
            no_recent = QAction("(なし)", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return

        for folder in self.recent_folders:
            folder_name = Path(folder).name
            action = QAction(f"{folder_name}  ({folder})", self)
            action.setData(folder)
            action.triggered.connect(lambda checked, f=folder: self.change_folder(f))
            self.recent_menu.addAction(action)

        self.recent_menu.addSeparator()
        clear_action = QAction("履歴をクリア", self)
        clear_action.triggered.connect(self.clear_recent_folders)
        self.recent_menu.addAction(clear_action)

    def clear_recent_folders(self):
        """最近使用したフォルダをクリア"""
        self.recent_folders = []
        self.save_recent_folders()
        self.update_recent_folders_menu()

    def center_dialog(self, dialog, width=None, height=None):
        """ダイアログをアプリウィンドウの中心に配置"""
        # サイズが指定されていればリサイズ
        if width and height:
            dialog.resize(width, height)
        else:
            dialog.adjustSize()
            # minimumSizeが設定されている場合はそれを使用
            min_size = dialog.minimumSize()
            if min_size.width() > 0 and min_size.height() > 0:
                dialog.resize(min_size)

        # メインウィンドウの中心を計算
        main_geo = self.geometry()
        main_center_x = main_geo.x() + main_geo.width() // 2
        main_center_y = main_geo.y() + main_geo.height() // 2

        # ダイアログの左上座標を計算（中心から半分ずらす）
        dialog_x = main_center_x - dialog.width() // 2
        dialog_y = main_center_y - dialog.height() // 2

        dialog.move(dialog_x, dialog_y)

    def show_message_box(self, icon, title, text, buttons=QMessageBox.Ok):
        """中央配置されたメッセージボックスを表示"""
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)

        # サイズを確定させてから中央配置
        msg.layout().setSizeConstraint(msg.layout().SetFixedSize)
        msg.adjustSize()

        # メインウィンドウの中心を計算
        main_geo = self.geometry()
        main_center_x = main_geo.x() + main_geo.width() // 2
        main_center_y = main_geo.y() + main_geo.height() // 2

        # ダイアログの左上座標を計算
        dialog_x = main_center_x - msg.sizeHint().width() // 2
        dialog_y = main_center_y - msg.sizeHint().height() // 2
        msg.move(dialog_x, dialog_y)

        return msg.exec()

    def open_folder(self):
        """フォルダを開くダイアログ"""
        dialog = QFileDialog(self, "フォルダを開く", self.start_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        self.center_dialog(dialog, 700, 500)

        if dialog.exec() == QFileDialog.Accepted:
            folders = dialog.selectedFiles()
            if folders:
                self.change_folder(folders[0])

    def change_folder(self, folder_path: str):
        """作業フォルダを変更"""
        if not Path(folder_path).is_dir():
            self.show_message_box(QMessageBox.Warning, "エラー", f"フォルダが見つかりません:\n{folder_path}")
            return

        # 現在の座標を保存
        self.save_current_corners()

        # フォルダ変更
        self.start_dir = folder_path
        self.current_image = None
        self.config = {}

        # 設定を読み込み
        self.load_config()

        # ファイル一覧を更新
        self.load_image_files()

        # キャンバスをクリア
        self.canvas.clear_image()
        self.update_coord_labels()

        # タイトル更新
        self.update_window_title()

        # ステータス更新
        self.update_status()

        # 最近使用したフォルダに追加
        self.add_recent_folder(folder_path)

    def closeEvent(self, event):
        """終了時に座標を保存"""
        self.save_current_corners()
        self.add_recent_folder(self.start_dir)
        event.accept()


def main():
    app = QApplication(sys.argv)

    # コマンドライン引数でディレクトリ指定可能（フォルダをexeにドロップした場合も対応）
    start_dir = None
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.is_dir():
            # フォルダが指定された場合
            start_dir = str(arg_path)
        elif arg_path.is_file():
            # ファイルが指定された場合は親フォルダを使用
            start_dir = str(arg_path.parent)

    window = PerspectiveCorrectorApp(start_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
