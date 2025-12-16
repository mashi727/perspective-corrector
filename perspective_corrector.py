#!/usr/bin/env python3
"""
Perspective Corrector - プレゼン写真 台形補正アプリケーション

概要:
    プロジェクターで投影されたスライドを斜めから撮影した写真の
    台形歪みを補正し、正面から見た状態に変換するツール。

主な機能:
    - 4隅座標の手動指定（クリック＆ドラッグ）
    - Cannyエッジ検出による四角形の自動認識
    - 拡大鏡による精密な座標調整
    - 色調補正（ホワイトバランス + CLAHE）
    - 一括処理（PNG個別出力 / PDF一括出力）
    - HEIC/HEIF形式のサポート（iPhone写真対応）

アーキテクチャ:
    - ImageCanvas: 画像表示と座標選択を担当するカスタムQLabel
    - FileListPanel: ファイル一覧と出力名編集を担当するパネル
    - MainWindow: 全体のレイアウトと処理フローを管理
    - ColorCorrectionSettingsDialog: 色調補正パラメータの調整ダイアログ

設定ファイル:
    - perspective_config.json: 作業ディレクトリに保存される座標情報
    - ~/.perspective_corrector_recent.json: 最近使用したフォルダの履歴

Author: mashi727
License: MIT
"""

# =============================================================================
# 標準ライブラリ
# =============================================================================
import sys
import json
import subprocess
import tempfile
import platform
from pathlib import Path

# =============================================================================
# サードパーティライブラリ
# =============================================================================
import cv2
import numpy as np

# =============================================================================
# Qt (PySide6)
# =============================================================================
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QGroupBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QProgressDialog, QSplitter, QStatusBar, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QSlider, QFileDialog, QMenuBar, QMenu
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QBrush,
    QAction, QKeySequence
)


# =============================================================================
# HEIC/HEIF サポート
# =============================================================================
# iPhoneで撮影した写真はHEIC形式で保存されることが多い。
# OpenCVやQtは標準ではHEICを読めないため、JPEGに変換して処理する。
# 変換方法は環境によって異なるため、複数のフォールバックを用意している。
#
# 優先順位:
#   1. pillow-heif (クロスプラットフォーム、推奨)
#   2. sips (macOS標準コマンド)
#   3. ImageMagick (Windows、要インストール)
# =============================================================================

_heif_registered = False
_heif_available = None  # None=未確認, True=利用可能, False=利用不可


def _init_heif_support() -> bool:
    """
    pillow-heifの初期化（一度だけ実行）

    Returns:
        pillow-heifが利用可能な場合True

    Note:
        register_heif_opener()は一度だけ呼ぶ必要がある。
        複数回呼ぶと警告が出るため、フラグで制御している。
    """
    global _heif_registered, _heif_available

    # 既に確認済みの場合はキャッシュを返す（毎回のインポート試行を避ける）
    if _heif_available is not None:
        return _heif_available

    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif_registered = True
        _heif_available = True
        return True
    except Exception as e:
        # ImportError以外にも、DLLが見つからない等の例外が発生しうる
        print(f"pillow-heif initialization failed: {e}")
        _heif_available = False
        return False


def convert_heic_to_temp_jpeg(heic_path: str) -> str:
    """
    HEICファイルを一時的なJPEGに変換

    Args:
        heic_path: 変換元のHEICファイルパス

    Returns:
        変換後の一時JPEGファイルパス。変換失敗時はNone。

    Note:
        一時ファイルは呼び出し元で削除する責任がある。
        delete=Falseで作成しているため、明示的な削除が必要。
    """
    # 方法1: pillow-heif（クロスプラットフォーム、最も信頼性が高い）
    if _init_heif_support():
        try:
            from PIL import Image
            img = Image.open(heic_path)
            # EXIF情報（撮影日時、カメラ情報等）を保持してJPEGに変換
            # これがないと、写真の向き情報などが失われる
            exif = img.info.get('exif', None)
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            if exif:
                img.save(temp_file.name, 'JPEG', quality=95, exif=exif)
            else:
                img.save(temp_file.name, 'JPEG', quality=95)
            return temp_file.name
        except Exception as e:
            print(f"HEIC conversion with pillow-heif failed: {e}")

    # 方法2: sips（macOS専用の画像処理コマンド）
    # macOSには標準でHEIC対応のsipsコマンドが含まれている
    if platform.system() == 'Darwin':
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Windowsではファイルを閉じないと別プロセスからアクセスできない
        try:
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', heic_path, '--out', temp_path],
                check=True, capture_output=True
            )
            return temp_path
        except subprocess.CalledProcessError as e:
            print(f"HEIC conversion with sips failed: {e}")

    # 方法3: ImageMagick（Windowsでpillow-heifが動かない場合のフォールバック）
    if platform.system() == 'Windows':
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # ImageMagick 7.x は magick コマンド
        try:
            subprocess.run(
                ['magick', heic_path, '-quality', '95', temp_path],
                check=True, capture_output=True
            )
            return temp_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # ImageMagick 6.x は convert コマンド（ただしWindowsのconvert.exeと競合注意）
        try:
            subprocess.run(
                ['convert', heic_path, '-quality', '95', temp_path],
                check=True, capture_output=True
            )
            return temp_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return None


# =============================================================================
# 画像処理関数
# =============================================================================


def auto_color_correction(img: np.ndarray, settings: dict = None) -> np.ndarray:
    """
    自動色調補正（ホワイトバランス + CLAHE）

    プロジェクター投影写真でよくある問題:
    - 照明の色かぶり（蛍光灯で青っぽい、白熱灯で黄色っぽい）
    - コントラスト不足（暗い部屋で撮影すると全体的に白っぽくなる）

    これらを自動補正するため、2段階の処理を行う:
    1. Gray World ホワイトバランス: 色かぶりを除去
    2. CLAHE: 局所的なコントラストを改善

    Args:
        img: 入力画像（BGR形式、OpenCVの標準フォーマット）
        settings: 色調補正パラメータ辞書
            - enabled: 色調補正の有効/無効 (デフォルト: True)
            - white_balance: ホワイトバランス補正の有効/無効 (デフォルト: True)
            - clahe_enabled: CLAHEの有効/無効 (デフォルト: True)
            - clahe_clip_limit: CLAHEのclipLimit (デフォルト: 2.0)
            - clahe_grid_size: CLAHEのtileGridSize (デフォルト: 8)

    Returns:
        色調補正後の画像（BGR形式）
    """
    if settings is None:
        settings = {}

    if not settings.get('enabled', True):
        return img

    result = img.copy()

    # -------------------------------------------------------------------------
    # ステップ1: ホワイトバランス補正（Gray World assumption）
    # -------------------------------------------------------------------------
    # 理論: 「自然なシーンでは、全ピクセルの平均色はグレーになるはず」
    # 実装: 各チャンネル(B,G,R)の平均を計算し、全体の平均輝度に揃える
    # 効果: 蛍光灯の青かぶりや白熱灯の黄かぶりを除去
    if settings.get('white_balance', True):
        result = result.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3

        # 各チャンネルを平均輝度に合わせてスケーリング
        # ゼロ除算を防ぐためにチェック
        if avg_b > 0:
            result[:, :, 0] = result[:, :, 0] * (avg_gray / avg_b)
        if avg_g > 0:
            result[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
        if avg_r > 0:
            result[:, :, 2] = result[:, :, 2] * (avg_gray / avg_r)

        # 0-255の範囲にクリップして整数に戻す
        result = np.clip(result, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # ステップ2: CLAHE（Contrast Limited Adaptive Histogram Equalization）
    # -------------------------------------------------------------------------
    # 通常のヒストグラム平坦化は画像全体に適用するため、
    # 局所的なコントラストが失われることがある。
    # CLAHEは画像を小さなタイルに分割し、各タイルで個別に平坦化する。
    # clip_limit: コントラスト増強の上限（高すぎるとノイズが増幅される）
    # grid_size: タイルのサイズ（小さいほど局所的な補正が強くなる）
    if settings.get('clahe_enabled', True):
        clip_limit = settings.get('clahe_clip_limit', 2.0)
        grid_size = settings.get('clahe_grid_size', 8)

        # LAB色空間に変換してL（輝度）チャンネルのみに適用
        # 理由: RGBで直接処理すると色相が変わってしまうため
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


def perspective_transform_cv(input_path: str, corners: list, output_path: str,
                              output_size: tuple = (1920, 1080),
                              color_settings: dict = None) -> bool:
    """
    OpenCVによる台形補正（透視変換）

    斜めから撮影した台形の画像を、正面から見た矩形に変換する。
    数学的には、4点の対応関係から3x3の透視変換行列を求め、
    その行列を使って画像全体を変換する。

    Args:
        input_path: 入力画像パス
        corners: 4隅の座標 [(x,y), ...] 左上→右上→右下→左下の順（時計回り）
        output_path: 出力画像パス
        output_size: 出力サイズ (width, height)、デフォルトはFullHD
        color_settings: 色調補正パラメータ辞書

    Returns:
        成功時True、失敗時False

    Note:
        cornersの順序が重要。左上から時計回りで指定しないと
        結果が回転したり歪んだりする。
    """
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False

        # 入力座標（ユーザーが指定した4隅）
        src = np.float32(corners)

        # 出力座標（補正後の矩形の4隅）
        # 同じ順序（左上→右上→右下→左下）で対応付ける
        w, h = output_size
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 透視変換行列を計算
        # 4点の対応から3x3の射影変換行列を求める
        matrix = cv2.getPerspectiveTransform(src, dst)

        # 台形補正を適用
        # INTER_LINEAR（デフォルト）でリサンプリング
        result = cv2.warpPerspective(img, matrix, output_size)

        # 色調補正を適用（プロジェクター写真の色かぶり等を補正）
        result = auto_color_correction(result, color_settings)

        # -------------------------------------------------------------------------
        # 出力形式に応じて保存
        # PNG: 可逆圧縮（品質劣化なし）
        #   - 圧縮レベル0-9（0=無圧縮/高速、9=最大圧縮/低速）
        #   - レベル3は圧縮速度とファイルサイズのバランスが良い
        #   - 圧縮レベルは画質に影響しない（PNGは常に可逆）
        # JPEG: 非可逆圧縮（ファイルサイズ重視）
        #   - 品質95%で視覚的にほぼ劣化なし
        # -------------------------------------------------------------------------
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

    処理フロー:
    1. 画像をリサイズ（高速化のため）
    2. グレースケール変換 → ぼかし → Cannyエッジ検出
    3. モルフォロジー処理でエッジを強化
    4. 輪郭検出 → 4点の多角形を探す
    5. 左上から時計回りの順序に並び替え

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

    Note:
        プロジェクター画面は通常、写真の中で最も大きな四角形なので、
        面積が大きい順に輪郭をチェックしている。
    """
    if settings is None:
        settings = {}

    # パラメータ取得
    canny_low = settings.get('canny_low', 50)
    canny_high = settings.get('canny_high', 150)
    blur_size = settings.get('blur_size', 5)
    approx_epsilon = settings.get('approx_epsilon', 0.02)
    min_area_ratio = settings.get('min_area_ratio', 0.05)

    img = cv2.imread(image_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    # -------------------------------------------------------------------------
    # ステップ1: リサイズ（処理高速化のため）
    # -------------------------------------------------------------------------
    # 4000x3000のような高解像度画像をそのまま処理すると遅いため、
    # 長辺を1000pxに縮小して処理し、最後に座標をスケールバックする
    max_dim = 1000
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    if scale < 1.0:
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        img_resized = img
        scale = 1.0

    # -------------------------------------------------------------------------
    # ステップ2: 前処理（グレースケール → ぼかし）
    # -------------------------------------------------------------------------
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーでノイズを除去（カーネルサイズは奇数でなければならない）
    blur_k = blur_size if blur_size % 2 == 1 else blur_size + 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # -------------------------------------------------------------------------
    # ステップ3: エッジ検出（Canny）
    # -------------------------------------------------------------------------
    # canny_low/highは勾配の閾値。この範囲の勾配を持つピクセルがエッジとして検出される
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # モルフォロジー処理: 膨張→収縮で途切れたエッジを接続
    # これによりプロジェクター画面の輪郭が閉じた四角形になりやすくなる
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # -------------------------------------------------------------------------
    # ステップ4: 輪郭検出と四角形の探索
    # -------------------------------------------------------------------------
    # RETR_EXTERNAL: 最も外側の輪郭のみ取得（入れ子の輪郭は無視）
    # CHAIN_APPROX_SIMPLE: 直線部分の中間点を省略してメモリ節約
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 面積の大きい順にソート（プロジェクター画面は通常最大の四角形）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 上位10個の輪郭から四角形を探す
    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        # approxPolyDPで輪郭を多角形に近似
        # epsilon（許容誤差）が大きいほど頂点数が減る
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)

        # 4頂点の多角形で、画像面積の一定比率以上の場合
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = img_resized.shape[0] * img_resized.shape[1]

            if area > img_area * min_area_ratio:
                pts = approx.reshape(4, 2)
                ordered = order_corners(pts)

                # 縮小した分をスケールバックして元の座標系に戻す
                ordered = [(int(x / scale), int(y / scale)) for x, y in ordered]
                return ordered

    return None


def order_corners(pts):
    """
    4点を左上、右上、右下、左下の順に並び替え

    透視変換では点の対応関係が重要なため、
    検出された4点を常に同じ順序に整列させる必要がある。

    アルゴリズム:
    - 左上: x+y が最小の点（左上隅に近い）
    - 右下: x+y が最大の点（右下隅に近い）
    - 右上: x-y が最小の点（右上隅に近い）
    - 左下: x-y が最大の点（左下隅に近い）

    Args:
        pts: 4点のnumpy配列 shape=(4, 2)

    Returns:
        [(x, y), ...] 左上、右上、右下、左下の順
    """
    # 重心を計算（角度ソート用、現在は使用していない）
    center = pts.mean(axis=0)

    # 角度でソート（デバッグ用に残している）
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    # 最も左上の点を見つける（x+yが最小）
    sums = sorted_pts.sum(axis=1)
    top_left_idx = np.argmin(sums)
    ordered = np.roll(sorted_pts, -top_left_idx, axis=0)

    # 実際の並び替えはx+y、x-yの性質を使う
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]      # 左上: x+yが最小
    rect[2] = pts[np.argmax(s)]      # 右下: x+yが最大
    rect[1] = pts[np.argmin(diff)]   # 右上: x-yが最小（xが大きくyが小さい）
    rect[3] = pts[np.argmax(diff)]   # 左下: x-yが最大（xが小さくyが大きい）

    return [(int(x), int(y)) for x, y in rect]


class DetectionSettingsDialog(QDialog):
    """
    自動認識パラメータ設定ダイアログ

    プレゼンテーション画面の4隅を自動検出するためのパラメータを調整する。
    ダイアログ内でリアルタイムプレビューを表示し、パラメータ変更の効果を
    即座に確認できる。

    主なパラメータ:
        - Canny閾値: エッジ検出の感度を調整。低いほど多くのエッジを検出
        - ブラーサイズ: ノイズ除去の強度。大きいほどノイズに強いが、
          細かいエッジも消えてしまう
        - 輪郭近似精度: 検出した輪郭を多角形に近似する際の許容誤差
        - 最小面積比率: 小さすぎる四角形を除外するための閾値

    使用シーン:
        - デフォルトパラメータで検出に失敗する場合
        - 照明条件や背景が特殊な写真を処理する場合
        - より精密な検出が必要な場合

    Note:
        OKボタンを押すと、検出結果がメインキャンバスに自動適用される。
        キャンセルするとパラメータ変更は破棄される。
    """

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

    def showEvent(self, event):
        """表示時に親ウィンドウの中央に配置"""
        super().showEvent(event)
        # レイアウト確定後に中央配置を実行
        QTimer.singleShot(0, self._center_on_parent)

    def _center_on_parent(self):
        """親ウィンドウの中央に配置"""
        if self.parent():
            main_geo = self.parent().geometry()
            main_center_x = main_geo.x() + main_geo.width() // 2
            main_center_y = main_geo.y() + main_geo.height() // 2
            dialog_x = main_center_x - self.width() // 2
            dialog_y = main_center_y - self.height() // 2
            self.move(dialog_x, dialog_y)

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


class ColorCorrectionSettingsDialog(QDialog):
    """
    色調補正パラメータ設定ダイアログ

    プロジェクター投影写真の色かぶりやコントラスト不足を補正するための
    パラメータを調整する。補正前/補正後のプレビューを並べて表示し、
    効果を視覚的に確認できる。

    補正処理:
        1. ホワイトバランス（Gray World）: 照明による色かぶりを除去
           - 蛍光灯の青っぽさや白熱灯の黄色っぽさを補正
        2. CLAHE: 局所的なコントラストを改善
           - 暗い部屋で撮影した写真のメリハリを強調

    パラメータ:
        - ホワイトバランス: ON/OFF
        - CLAHE有効化: ON/OFF
        - コントラスト制限 (clip_limit): 大きいほど効果が強い
        - グリッドサイズ: 小さいほど局所的な補正が強くなる

    Note:
        プレビューには台形補正も適用された状態で表示される。
        これは最終出力に近い状態を確認できるようにするため。
    """

    def __init__(self, parent=None, settings=None, image_path=None, corners=None):
        super().__init__(parent)
        self.setWindowTitle("色調補正設定")
        self.setMinimumSize(900, 600)
        self.main_app = parent
        self.image_path = image_path
        self.corners = corners  # 台形補正用の座標

        # 現在の設定（デフォルト値）
        self.settings = settings or {}

        layout = QHBoxLayout(self)

        # === 左側: パラメータ設定 ===
        left_panel = QVBoxLayout()
        from PySide6.QtWidgets import QCheckBox

        # ホワイトバランス
        self.white_balance_check = QCheckBox("ホワイトバランス補正")
        self.white_balance_check.setChecked(self.settings.get('white_balance', True))
        self.white_balance_check.setToolTip("プロジェクターの色かぶりを補正")
        self.white_balance_check.stateChanged.connect(self.on_value_changed)
        left_panel.addWidget(self.white_balance_check)

        left_panel.addSpacing(10)

        # CLAHE設定
        clahe_group = QGroupBox("CLAHE（コントラスト強調）")
        clahe_layout = QVBoxLayout()

        self.clahe_check = QCheckBox("CLAHEを有効にする")
        self.clahe_check.setChecked(self.settings.get('clahe_enabled', True))
        self.clahe_check.setToolTip("暗い部分のコントラストを改善")
        self.clahe_check.stateChanged.connect(self.on_value_changed)
        clahe_layout.addWidget(self.clahe_check)

        form = QFormLayout()

        # clipLimit
        self.clip_limit = QDoubleSpinBox()
        self.clip_limit.setRange(1.0, 10.0)
        self.clip_limit.setSingleStep(0.5)
        self.clip_limit.setDecimals(1)
        self.clip_limit.setValue(self.settings.get('clahe_clip_limit', 2.0))
        self.clip_limit.setToolTip("大きいほどコントラストが強調（推奨: 1.5-4.0）")
        self.clip_limit.valueChanged.connect(self.on_value_changed)
        form.addRow("コントラスト制限:", self.clip_limit)

        # gridSize
        self.grid_size = QSpinBox()
        self.grid_size.setRange(2, 16)
        self.grid_size.setValue(self.settings.get('clahe_grid_size', 8))
        self.grid_size.setToolTip("小さいほど局所的に補正（推奨: 4-12）")
        self.grid_size.valueChanged.connect(self.on_value_changed)
        form.addRow("グリッドサイズ:", self.grid_size)

        clahe_layout.addLayout(form)
        clahe_group.setLayout(clahe_layout)
        left_panel.addWidget(clahe_group)

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

        # === 右側: プレビュー（補正前/補正後） ===
        preview_panel = QVBoxLayout()

        # 補正前ラベル
        preview_panel.addWidget(QLabel("▼ 補正前"))
        self.before_label = QLabel()
        self.before_label.setMinimumSize(300, 200)
        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        preview_panel.addWidget(self.before_label, 1)

        # 補正後ラベル
        preview_panel.addWidget(QLabel("▼ 補正後"))
        self.after_label = QLabel()
        self.after_label.setMinimumSize(300, 200)
        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        preview_panel.addWidget(self.after_label, 1)

        layout.addLayout(preview_panel, 1)

        # 初期プレビュー
        self.load_preview_image()
        self.on_value_changed()

    def load_preview_image(self):
        """プレビュー用画像を読み込み（台形補正を適用）"""
        if not self.image_path or not self.corners or len(self.corners) != 4:
            self.before_label.setText("画像または座標が\n設定されていません")
            self.after_label.setText("")
            self.original_image = None
            return

        # HEICの場合は変換
        input_path = self.image_path
        if self.image_path.lower().endswith(('.heic', '.heif')):
            temp_path = convert_heic_to_temp_jpeg(self.image_path)
            if temp_path:
                input_path = temp_path

        # 画像を読み込み
        img = cv2.imread(input_path)
        if img is None:
            self.before_label.setText("画像を読み込めません")
            self.original_image = None
            return

        # 台形補正を適用（色調補正なし）
        src = np.float32(self.corners)
        output_size = (960, 540)  # プレビュー用に小さめ
        w, h = output_size
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        self.original_image = cv2.warpPerspective(img, matrix, output_size)

    def update_preview(self):
        """プレビュー画像を更新"""
        if self.original_image is None:
            return

        # 補正前画像を表示
        before_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w, ch = before_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        before_qimg = QImage(before_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        before_pixmap = QPixmap.fromImage(before_qimg)

        preview_w = self.before_label.width() - 10
        preview_h = self.before_label.height() - 10
        scaled_before = before_pixmap.scaled(preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.before_label.setPixmap(scaled_before)

        # 色調補正を適用
        current_settings = self.get_settings()
        corrected = auto_color_correction(self.original_image, current_settings)

        # 補正後画像を表示
        after_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        after_qimg = QImage(after_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        after_pixmap = QPixmap.fromImage(after_qimg)
        scaled_after = after_pixmap.scaled(preview_w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.after_label.setPixmap(scaled_after)

    def on_value_changed(self):
        """パラメータ変更時の処理"""
        # CLAHE関連のウィジェットの有効/無効
        clahe_enabled = self.clahe_check.isChecked()
        self.clip_limit.setEnabled(clahe_enabled)
        self.grid_size.setEnabled(clahe_enabled)

        self.update_preview()

    def reset_to_default(self):
        """デフォルト値に戻す"""
        self.white_balance_check.setChecked(True)
        self.clahe_check.setChecked(True)
        self.clip_limit.setValue(2.0)
        self.grid_size.setValue(8)

    def get_settings(self):
        """現在の設定を辞書で返す（enabledは外部から渡された値を維持）"""
        return {
            'enabled': self.settings.get('enabled', True),
            'white_balance': self.white_balance_check.isChecked(),
            'clahe_enabled': self.clahe_check.isChecked(),
            'clahe_clip_limit': self.clip_limit.value(),
            'clahe_grid_size': self.grid_size.value()
        }

    def showEvent(self, event):
        """表示時に親ウィンドウの中央に配置"""
        super().showEvent(event)
        QTimer.singleShot(0, self._center_on_parent)

    def _center_on_parent(self):
        """親ウィンドウの中央に配置"""
        if self.parent():
            main_geo = self.parent().geometry()
            main_center_x = main_geo.x() + main_geo.width() // 2
            main_center_y = main_geo.y() + main_geo.height() // 2
            dialog_x = main_center_x - self.width() // 2
            dialog_y = main_center_y - self.height() // 2
            self.move(dialog_x, dialog_y)


class ImageCanvas(QLabel):
    """
    画像表示と4隅座標選択用のカスタムキャンバス

    QLabel を継承し、以下の機能を提供:
        - 画像の表示とスケーリング（ウィジェットサイズに自動フィット）
        - 4隅座標のクリック指定とドラッグ調整
        - 拡大鏡表示（精密な座標調整のため）
        - 色調補正のプレビュー表示

    座標系:
        - 表示座標系: ウィジェット上のピクセル座標（スケーリング後）
        - 元画像座標系: 元画像のピクセル座標（実際の処理に使用）
        座標変換には scale と offset を使用する。

    拡大鏡の動作:
        - マウス位置周辺を拡大して画面中央に表示
        - 象限に応じたオフセット表示で、マウス位置が見やすくなるよう調整
        - 4隅設定済みの場合は選択領域をマゼンタでオーバーレイ

    ドラッグ操作:
        - 既存のコーナーをドラッグで移動可能
        - 精密モード（drag_precision）でマウス移動量を減衰させ、細かい調整が可能

    パフォーマンス最適化:
        - 色調補正済み画像をキャッシュして再計算を回避
        - ドラッグ中はコーナーマーカーのみ再描画（背景は再計算しない）
        - 画像切り替え時は遅延色調補正（即座にオリジナルを表示、50ms後に補正適用）

    Signals:
        coordinates_changed: 座標が変更された時に発火
    """

    coordinates_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2d2d2d; border: 1px solid #555;")
        self.setMouseTracking(True)  # マウス移動を追跡（拡大鏡表示のため）
        self.setCursor(Qt.CrossCursor)  # 精密選択用クロスヘアカーソル
        self.setAttribute(Qt.WA_Hover, True)  # ホバーイベントを有効化

        # ---------------------------------------------------------------------
        # 画像表示関連
        # ---------------------------------------------------------------------
        self.original_pixmap = None   # 元画像（フルサイズ）
        self.display_pixmap = None    # 表示用画像（スケール済み、コーナー描画後）
        self.base_pixmap = None       # 拡大鏡を描画する前のベース画像

        # 座標変換用: 表示座標 = 元座標 * scale + offset
        # ウィジェットサイズに合わせて動的に計算される
        self.scale = 1.0
        self.offset_x = 0  # 画像を中央配置するための水平オフセット
        self.offset_y = 0  # 画像を中央配置するための垂直オフセット

        self.image_path = None
        self.temp_file = None  # HEIC変換時の一時ファイル（終了時に削除）
        self.original_size = (0, 0)

        # ---------------------------------------------------------------------
        # 拡大鏡設定
        # 精密な座標指定を可能にするため、マウス位置周辺を拡大表示する
        # ---------------------------------------------------------------------
        self.magnifier_base_size = 360  # 拡大鏡の基準サイズ（長辺、ピクセル）
        self.magnifier_zoom = 1.75      # 拡大率（1.75倍で見やすいバランス）
        self.mouse_pos = None           # 現在のマウス位置（拡大鏡の中心）

        # ---------------------------------------------------------------------
        # ドラッグ操作用
        # コーナーをドラッグで微調整できるようにする
        # 精密モード: マウス移動量を減衰させ、細かい調整を可能にする
        # ---------------------------------------------------------------------
        self.dragging_corner = None     # ドラッグ中のコーナーインデックス (0-3)
        self.drag_threshold = 15        # ドラッグ判定の距離（この範囲内のクリックでドラッグ開始）
        self.drag_start_mouse = None    # ドラッグ開始時のマウス位置
        self.drag_start_corner = None   # ドラッグ開始時のコーナー位置
        # 精密モードの減衰率: マウスを100px動かしても、コーナーは15px（0.15倍）しか動かない
        # これにより、高解像度画像でも1ピクセル単位の調整が可能になる
        self.drag_precision = 0.15

        # ---------------------------------------------------------------------
        # ガイドメッセージ（自動検出失敗時など）
        # ---------------------------------------------------------------------
        self.show_guide_message = False
        self.guide_message = ""

        # ---------------------------------------------------------------------
        # 色調補正関連
        # パフォーマンス向上のため、色調補正結果をキャッシュする
        # ---------------------------------------------------------------------
        self.color_settings = {}
        self.preview_pixmap = None              # 台形補正+色調補正のプレビュー
        self.color_corrected_pixmap = None      # 色調補正済み画像（キャッシュ）
        self.color_correction_cache_valid = False  # キャッシュが有効かどうか
        self.scaled_image_cache = None          # スケール済み背景画像（ドラッグ中の高速描画用）

        # 遅延色調補正: 画像切り替え時、まずオリジナルを表示し、
        # 50ms後に色調補正を適用する。これにより体感的なレスポンスが向上する
        self.color_correction_pending = False
        self.color_correction_timer = QTimer()
        self.color_correction_timer.setSingleShot(True)
        self.color_correction_timer.timeout.connect(self._apply_deferred_color_correction)

        # ---------------------------------------------------------------------
        # 4隅の座標（表示座標系）
        # 左上→右上→右下→左下の順で格納（時計回り）
        # この順序は透視変換の入力として使用される
        # ---------------------------------------------------------------------
        self.corners = []  # [(x, y), ...]
        self.corner_labels = ["左上", "右上", "右下", "左下"]
        # 各コーナーを視覚的に区別するための色
        # 左上=赤、右上=緑、右下=青、左下=黄
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
        self.color_correction_cache_valid = False  # キャッシュを無効化
        self.color_corrected_pixmap = None
        self.scaled_image_cache = None  # スケーリングキャッシュも無効化

        # 最初はオリジナル画像を即座に表示（色調補正なし）
        self.color_correction_pending = True
        self.update_display()

        # 色調補正は遅延実行（50ms後）
        if self.color_settings.get('enabled', True):
            self.color_correction_timer.start(50)

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

    def set_color_settings(self, settings: dict):
        """色調補正設定を更新"""
        self.color_settings = settings.copy()  # 明示的にコピー
        self.preview_pixmap = None  # キャッシュをクリア
        self.color_correction_cache_valid = False  # 色調補正キャッシュを無効化
        self.update_display()
        self.repaint()  # 強制再描画

    def generate_preview(self):
        """台形補正+色調補正のプレビュー画像を生成"""
        if not self.original_pixmap or len(self.corners) != 4:
            return None

        # 元画像座標を取得
        corners_orig = self.get_corners_original()

        # 画像読み込みパス
        load_path = self.temp_file if self.temp_file else self.image_path
        if not load_path:
            return None

        # OpenCVで画像読み込み
        img = cv2.imread(load_path)
        if img is None:
            return None

        # プレビュー用にリサイズ（パフォーマンス向上）
        preview_size = (640, 360)
        src = np.float32(corners_orig)
        w, h = preview_size
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 台形補正
        matrix = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(img, matrix, preview_size)

        # 色調補正
        result = auto_color_correction(result, self.color_settings)

        # QPixmapに変換
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        qimg = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def generate_preview_full(self):
        """台形補正+色調補正のフルサイズプレビュー画像を生成（1920x1080）"""
        if not self.original_pixmap or len(self.corners) != 4:
            return None

        # 元画像座標を取得
        corners_orig = self.get_corners_original()

        # 画像読み込みパス
        load_path = self.temp_file if self.temp_file else self.image_path
        if not load_path:
            return None

        # OpenCVで画像読み込み
        img = cv2.imread(load_path)
        if img is None:
            return None

        # フルサイズ出力（出力サイズと同一）
        output_size = (1920, 1080)
        src = np.float32(corners_orig)
        w, h = output_size
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 台形補正
        matrix = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(img, matrix, output_size)

        # 色調補正
        result = auto_color_correction(result, self.color_settings)

        # QPixmapに変換
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        qimg = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def get_display_image(self):
        """表示用画像を取得（色調補正ON時は補正適用、キャッシュ使用）"""
        if not self.color_settings.get('enabled', True):
            # 色調補正OFFの場合はオリジナル画像
            return self.original_pixmap

        # 遅延処理待ちの場合はオリジナル画像を返す（高速化）
        if self.color_correction_pending:
            return self.original_pixmap

        # キャッシュが有効な場合はキャッシュを返す
        if self.color_correction_cache_valid and self.color_corrected_pixmap:
            return self.color_corrected_pixmap

        # 色調補正を計算
        self._compute_color_correction()
        return self.color_corrected_pixmap if self.color_corrected_pixmap else self.original_pixmap

    def _compute_color_correction(self):
        """色調補正を計算してキャッシュに保存"""
        load_path = self.temp_file if self.temp_file else self.image_path
        if not load_path:
            return

        # OpenCVで画像読み込み
        img = cv2.imread(load_path)
        if img is None:
            return

        # 色調補正を適用
        result = auto_color_correction(img, self.color_settings)

        # QPixmapに変換
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        qimg = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.color_corrected_pixmap = QPixmap.fromImage(qimg.copy())
        self.color_correction_cache_valid = True

    def _apply_deferred_color_correction(self):
        """遅延色調補正を適用"""
        if not self.original_pixmap or not self.color_settings.get('enabled', True):
            self.color_correction_pending = False
            return

        # 色調補正を計算
        self._compute_color_correction()
        self.color_correction_pending = False

        # キャッシュをクリアして再描画
        self.scaled_image_cache = None
        self.update_display()
        self._draw_corners()
        self.update()

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

        # 表示用画像を準備（色調補正適用の有無）
        display_source = self.get_display_image()

        # スケーリング（IgnoreAspectRatioで正確なサイズを指定）
        scaled = display_source.scaled(
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

        # オリジナル画像（色調補正適用済みまたは未適用）を表示
        painter.drawPixmap(self.offset_x, self.offset_y, scaled)
        painter.end()

        # スケーリング済み画像をキャッシュ（コーナーなし）
        self.scaled_image_cache = self.display_pixmap.copy()

        # 四隅マーカーを描画
        self._draw_corners()

    def _draw_corners(self):
        """四隅マーカーを描画（内部用）"""
        if not self.display_pixmap:
            return

        painter = QPainter(self.display_pixmap)

        # 四隅マーカーを描画（常に表示）
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
            # 4隅設定完了時は閉じる
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

    def update_corners_only(self):
        """コーナーマーカーのみを再描画（ドラッグ中の軽量更新用）"""
        if not self.scaled_image_cache:
            # キャッシュがない場合はフル更新
            self.update_display()
            return

        # キャッシュからコピーして使用
        self.display_pixmap = self.scaled_image_cache.copy()
        self._draw_corners()

    def draw_magnifier(self):
        """
        拡大鏡を描画

        拡大鏡は画面中央に固定表示され、マウス位置周辺を拡大して表示する。
        これにより、ユーザーは画像の細部を確認しながら精密な座標指定ができる。

        設計上の工夫:
            1. 象限オフセット: マウス位置が拡大鏡の中心ではなく、
               マウスが画面のどの象限にあるかに応じて少しずらして表示する。
               これにより、マウス位置の周囲（次にクリックしそうな場所）が
               より多く見えるようになる。
            2. 境界処理: 画像の端でも拡大鏡が正しく表示されるよう、
               表示範囲が画像外にはみ出る場合はクリッピングする。
            3. クロスヘア: 現在のマウス位置を赤い十字線で示す。
        """
        if not self.base_pixmap or not self.mouse_pos or not self.original_pixmap:
            if self.base_pixmap:
                self.setPixmap(self.base_pixmap)
            return

        mx, my = self.mouse_pos

        # 画像範囲内かチェック（画像外ではベース画像をそのまま表示）
        img_w = self.original_pixmap.width() * self.scale
        img_h = self.original_pixmap.height() * self.scale

        if not (self.offset_x <= mx <= self.offset_x + img_w and
                self.offset_y <= my <= self.offset_y + img_h):
            self.setPixmap(self.base_pixmap)
            return

        # 拡大鏡用のピクスマップを作成（ベース画像のコピーに描画）
        result = self.base_pixmap.copy()
        painter = QPainter(result)

        # 表示座標系から元画像座標系への変換
        corner_orig_x = (mx - self.offset_x) / self.scale
        corner_orig_y = (my - self.offset_y) / self.scale

        # -------------------------------------------------------------------------
        # 拡大鏡のサイズ決定
        # 画像のアスペクト比を維持した拡大鏡にする
        # これにより、画像と同じ形状の領域が表示され、違和感がない
        # -------------------------------------------------------------------------
        orig_w = self.original_pixmap.width()
        orig_h = self.original_pixmap.height()
        aspect_ratio = orig_w / orig_h

        if aspect_ratio >= 1:  # 横長画像
            mag_width = self.magnifier_base_size
            mag_height = int(self.magnifier_base_size / aspect_ratio)
        else:  # 縦長画像
            mag_height = self.magnifier_base_size
            mag_width = int(self.magnifier_base_size * aspect_ratio)

        # 拡大前の表示領域サイズ（元画像座標系）
        src_width = mag_width / self.magnifier_zoom
        src_height = mag_height / self.magnifier_zoom

        # -------------------------------------------------------------------------
        # 象限オフセット計算
        # なぜオフセットが必要か？
        # マウスを画像の左上隅に置くとき、ユーザーは左上隅の周辺を見たい。
        # もし拡大鏡がマウス位置を中心に表示すると、左上隅より左・上は
        # 画像外なので表示できない。そこで、マウス位置を少し端に寄せて表示し、
        # 画像内の有効な領域をより多く見えるようにする。
        # -------------------------------------------------------------------------
        image_center_x = orig_w / 2
        image_center_y = orig_h / 2

        is_left = corner_orig_x < image_center_x
        is_top = corner_orig_y < image_center_y

        # オフセット量は表示領域の25%
        # 左側にいる場合は右方向にオフセット（マウス位置が左側に表示される）
        # 上側にいる場合は下方向にオフセット（マウス位置が上側に表示される）
        offset_ratio = 0.25
        view_offset_x = src_width * offset_ratio if is_left else -src_width * offset_ratio
        view_offset_y = src_height * offset_ratio if is_top else -src_height * offset_ratio

        # 表示領域の中心（オフセット適用後）
        view_center_x = corner_orig_x + view_offset_x
        view_center_y = corner_orig_y + view_offset_y
        src_x = view_center_x - src_width / 2
        src_y = view_center_y - src_height / 2

        # -------------------------------------------------------------------------
        # 境界クリッピング処理
        # 表示領域が画像外にはみ出る場合、はみ出た分を切り詰める
        # clip_offset_x/y: 切り詰めた分を記録（クロスヘア位置計算に使用）
        # -------------------------------------------------------------------------
        src_rect_x = int(src_x)
        src_rect_y = int(src_y)
        src_rect_w = int(src_width)
        src_rect_h = int(src_height)

        clip_offset_x = 0
        clip_offset_y = 0

        # 左端・上端のクリッピング
        if src_rect_x < 0:
            clip_offset_x = -src_rect_x
            src_rect_w += src_rect_x
            src_rect_x = 0
        if src_rect_y < 0:
            clip_offset_y = -src_rect_y
            src_rect_h += src_rect_y
            src_rect_y = 0
        # 右端・下端のクリッピング
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
            # ドラッグ中は軽量更新（コーナーマーカーのみ再描画）
            self.update_corners_only()
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
    """
    台形補正ツールのメインウィンドウ

    アプリケーション全体の制御と、各コンポーネント間の連携を担当する。

    主な責務:
        - UI全体のレイアウト管理
        - ファイル一覧の表示と選択処理
        - 設定ファイル（JSON）の読み書き
        - 一括処理（PNG/PDF出力）の実行
        - メニューバーとショートカットの管理

    設定ファイル:
        - perspective_config.json: 作業ディレクトリに保存
          - 各画像の4隅座標
          - 出力ファイル名
          - 自動認識パラメータ
          - 色調補正パラメータ
        - ~/.perspective_corrector_recent.json: ホームディレクトリに保存
          - 最近使用したフォルダの履歴（最大10件）

    ワークフロー:
        1. フォルダを開く（メニューまたはドラッグ&ドロップ）
        2. ファイル一覧から画像を選択
        3. 自動検出または手動で4隅を指定
        4. 必要に応じて座標を微調整
        5. 一括処理でPNG/PDF出力

    Note:
        画像を切り替えると現在の座標が自動保存される。
        アプリ終了時も自動保存される。
    """

    CONFIG_FILE = "perspective_config.json"

    RECENT_FOLDERS_FILE = ".perspective_corrector_recent.json"  # ホームに保存

    def __init__(self, start_dir: str = None):
        super().__init__()
        self.setMinimumSize(1500, 800)

        self.start_dir = start_dir or str(Path.cwd())
        self.current_image = None
        self.config = {}
        self.detection_settings = {}  # 自動認識パラメータ
        self.color_settings = {}  # 色調補正パラメータ
        self.recent_folders = []  # 最近使用したフォルダ

        self.load_recent_folders()
        self.load_config()
        self.setup_ui()
        self.update_window_title()
        # キャンバスに色調補正設定を渡す
        self.canvas.color_settings = self.color_settings

    def load_config(self):
        """設定ファイルを読み込み"""
        config_path = Path(self.start_dir) / self.CONFIG_FILE
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 検出設定を分離
                    self.detection_settings = data.pop('_detection_settings', {})
                    self.color_settings = data.pop('_color_settings', {})
                    self.config = data
            except Exception as e:
                print(f"Config load error: {e}")
                self.config = {}
                self.detection_settings = {}
                self.color_settings = {}

    def save_config(self):
        """設定ファイルに保存"""
        config_path = Path(self.start_dir) / self.CONFIG_FILE
        try:
            # 検出設定・色調設定を含めて保存
            data = dict(self.config)
            if self.detection_settings:
                data['_detection_settings'] = self.detection_settings
            if self.color_settings:
                data['_color_settings'] = self.color_settings
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

        # 設定パネル（認識設定、色調設定、色調補正トグル）
        settings_panel = QWidget()
        settings_layout = QHBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 5)

        # 認識設定 - 紫（設定/調整）
        self.settings_btn = QPushButton("認識設定")
        self.settings_btn.setMinimumHeight(32)
        self.settings_btn.setMinimumWidth(90)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        self.settings_btn.clicked.connect(self.show_detection_settings)
        settings_layout.addWidget(self.settings_btn)

        # 色調設定 - シアン（色調整）
        self.color_settings_btn = QPushButton("色調設定")
        self.color_settings_btn.setMinimumHeight(32)
        self.color_settings_btn.setMinimumWidth(90)
        self.color_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #17A2B8;
                color: white;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.color_settings_btn.clicked.connect(self.show_color_settings)
        settings_layout.addWidget(self.color_settings_btn)

        settings_layout.addSpacing(20)

        # 色調補正トグル
        from PySide6.QtWidgets import QCheckBox
        self.color_correction_toggle = QCheckBox("色調補正")
        self.color_correction_toggle.setChecked(self.color_settings.get('enabled', True))
        self.color_correction_toggle.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                color: #333;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.color_correction_toggle.stateChanged.connect(self.on_color_correction_toggled)
        settings_layout.addWidget(self.color_correction_toggle)

        settings_layout.addStretch()
        right_layout.addWidget(settings_panel)

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

                # 4隅設定済みの場合は右カラムの背景色を変更
                if rel_path in self.config:
                    corners = self.config[rel_path].get("corners", [])
                    if len(corners) == 4:
                        output_item.setBackground(QColor(144, 238, 144, 100))  # 薄緑
                    elif len(corners) > 0:
                        output_item.setBackground(QColor(255, 200, 100, 100))  # 薄オレンジ（途中）

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

    def show_color_settings(self):
        """色調補正設定ダイアログを表示"""
        # 画像パスを取得（HEIC対応）
        image_path = None
        corners = None
        if self.current_image:
            image_path = self.current_image
            if image_path.lower().endswith(('.heic', '.heif')) and self.canvas.temp_file:
                image_path = self.canvas.temp_file
            # 現在の座標を取得
            corners = self.canvas.corners if len(self.canvas.corners) == 4 else None

        dialog = ColorCorrectionSettingsDialog(self, self.color_settings, image_path, corners)
        dialog.resize(900, 600)
        self.center_dialog(dialog, 900, 600)
        if dialog.exec() == QDialog.Accepted:
            self.color_settings = dialog.get_settings()
            self.save_config()
            # トグルを同期
            self.color_correction_toggle.blockSignals(True)
            self.color_correction_toggle.setChecked(self.color_settings.get('enabled', True))
            self.color_correction_toggle.blockSignals(False)
            # キャンバスのプレビューを更新
            self.canvas.set_color_settings(self.color_settings)
            self.statusBar.showMessage("色調補正設定を保存しました")

    def on_color_correction_toggled(self, state):
        """色調補正トグルの切り替え"""
        enabled = self.color_correction_toggle.isChecked()
        self.color_settings['enabled'] = enabled
        self.save_config()
        # キャンバスのプレビューを更新
        self.canvas.set_color_settings(self.color_settings.copy())
        status = "有効" if enabled else "無効"
        self.statusBar.showMessage(f"色調補正を{status}にしました")

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
        # 座標が設定されているファイルのみカウント（4隅すべて設定済み）
        count = sum(
            1 for rel_path, data in self.config.items()
            if not rel_path.startswith('_') and len(data.get("corners", [])) == 4
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
        # 座標が設定されているファイルのみカウント（4隅すべて設定済み）
        files_to_process = []
        for rel_path, data in self.config.items():
            # _で始まるキーは設定用なのでスキップ
            if rel_path.startswith('_'):
                continue
            corners = data.get("corners", [])
            if len(corners) == 4:
                files_to_process.append((rel_path, data))

        if not files_to_process:
            self.show_message_box(QMessageBox.Warning, "警告", "処理するファイルがありません")
            return

        # 現在の座標を保存
        self.save_current_corners()

        # 出力形式選択ダイアログ
        output_format = self.show_output_format_dialog(len(files_to_process))
        if output_format is None:
            return

        if output_format == "pdf":
            self.run_batch_process_pdf(files_to_process)
        else:
            self.run_batch_process_png(files_to_process)

    def show_output_format_dialog(self, file_count):
        """出力形式選択ダイアログを表示"""
        dialog = QDialog(self)
        dialog.setWindowTitle("出力形式の選択")
        layout = QVBoxLayout(dialog)

        # 説明
        label = QLabel(f"{file_count} ファイルの台形補正を実行します。\n出力形式を選択してください。")
        layout.addWidget(label)

        # ラジオボタン
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        self.format_group = QButtonGroup(dialog)

        png_radio = QRadioButton("個別PNG（各ファイルを [出力名]_corrected.png として保存）")
        png_radio.setChecked(True)
        self.format_group.addButton(png_radio, 0)
        layout.addWidget(png_radio)

        pdf_radio = QRadioButton("1つのPDF（全ファイルを1つのPDFにまとめる、A4横）")
        self.format_group.addButton(pdf_radio, 1)
        layout.addWidget(pdf_radio)

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("実行")
        ok_btn.setStyleSheet("background-color: #27AE60; color: white;")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        dialog.setMinimumWidth(450)
        self.center_dialog(dialog)

        if dialog.exec() == QDialog.Accepted:
            return "pdf" if self.format_group.checkedId() == 1 else "png"
        return None

    def run_batch_process_png(self, files_to_process):
        """個別PNG出力の一括処理"""
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
            try:
                if perspective_transform_cv(input_path, corners, str(output_path),
                                           color_settings=self.color_settings):
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

    def run_batch_process_pdf(self, files_to_process):
        """
        PDF出力の一括処理

        複数の台形補正済み画像を1つのPDFファイルにまとめて出力する。
        プレゼンテーション資料の印刷やアーカイブに便利。

        PDF仕様:
            - 用紙サイズ: A4横（297mm × 210mm）
            - 画像配置: アスペクト比を維持して最大サイズで中央配置
            - 背景: 白
            - 解像度: 72dpi（PDF標準）

        処理フロー:
            1. 各画像に台形補正＋色調補正を適用（一時ファイルに保存）
            2. A4横サイズの白背景に画像を中央配置
            3. 全ページをPillowでPDFとして保存
            4. 一時ファイルを削除
        """
        from PIL import Image
        import io

        # デフォルトファイル名をフォルダ名から生成（直感的な命名）
        default_name = Path(self.start_dir).name + ".pdf"
        dialog = QFileDialog(self, "PDFを保存", str(Path(self.start_dir) / default_name))
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("PDF Files (*.pdf)")
        dialog.setDefaultSuffix("pdf")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        self.center_dialog(dialog, 700, 500)

        if dialog.exec() != QFileDialog.Accepted:
            return

        pdf_path = dialog.selectedFiles()[0]

        # プログレスダイアログ
        progress = QProgressDialog("PDF作成中...", "キャンセル", 0, len(files_to_process) + 1, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setFixedSize(400, 100)
        self.center_dialog(progress, 400, 100)

        # -------------------------------------------------------------------------
        # A4横サイズの定義（高品質印刷用: 300dpi）
        # A4用紙は国際規格ISO 216で 210mm × 297mm と定義されている
        # 横向き（landscape）では 297mm × 210mm になる
        #
        # 解像度の考え方:
        #   - 72dpi: 画面表示用（Web、プレビュー）
        #   - 150dpi: 一般的な印刷（ニュースレター等）
        #   - 300dpi: 高品質印刷（書籍、写真）
        #
        # 300dpiでのピクセル数計算:
        #   297mm × (300/25.4) ≈ 3508 pixels
        #   210mm × (300/25.4) ≈ 2480 pixels
        # -------------------------------------------------------------------------
        PDF_DPI = 300  # 高品質印刷用解像度
        A4_WIDTH_PX = int(297 * PDF_DPI / 25.4)   # 3508 pixels
        A4_HEIGHT_PX = int(210 * PDF_DPI / 25.4)  # 2480 pixels

        corrected_images = []  # PDFに含める画像リスト
        error_files = []       # エラーが発生したファイルのリスト
        temp_files = []        # 後で削除する一時ファイルのリスト

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

            # HEICの場合は先に変換
            input_path = str(full_path)
            temp_heic = None
            if str(full_path).lower().endswith(('.heic', '.heif')):
                temp_heic = convert_heic_to_temp_jpeg(str(full_path))
                if temp_heic:
                    input_path = temp_heic
                    temp_files.append(temp_heic)
                else:
                    error_files.append((rel_path, "HEIC変換エラー"))
                    continue

            # 一時ファイルに台形補正結果を保存
            temp_output = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_output.close()
            temp_files.append(temp_output.name)

            try:
                if perspective_transform_cv(input_path, corners, temp_output.name,
                                           color_settings=self.color_settings):
                    # 台形補正済み画像をPILで読み込み
                    img = Image.open(temp_output.name)
                    img = img.convert('RGB')  # PDFはRGBモードが必要

                    # -----------------------------------------------------------------
                    # A4横にフィットするようにリサイズ（アスペクト比維持）
                    # 画像が用紙より大きい場合は縮小、小さい場合はそのまま
                    # 幅・高さそれぞれの縮小率を計算し、小さい方を採用することで
                    # 画像が用紙からはみ出ないようにする
                    # -----------------------------------------------------------------
                    img_width, img_height = img.size
                    scale = min(A4_WIDTH_PX / img_width, A4_HEIGHT_PX / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)

                    # A4横サイズの白背景を作成し、リサイズした画像を中央に配置
                    # 余白は白で埋められる（印刷時に自然に見える）
                    a4_img = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), 'white')
                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    x = (A4_WIDTH_PX - new_width) // 2
                    y = (A4_HEIGHT_PX - new_height) // 2
                    a4_img.paste(resized, (x, y))

                    corrected_images.append(a4_img)
                else:
                    error_files.append((rel_path, "変換に失敗しました"))
            except Exception as e:
                error_files.append((rel_path, str(e)))

        # -------------------------------------------------------------------------
        # PDF保存
        # Pillowのsave_all=Trueを使用してマルチページPDFを生成
        # 最初の画像のsaveメソッドを呼び、残りをappend_imagesで追加する
        # -------------------------------------------------------------------------
        if corrected_images and not progress.wasCanceled():
            progress.setLabelText("PDFを保存中...")
            progress.setValue(len(files_to_process))
            QApplication.processEvents()

            try:
                # Pillowのマルチページ出力: 最初の画像に残りを追加
                # -------------------------------------------------------------------------
                # PDF保存時の解像度設定
                # Pillowのresolutionパラメータは画像のDPI情報をPDFに埋め込む
                # 300dpiを指定することで、PDF内の画像が高解像度として扱われ、
                # 印刷時に鮮明な出力が得られる
                # -------------------------------------------------------------------------
                corrected_images[0].save(
                    pdf_path,
                    save_all=True,
                    append_images=corrected_images[1:] if len(corrected_images) > 1 else [],
                    resolution=PDF_DPI  # 高品質印刷用解像度（300dpi）
                )
            except Exception as e:
                error_files.append(("PDF保存", str(e)))

        progress.setValue(len(files_to_process) + 1)

        # 一時ファイル削除
        for temp_file in temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except:
                pass

        # 結果表示
        if error_files:
            msg = f"処理完了: {len(corrected_images)}/{len(files_to_process)} ファイル\n\nエラー:\n"
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
        # カスタムダイアログで確実に中央配置
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)

        # アイコンとテキストを横に並べる
        content_layout = QHBoxLayout()

        # アイコン
        icon_label = QLabel()
        style = dialog.style()
        if icon == QMessageBox.Warning:
            pixmap = style.standardIcon(style.StandardPixmap.SP_MessageBoxWarning).pixmap(48, 48)
        elif icon == QMessageBox.Question:
            pixmap = style.standardIcon(style.StandardPixmap.SP_MessageBoxQuestion).pixmap(48, 48)
        elif icon == QMessageBox.Information:
            pixmap = style.standardIcon(style.StandardPixmap.SP_MessageBoxInformation).pixmap(48, 48)
        elif icon == QMessageBox.Critical:
            pixmap = style.standardIcon(style.StandardPixmap.SP_MessageBoxCritical).pixmap(48, 48)
        else:
            pixmap = None
        if pixmap:
            icon_label.setPixmap(pixmap)
            content_layout.addWidget(icon_label)

        # テキスト
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        content_layout.addWidget(text_label, 1)
        layout.addLayout(content_layout)

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        result = [None]

        if buttons & QMessageBox.Yes:
            yes_btn = QPushButton("はい")
            yes_btn.clicked.connect(lambda: (result.__setitem__(0, QMessageBox.Yes), dialog.accept()))
            button_layout.addWidget(yes_btn)
        if buttons & QMessageBox.No:
            no_btn = QPushButton("いいえ")
            no_btn.clicked.connect(lambda: (result.__setitem__(0, QMessageBox.No), dialog.reject()))
            button_layout.addWidget(no_btn)
        if buttons & QMessageBox.Ok:
            ok_btn = QPushButton("OK")
            ok_btn.clicked.connect(lambda: (result.__setitem__(0, QMessageBox.Ok), dialog.accept()))
            button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        # サイズ調整と中央配置
        dialog.adjustSize()
        dialog.setMinimumWidth(350)
        self.center_dialog(dialog, dialog.width(), dialog.height())

        dialog.exec()
        return result[0] if result[0] else QMessageBox.No

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
