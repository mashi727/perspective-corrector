# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for perspective_corrector

import sys
import os
from PyInstaller.utils.hooks import collect_all

# specファイルのディレクトリを基準にアイコンパスを設定
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# プラットフォームに応じたアイコンファイルを選択
if sys.platform == 'darwin':
    app_icon = os.path.join(SPEC_DIR, 'icon.icns')
elif sys.platform == 'win32':
    app_icon = os.path.join(SPEC_DIR, 'icon.ico')
else:
    app_icon = None

# アイコンファイルの存在確認
if app_icon and not os.path.exists(app_icon):
    print(f"WARNING: Icon file not found: {app_icon}")
    app_icon = None
else:
    print(f"Using icon: {app_icon}")

block_cipher = None

# pillow-heif の依存関係を収集
pillow_heif_datas, pillow_heif_binaries, pillow_heif_hiddenimports = collect_all('pillow_heif')

# 不要なモジュールを除外してサイズ削減
excludes = [
    # PySide6の不要なモジュール
    'PySide6.QtQml',
    'PySide6.QtQuick',
    'PySide6.QtQuickWidgets',
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic',
    'PySide6.Qt3DAnimation',
    'PySide6.Qt3DExtras',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'PySide6.QtWebEngine',
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebChannel',
    'PySide6.QtWebSockets',
    'PySide6.QtPositioning',
    'PySide6.QtLocation',
    'PySide6.QtBluetooth',
    'PySide6.QtNfc',
    'PySide6.QtSensors',
    'PySide6.QtSerialPort',
    'PySide6.QtSql',
    'PySide6.QtTest',
    'PySide6.QtXml',
    'PySide6.QtDesigner',
    'PySide6.QtHelp',
    'PySide6.QtOpenGL',
    'PySide6.QtOpenGLWidgets',
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    'PySide6.QtRemoteObjects',
    'PySide6.QtScxml',
    'PySide6.QtStateMachine',
    'PySide6.QtSvgWidgets',
    'PySide6.QtCharts',
    'PySide6.QtDataVisualization',
    'PySide6.QtNetworkAuth',
    # その他不要なモジュール
    'tkinter',
    'unittest',
    'email',
    'html',
    'http',
    'xml',
    'pydoc',
    'doctest',
    'argparse',
    'difflib',
    'inspect',
]

a = Analysis(
    ['perspective_corrector.py'],
    pathex=[],
    binaries=pillow_heif_binaries,
    datas=pillow_heif_datas,
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'pillow_heif',
    ] + pillow_heif_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# macOS: .appバンドル用にone-dirモードで作成
# Windows: one-fileモードで.exeを作成
if sys.platform == 'darwin':
    # macOS用: one-dirモード（.appバンドル用）
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='PerspectiveCorrector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,  # シンボル削除でサイズ削減
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=app_icon,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=True,  # シンボル削除でサイズ削減
        upx=True,
        upx_exclude=[],
        name='PerspectiveCorrector',
    )

    app = BUNDLE(
        coll,
        name='PerspectiveCorrector.app',
        icon=app_icon,
        bundle_identifier='com.mashi.perspectivecorrector',
        info_plist={
            'CFBundleName': 'PerspectiveCorrector',
            'CFBundleDisplayName': 'Perspective Corrector',
            'CFBundleVersion': '1.1.0',
            'CFBundleShortVersionString': '1.1.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )
else:
    # Windows用: one-fileモード
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='PerspectiveCorrector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,  # シンボル削除でサイズ削減
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=app_icon,
    )
