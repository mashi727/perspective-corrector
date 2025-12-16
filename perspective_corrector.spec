# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for perspective_corrector

import sys
from PyInstaller.utils.hooks import collect_all

# プラットフォームに応じたアイコンファイルを選択
if sys.platform == 'darwin':
    app_icon = 'icon.icns'
elif sys.platform == 'win32':
    app_icon = 'icon.ico'
else:
    app_icon = None

block_cipher = None

# pillow-heif の依存関係を収集
pillow_heif_datas, pillow_heif_binaries, pillow_heif_hiddenimports = collect_all('pillow_heif')

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
    excludes=[],
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
        strip=False,
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
        strip=False,
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
        strip=False,
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
