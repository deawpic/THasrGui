# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for Typhoon ASR Desktop Transcriber.
Produces a standalone desktop executable (TyphoonTranscriber.exe on Windows).
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

SPEC_DIR = Path(SPECPATH).resolve()
if (SPEC_DIR / "src" / "typhoon_transcriber").is_dir():
    app_dir = SPEC_DIR / "src"
else:
    app_dir = SPEC_DIR

entry_script = str(app_dir / "typhoon_transcriber" / "main.py")

# Dynamic libraries (sounddevice/portaudio, soundfile, onnxruntime)
binaries = []
try:
    binaries += collect_dynamic_libs('sounddevice')
except Exception:
    pass

try:
    binaries += collect_dynamic_libs('soundfile')
except Exception:
    pass

try:
    binaries += collect_dynamic_libs('onnxruntime')
except Exception:
    pass

# Data files (PySide6, sentencepiece)
datas = []
try:
    datas += collect_data_files('PySide6')
except Exception:
    pass

try:
    datas += collect_data_files('sentencepiece')
except Exception:
    pass

# Include models/.gitkeep if exists
models_gitkeep = app_dir / "models" / ".gitkeep"
if models_gitkeep.exists():
    datas.append((str(models_gitkeep), "models"))

hidden_imports = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'onnxruntime',
    'sounddevice',
    'soundfile',
    'pydub',
    'sentencepiece',
    'numpy',
] + collect_submodules('typhoon_transcriber')

# Heavy libraries strictly excluded to keep footprint light
excludes = [
    'torch',
    'torchaudio',
    'torchvision',
    'nemo',
    'scipy',
    'matplotlib',
    'tkinter',
    'IPython',
    'jupyter',
    'notebook',
]

a = Analysis(
    [entry_script],
    pathex=[str(app_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TyphoonTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Keep false to prevent PySide6 DLL corruption
    console=False,  # Windowed GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TyphoonTranscriber',
)

