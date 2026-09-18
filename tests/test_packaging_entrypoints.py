"""
Packaging and Entry Point Verification Tests.
Validates that entry points, PyInstaller submodule collection, and direct execution
do not fail with relative import or missing module errors across Windows and Linux.
"""

import os
import subprocess
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


def test_main_script_direct_execution_no_relative_import_error():
    """Verify executing main.py directly (like PyInstaller does) does not fail with relative import error."""
    main_py_path = Path(__file__).resolve().parent.parent / "typhoon_transcriber" / "main.py"
    assert main_py_path.exists()

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, str(main_py_path), "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert result.returncode == 0, f"Failed with stderr: {result.stderr}"
    assert "Typhoon ASR Transcriber" in result.stdout
    assert "attempted relative import with no known parent package" not in result.stderr


def test_module_main_execution():
    """Verify python -m typhoon_transcriber module entrypoint can be invoked without error."""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-m", "typhoon_transcriber", "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert result.returncode == 0, f"Failed with stderr: {result.stderr}"
    assert "Typhoon ASR Transcriber" in result.stdout
    assert "ImportError" not in result.stderr


def test_pyinstaller_submodules_collection():
    """Verify PyInstaller collect_submodules collects all critical modules including ui.widgets."""
    submodules = collect_submodules("typhoon_transcriber")
    assert "typhoon_transcriber.main" in submodules
    assert "typhoon_transcriber.config" in submodules
    assert "typhoon_transcriber.ui.widgets" in submodules
    assert "typhoon_transcriber.ui.widgets.vu_meter" in submodules
    assert "typhoon_transcriber.ui.widgets.export_dialog" in submodules
    assert "typhoon_transcriber.ui.widgets.download_dialog" in submodules
    assert "typhoon_transcriber.ui.widgets.batch_dialog" in submodules
    assert "typhoon_transcriber.ui.widgets.post_processing_dialog" in submodules


def test_nuitka_plugins_and_module_resolution():
    """Verify Nuitka is available with PySide6 plugin support for standalone builds."""
    import nuitka.Version
    from nuitka.plugins.Plugins import getQtPluginNames, hasPluginName

    assert nuitka.Version.getNuitkaVersion() is not None
    assert "pyside6" in getQtPluginNames()
    assert hasPluginName("pyside6") is True


def test_package_init_lazy_main():
    """Verify importing typhoon_transcriber does not eagerly load main or PySide6 GUI module."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; import typhoon_transcriber.config; "
        "assert 'typhoon_transcriber.main' not in sys.modules, 'main eagerly imported!'; "
        "import typhoon_transcriber; "
        "assert callable(typhoon_transcriber.main); "
        "print('LAZY_OK')",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"Failed with: {result.stderr}"
    assert "LAZY_OK" in result.stdout

