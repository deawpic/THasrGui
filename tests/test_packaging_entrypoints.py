"""
Packaging and Entry Point Verification Tests.
Validates that entry points, PyInstaller submodule collection, and direct execution
do not fail with relative import or missing module errors.
"""

import subprocess
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


def test_main_script_direct_execution_no_relative_import_error():
    """Verify executing main.py directly (like PyInstaller does) does not fail with relative import error."""
    main_py_path = Path(__file__).resolve().parent.parent / "typhoon_transcriber" / "main.py"
    assert main_py_path.exists()

    # Execute main.py directly as __main__ with QApplication.exec patched to return immediately
    code = (
        "from unittest.mock import patch\n"
        "from PySide6.QtWidgets import QApplication\n"
        "import runpy\n"
        "with patch.object(QApplication, 'exec', return_value=0):\n"
        f"    runpy.run_path(r'{main_py_path}', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"QT_QPA_PLATFORM": "offscreen", "PATH": subprocess.os.environ.get("PATH", "")},
        timeout=15,
    )
    assert result.returncode == 0
    assert "attempted relative import with no known parent package" not in result.stderr


def test_module_main_execution():
    """Verify python -m typhoon_transcriber module entrypoint can be invoked without error."""
    result = subprocess.run(
        [sys.executable, "-c", "import typhoon_transcriber.__main__"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
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
