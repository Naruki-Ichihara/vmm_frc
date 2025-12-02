"""Runtime hook to set Qt plugin path for PyInstaller builds."""
import os
import sys

if getattr(sys, 'frozen', False):
    # Running as compiled executable
    app_path = os.path.dirname(sys.executable)
    # PyInstaller puts data files in _internal folder
    plugin_path = os.path.join(app_path, '_internal', 'PySide6', 'plugins')
    os.environ['QT_PLUGIN_PATH'] = plugin_path
