#!/usr/bin/env python3
"""Run the ACSC GUI"""

import sys
import os

# Setup for PyInstaller frozen executable
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    app_path = os.path.dirname(sys.executable)
    plugin_path = os.path.join(app_path, '_internal', 'PySide6', 'plugins')
    os.environ['QT_PLUGIN_PATH'] = plugin_path
else:
    # Running as script - only set xcb for Linux
    if sys.platform.startswith('linux'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

from acsc.gui import main

if __name__ == "__main__":
    main()