"""
Log viewer dialog for VMM-FRC application.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QSpinBox, QCheckBox, QFileDialog
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from vmm.logger import get_recent_logs, get_all_logs, get_log_file, get_log_directory
import subprocess
import sys


class LogViewerDialog(QDialog):
    """Dialog for viewing application logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VMM-FRC Log Viewer")
        self.resize(900, 600)
        self.auto_refresh = False
        self.initUI()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refreshLogs)

        # Load initial logs
        self.refreshLogs()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Top controls
        controls_layout = QHBoxLayout()

        # Log file info
        self.log_file_label = QLabel()
        log_file = get_log_file()
        if log_file:
            self.log_file_label.setText(f"Log file: {log_file}")
        else:
            self.log_file_label.setText("Log file: Not available")
        self.log_file_label.setStyleSheet("color: #666; font-size: 11px;")
        controls_layout.addWidget(self.log_file_label)

        controls_layout.addStretch()

        # Number of lines to show
        controls_layout.addWidget(QLabel("Show last:"))
        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(100, 10000)
        self.lines_spin.setValue(500)
        self.lines_spin.setSingleStep(100)
        self.lines_spin.setSuffix(" lines")
        self.lines_spin.valueChanged.connect(self.refreshLogs)
        controls_layout.addWidget(self.lines_spin)

        # Show all checkbox
        self.show_all_check = QCheckBox("Show all")
        self.show_all_check.stateChanged.connect(self.onShowAllChanged)
        controls_layout.addWidget(self.show_all_check)

        layout.addLayout(controls_layout)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)

        # Use monospace font for better readability
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(9)
        self.log_text.setFont(font)

        layout.addWidget(self.log_text)

        # Bottom buttons
        buttons_layout = QHBoxLayout()

        # Auto-refresh checkbox
        self.auto_refresh_check = QCheckBox("Auto-refresh")
        self.auto_refresh_check.setToolTip("Automatically refresh logs every 2 seconds")
        self.auto_refresh_check.stateChanged.connect(self.onAutoRefreshChanged)
        buttons_layout.addWidget(self.auto_refresh_check)

        buttons_layout.addStretch()

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refreshLogs)
        buttons_layout.addWidget(self.refresh_btn)

        # Save button
        self.save_btn = QPushButton("Save As...")
        self.save_btn.clicked.connect(self.saveLogs)
        buttons_layout.addWidget(self.save_btn)

        # Open log folder button
        self.open_folder_btn = QPushButton("Open Log Folder")
        self.open_folder_btn.clicked.connect(self.openLogFolder)
        buttons_layout.addWidget(self.open_folder_btn)

        # Clear button
        self.clear_btn = QPushButton("Clear Display")
        self.clear_btn.clicked.connect(self.clearDisplay)
        buttons_layout.addWidget(self.clear_btn)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_btn)

        layout.addLayout(buttons_layout)

    def refreshLogs(self):
        """Refresh the log display."""
        try:
            if self.show_all_check.isChecked():
                logs = get_all_logs()
            else:
                n_lines = self.lines_spin.value()
                logs = get_recent_logs(n_lines)

            self.log_text.setPlainText(logs)

            # Scroll to bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        except Exception as e:
            self.log_text.setPlainText(f"Error loading logs: {e}")

    def onShowAllChanged(self, state):
        """Handle show all checkbox state change."""
        self.lines_spin.setEnabled(state != Qt.Checked)
        self.refreshLogs()

    def onAutoRefreshChanged(self, state):
        """Handle auto-refresh checkbox state change."""
        if state == Qt.Checked:
            self.refresh_timer.start(2000)  # Refresh every 2 seconds
        else:
            self.refresh_timer.stop()

    def saveLogs(self):
        """Save logs to a file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            "vmm_logs.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())

                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Logs Saved",
                    f"Logs saved to:\n{filename}"
                )
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save logs:\n{e}"
                )

    def openLogFolder(self):
        """Open the log folder in file explorer."""
        log_dir = get_log_directory()

        try:
            if sys.platform == 'win32':
                subprocess.run(['explorer', str(log_dir)])
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(log_dir)])
            else:
                subprocess.run(['xdg-open', str(log_dir)])
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open log folder:\n{e}\n\nLog folder: {log_dir}"
            )

    def clearDisplay(self):
        """Clear the log display (does not delete log file)."""
        self.log_text.clear()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.refresh_timer.stop()
        super().closeEvent(event)
