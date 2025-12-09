import sys
import os
import colorsys
from pathlib import Path
from typing import Optional
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QSpinBox, QComboBox, QFileDialog, QGroupBox,
                               QGridLayout, QFormLayout, QTextEdit, QProgressBar, QMessageBox,
                               QCheckBox, QRadioButton, QSlider, QDoubleSpinBox, QStackedWidget,
                               QButtonGroup, QFrame, QScrollArea, QTabWidget,
                               QToolBar, QSizePolicy, QSplitter, QDialog)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QFont, QPalette, QColor, QAction, QIcon
import cv2 as cv
from acsc.io import import_image_sequence, trim_image
from acsc.analysis import compute_structure_tensor, compute_orientation, drop_edges_3D, _orientation_function, _orientation_function_reference
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pyvista as pv
from pyvistaqt import QtInteractor

class ImportDialog(QWidget):
    volume_imported = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Image Sequences")
        self.setFixedSize(800, 600)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Import Image Sequences")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)

        # Ribbon toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0; border-radius: 5px; padding: 10px; }")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(10)

        # File Operations Group
        file_group = QGroupBox("File")
        file_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        file_layout = QHBoxLayout(file_group)

        self.open_btn = RibbonButton("Open\nFolder")
        self.open_btn.clicked.connect(self.browseFolder)
        file_layout.addWidget(self.open_btn)

        toolbar_layout.addWidget(file_group)

        # Format Group
        format_group = QGroupBox("Format")
        format_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        format_layout = QVBoxLayout(format_group)

        format_layout.addWidget(QLabel("File Format:"))
        self.format_combo = RibbonComboBox()
        self.format_combo.addItems(["tif", "tiff", "png", "jpg", "jpeg", "bmp", "dcm"])
        format_layout.addWidget(self.format_combo)

        toolbar_layout.addWidget(format_group)

        # Processing Group
        process_group = QGroupBox("Processing")
        process_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        process_layout = QVBoxLayout(process_group)

        self.grayscale_check = QCheckBox("Convert to Grayscale")
        self.grayscale_check.setChecked(True)
        process_layout.addWidget(self.grayscale_check)

        self.crop_check = QCheckBox("Enable Cropping")
        process_layout.addWidget(self.crop_check)

        toolbar_layout.addWidget(process_group)

        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

        # Parameters area
        params_widget = QWidget()
        params_layout = QHBoxLayout(params_widget)

        # Left column
        left_params = QGroupBox("File Parameters")
        left_layout = QGridLayout(left_params)

        left_layout.addWidget(QLabel("Path Template:"), 0, 0)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("e.g., /data/images/img_")
        left_layout.addWidget(self.path_input, 0, 1)

        left_layout.addWidget(QLabel("Number of Images:"), 1, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 10000)
        self.num_images_spin.setValue(100)
        left_layout.addWidget(self.num_images_spin, 1, 1)

        left_layout.addWidget(QLabel("Number of Digits:"), 2, 0)
        self.num_digits_spin = QSpinBox()
        self.num_digits_spin.setRange(1, 10)
        self.num_digits_spin.setValue(4)
        left_layout.addWidget(self.num_digits_spin, 2, 1)

        left_layout.addWidget(QLabel("Start Index:"), 3, 0)
        self.initial_number_spin = QSpinBox()
        self.initial_number_spin.setRange(0, 99999)
        self.initial_number_spin.setValue(0)
        self.initial_number_spin.setToolTip("Starting file number (e.g., 100 to start from img_0100.tif)")
        left_layout.addWidget(self.initial_number_spin, 3, 1)

        params_layout.addWidget(left_params)

        # Right column
        right_params = QGroupBox("Cropping Parameters")
        right_layout = QGridLayout(right_params)

        right_layout.addWidget(QLabel("Crop Start (x,y):"), 0, 0)
        self.crop_start_input = QLineEdit()
        self.crop_start_input.setPlaceholderText("e.g., 10,10")
        self.crop_start_input.setEnabled(False)
        right_layout.addWidget(self.crop_start_input, 0, 1)

        right_layout.addWidget(QLabel("Crop End (x,y):"), 1, 0)
        self.crop_end_input = QLineEdit()
        self.crop_end_input.setPlaceholderText("e.g., 500,500")
        self.crop_end_input.setEnabled(False)
        right_layout.addWidget(self.crop_end_input, 1, 1)

        params_layout.addWidget(right_params)

        layout.addWidget(params_widget)

        # Enable cropping controls
        self.crop_check.toggled.connect(self.crop_start_input.setEnabled)
        self.crop_check.toggled.connect(self.crop_end_input.setEnabled)

        # Status area
        status_group = QGroupBox("Import Information")
        status_layout = QVBoxLayout(status_group)

        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setPlainText("Import information will appear here...")
        status_layout.addWidget(self.info_text)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_group)

        # Button area
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        button_layout.addStretch()

        # Import button (same size and style as close button)
        self.import_btn = QPushButton("Import")
        self.import_btn.setMinimumHeight(35)
        self.import_btn.setMinimumWidth(80)
        self.import_btn.setStyleSheet("""
            QPushButton {
                text-align: center;
                padding: 8px 16px;
                border: 1px solid #2e7d32;
                background-color: #4caf50;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border: 1px solid #999999;
                color: #666666;
            }
        """)
        self.import_btn.clicked.connect(self.importImages)
        button_layout.addWidget(self.import_btn)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumHeight(35)
        self.close_btn.setMinimumWidth(80)
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addWidget(button_widget)

    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            path = Path(folder)

            # Find image files in the folder
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.dcm']
            image_files = []

            for ext in image_extensions:
                image_files.extend(sorted(path.glob(f"*{ext}")))
                # On case-sensitive filesystems, also check uppercase extensions
                import os
                if os.name != 'nt':  # Not Windows
                    image_files.extend(sorted(path.glob(f"*{ext.upper()}")))

            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for f in image_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            image_files = unique_files

            if image_files:
                # Take first image file to analyze pattern
                first_file = image_files[0]
                file_stem = first_file.stem
                file_ext = first_file.suffix[1:].lower()

                # Try to detect numeric pattern at end of filename
                import re
                match = re.search(r'(.+?)(\d+)$', file_stem)
                if match:
                    prefix = match.group(1)
                    number = match.group(2)

                    # Set path template
                    template_path = str(path / prefix)
                    self.path_input.setText(template_path)

                    # Update format combo
                    format_index = self.format_combo.findText(file_ext)
                    if format_index >= 0:
                        self.format_combo.setCurrentIndex(format_index)

                    # Update initial number and digits
                    self.initial_number_spin.setValue(int(number))
                    self.num_digits_spin.setValue(len(number))

                    # Estimate number of images
                    self.num_images_spin.setValue(len(image_files))

                    self.info_text.append(f"Selected folder: {folder}")
                    self.info_text.append(f"Detected pattern: {prefix}####.{file_ext}")
                    self.info_text.append(f"Found {len(image_files)} image files")
                else:
                    # Fallback to generic pattern
                    self.path_input.setText(str(path / "img_"))
                    self.info_text.append(f"Selected folder: {folder}")
                    self.info_text.append(f"Could not detect pattern, using default")
            else:
                # No images found, use generic pattern
                self.path_input.setText(str(path / "img_"))
                self.info_text.append(f"Selected folder: {folder}")
                self.info_text.append("No image files found in folder")

    def importImages(self):
        # Get parameters
        params = {
            'path_template': self.path_input.text(),
            'num_images': self.num_images_spin.value(),
            'num_digits': self.num_digits_spin.value(),
            'format': self.format_combo.currentText(),
            'initial_number': self.initial_number_spin.value(),
            'convert_grayscale': self.grayscale_check.isChecked(),
            'crop_enabled': self.crop_check.isChecked(),
            'crop_start': self.crop_start_input.text(),
            'crop_end': self.crop_end_input.text()
        }

        # Validate parameters
        if not params['path_template']:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Please specify a path template")
            msg.setIcon(QMessageBox.Warning)
            msg.move(self.geometry().center() - msg.rect().center())
            msg.exec_()
            return

        if params['crop_enabled']:
            if not params['crop_start'] or not params['crop_end']:
                msg = QMessageBox(self)
                msg.setWindowTitle("Error")
                msg.setText("Please specify crop coordinates")
                msg.setIcon(QMessageBox.Warning)
                msg.move(self.geometry().center() - msg.rect().center())
                msg.exec_()
                return

        # Start import
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.import_btn.setEnabled(False)

        self.worker = ImportWorker(params)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.info_text.append)
        self.worker.finished.connect(self.importFinished)
        self.worker.error.connect(self.importError)
        self.worker.start()

    def importFinished(self, volume):
        self.progress_bar.setVisible(False)
        self.import_btn.setEnabled(True)
        self.info_text.append(f"✓ Import completed successfully!")
        self.info_text.append(f"Volume shape: {volume.shape}")
        self.info_text.append(f"Data type: {volume.dtype}")

        # Emit signal with volume data
        self.volume_imported.emit(volume)

        # Show success message
        msg = QMessageBox(self)
        msg.setWindowTitle("Import Complete")
        msg.setText(f"Successfully imported volume with shape {volume.shape}")
        msg.setIcon(QMessageBox.Information)
        msg.move(self.geometry().center() - msg.rect().center())
        msg.exec_()

    def importError(self, error_msg):
        self.progress_bar.setVisible(False)
        self.import_btn.setEnabled(True)
        self.info_text.append(f"✗ Import failed: {error_msg}")
        msg = QMessageBox(self)
        msg.setWindowTitle("Import Error")
        msg.setText(error_msg)
        msg.setIcon(QMessageBox.Critical)
        msg.move(self.geometry().center() - msg.rect().center())
        msg.exec_()

class ImportWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.status.emit("Starting import...")

            # Parse crop coordinates
            if self.params['crop_enabled']:
                start_coords = tuple(map(int, self.params['crop_start'].split(',')))
                end_coords = tuple(map(int, self.params['crop_end'].split(',')))
                process_func = lambda x: trim_image(start_coords, end_coords, x)
            else:
                process_func = None

            # Determine color conversion
            cvt_control = None
            if self.params['convert_grayscale']:
                cvt_control = cv.COLOR_BGR2GRAY

            self.status.emit(f"Importing {self.params['num_images']} images...")
            self.progress.emit(25)

            # Import image sequence
            volume = import_image_sequence(
                path_template=self.params['path_template'],
                number_of_images=self.params['num_images'],
                number_of_digits=self.params['num_digits'],
                format=self.params['format'],
                initial_number=self.params['initial_number'],
                process=process_func,
                cvt_control=cvt_control
            )

            self.progress.emit(100)
            self.status.emit(f"Import complete! Volume shape: {volume.shape}")
            self.finished.emit(volume)

        except Exception as e:
            self.error.emit(str(e))


class ExportScreenshotDialog(QDialog):
    """Dialog for configuring screenshot export settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Screenshot")
        self.setMinimumWidth(300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # View selection
        view_group = QGroupBox("Select View")
        view_layout = QVBoxLayout(view_group)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["3D View", "XY Slice", "XZ Slice", "YZ Slice"])
        self.view_combo.currentTextChanged.connect(self.onViewChanged)
        view_layout.addWidget(self.view_combo)

        layout.addWidget(view_group)

        # Background color
        bg_group = QGroupBox("Background Color")
        bg_layout = QVBoxLayout(bg_group)

        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["White", "Transparent", "Gray"])
        bg_layout.addWidget(self.bg_combo)

        layout.addWidget(bg_group)

        # CT overlay option (only for slice views)
        self.ct_group = QGroupBox("Slice Options")
        ct_layout = QVBoxLayout(self.ct_group)

        self.include_ct_check = QCheckBox("Include CT Image Overlay")
        self.include_ct_check.setChecked(True)
        ct_layout.addWidget(self.include_ct_check)

        layout.addWidget(self.ct_group)

        # Resolution selection
        res_group = QGroupBox("Resolution")
        res_layout = QVBoxLayout(res_group)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Current View",
            "800 x 600",
            "1024 x 768",
            "1280 x 960",
            "1600 x 1200",
            "1920 x 1440",
            "2560 x 1920",
            "3840 x 2880"
        ])
        self.resolution_combo.setCurrentIndex(0)  # Default: Current View
        res_layout.addWidget(self.resolution_combo)

        layout.addWidget(res_group)

        # Legend options
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout(legend_group)

        self.include_legend_check = QCheckBox("Include Color Legend")
        self.include_legend_check.setChecked(True)
        self.include_legend_check.setToolTip("Include color wheel (azimuth mode) or colorbar (tilt mode)")
        legend_layout.addWidget(self.include_legend_check)

        layout.addWidget(legend_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Export")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Initial state
        self.onViewChanged(self.view_combo.currentText())

    def onViewChanged(self, view):
        """Enable/disable CT overlay option based on view selection."""
        is_slice = view != "3D View"
        self.ct_group.setEnabled(is_slice)
        if not is_slice:
            self.include_ct_check.setChecked(False)

    def getSettings(self):
        """Return the current dialog settings."""
        # Parse resolution string (e.g., "1280 x 960" -> (1280, 960))
        res_text = self.resolution_combo.currentText()
        if res_text == "Current View":
            resolution = None  # Use current view size
        else:
            width, height = map(int, res_text.replace(" ", "").split("x"))
            resolution = (width, height)
        return {
            'view': self.view_combo.currentText(),
            'background': self.bg_combo.currentText(),
            'include_ct': self.include_ct_check.isChecked(),
            'resolution': resolution,
            'include_legend': self.include_legend_check.isChecked()
        }


class RibbonButton(QPushButton):
    def __init__(self, text, icon_name=None):
        super().__init__()
        self.setText(text)
        self.setMinimumSize(80, 60)
        self.setMaximumSize(120, 60)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                text-align: center;
                padding: 5px;
                border: 1px solid #d0d0d0;
                background-color: white;
                font-size: 11px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e5f3ff;
                border: 1px solid #005499;
            }
            QPushButton:pressed {
                background-color: #cce8ff;
                border: 1px solid #005499;
            }
        """)

class RibbonComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(120)
        self.setStyleSheet("""
            QComboBox {
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                padding: 3px 5px;
                background-color: white;
                font-size: 11px;
            }
            QComboBox:hover {
                border: 1px solid #005499;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-style: solid;
                border-width: 4px 3px 0px 3px;
                border-color: #666 transparent transparent transparent;
            }
        """)

class ColorBarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(80)
        self.current_range = (0, 255)
        self.colormap = "viridis"
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        self.title_label = QLabel("Intensity")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        layout.addWidget(self.title_label)

        # Color bar placeholder (would need custom painting for real colorbar)
        self.colorbar_widget = QFrame()
        self.colorbar_widget.setStyleSheet("""
            QFrame {
                border: 1px solid #d0d0d0;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #440154, stop: 0.25 #31688e, stop: 0.5 #35b779,
                    stop: 0.75 #fde725, stop: 1 #fff200);
            }
        """)
        self.colorbar_widget.setMinimumHeight(200)
        layout.addWidget(self.colorbar_widget)

        # Max value label
        self.max_label = QLabel(f"{self.current_range[1]:.0f}")
        self.max_label.setAlignment(Qt.AlignCenter)
        self.max_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.max_label)

        layout.addStretch()

        # Min value label
        self.min_label = QLabel(f"{self.current_range[0]:.0f}")
        self.min_label.setAlignment(Qt.AlignCenter)
        self.min_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.min_label)

    def updateRange(self, min_val, max_val):
        self.current_range = (min_val, max_val)
        self.min_label.setText(f"{min_val:.0f}")
        self.max_label.setText(f"{max_val:.0f}")

    def update_colormap(self, colormap):
        self.colormap = colormap
        # Update gradient based on colormap
        gradients = {
            "viridis": "stop: 0 #440154, stop: 0.25 #31688e, stop: 0.5 #35b779, stop: 0.75 #fde725, stop: 1 #fff200",
            "gray": "stop: 0 #000000, stop: 1 #ffffff",
            "jet": "stop: 0 #000080, stop: 0.25 #0000ff, stop: 0.5 #00ff00, stop: 0.75 #ffff00, stop: 1 #ff0000",
            "coolwarm": "stop: 0 #3b4cc0, stop: 0.5 #ffffff, stop: 1 #b40426",
            "rainbow": "stop: 0 #9400d3, stop: 0.17 #0000ff, stop: 0.33 #00ff00, stop: 0.5 #ffff00, stop: 0.67 #ff7f00, stop: 1 #ff0000",
            "bone": "stop: 0 #000000, stop: 0.33 #545474, stop: 0.67 #a8a8bc, stop: 1 #ffffff"
        }
        gradient = gradients.get(colormap, gradients["viridis"])
        self.colorbar_widget.setStyleSheet(f"""
            QFrame {{
                border: 1px solid #d0d0d0;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, {gradient});
            }}
        """)

    def setTitle(self, title):
        """Update the color bar title/label"""
        self.title_label.setText(title)

class HistogramPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 10))
        self.canvas = FigureCanvas(self.figure)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        self.title_label = QLabel("Orientation Histogram")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(self.title_label)

        # Canvas for matplotlib
        layout.addWidget(self.canvas)

        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.stats_text)

        # Button layout
        button_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.exportToCSV)
        button_layout.addWidget(self.export_btn)

        self.hide_btn = QPushButton("Hide Histogram")
        self.hide_btn.clicked.connect(self.hidePanel)
        button_layout.addWidget(self.hide_btn)

        layout.addLayout(button_layout)

        self.setMaximumWidth(400)
        self.setMinimumWidth(300)

        # Store data for export
        self.histogram_data = {}
        self.statistics_data = {}

    def plotHistogram(self, config, orientation_data):
        """Plot histogram based on configuration"""
        self.figure.clear()

        # Clear stored data for export
        self.histogram_data = {}
        self.statistics_data = {}
        self.current_config = config

        # Determine number of subplots needed
        plots_needed = sum([
            config['orientations']['reference'],
            config['orientations']['theta'],
            config['orientations']['phi']
        ])

        if plots_needed == 0:
            return

        # Create subplots
        axes = []
        for i in range(plots_needed):
            ax = self.figure.add_subplot(plots_needed, 1, i+1)
            axes.append(ax)

        # Plot counter
        plot_idx = 0
        stats_text = "Statistical Analysis:\n" + "="*40 + "\n\n"

        # Plot reference orientation
        if config['orientations']['reference']:
            data = orientation_data.get('reference')
            if data is not None:
                ax = axes[plot_idx]
                data_flat = data.flatten()

                # Determine range
                if config['auto_range']:
                    hist_range = (np.min(data_flat), np.max(data_flat))
                else:
                    hist_range = config['range']

                # Plot histogram
                n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                          range=hist_range, color='blue',
                                          alpha=0.7, edgecolor='black')

                ax.set_title('Reference Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Calculate statistics
                mean_val = np.mean(data_flat)
                std_val = np.std(data_flat)
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                # Store data for export
                self.histogram_data['Reference'] = {
                    'data': data_flat,
                    'bins': bins,
                    'counts': n
                }
                self.statistics_data['Reference'] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val
                }

                stats_text += "Reference Orientation:\n"
                if config['statistics']['mean']:
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}°')
                    stats_text += f"  Mean: {mean_val:.2f}°\n"
                if config['statistics']['std']:
                    stats_text += f"  Std Dev: {std_val:.2f}°\n"
                if config['statistics']['cv']:
                    stats_text += f"  CV: {cv_val:.2f}%\n"

                if config['statistics']['mean']:
                    ax.legend(fontsize=9)

                stats_text += "\n"
                plot_idx += 1

        # Plot theta orientation
        if config['orientations']['theta']:
            data = orientation_data.get('theta')
            if data is not None:
                ax = axes[plot_idx]
                data_flat = data.flatten()

                # Determine range
                if config['auto_range']:
                    hist_range = (np.min(data_flat), np.max(data_flat))
                else:
                    hist_range = config['range']

                # Plot histogram
                n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                          range=hist_range, color='green',
                                          alpha=0.7, edgecolor='black')

                ax.set_title('X-Z Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Calculate statistics
                mean_val = np.mean(data_flat)
                std_val = np.std(data_flat)
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                # Store data for export
                self.histogram_data['X-Z'] = {
                    'data': data_flat,
                    'bins': bins,
                    'counts': n
                }
                self.statistics_data['X-Z'] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val
                }

                stats_text += "X-Z Orientation:\n"
                if config['statistics']['mean']:
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}°')
                    stats_text += f"  Mean: {mean_val:.2f}°\n"
                if config['statistics']['std']:
                    stats_text += f"  Std Dev: {std_val:.2f}°\n"
                if config['statistics']['cv']:
                    stats_text += f"  CV: {cv_val:.2f}%\n"

                if config['statistics']['mean']:
                    ax.legend(fontsize=9)

                stats_text += "\n"
                plot_idx += 1

        # Plot phi orientation
        if config['orientations']['phi']:
            data = orientation_data.get('phi')
            if data is not None:
                ax = axes[plot_idx]
                data_flat = data.flatten()

                # Determine range
                if config['auto_range']:
                    hist_range = (np.min(data_flat), np.max(data_flat))
                else:
                    hist_range = config['range']

                # Plot histogram
                n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                          range=hist_range, color='orange',
                                          alpha=0.7, edgecolor='black')

                ax.set_title('Y-Z Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Calculate statistics
                mean_val = np.mean(data_flat)
                std_val = np.std(data_flat)
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                # Store data for export
                self.histogram_data['Y-Z'] = {
                    'data': data_flat,
                    'bins': bins,
                    'counts': n
                }
                self.statistics_data['Y-Z'] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val
                }

                stats_text += "Y-Z Orientation:\n"
                if config['statistics']['mean']:
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}°')
                    stats_text += f"  Mean: {mean_val:.2f}°\n"
                if config['statistics']['std']:
                    stats_text += f"  Std Dev: {std_val:.2f}°\n"
                if config['statistics']['cv']:
                    stats_text += f"  CV: {cv_val:.2f}%\n"

                if config['statistics']['mean']:
                    ax.legend(fontsize=9)

                stats_text += "\n"

        # Update statistics display
        self.stats_text.setText(stats_text)

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def plotHistogramMultiROI(self, config, rois_data):
        """Plot histograms for multiple ROIs overlaid on the same canvas"""
        self.figure.clear()

        # Clear stored data for export
        self.histogram_data = {}
        self.statistics_data = {}
        self.current_config = config

        # Get selected ROIs
        selected_rois = config.get('rois', [])
        if not selected_rois:
            return

        # Count total number of plots needed (one subplot per orientation type)
        num_orientations = sum([
            config['orientations']['reference'],
            config['orientations']['theta'],
            config['orientations']['phi']
        ])

        if num_orientations == 0:
            return

        # Define colors for each ROI (cycle through if more than available)
        roi_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Create subplots (one per orientation type)
        axes = []
        for i in range(num_orientations):
            ax = self.figure.add_subplot(num_orientations, 1, i+1)
            axes.append(ax)

        # Build list of orientation types to plot
        orientation_types = []
        orientation_labels = []
        if config['orientations']['reference']:
            orientation_types.append('angle')
            orientation_labels.append('Reference Orientation')
        if config['orientations']['theta']:
            orientation_types.append('theta')
            orientation_labels.append('X-Z Orientation')
        if config['orientations']['phi']:
            orientation_types.append('phi')
            orientation_labels.append('Y-Z Orientation')

        # Statistics text
        stats_text = "Statistical Analysis:\n" + "="*60 + "\n\n"

        # For each orientation type, overlay all ROIs on the same subplot
        for orient_idx, (orient_type, orient_label) in enumerate(zip(orientation_types, orientation_labels)):
            ax = axes[orient_idx]

            # Store max count for y-axis scaling
            max_count = 0

            # Plot each ROI on the same axis
            for roi_idx, roi_name in enumerate(selected_rois):
                if roi_name not in rois_data:
                    continue

                roi_data = rois_data[roi_name]
                data = roi_data.get(orient_type)

                if data is None:
                    continue

                # Flatten data and remove NaN values
                data_flat = data.flatten()
                data_flat = data_flat[~np.isnan(data_flat)]

                if len(data_flat) == 0:
                    continue

                # Determine range
                if config['auto_range']:
                    hist_range = (np.min(data_flat), np.max(data_flat))
                else:
                    hist_range = config['range']

                # Plot histogram with ROI-specific color
                color = roi_colors[roi_idx % len(roi_colors)]
                n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                          range=hist_range, color=color,
                                          alpha=0.5, edgecolor=color, linewidth=1.5,
                                          label=roi_name, histtype='stepfilled')

                # Track max count for y-axis
                max_count = max(max_count, np.max(n))

                # Calculate statistics
                mean_val = np.mean(data_flat)
                std_val = np.std(data_flat)
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                # Store data for export
                key = f'{roi_name}_{orient_label}'
                self.histogram_data[key] = {
                    'data': data_flat,
                    'bins': bins,
                    'counts': n
                }
                self.statistics_data[key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val
                }

                # Add statistics to text
                stats_text += f"{roi_name} - {orient_label}:\n"

                # Plot mean line if requested
                if config['statistics']['mean']:
                    ax.axvline(mean_val, color=color, linestyle='--',
                              linewidth=2, alpha=0.8)
                    stats_text += f"  Mean: {mean_val:.2f}°\n"

                if config['statistics']['std']:
                    stats_text += f"  Std Dev: {std_val:.2f}°\n"
                if config['statistics']['cv']:
                    stats_text += f"  CV: {cv_val:.2f}%\n"

                stats_text += "\n"

            # Configure the subplot
            ax.set_title(orient_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Angle (degrees)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max_count * 1.1)  # Add 10% headroom

            # Add legend with ROI names and colors
            if len(selected_rois) > 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # Update statistics display
        self.stats_text.setText(stats_text)

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def plotTrajectoryHistogram(self, config, angle_data):
        """Plot trajectory angle histogram based on configuration."""
        self.figure.clear()

        # Clear stored data for export
        self.histogram_data = {}
        self.statistics_data = {}
        self.current_config = config

        # Update title
        self.title_label.setText("Fiber Trajectory Angle Histogram")

        # Determine number of subplots needed
        plots_needed = sum([
            config['angles'].get('tilt', False) and angle_data.get('tilt') is not None,
            config['angles'].get('azimuth', False) and angle_data.get('azimuth') is not None
        ])

        if plots_needed == 0:
            self.stats_text.setText("No angle data available for selected options.")
            return

        # Create subplots
        axes = []
        for i in range(plots_needed):
            ax = self.figure.add_subplot(plots_needed, 1, i+1)
            axes.append(ax)

        # Plot counter
        plot_idx = 0
        stats_text = "Fiber Trajectory Statistical Analysis:\n" + "="*40 + "\n\n"

        # Plot tilt angle histogram
        if config['angles'].get('tilt', False) and angle_data.get('tilt') is not None:
            data = angle_data['tilt']
            if len(data) > 0:
                ax = axes[plot_idx]
                data_flat = np.array(data).flatten()
                # Remove NaN values
                data_flat = data_flat[~np.isnan(data_flat)]

                if len(data_flat) > 0:
                    # Determine range
                    if config['auto_range']:
                        hist_range = (np.min(data_flat), np.max(data_flat))
                    else:
                        hist_range = config['range']

                    # Plot histogram
                    n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                              range=hist_range, color='#1f77b4',
                                              alpha=0.7, edgecolor='black')

                    ax.set_title('Tilt Angle (from fiber axis)', fontsize=12)
                    ax.set_xlabel('Angle (degrees)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Calculate statistics
                    mean_val = np.mean(data_flat)
                    std_val = np.std(data_flat)
                    cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0

                    # Store data for export
                    self.histogram_data['Tilt'] = {
                        'data': data_flat,
                        'bins': bins,
                        'counts': n
                    }
                    self.statistics_data['Tilt'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv_val
                    }

                    stats_text += "Tilt Angle:\n"
                    stats_text += f"  N: {len(data_flat)}\n"
                    if config['statistics']['mean']:
                        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}°')
                        stats_text += f"  Mean: {mean_val:.2f}°\n"
                    if config['statistics']['std']:
                        stats_text += f"  Std Dev: {std_val:.2f}°\n"
                    if config['statistics']['cv']:
                        stats_text += f"  CV: {cv_val:.2f}%\n"
                    stats_text += f"  Range: [{np.min(data_flat):.2f}°, {np.max(data_flat):.2f}°]\n"

                    if config['statistics']['mean']:
                        ax.legend(fontsize=9)

                    stats_text += "\n"
                    plot_idx += 1

        # Plot azimuth angle histogram
        if config['angles'].get('azimuth', False) and angle_data.get('azimuth') is not None:
            data = angle_data['azimuth']
            if len(data) > 0:
                ax = axes[plot_idx]
                data_flat = np.array(data).flatten()
                # Remove NaN values
                data_flat = data_flat[~np.isnan(data_flat)]

                if len(data_flat) > 0:
                    # Determine range
                    if config['auto_range']:
                        hist_range = (np.min(data_flat), np.max(data_flat))
                    else:
                        hist_range = (-180, 180)  # Full azimuth range

                    # Plot histogram
                    n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                              range=hist_range, color='#2ca02c',
                                              alpha=0.7, edgecolor='black')

                    ax.set_title('Azimuth Angle (in cross-section)', fontsize=12)
                    ax.set_xlabel('Angle (degrees)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Calculate statistics (circular mean for azimuth)
                    mean_val = np.mean(data_flat)
                    std_val = np.std(data_flat)
                    cv_val = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0

                    # Store data for export
                    self.histogram_data['Azimuth'] = {
                        'data': data_flat,
                        'bins': bins,
                        'counts': n
                    }
                    self.statistics_data['Azimuth'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv_val
                    }

                    stats_text += "Azimuth Angle:\n"
                    stats_text += f"  N: {len(data_flat)}\n"
                    if config['statistics']['mean']:
                        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}°')
                        stats_text += f"  Mean: {mean_val:.2f}°\n"
                    if config['statistics']['std']:
                        stats_text += f"  Std Dev: {std_val:.2f}°\n"
                    if config['statistics']['cv']:
                        stats_text += f"  CV: {cv_val:.2f}%\n"
                    stats_text += f"  Range: [{np.min(data_flat):.2f}°, {np.max(data_flat):.2f}°]\n"

                    if config['statistics']['mean']:
                        ax.legend(fontsize=9)

                    stats_text += "\n"
                    plot_idx += 1

        # Update statistics text
        self.stats_text.setText(stats_text)

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def hidePanel(self):
        """Hide the histogram panel"""
        # Find the main window
        main_window = self.window()
        if hasattr(main_window, 'hideHistogramPanel'):
            main_window.hideHistogramPanel()

    def exportToCSV(self):
        """Export histogram data and statistics to CSV"""
        if not self.histogram_data:
            QMessageBox.warning(self, "No Data", "No histogram data to export.")
            return

        # Get save file name
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Histogram Data", "", "CSV Files (*.csv)"
        )

        if not filename:
            return

        # Automatically add .csv extension if not present
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        try:
            import csv

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(['Orientation Histogram Data Export'])
                writer.writerow(['Generated from ACSC Analysis'])
                writer.writerow([])

                # Write configuration info
                if hasattr(self, 'current_config'):
                    config = self.current_config
                    writer.writerow(['Configuration:'])
                    writer.writerow(['Bins:', config['bins']])
                    writer.writerow(['Range:', f"{config['range'][0]:.2f} to {config['range'][1]:.2f}"])
                    writer.writerow([])

                # Write statistics summary
                writer.writerow(['Statistical Summary:'])
                writer.writerow(['Orientation', 'Mean (degrees)', 'Std Dev (degrees)', 'CV (%)'])

                for orientation_name in self.statistics_data:
                    stats = self.statistics_data[orientation_name]
                    writer.writerow([
                        orientation_name,
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['cv']:.2f}"
                    ])
                writer.writerow([])

                # Write histogram data for each orientation
                for orientation_name in self.histogram_data:
                    hist_data = self.histogram_data[orientation_name]

                    writer.writerow([f'{orientation_name} Orientation Histogram:'])
                    writer.writerow(['Bin Center (degrees)', 'Count'])

                    # Calculate bin centers
                    bins = hist_data['bins']
                    counts = hist_data['counts']
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    for center, count in zip(bin_centers, counts):
                        writer.writerow([f"{center:.2f}", int(count)])
                    writer.writerow([])

                    # Write raw data summary
                    writer.writerow([f'{orientation_name} Raw Data Summary:'])
                    data = hist_data['data']
                    writer.writerow(['Total Points:', len(data)])
                    writer.writerow(['Min Value:', f"{np.min(data):.2f}"])
                    writer.writerow(['Max Value:', f"{np.max(data):.2f}"])
                    writer.writerow(['25th Percentile:', f"{np.percentile(data, 25):.2f}"])
                    writer.writerow(['Median:', f"{np.median(data):.2f}"])
                    writer.writerow(['75th Percentile:', f"{np.percentile(data, 75):.2f}"])
                    writer.writerow([])

            QMessageBox.information(self, "Export Successful",
                                  f"Histogram data exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export data:\n{str(e)}")


class ModellingHistogramPanel(QWidget):
    """Histogram panel for Modelling tab - displays fiber trajectory angle histograms (vertical layout)."""
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(4, 6))
        self.canvas = FigureCanvas(self.figure)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Header with title and close button
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Trajectory Histogram")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.hide_btn = QPushButton("×")
        self.hide_btn.setFixedSize(20, 20)
        self.hide_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self.hide_btn.clicked.connect(self.hidePanel)
        header_layout.addWidget(self.hide_btn)

        layout.addLayout(header_layout)

        # Canvas for matplotlib (vertical layout - stacked plots)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=3)

        # Statistics display at bottom
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.stats_text, stretch=1)

        # Button layout
        button_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.exportToCSV)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)

        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        # Store data for export
        self.histogram_data = {}
        self.statistics_data = {}
        self.current_config = {}

    def plotTrajectoryHistogram(self, config, angle_data):
        """Plot trajectory angle histogram based on configuration."""
        self.figure.clear()

        # Clear stored data for export
        self.histogram_data = {}
        self.statistics_data = {}
        self.current_config = config

        # Determine number of subplots needed
        plots_needed = sum([
            config['angles'].get('tilt', False) and angle_data.get('tilt') is not None,
            config['angles'].get('azimuth', False) and angle_data.get('azimuth') is not None,
            config['angles'].get('xz_projection', False) and angle_data.get('xz_projection') is not None,
            config['angles'].get('yz_projection', False) and angle_data.get('yz_projection') is not None
        ])

        if plots_needed == 0:
            self.stats_text.setText("No angle data available for selected options.")
            return

        # Create subplots (stacked vertically for vertical panel layout)
        axes = []
        for i in range(plots_needed):
            ax = self.figure.add_subplot(plots_needed, 1, i+1)
            axes.append(ax)

        # Plot counter
        plot_idx = 0
        stats_text = "Statistical Analysis:\n" + "="*30 + "\n\n"

        # Helper function to plot a histogram
        def plot_angle_histogram(data, title, color, key):
            nonlocal plot_idx, stats_text
            if data is None or len(data) == 0:
                return
            ax = axes[plot_idx]
            data_flat = np.array(data).flatten()
            data_flat = data_flat[~np.isnan(data_flat)]

            if len(data_flat) == 0:
                return

            if config['auto_range']:
                hist_range = (np.min(data_flat), np.max(data_flat))
            else:
                hist_range = config['range']

            n, bins, patches = ax.hist(data_flat, bins=config['bins'],
                                      range=hist_range, color=color,
                                      alpha=0.7, edgecolor='black')

            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Angle (°)', fontsize=9)
            ax.set_ylabel('Freq', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            mean_val = np.mean(data_flat)
            std_val = np.std(data_flat)
            cv_val = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0

            self.histogram_data[key] = {
                'data': data_flat,
                'bins': bins,
                'counts': n
            }
            self.statistics_data[key] = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv_val
            }

            stats_text += f"{key}:\n"
            stats_text += f"  N: {len(data_flat)}\n"
            if config['statistics']['mean']:
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}°')
                stats_text += f"  Mean: {mean_val:.2f}°\n"
            if config['statistics']['std']:
                stats_text += f"  Std: {std_val:.2f}°\n"
            if config['statistics']['cv']:
                stats_text += f"  CV: {cv_val:.2f}%\n"

            if config['statistics']['mean']:
                ax.legend(fontsize=7, loc='upper right')

            stats_text += "\n"
            plot_idx += 1

        # Plot tilt angle histogram
        if config['angles'].get('tilt', False) and angle_data.get('tilt') is not None:
            plot_angle_histogram(angle_data['tilt'], 'Tilt Angle', '#1f77b4', 'Tilt')

        # Plot azimuth angle histogram
        if config['angles'].get('azimuth', False) and angle_data.get('azimuth') is not None:
            plot_angle_histogram(angle_data['azimuth'], 'Azimuth Angle', '#2ca02c', 'Azimuth')

        # Plot XZ projection angle histogram
        if config['angles'].get('xz_projection', False) and angle_data.get('xz_projection') is not None:
            plot_angle_histogram(angle_data['xz_projection'], 'XZ Projection', '#d62728', 'XZ Proj')

        # Plot YZ projection angle histogram
        if config['angles'].get('yz_projection', False) and angle_data.get('yz_projection') is not None:
            plot_angle_histogram(angle_data['yz_projection'], 'YZ Projection', '#9467bd', 'YZ Proj')

        self.stats_text.setText(stats_text)
        self.figure.tight_layout()
        self.canvas.draw()

    def hidePanel(self):
        """Hide the histogram panel."""
        self.setVisible(False)

    def exportToCSV(self):
        """Export histogram data and statistics to CSV."""
        if not self.histogram_data:
            QMessageBox.warning(self, "No Data", "No histogram data to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Histogram Data", "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return

        try:
            import csv
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(['Fiber Trajectory Angle Histogram Export'])
                writer.writerow([])

                for name, data_dict in self.histogram_data.items():
                    writer.writerow([f'{name} Angle Data'])
                    writer.writerow(['Bin Start', 'Bin End', 'Count'])

                    bins = data_dict['bins']
                    counts = data_dict['counts']

                    for i in range(len(counts)):
                        writer.writerow([f"{bins[i]:.2f}", f"{bins[i+1]:.2f}", int(counts[i])])

                    writer.writerow([])

                    # Statistics
                    if name in self.statistics_data:
                        stats = self.statistics_data[name]
                        data = data_dict['data']
                        writer.writerow([f'{name} Statistics'])
                        writer.writerow(['Mean:', f"{stats['mean']:.2f}"])
                        writer.writerow(['Std Dev:', f"{stats['std']:.2f}"])
                        writer.writerow(['CV (%):', f"{stats['cv']:.2f}"])
                        writer.writerow(['Total Points:', len(data)])
                        writer.writerow(['Min Value:', f"{np.min(data):.2f}"])
                        writer.writerow(['Max Value:', f"{np.max(data):.2f}"])
                        writer.writerow([])

            QMessageBox.information(self, "Export Successful",
                                  f"Histogram data exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export data:\n{str(e)}")


class Viewer2D(QWidget):
    """Lightweight 2D slice viewer showing three orthogonal views"""
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window  # Reference to main window for slider sync
        self.current_volume = None
        self.base_volume = None
        self.overlay_volume = None
        self.overlay_name = None
        self.trim_width = 0

        # Store separate orientation volumes
        self.theta_volume = None
        self.phi_volume = None
        self.angle_volume = None
        self.colormap = "gray"
        self.orientation_colormap = "jet"
        self.opacity = 1.0
        self.orientation_opacity = 0.7
        self.scalar_bar_title = "Volume"


        # Slice positions (initialized when volume is loaded)
        self.slice_x = 0
        self.slice_y = 0
        self.slice_z = 0

        # Colorbar ranges
        self.intensity_range = None
        self.orientation_range = None

        # Colorbar reference (to prevent duplicates)
        self.histogram_colorbar = None

        # Void analysis
        self.void_threshold = 50  # Default threshold value
        self.void_analysis_active = False  # Only active when button clicked

        # ROI (Region of Interest) for orientation computation
        self.roi_enabled = False
        # Dictionary to store multiple ROIs: {name: {'bounds': [z_min, z_max, y_min, y_max, x_min, x_max], 'color': 'red'}}
        self.rois = {}
        self.roi_counter = 0  # Counter for automatic ROI naming
        self.current_roi_name = None  # Currently selected ROI for editing
        # Rectangle selectors for current ROI being edited
        self.roi_selector_xy = None
        self.roi_selector_xz = None
        self.roi_selector_yz = None
        # Colors for ROIs
        self.roi_colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

        # Zoom/Pan state
        self.zoom_enabled = False
        self.zoom_factor_xy = 1.0
        self.zoom_factor_xz = 1.0
        self.zoom_factor_yz = 1.0
        self.pan_offset_xy = [0, 0]  # [x_offset, y_offset]
        self.pan_offset_xz = [0, 0]
        self.pan_offset_yz = [0, 0]
        # Store original view limits for reset
        self.original_xlim_xy = None
        self.original_ylim_xy = None
        self.original_xlim_xz = None
        self.original_ylim_xz = None
        self.original_xlim_yz = None
        self.original_ylim_yz = None
        # Pan state
        self.pan_active = False
        self.pan_start = None
        self.pan_view = None

        # Fiber detection visualization
        self.fiber_detection_result = None
        self.show_fiber_detection = False

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Create main vertical splitter (top: viewers, bottom: histogram)
        main_splitter = QSplitter(Qt.Vertical)

        # Top section: Three slice viewers in horizontal layout
        viewers_splitter = QSplitter(Qt.Horizontal)

        # XY view (left)
        self.figure_xy = Figure(figsize=(4, 4), facecolor='#f5f5f5')
        # Add extra space on left and bottom for axis arrows
        self.figure_xy.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_xy = FigureCanvas(self.figure_xy)
        self.ax_xy = self.figure_xy.add_subplot(111)
        self.ax_xy.set_title('XY Plane (Z slice)', fontsize=10, fontweight='bold')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.axis('off')
        # Draw axis arrows even before image is loaded
        self._drawAxisArrows(self.ax_xy, 'xy', None)
        self.canvas_xy.draw()
        viewers_splitter.addWidget(self.canvas_xy)
        # Connect mouse wheel event for Z slice control
        self.canvas_xy.mpl_connect('scroll_event', self.onScrollXY)

        # XZ view (center)
        self.figure_xz = Figure(figsize=(4, 4), facecolor='#f5f5f5')
        # Add extra space on left and bottom for axis arrows
        self.figure_xz.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_xz = FigureCanvas(self.figure_xz)
        self.ax_xz = self.figure_xz.add_subplot(111)
        self.ax_xz.set_title('XZ Plane (Y slice)', fontsize=10, fontweight='bold')
        self.ax_xz.set_aspect('equal')
        self.ax_xz.axis('off')
        # Draw axis arrows even before image is loaded
        self._drawAxisArrows(self.ax_xz, 'xz', None)
        self.canvas_xz.draw()
        viewers_splitter.addWidget(self.canvas_xz)
        # Connect mouse wheel event for Y slice control
        self.canvas_xz.mpl_connect('scroll_event', self.onScrollXZ)

        # YZ view (right)
        self.figure_yz = Figure(figsize=(4, 4), facecolor='#f5f5f5')
        # Add extra space on left and bottom for axis arrows
        self.figure_yz.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_yz = FigureCanvas(self.figure_yz)
        self.ax_yz = self.figure_yz.add_subplot(111)
        self.ax_yz.set_title('YZ Plane (X slice)', fontsize=10, fontweight='bold')
        self.ax_yz.set_aspect('equal')
        self.ax_yz.axis('off')
        # Draw axis arrows even before image is loaded
        self._drawAxisArrows(self.ax_yz, 'yz', None)
        self.canvas_yz.draw()
        viewers_splitter.addWidget(self.canvas_yz)
        # Connect mouse wheel event for X slice control
        self.canvas_yz.mpl_connect('scroll_event', self.onScrollYZ)

        main_splitter.addWidget(viewers_splitter)

        # Bottom section: Dual Histograms (Intensity on left, Orientation on right)
        self.figure_hist = Figure(figsize=(12, 3), facecolor='#f5f5f5')
        self.canvas_hist = FigureCanvas(self.figure_hist)

        # Create two histogram axes side by side
        # Left: Intensity histogram with colorbar
        # [left, bottom, width, height] in figure coordinates (0-1)
        self.ax_hist_intensity = self.figure_hist.add_axes([0.05, 0.20, 0.38, 0.65])
        self.ax_hist_intensity.set_title('Intensity Histogram', fontsize=10, fontweight='bold')
        self.ax_hist_intensity.set_xlabel('Intensity', fontsize=9)
        self.ax_hist_intensity.set_ylabel('Density', fontsize=9)
        self.ax_hist_intensity.set_xlim(0, 255)
        self.ax_hist_intensity.grid(True, alpha=0.3)

        # Intensity colorbar axis
        self.ax_hist_intensity_cbar = self.figure_hist.add_axes([0.44, 0.20, 0.01, 0.65])

        # Right: Orientation histogram with colorbar
        self.ax_hist_orientation = self.figure_hist.add_axes([0.52, 0.20, 0.38, 0.65])
        self.ax_hist_orientation.set_title('Orientation Histogram', fontsize=10, fontweight='bold')
        self.ax_hist_orientation.set_xlabel('Angle (degrees)', fontsize=9)
        self.ax_hist_orientation.set_ylabel('Density', fontsize=9)
        self.ax_hist_orientation.grid(True, alpha=0.3)

        # Orientation colorbar axis
        self.ax_hist_orientation_cbar = self.figure_hist.add_axes([0.91, 0.20, 0.01, 0.65])

        main_splitter.addWidget(self.canvas_hist)

        # Set initial sizes: viewers = 60%, histogram = 40%
        main_splitter.setSizes([600, 400])

        # Pipeline widgets (to be added to main window's slider panel)
        # Note: Intensity is always shown in the slice viewer, no toggle needed

        # Dictionary to store orientation checkboxes for each ROI
        # {roi_name: {'check': QCheckBox, 'children_widget': QWidget}}
        self.orientation_roi_widgets = {}

        # Container for dynamically added orientation toggles
        self.orientation_container = QWidget()
        self.orientation_container_layout = QVBoxLayout(self.orientation_container)
        self.orientation_container_layout.setContentsMargins(0, 0, 0, 0)
        self.orientation_container_layout.setSpacing(2)

        main_layout.addWidget(main_splitter)

    def setVolume(self, volume):
        """Set volume data and initialize slice positions"""
        self.current_volume = volume
        self.base_volume = None
        self.overlay_volume = None

        if volume is not None:
            # Initialize slice positions to center of volume
            if len(volume.shape) == 3:
                self.slice_z = volume.shape[0] // 2
                self.slice_y = volume.shape[1] // 2
                self.slice_x = volume.shape[2] // 2

                if volume.dtype == np.uint8:
                    self.scalar_bar_title = "Intensity"
                elif volume.dtype in [np.float32, np.float64]:
                    self.scalar_bar_title = "Value"
                else:
                    self.scalar_bar_title = "Data"
            else:
                self.scalar_bar_title = "RGB"
        else:
            self.scalar_bar_title = "Volume"

        self.renderVolume()

    def setOverlayVolume(self, base_volume, orientation_volume, orientation_name, trim_width=0, theta_vol=None, phi_vol=None, angle_vol=None):
        """Display base volume with orientation overlay as 2D slices"""
        if base_volume is None or orientation_volume is None:
            return

        # Store volumes for rendering
        self.base_volume = base_volume if len(base_volume.shape) == 3 else np.mean(base_volume, axis=-1)
        self.overlay_volume = orientation_volume.astype(np.float32)
        self.overlay_name = orientation_name
        self.trim_width = trim_width

        # Store all orientation types for histogram display
        if theta_vol is not None:
            self.theta_volume = theta_vol.astype(np.float32)
        if phi_vol is not None:
            self.phi_volume = phi_vol.astype(np.float32)
        if angle_vol is not None:
            self.angle_volume = angle_vol.astype(np.float32)

        # Initialize slice positions if not set
        if self.slice_z == 0:
            self.slice_z = base_volume.shape[0] // 2
            self.slice_y = base_volume.shape[1] // 2
            self.slice_x = base_volume.shape[2] // 2

        # Get custom ranges
        main_window = getattr(self, 'main_window', None)
        if main_window and hasattr(main_window, 'colorbar_ranges'):
            ranges = main_window.colorbar_ranges
            if not ranges.get('intensity_auto', True):
                self.intensity_range = [ranges.get('intensity_min', 0), ranges.get('intensity_max', 255)]
            if not ranges.get('orientation_auto', True):
                self.orientation_range = [ranges.get('orientation_min', 0), ranges.get('orientation_max', 180)]

        self.renderVolume()

    def addOrientationROIToggle(self, roi_name):
        """Dynamically add an Orientation-ROI toggle to the pipeline"""
        if roi_name in self.orientation_roi_widgets:
            return  # Already exists

        # Create checkbox for this ROI's orientation
        roi_label = f"Orientation-{roi_name}"
        roi_check = QCheckBox(roi_label)
        roi_check.setChecked(False)  # Default unchecked

        # Create container for child toggles (X-Z, Y-Z, Reference) - radio buttons for single selection
        children_widget = QWidget()
        children_layout = QVBoxLayout(children_widget)
        children_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        children_layout.setSpacing(2)

        # Create button group for exclusive selection
        orientation_type_group = QButtonGroup(children_widget)

        # Create radio buttons for theta, phi, angle (only one can be selected)
        theta_radio = QRadioButton("X-Z")
        theta_radio.setChecked(True)  # Default selection
        theta_radio.toggled.connect(self.updateHistogramDisplay)
        orientation_type_group.addButton(theta_radio)
        children_layout.addWidget(theta_radio)

        phi_radio = QRadioButton("Y-Z")
        phi_radio.toggled.connect(self.updateHistogramDisplay)
        orientation_type_group.addButton(phi_radio)
        children_layout.addWidget(phi_radio)

        angle_radio = QRadioButton("Reference")
        angle_radio.toggled.connect(self.updateHistogramDisplay)
        orientation_type_group.addButton(angle_radio)
        children_layout.addWidget(angle_radio)

        # Connect checkbox to show/hide children and update display
        def on_toggled(checked):
            children_widget.setVisible(checked)
            self.updateHistogramDisplay()
        roi_check.toggled.connect(on_toggled)

        # Add to container
        self.orientation_container_layout.addWidget(roi_check)
        self.orientation_container_layout.addWidget(children_widget)

        # Store references
        self.orientation_roi_widgets[roi_name] = {
            'check': roi_check,
            'children_widget': children_widget,
            'button_group': orientation_type_group,
            'theta_radio': theta_radio,
            'phi_radio': phi_radio,
            'angle_radio': angle_radio
        }

        # Initially hide children (since checkbox is unchecked by default)
        children_widget.setVisible(False)

        # Show the toggle
        roi_check.setVisible(True)

    def onScrollXY(self, event):
        """Handle mouse wheel scroll on XY view to change Z slice or zoom"""
        if self.current_volume is None:
            return

        if self.zoom_enabled:
            # Zoom mode - adjust zoom level
            zoom_speed = 1.1
            if event.button == 'up':
                self.zoom_factor_xy *= zoom_speed
            elif event.button == 'down':
                self.zoom_factor_xy /= zoom_speed
            self._applyZoom(self.ax_xy, self.zoom_factor_xy, event)
            self.canvas_xy.draw_idle()
        else:
            # Normal mode - scroll through slices
            if event.button == 'up':
                self.slice_z = min(self.slice_z + 1, self.current_volume.shape[0] - 1)
            elif event.button == 'down':
                self.slice_z = max(self.slice_z - 1, 0)
            self.renderSlicesOnly()
            # Update main window slider if it exists
            self._updateMainWindowSliders()

    def onScrollXZ(self, event):
        """Handle mouse wheel scroll on XZ view to change Y slice or zoom"""
        if self.current_volume is None:
            return

        if self.zoom_enabled:
            # Zoom mode - adjust zoom level
            zoom_speed = 1.1
            if event.button == 'up':
                self.zoom_factor_xz *= zoom_speed
            elif event.button == 'down':
                self.zoom_factor_xz /= zoom_speed
            self._applyZoom(self.ax_xz, self.zoom_factor_xz, event)
            self.canvas_xz.draw_idle()
        else:
            # Normal mode - scroll through slices
            if event.button == 'up':
                self.slice_y = min(self.slice_y + 1, self.current_volume.shape[1] - 1)
            elif event.button == 'down':
                self.slice_y = max(self.slice_y - 1, 0)
            self.renderSlicesOnly()
            # Update main window slider if it exists
            self._updateMainWindowSliders()

    def onScrollYZ(self, event):
        """Handle mouse wheel scroll on YZ view to change X slice or zoom"""
        if self.current_volume is None:
            return

        if self.zoom_enabled:
            # Zoom mode - adjust zoom level
            zoom_speed = 1.1
            if event.button == 'up':
                self.zoom_factor_yz *= zoom_speed
            elif event.button == 'down':
                self.zoom_factor_yz /= zoom_speed
            self._applyZoom(self.ax_yz, self.zoom_factor_yz, event)
            self.canvas_yz.draw_idle()
        else:
            # Normal mode - scroll through slices
            if event.button == 'up':
                self.slice_x = min(self.slice_x + 1, self.current_volume.shape[2] - 1)
            elif event.button == 'down':
                self.slice_x = max(self.slice_x - 1, 0)
            self.renderSlicesOnly()
            # Update main window slider if it exists
            self._updateMainWindowSliders()

    def updateHistogramDisplay(self):
        """Update histogram display based on selected graph type"""
        if self.current_volume is not None:
            self.renderVolume()

    def toggleROI(self, enabled):
        """Toggle ROI (Region of Interest) editing mode - creates new ROI each time"""
        if enabled and self.current_volume is not None:
            # If selector is active, finalize current ROI before creating new one
            if self.roi_selector_xy is not None:
                self._finalizeCurrentROI()

            # Create new ROI with automatic naming
            self.roi_counter += 1
            roi_name = f"ROI{self.roi_counter}"
            self.current_roi_name = roi_name

            # Initialize new ROI to full volume
            volume_shape = self.current_volume.shape
            initial_bounds = [0, volume_shape[0], 0, volume_shape[1], 0, volume_shape[2]]

            # Assign color
            color = self.roi_colors[(self.roi_counter - 1) % len(self.roi_colors)]

            self.rois[roi_name] = {
                'bounds': initial_bounds,
                'color': color,
                'theta': None,  # Will be filled after orientation computation
                'phi': None,
                'angle': None
            }

            self.roi_enabled = True
            # Create interactive selector for new ROI
            self._createROISelector(roi_name)
            self.renderVolume()
        else:
            # Finalize current ROI and disable editing mode
            if self.roi_selector_xy is not None:
                self._finalizeCurrentROI()
            self.roi_enabled = False
            self.renderVolume()

    def _finalizeCurrentROI(self):
        """Finalize the currently editing ROI and remove selector"""
        if self.roi_selector_xy is not None:
            self.roi_selector_xy.set_active(False)
            self.roi_selector_xy = None
        if self.roi_selector_xz is not None:
            self.roi_selector_xz.set_active(False)
            self.roi_selector_xz = None
        if self.roi_selector_yz is not None:
            self.roi_selector_yz.set_active(False)
            self.roi_selector_yz = None
        self.current_roi_name = None

    def _createROISelector(self, roi_name):
        """Create interactive ROI selector for the specified ROI"""
        from matplotlib.widgets import RectangleSelector

        if roi_name not in self.rois or self.current_volume is None:
            return

        roi_data = self.rois[roi_name]
        bounds = roi_data['bounds']
        color = roi_data['color']
        z_min, z_max, y_min, y_max, x_min, x_max = bounds

        # Remove old selectors if they exist
        if self.roi_selector_xy is not None:
            self.roi_selector_xy.set_active(False)
        if self.roi_selector_xz is not None:
            self.roi_selector_xz.set_active(False)
        if self.roi_selector_yz is not None:
            self.roi_selector_yz.set_active(False)

        # XY view (Z slice) - shows X and Y bounds
        self.roi_selector_xy = RectangleSelector(
            self.ax_xy,
            self._onROISelectXY,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor=color, edgecolor=color, alpha=0.3, fill=True, linewidth=2)
        )
        self.roi_selector_xy.extents = (x_min, x_max, y_min, y_max)

        # XZ view (Y slice) - shows X and Z bounds
        self.roi_selector_xz = RectangleSelector(
            self.ax_xz,
            self._onROISelectXZ,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor=color, edgecolor=color, alpha=0.3, fill=True, linewidth=2)
        )
        self.roi_selector_xz.extents = (x_min, x_max, z_min, z_max)

        # YZ view (X slice) - shows Y and Z bounds
        self.roi_selector_yz = RectangleSelector(
            self.ax_yz,
            self._onROISelectYZ,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor=color, edgecolor=color, alpha=0.3, fill=True, linewidth=2)
        )
        self.roi_selector_yz.extents = (y_min, y_max, z_min, z_max)

    def _onROISelectXY(self, eclick, erelease):
        """Handle ROI selection in XY view (updates X and Y bounds)"""
        if self.current_roi_name is None or self.current_roi_name not in self.rois:
            return

        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])

        # Clamp to volume bounds
        if self.current_volume is not None:
            volume_shape = self.current_volume.shape
            x_min = max(0, int(x_min))
            x_max = min(volume_shape[2], int(x_max))
            y_min = max(0, int(y_min))
            y_max = min(volume_shape[1], int(y_max))

        # Update current ROI bounds (keep Z unchanged)
        bounds = self.rois[self.current_roi_name]['bounds']
        z_min, z_max = bounds[0], bounds[1]
        self.rois[self.current_roi_name]['bounds'] = [z_min, z_max, y_min, y_max, x_min, x_max]
        self._syncROISelectors()

    def _onROISelectXZ(self, eclick, erelease):
        """Handle ROI selection in XZ view (updates X and Z bounds)"""
        if self.current_roi_name is None or self.current_roi_name not in self.rois:
            return

        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        z_min, z_max = sorted([eclick.ydata, erelease.ydata])

        # Clamp to volume bounds
        if self.current_volume is not None:
            volume_shape = self.current_volume.shape
            x_min = max(0, int(x_min))
            x_max = min(volume_shape[2], int(x_max))
            z_min = max(0, int(z_min))
            z_max = min(volume_shape[0], int(z_max))

        # Update current ROI bounds (keep Y unchanged)
        bounds = self.rois[self.current_roi_name]['bounds']
        y_min, y_max = bounds[2], bounds[3]
        self.rois[self.current_roi_name]['bounds'] = [z_min, z_max, y_min, y_max, x_min, x_max]
        self._syncROISelectors()

    def _onROISelectYZ(self, eclick, erelease):
        """Handle ROI selection in YZ view (updates Y and Z bounds)"""
        if self.current_roi_name is None or self.current_roi_name not in self.rois:
            return

        y_min, y_max = sorted([eclick.xdata, erelease.xdata])
        z_min, z_max = sorted([eclick.ydata, erelease.ydata])

        # Clamp to volume bounds
        if self.current_volume is not None:
            volume_shape = self.current_volume.shape
            y_min = max(0, int(y_min))
            y_max = min(volume_shape[1], int(y_max))
            z_min = max(0, int(z_min))
            z_max = min(volume_shape[0], int(z_max))

        # Update current ROI bounds (keep X unchanged)
        bounds = self.rois[self.current_roi_name]['bounds']
        x_min, x_max = bounds[4], bounds[5]
        self.rois[self.current_roi_name]['bounds'] = [z_min, z_max, y_min, y_max, x_min, x_max]
        self._syncROISelectors()

    def _syncROISelectors(self):
        """Synchronize all ROI selectors after one changes"""
        if self.current_roi_name is None or self.current_roi_name not in self.rois:
            return

        bounds = self.rois[self.current_roi_name]['bounds']
        z_min, z_max, y_min, y_max, x_min, x_max = bounds

        # Update XY selector
        if self.roi_selector_xy is not None:
            self.roi_selector_xy.extents = (x_min, x_max, y_min, y_max)
            self.canvas_xy.draw_idle()

        # Update XZ selector
        if self.roi_selector_xz is not None:
            self.roi_selector_xz.extents = (x_min, x_max, z_min, z_max)
            self.canvas_xz.draw_idle()

        # Update YZ selector
        if self.roi_selector_yz is not None:
            self.roi_selector_yz.extents = (y_min, y_max, z_min, z_max)
            self.canvas_yz.draw_idle()

    def _removeROIRectangles(self):
        """Remove ROI rectangles and selectors from all views"""
        # Remove selectors
        if self.roi_selector_xy is not None:
            self.roi_selector_xy.set_active(False)
            self.roi_selector_xy = None
        if self.roi_selector_xz is not None:
            self.roi_selector_xz.set_active(False)
            self.roi_selector_xz = None
        if self.roi_selector_yz is not None:
            self.roi_selector_yz.set_active(False)
            self.roi_selector_yz = None

        # Remove rectangle patches (if any exist)
        if self.roi_rect_xy is not None:
            self.roi_rect_xy.remove()
            self.roi_rect_xy = None
        if self.roi_rect_xz is not None:
            self.roi_rect_xz.remove()
            self.roi_rect_xz = None
        if self.roi_rect_yz is not None:
            self.roi_rect_yz.remove()
            self.roi_rect_yz = None

    def _renderOrientationOverlay(self):
        """Render orientation colormap overlay on full-size image (only ROI region colored)"""
        import matplotlib.cm as cm

        # Collect all checked ROIs and their selected orientation type (radio button)
        checked_rois = []
        for roi_name, widgets in self.orientation_roi_widgets.items():
            if widgets['check'].isChecked():
                # Check which radio button is selected
                orientation_type = None
                if widgets['theta_radio'].isChecked():
                    orientation_type = 'theta'
                elif widgets['phi_radio'].isChecked():
                    orientation_type = 'phi'
                elif widgets['angle_radio'].isChecked():
                    orientation_type = 'angle'

                if orientation_type:  # Only add if an orientation type is selected
                    checked_rois.append((roi_name, [orientation_type]))

        if not checked_rois:
            return  # No orientation selected

        full_shape = self.current_volume.shape

        # Use single orientation colormap for all overlays
        cmap = cm.get_cmap(self.orientation_colormap)

        # Calculate alpha based on total number of overlays
        total_overlays = sum(len(orient_types) for _, orient_types in checked_rois)
        base_alpha = 0.6 if total_overlays == 1 else max(0.3, 0.6 / (total_overlays ** 0.5))

        # Render each checked ROI and its orientation types
        for roi_name, orientation_types in checked_rois:
            if roi_name not in self.rois:
                continue

            roi_data = self.rois[roi_name]

            # Get ROI bounds and trim width
            bounds = roi_data['bounds']
            z_min, z_max, y_min, y_max, x_min, x_max = bounds
            trim_width = roi_data.get('trim_width', 0)

            # Apply offset to center the trimmed orientation data within the ROI
            z_offset = z_min + trim_width
            y_offset = y_min + trim_width
            x_offset = x_min + trim_width

            # Render each selected orientation type for this ROI
            for orient_type in orientation_types:
                orientation_volume = roi_data.get(orient_type)

                if orientation_volume is None:
                    continue

                # Create full-size arrays filled with NaN (transparent)
                orientation_full_xy = np.full((full_shape[1], full_shape[2]), np.nan)
                orientation_full_xz = np.full((full_shape[0], full_shape[2]), np.nan)
                orientation_full_yz = np.full((full_shape[0], full_shape[1]), np.nan)

                # Fill ROI region with orientation data (centered)
                try:
                    # XY plane (Z slice)
                    if z_offset <= self.slice_z < z_max - trim_width:
                        z_roi = self.slice_z - z_offset
                        if 0 <= z_roi < orientation_volume.shape[0]:
                            roi_slice = orientation_volume[z_roi, :, :]
                            h, w = roi_slice.shape
                            y_end = min(y_offset + h, y_max - trim_width, full_shape[1])
                            x_end = min(x_offset + w, x_max - trim_width, full_shape[2])
                            h_use = y_end - y_offset
                            w_use = x_end - x_offset
                            if h_use > 0 and w_use > 0:
                                orientation_full_xy[y_offset:y_end, x_offset:x_end] = roi_slice[:h_use, :w_use]

                    # XZ plane (Y slice)
                    if y_offset <= self.slice_y < y_max - trim_width:
                        y_roi = self.slice_y - y_offset
                        if 0 <= y_roi < orientation_volume.shape[1]:
                            roi_slice = orientation_volume[:, y_roi, :]
                            d, w = roi_slice.shape
                            z_end = min(z_offset + d, z_max - trim_width, full_shape[0])
                            x_end = min(x_offset + w, x_max - trim_width, full_shape[2])
                            d_use = z_end - z_offset
                            w_use = x_end - x_offset
                            if d_use > 0 and w_use > 0:
                                orientation_full_xz[z_offset:z_end, x_offset:x_end] = roi_slice[:d_use, :w_use]

                    # YZ plane (X slice)
                    if x_offset <= self.slice_x < x_max - trim_width:
                        x_roi = self.slice_x - x_offset
                        if 0 <= x_roi < orientation_volume.shape[2]:
                            roi_slice = orientation_volume[:, :, x_roi]
                            d, h = roi_slice.shape
                            z_end = min(z_offset + d, z_max - trim_width, full_shape[0])
                            y_end = min(y_offset + h, y_max - trim_width, full_shape[1])
                            d_use = z_end - z_offset
                            h_use = y_end - y_offset
                            if d_use > 0 and h_use > 0:
                                orientation_full_yz[z_offset:z_end, y_offset:y_end] = roi_slice[:d_use, :h_use]
                except Exception as e:
                    print(f"Error placing orientation overlay for {roi_name}-{orient_type}: {e}")
                    continue

                # Get data range for normalization (use custom range if set)
                if self.orientation_range:
                    data_min, data_max = self.orientation_range
                elif orientation_volume.size == 0:
                    # Skip if empty volume
                    continue
                else:
                    data_min = np.nanmin(orientation_volume)
                    data_max = np.nanmax(orientation_volume)

                # Overlay on XY view
                self.ax_xy.imshow(orientation_full_xy, cmap=cmap, origin='lower',
                                 vmin=data_min, vmax=data_max, alpha=base_alpha, aspect='equal')

                # Overlay on XZ view
                self.ax_xz.imshow(orientation_full_xz, cmap=cmap, origin='lower',
                                 vmin=data_min, vmax=data_max, alpha=base_alpha, aspect='equal')

                # Overlay on YZ view
                self.ax_yz.imshow(orientation_full_yz, cmap=cmap, origin='lower',
                                 vmin=data_min, vmax=data_max, alpha=base_alpha, aspect='equal')

    def _drawAllROIOverlays(self):
        """Draw colored overlay rectangles for all ROIs on all three views"""
        from matplotlib.patches import Rectangle

        # Draw each ROI as a colored rectangle
        for roi_name, roi_data in self.rois.items():
            bounds = roi_data['bounds']
            color = roi_data['color']
            z_min, z_max, y_min, y_max, x_min, x_max = bounds

            # XY view
            rect_xy = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            self.ax_xy.add_patch(rect_xy)
            # Add label
            self.ax_xy.text(x_min + 5, y_min + 5, roi_name, color=color, fontsize=9,
                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # XZ view
            rect_xz = Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            self.ax_xz.add_patch(rect_xz)
            self.ax_xz.text(x_min + 5, z_min + 5, roi_name, color=color, fontsize=9,
                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # YZ view
            rect_yz = Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            self.ax_yz.add_patch(rect_yz)
            self.ax_yz.text(y_min + 5, z_min + 5, roi_name, color=color, fontsize=9,
                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _renderVoidOverlay(self):
        """Render void analysis overlay only within ROIs and display void ratio for each ROI"""
        if not self.void_analysis_active or self.current_volume is None or not self.rois:
            return

        volume = self.current_volume

        # Iterate through all ROIs
        for roi_name, roi_data in self.rois.items():
            bounds = roi_data['bounds']
            z_min, z_max, y_min, y_max, x_min, x_max = bounds

            # Extract ROI subvolume
            roi_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]

            # Calculate void ratio for this ROI
            void_voxels = np.sum(roi_volume < self.void_threshold)
            total_voxels = roi_volume.size
            void_ratio = (void_voxels / total_voxels) * 100 if total_voxels > 0 else 0

            # Store void ratio in ROI data
            roi_data['void_ratio'] = void_ratio

            # Render void overlay on slices if they intersect the ROI
            # XY plane
            if z_min <= self.slice_z < z_max:
                slice_xy = volume[self.slice_z, y_min:y_max, x_min:x_max]
                void_mask = slice_xy < self.void_threshold
                if np.any(void_mask):
                    # Create full-size masked array
                    void_overlay = np.full(volume.shape[1:], np.nan)
                    void_overlay[y_min:y_max, x_min:x_max] = np.where(void_mask, slice_xy, np.nan)
                    void_overlay_masked = np.ma.masked_invalid(void_overlay)
                    self.ax_xy.imshow(void_overlay_masked, cmap='Reds', origin='lower',
                                     alpha=0.5, vmin=0, vmax=self.void_threshold, aspect='equal')

            # XZ plane
            if y_min <= self.slice_y < y_max:
                slice_xz = volume[z_min:z_max, self.slice_y, x_min:x_max]
                void_mask = slice_xz < self.void_threshold
                if np.any(void_mask):
                    void_overlay = np.full((volume.shape[0], volume.shape[2]), np.nan)
                    void_overlay[z_min:z_max, x_min:x_max] = np.where(void_mask, slice_xz, np.nan)
                    void_overlay_masked = np.ma.masked_invalid(void_overlay)
                    self.ax_xz.imshow(void_overlay_masked, cmap='Reds', origin='lower',
                                     alpha=0.5, vmin=0, vmax=self.void_threshold, aspect='equal')

            # YZ plane
            if x_min <= self.slice_x < x_max:
                slice_yz = volume[z_min:z_max, y_min:y_max, self.slice_x]
                void_mask = slice_yz < self.void_threshold
                if np.any(void_mask):
                    void_overlay = np.full((volume.shape[0], volume.shape[1]), np.nan)
                    void_overlay[z_min:z_max, y_min:y_max] = np.where(void_mask, slice_yz, np.nan)
                    void_overlay_masked = np.ma.masked_invalid(void_overlay)
                    self.ax_yz.imshow(void_overlay_masked, cmap='Reds', origin='lower',
                                     alpha=0.5, vmin=0, vmax=self.void_threshold, aspect='equal')

            # Display void ratio as text near the ROI (on XY view)
            if z_min <= self.slice_z < z_max:
                self.ax_xy.text(x_min + 5, y_max - 10, f'{roi_name}: {void_ratio:.1f}%',
                               color='red', fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

    def _renderFiberDetection(self):
        """Render fiber detection results on XY view (optimized)"""
        if not self.show_fiber_detection or self.fiber_detection_result is None:
            return

        result = self.fiber_detection_result
        roi_offset = result['roi_offset']
        settings = result['settings']
        all_slices = result.get('all_slices', {})

        # Check if current slice has detection results
        if self.slice_z not in all_slices:
            return

        slice_data = all_slices[self.slice_z]
        centers = slice_data['centers']
        diameters = slice_data['diameters']
        labels = slice_data.get('labels', None)

        if len(centers) == 0:
            return

        x_offset, y_offset = roi_offset

        # Pre-compute global coordinates
        global_centers_x = centers[:, 0] + x_offset
        global_centers_y = centers[:, 1] + y_offset

        # Draw watershed regions with color-coded overlay
        if settings.get('show_watershed', True) and labels is not None:
            # Create a colormap for the labels
            from matplotlib.colors import ListedColormap
            import matplotlib.cm as cm

            # Get number of labels
            n_labels = labels.max()
            if n_labels > 0:
                # Create random colormap for distinct fiber colors
                np.random.seed(42)  # For consistent colors
                colors = cm.tab20(np.linspace(0, 1, 20))
                # Extend colors if needed
                if n_labels > 20:
                    colors = np.tile(colors, (n_labels // 20 + 1, 1))[:n_labels]

                # Create masked array for labels (0 = background = transparent)
                labels_masked = np.ma.masked_where(labels == 0, labels)

                # Create extent for proper positioning
                roi_bounds = result.get('roi_bounds')
                if roi_bounds:
                    z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
                    extent = [x_min, x_min + labels.shape[1], y_min, y_min + labels.shape[0]]
                else:
                    extent = [x_offset, x_offset + labels.shape[1],
                              y_offset, y_offset + labels.shape[0]]

                # Display watershed regions with colormap
                self.ax_xy.imshow(labels_masked, cmap='tab20', origin='lower',
                                 extent=extent, alpha=0.5, zorder=3,
                                 interpolation='nearest')

        # Draw detected fiber centers
        if settings.get('show_centers', True):
            marker_size = settings.get('center_marker_size', 3)
            self.ax_xy.scatter(global_centers_x, global_centers_y,
                              c='red', s=marker_size**2, marker='o',
                              edgecolors='white', linewidths=0.5,
                              zorder=10, label=f'{len(centers)} fibers')
            # Add legend only when centers are shown
            self.ax_xy.legend(loc='upper right', fontsize=8,
                             facecolor='white', edgecolor='gray', framealpha=0.9)

    def enableZoom(self, enabled):
        """Enable or disable zoom mode"""
        self.zoom_enabled = enabled

        if enabled:
            # Store original view limits for reset
            self.original_xlim_xy = self.ax_xy.get_xlim()
            self.original_ylim_xy = self.ax_xy.get_ylim()
            self.original_xlim_xz = self.ax_xz.get_xlim()
            self.original_ylim_xz = self.ax_xz.get_ylim()
            self.original_xlim_yz = self.ax_yz.get_xlim()
            self.original_ylim_yz = self.ax_yz.get_ylim()

            # Connect mouse button events for panning
            self.cid_press_xy = self.canvas_xy.mpl_connect('button_press_event', self.onPressXY)
            self.cid_release_xy = self.canvas_xy.mpl_connect('button_release_event', self.onReleaseXY)
            self.cid_motion_xy = self.canvas_xy.mpl_connect('motion_notify_event', self.onMotionXY)

            self.cid_press_xz = self.canvas_xz.mpl_connect('button_press_event', self.onPressXZ)
            self.cid_release_xz = self.canvas_xz.mpl_connect('button_release_event', self.onReleaseXZ)
            self.cid_motion_xz = self.canvas_xz.mpl_connect('motion_notify_event', self.onMotionXZ)

            self.cid_press_yz = self.canvas_yz.mpl_connect('button_press_event', self.onPressYZ)
            self.cid_release_yz = self.canvas_yz.mpl_connect('button_release_event', self.onReleaseYZ)
            self.cid_motion_yz = self.canvas_yz.mpl_connect('motion_notify_event', self.onMotionYZ)
        else:
            # Reset zoom
            self.zoom_factor_xy = 1.0
            self.zoom_factor_xz = 1.0
            self.zoom_factor_yz = 1.0
            self.pan_offset_xy = [0, 0]
            self.pan_offset_xz = [0, 0]
            self.pan_offset_yz = [0, 0]

            # Restore original view limits if they exist
            if self.original_xlim_xy:
                self.ax_xy.set_xlim(self.original_xlim_xy)
                self.ax_xy.set_ylim(self.original_ylim_xy)
                self.ax_xz.set_xlim(self.original_xlim_xz)
                self.ax_xz.set_ylim(self.original_ylim_xz)
                self.ax_yz.set_xlim(self.original_xlim_yz)
                self.ax_yz.set_ylim(self.original_ylim_yz)

                self.canvas_xy.draw_idle()
                self.canvas_xz.draw_idle()
                self.canvas_yz.draw_idle()

            # Disconnect mouse events
            if hasattr(self, 'cid_press_xy'):
                self.canvas_xy.mpl_disconnect(self.cid_press_xy)
                self.canvas_xy.mpl_disconnect(self.cid_release_xy)
                self.canvas_xy.mpl_disconnect(self.cid_motion_xy)

                self.canvas_xz.mpl_disconnect(self.cid_press_xz)
                self.canvas_xz.mpl_disconnect(self.cid_release_xz)
                self.canvas_xz.mpl_disconnect(self.cid_motion_xz)

                self.canvas_yz.mpl_disconnect(self.cid_press_yz)
                self.canvas_yz.mpl_disconnect(self.cid_release_yz)
                self.canvas_yz.mpl_disconnect(self.cid_motion_yz)

    def _applyZoom(self, ax, zoom_factor, event):
        """Apply zoom to axis centered on mouse position"""
        if event.xdata is None or event.ydata is None:
            return

        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new limits centered on mouse position
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor

        # Center on mouse position
        new_xlim = [xdata - x_range * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
                    xdata + x_range * (xlim[1] - xdata) / (xlim[1] - xlim[0])]
        new_ylim = [ydata - y_range * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                    ydata + y_range * (ylim[1] - ydata) / (ylim[1] - ylim[0])]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

    def onPressXY(self, event):
        """Start panning on XY view"""
        if event.inaxes != self.ax_xy or event.button != 1:
            return
        self.pan_active = True
        self.pan_start = (event.xdata, event.ydata)
        self.pan_view = (self.ax_xy.get_xlim(), self.ax_xy.get_ylim())

    def onReleaseXY(self, event):
        """Stop panning on XY view"""
        self.pan_active = False
        self.pan_start = None
        self.pan_view = None

    def onMotionXY(self, event):
        """Handle panning motion on XY view"""
        if not self.pan_active or self.pan_start is None or event.xdata is None:
            return

        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        xlim, ylim = self.pan_view
        self.ax_xy.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax_xy.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.canvas_xy.draw_idle()

    def onPressXZ(self, event):
        """Start panning on XZ view"""
        if event.inaxes != self.ax_xz or event.button != 1:
            return
        self.pan_active = True
        self.pan_start = (event.xdata, event.ydata)
        self.pan_view = (self.ax_xz.get_xlim(), self.ax_xz.get_ylim())

    def onReleaseXZ(self, event):
        """Stop panning on XZ view"""
        self.pan_active = False
        self.pan_start = None
        self.pan_view = None

    def onMotionXZ(self, event):
        """Handle panning motion on XZ view"""
        if not self.pan_active or self.pan_start is None or event.xdata is None:
            return

        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        xlim, ylim = self.pan_view
        self.ax_xz.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax_xz.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.canvas_xz.draw_idle()

    def onPressYZ(self, event):
        """Start panning on YZ view"""
        if event.inaxes != self.ax_yz or event.button != 1:
            return
        self.pan_active = True
        self.pan_start = (event.xdata, event.ydata)
        self.pan_view = (self.ax_yz.get_xlim(), self.ax_yz.get_ylim())

    def onReleaseYZ(self, event):
        """Stop panning on YZ view"""
        self.pan_active = False
        self.pan_start = None
        self.pan_view = None

    def onMotionYZ(self, event):
        """Handle panning motion on YZ view"""
        if not self.pan_active or self.pan_start is None or event.xdata is None:
            return

        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        xlim, ylim = self.pan_view
        self.ax_yz.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax_yz.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.canvas_yz.draw_idle()

    def _setSquareAspect(self, ax, slice_shape):
        """Force axes to display as square regardless of image aspect ratio

        Args:
            ax: Matplotlib axis
            slice_shape: Shape of the slice (height, width)
        """
        if slice_shape is None or len(slice_shape) < 2:
            return

        height, width = slice_shape
        max_dim = max(height, width)

        # Center the image and make the view square
        x_center = width / 2
        y_center = height / 2
        half_size = max_dim / 2

        ax.set_xlim(x_center - half_size, x_center + half_size)
        ax.set_ylim(y_center - half_size, y_center + half_size)
        ax.set_aspect('equal')

    def _drawAxisArrows(self, ax, plane_type, slice_shape):
        """Draw short axis arrows OUTSIDE the image area in the margin

        Args:
            ax: Matplotlib axis to draw on
            plane_type: 'xy', 'xz', or 'yz'
            slice_shape: Shape of the slice (height, width) - not used
        """
        from matplotlib.patches import FancyArrow

        # Define colors: X=red, Y=green, Z=blue
        colors = {'x': '#FF0000', 'y': '#00FF00', 'z': '#0000FF'}

        # Position OUTSIDE the axes (negative coordinates = outside image area!)
        # These coordinates are in axes fraction, where (0,0)-(1,1) is the image area
        # Position on LEFT side to avoid histogram below
        start_x = -0.12  # OUTSIDE - 12% left of image edge
        start_y = 0.02   # Just inside bottom edge (to avoid histogram)
        arrow_length = 0.08  # 8% of axes size

        # Arrow appearance
        arrow_width = 0.01   # width of arrow shaft
        head_width = 0.03    # width of arrow head (3x shaft)
        head_length = 0.02   # length of arrow head

        if plane_type == 'xy':
            # XY plane: show X (horizontal, red) and Y (vertical, green)
            # X axis arrow - horizontal, OUTSIDE image
            arrow_x = FancyArrow(start_x, start_y, arrow_length, 0,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['x'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_x)
            ax.text(start_x + arrow_length + 0.01, start_y, 'X',
                   transform=ax.transAxes,
                   color=colors['x'], fontsize=10, fontweight='bold',
                   ha='left', va='center', clip_on=False)

            # Y axis arrow - vertical, OUTSIDE image
            arrow_y = FancyArrow(start_x, start_y, 0, arrow_length,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['y'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_y)
            ax.text(start_x, start_y + arrow_length + 0.01, 'Y',
                   transform=ax.transAxes,
                   color=colors['y'], fontsize=10, fontweight='bold',
                   ha='center', va='bottom', clip_on=False)

        elif plane_type == 'xz':
            # XZ plane: show X (horizontal, red) and Z (vertical, blue)
            # X axis arrow - horizontal, OUTSIDE image
            arrow_x = FancyArrow(start_x, start_y, arrow_length, 0,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['x'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_x)
            ax.text(start_x + arrow_length + 0.01, start_y, 'X',
                   transform=ax.transAxes,
                   color=colors['x'], fontsize=10, fontweight='bold',
                   ha='left', va='center', clip_on=False)

            # Z axis arrow - vertical, OUTSIDE image
            arrow_z = FancyArrow(start_x, start_y, 0, arrow_length,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['z'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_z)
            ax.text(start_x, start_y + arrow_length + 0.01, 'Z',
                   transform=ax.transAxes,
                   color=colors['z'], fontsize=10, fontweight='bold',
                   ha='center', va='bottom', clip_on=False)

        elif plane_type == 'yz':
            # YZ plane: show Y (horizontal, green) and Z (vertical, blue)
            # Y axis arrow - horizontal, OUTSIDE image
            arrow_y = FancyArrow(start_x, start_y, arrow_length, 0,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['y'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_y)
            ax.text(start_x + arrow_length + 0.01, start_y, 'Y',
                   transform=ax.transAxes,
                   color=colors['y'], fontsize=10, fontweight='bold',
                   ha='left', va='center', clip_on=False)

            # Z axis arrow - vertical, OUTSIDE image
            arrow_z = FancyArrow(start_x, start_y, 0, arrow_length,
                                width=arrow_width, head_width=head_width,
                                head_length=head_length,
                                transform=ax.transAxes, color=colors['z'],
                                clip_on=False, zorder=1000)
            ax.add_patch(arrow_z)
            ax.text(start_x, start_y + arrow_length + 0.01, 'Z',
                   transform=ax.transAxes,
                   color=colors['z'], fontsize=10, fontweight='bold',
                   ha='center', va='bottom', clip_on=False)

    def _updateMainWindowSliders(self):
        """Update main window sliders to match current slice positions from mouse scroll"""
        if self.parent_window is None:
            return

        # Temporarily disconnect signals to avoid triggering updateSlices
        try:
            self.parent_window.x_slice_slider.valueChanged.disconnect()
            self.parent_window.y_slice_slider.valueChanged.disconnect()
            self.parent_window.z_slice_slider.valueChanged.disconnect()
        except:
            pass  # Signals may not be connected yet

        # Update slider and spin box values
        self.parent_window.x_slice_slider.setValue(self.slice_x)
        self.parent_window.x_slice_spin.setValue(self.slice_x)
        self.parent_window.y_slice_slider.setValue(self.slice_y)
        self.parent_window.y_slice_spin.setValue(self.slice_y)
        self.parent_window.z_slice_slider.setValue(self.slice_z)
        self.parent_window.z_slice_spin.setValue(self.slice_z)

        # Reconnect signals
        self.parent_window.x_slice_slider.valueChanged.connect(self.parent_window.updateSlices)
        self.parent_window.y_slice_slider.valueChanged.connect(self.parent_window.updateSlices)
        self.parent_window.z_slice_slider.valueChanged.connect(self.parent_window.updateSlices)

    def renderSlicesOnly(self):
        """Fast rendering - only update slice views without histograms"""
        if self.current_volume is None:
            return

        volume = self.current_volume

        # Get intensity range
        if self.intensity_range:
            vmin, vmax = self.intensity_range
        else:
            vmin, vmax = 0, 255

        # Clear and render XY plane
        self.ax_xy.clear()
        slice_xy = volume[self.slice_z, :, :]
        self.ax_xy.imshow(slice_xy, cmap=self.colormap, origin='lower',
                         vmin=vmin, vmax=vmax, aspect='equal')
        self._setSquareAspect(self.ax_xy, slice_xy.shape)
        self.ax_xy.set_title(f'XY Plane (Z={self.slice_z})', fontsize=10, fontweight='bold')
        self.ax_xy.axis('off')
        # Draw axis arrows
        self._drawAxisArrows(self.ax_xy, 'xy', slice_xy.shape)

        # Clear and render XZ plane
        self.ax_xz.clear()
        slice_xz = volume[:, self.slice_y, :]
        self.ax_xz.imshow(slice_xz, cmap=self.colormap, origin='lower',
                         vmin=vmin, vmax=vmax, aspect='equal')
        self._setSquareAspect(self.ax_xz, slice_xz.shape)
        self.ax_xz.set_title(f'XZ Plane (Y={self.slice_y})', fontsize=10, fontweight='bold')
        self.ax_xz.axis('off')
        # Draw axis arrows
        self._drawAxisArrows(self.ax_xz, 'xz', slice_xz.shape)

        # Clear and render YZ plane
        self.ax_yz.clear()
        slice_yz = volume[:, :, self.slice_x]
        self.ax_yz.imshow(slice_yz, cmap=self.colormap, origin='lower',
                         vmin=vmin, vmax=vmax, aspect='equal')
        self._setSquareAspect(self.ax_yz, slice_yz.shape)
        self.ax_yz.set_title(f'YZ Plane (X={self.slice_x})', fontsize=10, fontweight='bold')
        self.ax_yz.axis('off')
        # Draw axis arrows
        self._drawAxisArrows(self.ax_yz, 'yz', slice_yz.shape)

        # Add void overlay only within ROIs
        self._renderVoidOverlay()

        # Render orientation overlays if active
        self._renderOrientationOverlay()

        # Render fiber detection results if available
        self._renderFiberDetection()

        # Draw ROI rectangles
        self._drawAllROIOverlays()

        # Refresh only the slice canvases (not histograms)
        self.canvas_xy.draw_idle()
        self.canvas_xz.draw_idle()
        self.canvas_yz.draw_idle()

    def renderVolume(self):
        """Render 2D slices of the volume"""
        if self.current_volume is None:
            return

        try:
            # Clear all axes
            self.ax_xy.clear()
            self.ax_xz.clear()
            self.ax_yz.clear()
            self.ax_hist_intensity.clear()
            self.ax_hist_orientation.clear()

            # Get volume data
            if len(self.current_volume.shape) == 3:
                volume = self.current_volume.astype(np.float32)
            else:
                volume = np.mean(self.current_volume, axis=-1).astype(np.float32)

            # Ensure slice positions are within bounds
            self.slice_z = max(0, min(self.slice_z, volume.shape[0] - 1))
            self.slice_y = max(0, min(self.slice_y, volume.shape[1] - 1))
            self.slice_x = max(0, min(self.slice_x, volume.shape[2] - 1))

            # If an ROI overlay is active, ensure slices are within the valid ROI range
            # Use first checked ROI for slice clamping
            selected_roi = None
            for roi_name, widgets in self.orientation_roi_widgets.items():
                if widgets['check'].isChecked():
                    selected_roi = roi_name
                    break

            if selected_roi and selected_roi in self.rois:
                roi_data = self.rois[selected_roi]
                bounds = roi_data['bounds']
                z_min, z_max, y_min, y_max, x_min, x_max = bounds
                trim_width = roi_data.get('trim_width', 0)

                # Clamp slice positions to valid ROI range (accounting for trim)
                z_roi_min = z_min + trim_width
                z_roi_max = z_max - trim_width - 1
                y_roi_min = y_min + trim_width
                y_roi_max = y_max - trim_width - 1
                x_roi_min = x_min + trim_width
                x_roi_max = x_max - trim_width - 1

                if z_roi_min <= z_roi_max:
                    self.slice_z = max(z_roi_min, min(self.slice_z, z_roi_max))
                if y_roi_min <= y_roi_max:
                    self.slice_y = max(y_roi_min, min(self.slice_y, y_roi_max))
                if x_roi_min <= x_roi_max:
                    self.slice_x = max(x_roi_min, min(self.slice_x, x_roi_max))

            # Get intensity range (custom or auto)
            if self.intensity_range:
                vmin, vmax = self.intensity_range
            else:
                vmin, vmax = 0, 255

            # Render XY plane (Z slice)
            slice_xy = volume[self.slice_z, :, :]
            im_xy = self.ax_xy.imshow(slice_xy, cmap=self.colormap, origin='lower',
                                       vmin=vmin, vmax=vmax, aspect='equal')
            self._setSquareAspect(self.ax_xy, slice_xy.shape)
            self.ax_xy.set_title(f'XY Plane (Z={self.slice_z})', fontsize=10, fontweight='bold')
            self.ax_xy.axis('off')
            # Draw axis arrows
            self._drawAxisArrows(self.ax_xy, 'xy', slice_xy.shape)

            # Render fiber detection results if available
            self._renderFiberDetection()

            self.figure_xy.tight_layout()

            # Render XZ plane (Y slice)
            slice_xz = volume[:, self.slice_y, :]
            im_xz = self.ax_xz.imshow(slice_xz, cmap=self.colormap, origin='lower',
                                       vmin=vmin, vmax=vmax, aspect='equal')
            self._setSquareAspect(self.ax_xz, slice_xz.shape)
            self.ax_xz.set_title(f'XZ Plane (Y={self.slice_y})', fontsize=10, fontweight='bold')
            self.ax_xz.axis('off')
            # Draw axis arrows
            self._drawAxisArrows(self.ax_xz, 'xz', slice_xz.shape)
            self.figure_xz.tight_layout()

            # Render YZ plane (X slice)
            slice_yz = volume[:, :, self.slice_x]
            im_yz = self.ax_yz.imshow(slice_yz, cmap=self.colormap, origin='lower',
                                       vmin=vmin, vmax=vmax, aspect='equal')
            self._setSquareAspect(self.ax_yz, slice_yz.shape)
            self.ax_yz.set_title(f'YZ Plane (X={self.slice_x})', fontsize=10, fontweight='bold')
            self.ax_yz.axis('off')
            # Draw axis arrows
            self._drawAxisArrows(self.ax_yz, 'yz', slice_yz.shape)
            self.figure_yz.tight_layout()

            # Add void overlay only within ROIs
            self._renderVoidOverlay()

            # Check if orientation overlay should be shown
            self._renderOrientationOverlay()

            # Render dual histograms - intensity (left) and orientation (right)
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Always render intensity histogram on the left
            self.ax_hist_intensity.clear()
            self.ax_hist_intensity.set_title('Intensity Histogram', fontsize=10, fontweight='bold')
            self.ax_hist_intensity.set_xlabel('Intensity', fontsize=9)
            self.ax_hist_intensity.set_ylabel('Density', fontsize=9)
            self.ax_hist_intensity.grid(True, alpha=0.3)

            # Get intensity range
            if self.intensity_range:
                hist_vmin, hist_vmax = self.intensity_range
            else:
                hist_vmin, hist_vmax = 0, 255

            # Render intensity histogram (density mode)
            hist_data = volume.flatten()
            density, bins = np.histogram(hist_data, bins=256, range=(hist_vmin, hist_vmax), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            self.ax_hist_intensity.plot(bin_centers, density, linewidth=2, color='black')
            self.ax_hist_intensity.fill_between(bin_centers, density, alpha=0.3, color='gray')
            self.ax_hist_intensity.set_xlim(hist_vmin, hist_vmax)
            self.ax_hist_intensity.set_ylim(bottom=0)

            # Add static red threshold line (not draggable)
            self.ax_hist_intensity.axvline(x=self.void_threshold, color='red',
                                          linewidth=2, linestyle='--', alpha=0.8)

            # Clear and update intensity colorbar
            self.ax_hist_intensity_cbar.clear()
            cmap = cm.get_cmap(self.colormap)
            norm = plt.Normalize(vmin=hist_vmin, vmax=hist_vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.figure_hist.colorbar(sm, cax=self.ax_hist_intensity_cbar, label='Intensity')

            # Always render orientation histogram on the right
            self.ax_hist_orientation.clear()
            self.ax_hist_orientation.set_title('Orientation Histogram', fontsize=10, fontweight='bold')
            self.ax_hist_orientation.set_xlabel('Angle (degrees)', fontsize=9)
            self.ax_hist_orientation.set_ylabel('Density', fontsize=9)
            self.ax_hist_orientation.grid(True, alpha=0.3)

            # Collect all checked ROIs for histogram display
            selected_rois = []
            for roi_name, widgets in self.orientation_roi_widgets.items():
                if widgets['check'].isChecked():
                    selected_rois.append(roi_name)

            if selected_rois:
                # Show orientation histograms for all checked ROIs
                # Get orientation range (custom or auto)
                if self.orientation_range:
                    ori_vmin, ori_vmax = self.orientation_range
                    data_min_global = ori_vmin
                    data_max_global = ori_vmax
                else:
                    data_min_global = float('inf')
                    data_max_global = float('-inf')

                max_count = 0

                # Define colors for different ROIs
                roi_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                # Plot histogram for each checked ROI
                for roi_idx, roi_name in enumerate(selected_rois):
                    if roi_name not in self.rois:
                        continue

                    roi_data = self.rois[roi_name]
                    widgets = self.orientation_roi_widgets.get(roi_name)
                    if not widgets:
                        continue

                    # Determine which orientation type is selected (via radio buttons)
                    orientation_type = None
                    orientation_label = None
                    if widgets['theta_radio'].isChecked():
                        orientation_type = 'theta'
                        orientation_label = 'X-Z'
                    elif widgets['phi_radio'].isChecked():
                        orientation_type = 'phi'
                        orientation_label = 'Y-Z'
                    elif widgets['angle_radio'].isChecked():
                        orientation_type = 'angle'
                        orientation_label = 'Reference'

                    if orientation_type and roi_data.get(orientation_type) is not None:
                        hist_data = roi_data[orientation_type].flatten()

                        # Use custom range or calculate from data
                        if self.orientation_range:
                            data_min, data_max = ori_vmin, ori_vmax
                        else:
                            data_min = np.nanmin(hist_data)
                            data_max = np.nanmax(hist_data)
                            data_min_global = min(data_min_global, data_min)
                            data_max_global = max(data_max_global, data_max)

                        density, bins = np.histogram(hist_data[~np.isnan(hist_data)], bins=180, range=(data_min, data_max), density=True)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        max_count = max(max_count, np.max(density))

                        # Use different color for each ROI
                        color = roi_colors[roi_idx % len(roi_colors)]
                        label = f'{roi_name} ({orientation_label})'

                        self.ax_hist_orientation.plot(bin_centers, density, linewidth=2, color=color, label=label)
                        self.ax_hist_orientation.fill_between(bin_centers, density, alpha=0.3, color=color)

                # Set up axes if any histogram was plotted
                if data_min_global != float('inf'):
                    if len(selected_rois) > 1:
                        self.ax_hist_orientation.set_title('Orientation Histograms - Multiple ROIs', fontsize=10, fontweight='bold')
                    else:
                        self.ax_hist_orientation.set_title(f'Orientation Histogram - {selected_rois[0]}', fontsize=10, fontweight='bold')

                    self.ax_hist_orientation.set_xlim(data_min_global, data_max_global)
                    self.ax_hist_orientation.set_ylim(0, max_count * 1.1)  # Add 10% headroom
                    self.ax_hist_orientation.legend(loc='upper right', fontsize=8)

                    # Clear and update orientation colorbar
                    self.ax_hist_orientation_cbar.clear()
                    cmap = cm.get_cmap(self.orientation_colormap)
                    norm = plt.Normalize(vmin=data_min_global, vmax=data_max_global)
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    self.figure_hist.colorbar(sm, cax=self.ax_hist_orientation_cbar, label='Angle')

            # If overlay mode, add orientation overlay
            if self.base_volume is not None and self.overlay_volume is not None:
                self._renderOverlay()

            # Draw all ROI overlays (colored rectangles with labels)
            self._drawAllROIOverlays()

            # Draw all canvases
            self.canvas_xy.draw()
            self.canvas_xz.draw()
            self.canvas_yz.draw()
            self.canvas_hist.draw()

        except Exception as e:
            print(f"Rendering error: {e}")

    def _renderOverlay(self):
        """Helper method to render orientation overlay on base volume slices"""
        # Adjust slice positions for overlay volume (accounting for trim)
        trim = self.trim_width
        overlay_z = max(0, min(self.slice_z - trim, self.overlay_volume.shape[0] - 1))
        overlay_y = max(0, min(self.slice_y - trim, self.overlay_volume.shape[1] - 1))
        overlay_x = max(0, min(self.slice_x - trim, self.overlay_volume.shape[2] - 1))

        # Get orientation ranges
        ovmin, ovmax = self.orientation_range if self.orientation_range else (None, None)

        # Overlay on XY plane
        if 0 <= overlay_z < self.overlay_volume.shape[0]:
            overlay_xy = self.overlay_volume[overlay_z, :, :]
            im_overlay_xy = self.ax_xy.imshow(overlay_xy, cmap=self.orientation_colormap,
                                               origin='lower', alpha=self.orientation_opacity,
                                               vmin=ovmin, vmax=ovmax, aspect='equal',
                                               extent=[trim, trim + overlay_xy.shape[1],
                                                       trim, trim + overlay_xy.shape[0]])

        # Overlay on XZ plane
        if 0 <= overlay_y < self.overlay_volume.shape[1]:
            overlay_xz = self.overlay_volume[:, overlay_y, :]
            im_overlay_xz = self.ax_xz.imshow(overlay_xz, cmap=self.orientation_colormap,
                                               origin='lower', alpha=self.orientation_opacity,
                                               vmin=ovmin, vmax=ovmax, aspect='equal',
                                               extent=[trim, trim + overlay_xz.shape[1],
                                                       trim, trim + overlay_xz.shape[0]])

        # Overlay on YZ plane
        if 0 <= overlay_x < self.overlay_volume.shape[2]:
            overlay_yz = self.overlay_volume[:, :, overlay_x]
            im_overlay_yz = self.ax_yz.imshow(overlay_yz, cmap=self.orientation_colormap,
                                               origin='lower', alpha=self.orientation_opacity,
                                               vmin=ovmin, vmax=ovmax, aspect='equal',
                                               extent=[trim, trim + overlay_yz.shape[1],
                                                       trim, trim + overlay_yz.shape[0]])

    def setRenderMethod(self, method):
        """Compatibility method - 2D viewer doesn't use render methods"""
        pass

    def setColormap(self, cmap):
        self.colormap = cmap
        self.renderVolume()

    def setOpacity(self, opacity):
        self.opacity = opacity
        self.renderVolume()

    def setOrientationOpacity(self, opacity):
        self.orientation_opacity = opacity
        # Re-render if there's current content to update
        if self.current_volume is not None:
            self.renderVolume()

    def findMainWindow(self):
        """Find the main window from the widget hierarchy"""
        widget = self
        while widget:
            if isinstance(widget, ACSCMainWindow):
                return widget
            widget = widget.parent()
        return None

    def setOrientationColormap(self, colormap):
        self.orientation_colormap = colormap
        # Re-render if there's current content to update
        if self.current_volume is not None:
            self.renderVolume()

    def updateSlicePositions(self, x, y, z):
        """Update slice positions for the 2D views"""
        self.slice_x = x
        self.slice_y = y
        self.slice_z = z
        # Use fast slice-only rendering for responsive slider updates
        self.renderSlicesOnly()

    def updateSlices(self, x, y, z):
        """Alias for updateSlicePositions for compatibility"""
        self.updateSlicePositions(x, y, z)

    def setIsosurfaceValue(self, value):
        """Compatibility method - not used in 2D viewer"""
        pass

    def updateIsosurface(self, value):
        """Compatibility method - not used in 2D viewer"""
        pass

    def resetCamera(self):
        """Compatibility method - resets view to default"""
        if self.current_volume is not None:
            self.renderVolume()

    def screenshot(self, filename):
        """Save current view as image - saves all three slice views"""
        if self.current_volume is not None:
            # Create a combined figure with all three views
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            fig = plt.figure(figsize=(15, 5))
            gs = GridSpec(1, 3, figure=fig)

            # Copy XY view
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(self.current_volume[self.slice_z, :, :], cmap=self.colormap, origin='lower')
            ax1.set_title(f'XY Plane (Z={self.slice_z})', fontweight='bold')
            ax1.axis('off')

            # Copy XZ view
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(self.current_volume[:, self.slice_y, :], cmap=self.colormap, origin='lower')
            ax2.set_title(f'XZ Plane (Y={self.slice_y})', fontweight='bold')
            ax2.axis('off')

            # Copy YZ view
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(self.current_volume[:, :, self.slice_x], cmap=self.colormap, origin='lower')
            ax3.set_title(f'YZ Plane (X={self.slice_x})', fontweight='bold')
            ax3.axis('off')

            fig.tight_layout()
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)

            QMessageBox.information(None, "Success", f"Screenshot saved to {filename}")

    def export3D(self, filename, iso_value):
        """Export functionality removed - 2D viewer only"""
        QMessageBox.warning(None, "Not Available",
                            "3D export is not available in 2D slice viewer mode.\n"
                            "This feature requires the 3D PyVista viewer.")


class VolumeTab(QWidget):
    """Tab for volume data visualization and import."""
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Ribbon toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0; }")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(10)

        # File Operations Group
        file_group = QGroupBox("File")
        file_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        file_layout = QHBoxLayout(file_group)

        self.open_btn = RibbonButton("Open\nFiles")
        self.open_btn.clicked.connect(self.openImportDialog)
        file_layout.addWidget(self.open_btn)

        toolbar_layout.addWidget(file_group)

        # Render Mode Group removed - 2D viewer always shows slices

        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        appearance_layout = QVBoxLayout(appearance_group)

        appearance_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = RibbonComboBox()
        self.colormap_combo.addItems(["gray", "viridis", "jet", "coolwarm", "rainbow", "bone"])
        self.colormap_combo.currentTextChanged.connect(self.updateColormap)
        appearance_layout.addWidget(self.colormap_combo)

        toolbar_layout.addWidget(appearance_group)

        # Camera Group
        camera_group = QGroupBox("Camera")
        camera_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        camera_layout = QHBoxLayout(camera_group)

        self.reset_view_btn = RibbonButton("Reset\nView")
        self.reset_view_btn.clicked.connect(self.resetView)
        camera_layout.addWidget(self.reset_view_btn)

        toolbar_layout.addWidget(camera_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QHBoxLayout(export_group)

        self.screenshot_btn = RibbonButton("Screen\nshot")
        self.screenshot_btn.clicked.connect(self.takeScreenshot)
        export_layout.addWidget(self.screenshot_btn)

        self.export_btn = RibbonButton("Export\n3D")
        self.export_btn.clicked.connect(self.export3D)
        export_layout.addWidget(self.export_btn)

        toolbar_layout.addWidget(export_group)

        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

        # Note: Control widgets (opacity, slices, isosurface) are now in the left panel

        layout.addStretch()

    def connectViewer(self, viewer):
        self.viewer = viewer

    def openImportDialog(self):
        """Open the import dialog window"""
        if not hasattr(self, 'import_dialog') or not self.import_dialog:
            self.import_dialog = ImportDialog(self)
            self.import_dialog.volume_imported.connect(self.onVolumeImported)

        self.import_dialog.show()
        self.import_dialog.raise_()
        self.import_dialog.activateWindow()

    def onVolumeImported(self, volume):
        """Handle imported volume from dialog"""
        # Find main window and use its setVolume method
        main_window = self.parent()
        while main_window and not hasattr(main_window, 'setVolume'):
            main_window = main_window.parent()

        if main_window:
            # Use main window's setVolume to properly initialize everything
            main_window.setVolume(volume)
        elif self.viewer:
            # Fallback if main window not found
            self.viewer.setVolume(volume)

    def updateRenderMethod(self):
        """Compatibility method - no render modes in 2D viewer"""
        pass

    def updateColormap(self):
        if self.viewer:
            colormap = self.colormap_combo.currentText()
            self.viewer.setColormap(colormap)

# Color bar colormap sync removed - PyVista handles color bar automatically

# Control methods removed - now handled by main window

    def resetView(self):
        if self.viewer:
            self.viewer.resetCamera()

    def takeScreenshot(self):
        if self.viewer:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Screenshot", "", "PNG Files (*.png);;All Files (*)"
            )
            if filename:
                self.viewer.screenshot(filename)
                msg = QMessageBox(self)
                msg.setWindowTitle("Success")
                msg.setText(f"Screenshot saved to {filename}")
                msg.setIcon(QMessageBox.Information)
                msg.move(self.geometry().center() - msg.rect().center())
                msg.exec_()

    def export3D(self):
        if self.viewer:
            filename, selected_filter = QFileDialog.getSaveFileName(
                self, "Export 3D Model", "", "VTK Files (*.vtk);;STL Files (*.stl);;VTI Files (*.vti)"
            )
            if filename:
                # Add correct extension based on selected filter if not present
                if selected_filter == "VTK Files (*.vtk)" and not filename.endswith('.vtk'):
                    filename += '.vtk'
                elif selected_filter == "STL Files (*.stl)" and not filename.endswith('.stl'):
                    filename += '.stl'
                elif selected_filter == "VTI Files (*.vti)" and not filename.endswith('.vti'):
                    filename += '.vti'
                elif not any(filename.endswith(ext) for ext in ['.vtk', '.stl', '.vti']):
                    # Default to .vtk if no supported extension
                    filename += '.vtk'

                # Use default iso value for 3D export (mean of volume)
                iso_value = np.mean(self.viewer.current_volume) if self.viewer.current_volume is not None else 128
                self.viewer.export3D(filename, iso_value)

# Volume info setting removed - now handled by main window


class VisualizationTab(QWidget):
    """Tab for fiber trajectory visualization with 2x2 viewport layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None
        self.fiber_trajectory = None
        self.structure_tensor = None
        self.volume_data = None  # CT volume for overlay
        self.current_slice = {'x': 0, 'y': 0, 'z': 0}
        self.volume_shape = None
        self.roi_trajectories = {}  # Store trajectories for each ROI
        # Fiber trajectory settings (stored for use by settings dialog)
        self.trajectory_settings = {
            'fiber_diameter': 12.0,
            'volume_fraction': 0.5,
            'propagation_axis': 'Z (default)',
            'integration_method': 'RK4',
            'tilt_min': 0.0,
            'tilt_max': 20.0,
            'sat_min': 0.0,
            'sat_max': 20.0,
            'relax': True,
            'color_by_angle': True,
            'show_fiber_diameter': False,
            'resample': False,
            'resample_interval': 20
        }
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Ribbon toolbar (matching other tabs - simplified)
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0; }")
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(10)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # Settings Group (first)
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)

        self.settings_btn = RibbonButton("Settings")
        self.settings_btn.clicked.connect(self.openSettingsDialog)
        settings_layout.addWidget(self.settings_btn)

        toolbar_layout.addWidget(settings_group)

        # Fiber Trajectory Group
        fiber_group = QGroupBox("Fiber Trajectory")
        fiber_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        fiber_layout = QHBoxLayout(fiber_group)

        self.generate_btn = RibbonButton("Generate\nTrajectory")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generateTrajectory)
        fiber_layout.addWidget(self.generate_btn)

        toolbar_layout.addWidget(fiber_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QHBoxLayout(export_group)

        self.export_screenshot_btn = RibbonButton("Export\nScreenshot")
        self.export_screenshot_btn.setEnabled(False)
        self.export_screenshot_btn.clicked.connect(self.showExportScreenshotDialog)
        export_layout.addWidget(self.export_screenshot_btn)

        toolbar_layout.addWidget(export_group)

        # Analysis Group
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QHBoxLayout(analysis_group)

        self.trajectory_histogram_btn = RibbonButton("Histogram")
        self.trajectory_histogram_btn.setEnabled(False)
        self.trajectory_histogram_btn.clicked.connect(self.openTrajectoryHistogramDialog)
        analysis_layout.addWidget(self.trajectory_histogram_btn)

        toolbar_layout.addWidget(analysis_group)

        toolbar_layout.addStretch()
        layout.addWidget(toolbar)

        # Main content: left panel + 2x2 viewports
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel for controls - use scroll area for responsiveness
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(180)
        left_scroll.setMaximumWidth(280)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # ROI Pipeline Group
        roi_group = QGroupBox("ROI Pipeline")
        roi_layout = QVBoxLayout(roi_group)

        roi_info_label = QLabel("Select ROIs to display:")
        roi_info_label.setStyleSheet("color: #666; font-size: 10px;")
        roi_layout.addWidget(roi_info_label)

        # Scroll area for ROI checkboxes
        self.roi_scroll = QScrollArea()
        self.roi_scroll.setWidgetResizable(True)
        self.roi_scroll.setMaximumHeight(120)
        self.roi_list_widget = QWidget()
        self.roi_list_layout = QVBoxLayout(self.roi_list_widget)
        self.roi_list_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_list_layout.setSpacing(2)
        self.roi_checkboxes = {}  # Store ROI checkboxes
        # Initial message
        no_roi_label = QLabel("No ROIs available")
        no_roi_label.setStyleSheet("color: #888; font-style: italic;")
        self.roi_list_layout.addWidget(no_roi_label)
        self.roi_checkboxes['_no_roi'] = no_roi_label
        self.roi_list_layout.addStretch()
        self.roi_scroll.setWidget(self.roi_list_widget)
        roi_layout.addWidget(self.roi_scroll)

        # Refresh ROI list button
        self.refresh_roi_btn = QPushButton("Refresh ROI List")
        self.refresh_roi_btn.clicked.connect(self.refreshROIList)
        roi_layout.addWidget(self.refresh_roi_btn)

        left_layout.addWidget(roi_group)

        # Display Options Group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        self.show_ct_check = QCheckBox("Show CT Image")
        self.show_ct_check.setChecked(False)
        self.show_ct_check.stateChanged.connect(self.updateSliceViews)
        display_layout.addWidget(self.show_ct_check)

        self.ct_opacity_label = QLabel("CT Opacity: 50%")
        display_layout.addWidget(self.ct_opacity_label)
        self.ct_opacity_slider = QSlider(Qt.Horizontal)
        self.ct_opacity_slider.setRange(0, 100)
        self.ct_opacity_slider.setValue(50)
        self.ct_opacity_slider.valueChanged.connect(self.onCTOpacityChanged)
        display_layout.addWidget(self.ct_opacity_slider)

        left_layout.addWidget(display_group)

        # 3D Visualization Mode Group
        vis_mode_group = QGroupBox("3D Visualization")
        vis_mode_layout = QVBoxLayout(vis_mode_group)

        # Line width slider for fiber trajectories
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("Line Width:"))
        self.line_width_slider = QSlider(Qt.Horizontal)
        self.line_width_slider.setRange(1, 10)
        self.line_width_slider.setValue(2)
        self.line_width_slider.valueChanged.connect(self.updateVisualization)
        line_width_layout.addWidget(self.line_width_slider)
        self.line_width_label = QLabel("2")
        self.line_width_label.setMinimumWidth(20)
        self.line_width_slider.valueChanged.connect(lambda v: self.line_width_label.setText(f"{v}"))
        line_width_layout.addWidget(self.line_width_label)
        vis_mode_layout.addLayout(line_width_layout)

        left_layout.addWidget(vis_mode_group)

        # Camera Preset Group
        camera_group = QGroupBox("3D Camera")
        camera_layout = QVBoxLayout(camera_group)

        camera_layout.addWidget(QLabel("View Preset:"))
        self.camera_preset_combo = QComboBox()
        self.camera_preset_combo.addItems([
            "Isometric",
            "Front (XY)",
            "Back (-XY)",
            "Right (YZ)",
            "Left (-YZ)",
            "Top (XZ)",
            "Bottom (-XZ)"
        ])
        self.camera_preset_combo.currentTextChanged.connect(self.setCameraPreset)
        self.camera_preset_combo.setToolTip("Select a preset camera view for 3D visualization")
        camera_layout.addWidget(self.camera_preset_combo)

        # Reset camera button
        self.reset_camera_btn = QPushButton("Reset Camera")
        self.reset_camera_btn.clicked.connect(self.resetCamera)
        self.reset_camera_btn.setToolTip("Reset camera to fit all visible objects")
        camera_layout.addWidget(self.reset_camera_btn)

        # Perspective projection toggle
        self.perspective_check = QCheckBox("Perspective Projection")
        self.perspective_check.setChecked(True)
        self.perspective_check.stateChanged.connect(self.togglePerspective)
        self.perspective_check.setToolTip("ON: Perspective projection (realistic)\nOFF: Parallel/Orthographic projection")
        camera_layout.addWidget(self.perspective_check)

        left_layout.addWidget(camera_group)

        # Colormap Group
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QVBoxLayout(colormap_group)

        colormap_layout.addWidget(QLabel("Color Mode:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Tilt Angle", "Azimuth (HSV)"])
        self.color_mode_combo.currentTextChanged.connect(self.updateVisualization)
        self.color_mode_combo.setToolTip("Tilt: color by angle from fiber axis\nAzimuth: color by direction in cross-section (HSV)")
        colormap_layout.addWidget(self.color_mode_combo)

        colormap_layout.addWidget(QLabel("Colormap (Tilt):"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["coolwarm", "viridis", "jet", "rainbow", "plasma", "turbo"])
        self.colormap_combo.currentTextChanged.connect(self.updateVisualization)
        colormap_layout.addWidget(self.colormap_combo)

        left_layout.addWidget(colormap_group)

        # Slice Controls Group
        slice_group = QGroupBox("Slice Position")
        slice_layout = QGridLayout(slice_group)

        slice_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_slice_slider = QSlider(Qt.Horizontal)
        self.x_slice_slider.setEnabled(False)
        self.x_slice_slider.valueChanged.connect(lambda v: self.updateSlicePosition('x', v))
        slice_layout.addWidget(self.x_slice_slider, 0, 1)
        self.x_slice_label = QLabel("0")
        self.x_slice_label.setMinimumWidth(40)
        slice_layout.addWidget(self.x_slice_label, 0, 2)

        slice_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_slice_slider = QSlider(Qt.Horizontal)
        self.y_slice_slider.setEnabled(False)
        self.y_slice_slider.valueChanged.connect(lambda v: self.updateSlicePosition('y', v))
        slice_layout.addWidget(self.y_slice_slider, 1, 1)
        self.y_slice_label = QLabel("0")
        self.y_slice_label.setMinimumWidth(40)
        slice_layout.addWidget(self.y_slice_label, 1, 2)

        slice_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_slice_slider = QSlider(Qt.Horizontal)
        self.z_slice_slider.setEnabled(False)
        self.z_slice_slider.valueChanged.connect(lambda v: self.updateSlicePosition('z', v))
        slice_layout.addWidget(self.z_slice_slider, 2, 1)
        self.z_slice_label = QLabel("0")
        self.z_slice_label.setMinimumWidth(40)
        slice_layout.addWidget(self.z_slice_label, 2, 2)

        left_layout.addWidget(slice_group)

        # Statistics Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("No trajectory generated")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self.stats_label)

        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        content_splitter.addWidget(left_scroll)

        # Right side: 2x2 viewport grid
        viewport_widget = QWidget()
        viewport_widget.setMinimumSize(300, 300)  # Minimum size for viewports
        viewport_layout = QGridLayout(viewport_widget)
        viewport_layout.setSpacing(2)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        # Create 4 viewport frames with labels
        self.viewport_frames = []
        viewport_titles = ["3D View", "XY Slice (Z)", "XZ Slice (Y)", "YZ Slice (X)"]

        for i, title in enumerate(viewport_titles):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box | QFrame.Plain)
            frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #555; }")
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(2, 2, 2, 2)
            frame_layout.setSpacing(0)

            # Title label
            title_label = QLabel(title)
            title_label.setStyleSheet("color: white; font-weight: bold; background-color: #444; padding: 2px;")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setMaximumHeight(20)
            frame_layout.addWidget(title_label)

            if i == 0:
                # 3D View: Use PyVista QtInteractor
                self.plotter_3d = QtInteractor(frame)
                self.plotter_3d.set_background('#2a2a2a')
                self.plotter_3d.add_axes()
                frame_layout.addWidget(self.plotter_3d.interactor)
                canvas = self.plotter_3d

                # Add color wheel legend overlay
                self.color_wheel_label = QLabel(frame)
                self.color_wheel_label.setStyleSheet("background-color: transparent;")
                self.color_wheel_label.setFixedSize(120, 120)
                self.color_wheel_label.hide()  # Initially hidden
                self._create_color_wheel_legend()
            else:
                # Slice views: Use Matplotlib canvas
                fig = Figure(figsize=(4, 4), facecolor='#2a2a2a')
                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                ax = fig.add_subplot(111)
                ax.set_facecolor('#2a2a2a')
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#555')
                canvas = FigureCanvas(fig)
                canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                frame_layout.addWidget(canvas)
                canvas.figure = fig
                canvas.axes = ax

            row, col = divmod(i, 2)
            viewport_layout.addWidget(frame, row, col)
            self.viewport_frames.append({'frame': frame, 'canvas': canvas, 'title': title})

        # Histogram panel (initially hidden, placed between left panel and viewport)
        self.histogram_panel = ModellingHistogramPanel()
        self.histogram_panel.setVisible(False)
        content_splitter.addWidget(self.histogram_panel)

        content_splitter.addWidget(viewport_widget)
        content_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        content_splitter.setStretchFactor(1, 0)  # Histogram panel doesn't stretch when hidden
        content_splitter.setStretchFactor(2, 1)  # Viewport area stretches
        content_splitter.setSizes([200, 0, 800])
        content_splitter.setCollapsible(0, False)  # Prevent left panel from collapsing completely
        content_splitter.setCollapsible(1, True)   # Histogram panel can collapse
        content_splitter.setCollapsible(2, False)  # Prevent viewport from collapsing

        # Store reference to content_splitter for histogram panel visibility
        self.content_splitter = content_splitter

        layout.addWidget(content_splitter)

    def _create_color_wheel_legend(self):
        """Create a color wheel legend image with axis labels for azimuth visualization.

        The color mapping follows the image coordinate system:
        - azimuth = arctan2(y, x) where y is row direction (down = positive in numpy)
        - hue = azimuth / 360
        - 0° (red) = +X direction (right)
        - 90° (green/yellow) = up direction (visual)
        - 180° (cyan) = -X direction (left)
        - 270° (magenta) = down direction (visual)

        Fiber azimuths are converted from numpy to display coordinates using:
        display_az = (360 - numpy_az) % 360
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Wedge

        # Create figure with dark background (no polar projection)
        fig = Figure(figsize=(1.5, 1.5), dpi=80, facecolor='#2a2a2a')
        ax = fig.add_subplot(111)  # Standard Cartesian axes

        # Draw color wheel using wedges in Cartesian coordinates
        # The color wheel shows display coordinates where:
        # - 0° = +X direction (right) -> red (hue=0)
        # - 90° = visual up direction -> green/yellow (hue=0.25)
        # - 180° = -X direction (left) -> cyan (hue=0.5)
        # - 270° = visual down direction -> magenta (hue=0.75)
        #
        # Fiber colors are converted from numpy to display using: (360 - az) % 360
        n_seg = 72  # Number of angular segments
        n_rad = 15  # Number of radial segments
        for i_theta in range(n_seg):
            # hue from segment index
            hue = (i_theta + 0.5) / n_seg
            # Flip across X-axis: use negative angles (clockwise from +X)
            theta1 = -(i_theta + 1) * 360 / n_seg
            theta2 = -i_theta * 360 / n_seg
            for i_r in range(n_rad):
                r2 = (i_r + 1) / n_rad
                sat = (i_r + 0.5) / n_rad
                color = colorsys.hsv_to_rgb(hue, sat, 1.0)
                wedge = Wedge((0, 0), r2, theta1, theta2, width=1.0 / n_rad,
                             facecolor=color, edgecolor='none')
                ax.add_patch(wedge)

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#2a2a2a')

        # Add axis labels in display coordinates
        # Standard math convention: counter-clockwise from +X
        ax.text(1.2, 0, '+X', fontsize=8, color='white', ha='left', va='center',
               fontweight='bold')
        ax.text(-1.2, 0, '-X', fontsize=8, color='white', ha='right', va='center',
               fontweight='bold')
        ax.text(0, 1.2, 'Up', fontsize=8, color='white', ha='center', va='bottom',
               fontweight='bold')
        ax.text(0, -1.2, 'Down', fontsize=8, color='white', ha='center', va='top',
               fontweight='bold')

        # Convert to QPixmap
        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()

        # Get the RGBA buffer
        buf = canvas_agg.buffer_rgba()
        width, height = fig.canvas.get_width_height()

        # Create QImage and QPixmap
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        self.color_wheel_label.setPixmap(pixmap)
        plt.close(fig)

    def _update_color_wheel_position(self):
        """Update the position of the color wheel legend in the 3D view."""
        if hasattr(self, 'color_wheel_label') and self.color_wheel_label:
            # Position at bottom-right corner of the 3D viewport
            parent = self.color_wheel_label.parent()
            if parent:
                parent_rect = parent.rect()
                label_size = self.color_wheel_label.size()
                x = parent_rect.width() - label_size.width() - 10
                y = parent_rect.height() - label_size.height() - 10
                self.color_wheel_label.move(x, y)

    def _show_color_wheel_legend(self, show: bool):
        """Show or hide the color wheel legend."""
        if hasattr(self, 'color_wheel_label') and self.color_wheel_label:
            if show:
                self._update_color_wheel_position()
                self.color_wheel_label.show()
                self.color_wheel_label.raise_()
            else:
                self.color_wheel_label.hide()

    def resizeEvent(self, event):
        """Handle resize events to reposition the color wheel legend."""
        super().resizeEvent(event)
        if hasattr(self, 'color_wheel_label') and self.color_wheel_label.isVisible():
            self._update_color_wheel_position()

    def updateMainStatus(self, message: str):
        """Update the main window status bar with a message."""
        if self.main_window and hasattr(self.main_window, 'status_label'):
            self.main_window.status_label.setText(message)
            QApplication.processEvents()

    def setMainWindow(self, main_window):
        """Set reference to main window for accessing shared data."""
        self.main_window = main_window

    def setStructureTensor(self, structure_tensor, volume_shape, volume_data=None):
        """Set structure tensor data for trajectory generation."""
        self.structure_tensor = structure_tensor
        self.volume_shape = volume_shape
        self.volume_data = volume_data
        self.generate_btn.setEnabled(True)

        # Update slice sliders
        if volume_shape is not None:
            self.x_slice_slider.setRange(0, volume_shape[2] - 1)
            self.x_slice_slider.setValue(volume_shape[2] // 2)
            self.x_slice_slider.setEnabled(True)

            self.y_slice_slider.setRange(0, volume_shape[1] - 1)
            self.y_slice_slider.setValue(volume_shape[1] // 2)
            self.y_slice_slider.setEnabled(True)

            self.z_slice_slider.setRange(0, volume_shape[0] - 1)
            self.z_slice_slider.setValue(volume_shape[0] // 2)
            self.z_slice_slider.setEnabled(True)

    def updateSlicePosition(self, axis, value):
        """Update slice position for a given axis."""
        self.current_slice[axis] = value
        if axis == 'x':
            self.x_slice_label.setText(str(value))
        elif axis == 'y':
            self.y_slice_label.setText(str(value))
        elif axis == 'z':
            self.z_slice_label.setText(str(value))
        self.updateSliceViews()

    def onCTOpacityChanged(self, value):
        """Handle CT opacity slider change."""
        self.ct_opacity_label.setText(f"CT Opacity: {value}%")
        if self.show_ct_check.isChecked():
            self.updateSliceViews()

    def showEvent(self, event):
        """Called when the tab becomes visible."""
        super().showEvent(event)
        # Auto-refresh ROI list when tab is shown
        self.refreshROIList()

    def refreshROIList(self):
        """Refresh the list of available ROIs from the main window."""
        if self.main_window is None or not hasattr(self.main_window, 'viewer'):
            return

        # Clear all widgets from layout
        while self.roi_list_layout.count():
            item = self.roi_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.roi_checkboxes.clear()

        # Get ROIs from viewer
        rois = self.main_window.viewer.rois if hasattr(self.main_window.viewer, 'rois') else {}

        if not rois:
            no_roi_label = QLabel("No ROIs available")
            no_roi_label.setStyleSheet("color: #888; font-style: italic;")
            self.roi_list_layout.addWidget(no_roi_label)
            self.roi_checkboxes['_no_roi'] = no_roi_label
            self.roi_list_layout.addStretch()
            return

        # Add checkbox for each ROI, preserving previous selection state
        # By default, all ROIs are selected
        previously_selected = list(self.roi_trajectories.keys()) if self.roi_trajectories else None

        for i, roi_name in enumerate(sorted(rois.keys())):
            checkbox = QCheckBox(roi_name)
            # Check if was previously selected, otherwise check all ROIs by default
            if previously_selected is not None:
                checkbox.setChecked(roi_name in previously_selected)
            else:
                checkbox.setChecked(True)  # Select all ROIs by default
            checkbox.stateChanged.connect(self.onROISelectionChanged)
            self.roi_list_layout.addWidget(checkbox)
            self.roi_checkboxes[roi_name] = checkbox

        self.roi_list_layout.addStretch()

    def onROISelectionChanged(self):
        """Handle ROI selection change - regenerate trajectories for selected ROIs."""
        self.updateVisualization()

    def getSelectedROIs(self):
        """Get list of selected ROI names."""
        selected = []
        for roi_name, checkbox in self.roi_checkboxes.items():
            if roi_name != '_no_roi' and isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                selected.append(roi_name)
        return selected

    def generateTrajectory(self):
        """Generate fiber trajectory for selected ROIs."""
        from acsc.fiber_trajectory import create_fiber_distribution, FiberTrajectory, detect_fiber_centers
        from acsc.analysis import compute_structure_tensor

        # Get selected ROIs
        selected_rois = self.getSelectedROIs()

        if not selected_rois and self.structure_tensor is None:
            QMessageBox.warning(self, "Warning", "No ROIs selected and no structure tensor available.\nPlease select ROIs or compute orientation in Analysis tab first.")
            return

        # Get settings from trajectory_settings
        fiber_diameter = self.trajectory_settings['fiber_diameter']
        volume_fraction = self.trajectory_settings['volume_fraction']
        relax = self.trajectory_settings['relax']
        use_detected_centers = self.trajectory_settings.get('use_detected_centers', False)
        detection_interval = self.trajectory_settings.get('detection_interval', 1)
        max_matching_distance = self.trajectory_settings.get('max_matching_distance', 10.0)

        # Check if detected fiber centers are available when option is enabled
        fiber_detection_result = None
        if use_detected_centers and self.main_window and hasattr(self.main_window, 'viewer'):
            fiber_detection_result = self.main_window.viewer.fiber_detection_result
            if fiber_detection_result is None:
                QMessageBox.warning(self, "Warning",
                    "Image-based tracking is enabled but no fiber detection results available.\n"
                    "Please run 'Detect Fibers' in the Analysis tab first, or disable this option.")
                return

        # Get reference vector based on propagation axis selection
        prop_axis = self.trajectory_settings['propagation_axis']
        if prop_axis == "X":
            prop_axis_idx = 2
        elif prop_axis == "Y":
            prop_axis_idx = 1
        else:
            prop_axis_idx = 0
        if prop_axis_idx == 2:  # X
            reference_vector = [1.0, 0.0, 0.0]
        elif prop_axis_idx == 1:  # Y
            reference_vector = [0.0, 1.0, 0.0]
        else:  # Z (default)
            reference_vector = [0.0, 0.0, 1.0]

        # Clear previous trajectories
        self.roi_trajectories.clear()

        if selected_rois and self.main_window and hasattr(self.main_window, 'viewer'):
            # Generate trajectories for each selected ROI
            rois = self.main_window.viewer.rois
            total_fibers = 0
            total_slices = 0
            all_angles_list = []

            num_rois = len(selected_rois)
            for roi_idx, roi_name in enumerate(selected_rois):
                if roi_name not in rois:
                    continue

                roi_data = rois[roi_name]
                bounds = roi_data.get('bounds')
                if bounds is None:
                    continue

                z_min, z_max, y_min, y_max, x_min, x_max = bounds

                # Get volume data for this ROI
                if self.main_window.current_volume is not None:
                    roi_volume = self.main_window.current_volume[z_min:z_max, y_min:y_max, x_min:x_max]
                    roi_shape = roi_volume.shape

                    # Try to reuse structure tensor from Analysis tab if available
                    global_st = self.main_window.orientation_data.get('structure_tensor')
                    roi_structure_tensor = None

                    if global_st is not None:
                        # Check if ROI bounds are within global structure tensor bounds
                        # structure_tensor shape is (6, Z, Y, X)
                        st_z, st_y, st_x = global_st.shape[1], global_st.shape[2], global_st.shape[3]
                        if z_max <= st_z and y_max <= st_y and x_max <= st_x:
                            # Extract ROI region from global structure tensor
                            self.updateMainStatus(f"[{roi_name}] Extracting structure tensor from analysis...")
                            roi_structure_tensor = global_st[:, z_min:z_max, y_min:y_max, x_min:x_max]

                            # Verify extracted tensor has valid dimensions
                            if roi_structure_tensor.shape[1] == 0 or roi_structure_tensor.shape[2] == 0 or roi_structure_tensor.shape[3] == 0:
                                roi_structure_tensor = None

                    if roi_structure_tensor is None:
                        # Compute structure tensor for this ROI (fallback)
                        self.updateMainStatus(f"[{roi_name}] Computing structure tensor...")
                        noise_scale = self.main_window.noise_scale_slider.value() if hasattr(self.main_window, 'noise_scale_slider') else 10
                        roi_structure_tensor = compute_structure_tensor(roi_volume, noise_scale=noise_scale)

                    # Create fiber trajectory object
                    fiber_traj = FiberTrajectory(
                        status_callback=lambda msg, r=roi_name: self.updateMainStatus(f"[{r}] {msg}")
                    )

                    # Check if we should use detected fiber centers
                    if use_detected_centers and fiber_detection_result is not None:
                        # Get initial fiber centers from detection result (first slice of ROI)
                        all_slices = fiber_detection_result.get('all_slices', {})
                        detection_roi_bounds = fiber_detection_result.get('roi_bounds')

                        # Find the first slice with detected centers within this ROI
                        initial_slice = None
                        initial_centers = None
                        for z in range(z_min, z_max):
                            if z in all_slices:
                                slice_data = all_slices[z]
                                centers = slice_data['centers']
                                if len(centers) > 0:
                                    initial_slice = z
                                    initial_centers = centers.copy()
                                    break

                        if initial_centers is None or len(initial_centers) == 0:
                            self.updateMainStatus(f"[{roi_name}] No detected centers found, using Poisson disk sampling...")
                            # Fall back to Poisson disk sampling
                            fiber_traj.initialize(
                                shape=roi_shape,
                                fiber_diameter=fiber_diameter,
                                fiber_volume_fraction=volume_fraction,
                                scale=1.0,
                                seed=42 + hash(roi_name) % 1000,
                                reference_vector=reference_vector
                            )
                        else:
                            self.updateMainStatus(f"[{roi_name}] Using {len(initial_centers)} detected fiber centers...")

                            # Initialize with detected centers
                            # Centers are in ROI-local coordinates already
                            fiber_traj.bounds = roi_shape

                            # Determine propagation axis
                            fiber_traj.reference_vector = np.array(reference_vector, dtype=np.float32)
                            ref_norm = np.linalg.norm(fiber_traj.reference_vector)
                            if ref_norm > 0:
                                fiber_traj.reference_vector = fiber_traj.reference_vector / ref_norm
                            fiber_traj.propagation_axis = np.argmax(np.abs(fiber_traj.reference_vector))

                            # Set fiber diameter from detection settings
                            diameters = all_slices[initial_slice]['diameters']
                            fiber_traj.fiber_diameter = np.mean(diameters) if len(diameters) > 0 else fiber_diameter

                            # Initialize points
                            fiber_traj.points = initial_centers
                            fiber_traj.trajectories = [(0, initial_centers.copy())]
                            fiber_traj.angles = [np.zeros(len(initial_centers))]
                            fiber_traj.azimuths = [np.zeros(len(initial_centers))]

                            # Initialize per-fiber trajectory data
                            n_fibers = len(initial_centers)
                            fiber_traj.fiber_trajectories = [[(0, initial_centers[i].copy())] for i in range(n_fibers)]
                            fiber_traj.fiber_angles = [[0.0] for _ in range(n_fibers)]
                            fiber_traj.fiber_azimuths = [[0.0] for _ in range(n_fibers)]
                            fiber_traj.active_fibers = np.ones(n_fibers, dtype=bool)

                            # Exclude fibers near the boundary from tracking (same as propagation boundary check)
                            boundary_margin = fiber_traj.fiber_diameter / 2.0
                            prop_axis = fiber_traj.propagation_axis
                            if prop_axis == 2:  # Z-axis
                                dim0_max = roi_shape[1]  # y
                                dim1_max = roi_shape[2]  # x
                            elif prop_axis == 1:  # Y-axis
                                dim0_max = roi_shape[0]  # z
                                dim1_max = roi_shape[2]  # x
                            else:  # X-axis
                                dim0_max = roi_shape[0]  # z
                                dim1_max = roi_shape[1]  # y

                            near_boundary = (
                                (initial_centers[:, 0] < boundary_margin) |
                                (initial_centers[:, 0] > dim1_max - boundary_margin) |
                                (initial_centers[:, 1] < boundary_margin) |
                                (initial_centers[:, 1] > dim0_max - boundary_margin)
                            )
                            fiber_traj.active_fibers = ~near_boundary
                            n_excluded = np.sum(near_boundary)
                            if n_excluded > 0:
                                print(f"[INFO] [{roi_name}] Excluded {n_excluded} fibers near boundary (margin={boundary_margin:.1f}px)")
                    else:
                        # Use Poisson disk sampling (original behavior)
                        self.updateMainStatus(f"[{roi_name}] Creating fiber distribution...")
                        fiber_traj.initialize(
                            shape=roi_shape,
                            fiber_diameter=fiber_diameter,
                            fiber_volume_fraction=volume_fraction,
                            scale=1.0,
                            seed=42 + hash(roi_name) % 1000,
                            reference_vector=reference_vector
                        )

                    # Get resample settings
                    resample_interval = self.trajectory_settings['resample_interval'] if self.trajectory_settings['resample'] else 0

                    # Update status - Propagating trajectories
                    self.updateMainStatus(f"[{roi_name}] Propagating trajectories...")

                    # Propagate through structure tensor
                    use_rk4 = 'RK4' in self.trajectory_settings.get('integration_method', 'Euler')

                    # Use image-based tracking if enabled
                    if use_detected_centers and fiber_detection_result is not None:
                        all_slices = fiber_detection_result.get('all_slices', {})
                        self.updateMainStatus(f"[{roi_name}] Propagating with image-based tracking (interval={detection_interval})...")

                        # Get new fiber options
                        add_new_fibers = self.trajectory_settings.get('add_new_fibers', False)
                        new_fiber_interval = self.trajectory_settings.get('new_fiber_interval', 10)

                        # Use propagate_with_detection method
                        fiber_traj.propagate_with_detection(
                            volume=roi_volume,
                            structure_tensor=roi_structure_tensor,
                            detection_interval=detection_interval,
                            max_matching_distance=max_matching_distance,
                            min_diameter=fiber_diameter * 0.5,
                            max_diameter=fiber_diameter * 2.0,
                            min_peak_distance=int(fiber_diameter * 0.4),
                            relax=relax,
                            relax_iterations=50,
                            stop_at_boundary=True,
                            boundary_margin=0.5,
                            add_new_fibers=add_new_fibers,
                            new_fiber_interval=new_fiber_interval
                        )
                    elif use_rk4:
                        fiber_traj.propagate_rk4(
                            roi_structure_tensor,
                            relax=relax,
                            relax_iterations=50,
                            stop_at_boundary=True,
                            resample_interval=resample_interval
                        )
                    else:
                        fiber_traj.propagate(
                            roi_structure_tensor,
                            relax=relax,
                            relax_iterations=50,
                            stop_at_boundary=True,
                            resample_interval=resample_interval
                        )

                    # Apply trajectory smoothing if enabled
                    if self.trajectory_settings.get('smooth_trajectories', False):
                        smooth_method = self.trajectory_settings.get('smooth_method', 'gaussian')
                        smooth_sigma = self.trajectory_settings.get('smooth_sigma', 1.0)
                        smooth_window = self.trajectory_settings.get('smooth_window', 5)
                        self.updateMainStatus(f"[{roi_name}] Smoothing trajectories...")
                        fiber_traj.smooth_trajectories(
                            method=smooth_method,
                            window_size=smooth_window,
                            sigma=smooth_sigma
                        )

                    # Store with ROI offset information
                    self.roi_trajectories[roi_name] = {
                        'trajectory': fiber_traj,
                        'offset': (z_min, y_min, x_min),
                        'bounds': bounds,
                        'volume': roi_volume,
                        'structure_tensor': roi_structure_tensor
                    }

                    # Collect statistics
                    total_fibers += fiber_traj.get_num_fibers()
                    total_slices = max(total_slices, len(fiber_traj.trajectories))
                    angles = fiber_traj.get_angles()
                    if angles:
                        all_angles_list.extend([np.concatenate(angles)])

            # Update statistics
            if all_angles_list:
                all_angles = np.concatenate(all_angles_list)
                stats_text = f"ROIs: {len(self.roi_trajectories)}\n"
                stats_text += f"Total Fibers: {total_fibers}\n"
                stats_text += f"Angle: {all_angles.mean():.2f}° ± {all_angles.std():.2f}°\n"
                stats_text += f"Range: [{all_angles.min():.2f}°, {all_angles.max():.2f}°]"
                self.stats_label.setText(stats_text)

            # Use the first ROI's trajectory as the main one for backward compatibility
            if self.roi_trajectories:
                first_roi = list(self.roi_trajectories.keys())[0]
                self.fiber_trajectory = self.roi_trajectories[first_roi]['trajectory']
                self.volume_data = self.roi_trajectories[first_roi]['volume']
        else:
            # Fall back to using the global structure tensor
            if self.structure_tensor is None:
                self.updateMainStatus("Ready")
                QMessageBox.warning(self, "Warning", "No structure tensor available.")
                return

            # Update status - Creating fiber distribution
            self.updateMainStatus("Creating fiber distribution...")

            # Create fiber trajectory object
            self.fiber_trajectory = FiberTrajectory(
                status_callback=lambda msg: self.updateMainStatus(msg)
            )

            # Use Poisson disk sampling
            self.fiber_trajectory.initialize(
                shape=self.volume_shape,
                fiber_diameter=fiber_diameter,
                fiber_volume_fraction=volume_fraction,
                scale=1.0,
                seed=42,
                reference_vector=reference_vector
            )

            # Get resample settings
            resample_interval = self.trajectory_settings['resample_interval'] if self.trajectory_settings['resample'] else 0

            # Update status - Propagating trajectories
            self.updateMainStatus("Propagating trajectories...")

            use_rk4 = 'RK4' in self.trajectory_settings.get('integration_method', 'Euler')

            if use_rk4:
                self.fiber_trajectory.propagate_rk4(
                    self.structure_tensor,
                    relax=relax,
                    relax_iterations=50,
                    stop_at_boundary=True,
                    resample_interval=resample_interval
                )
            else:
                self.fiber_trajectory.propagate(
                    self.structure_tensor,
                    relax=relax,
                    relax_iterations=50,
                    stop_at_boundary=True,
                    resample_interval=resample_interval
                )

            # Apply trajectory smoothing if enabled
            if self.trajectory_settings.get('smooth_trajectories', False):
                smooth_method = self.trajectory_settings.get('smooth_method', 'gaussian')
                smooth_sigma = self.trajectory_settings.get('smooth_sigma', 1.0)
                smooth_window = self.trajectory_settings.get('smooth_window', 5)
                self.updateMainStatus("Smoothing trajectories...")
                self.fiber_trajectory.smooth_trajectories(
                    method=smooth_method,
                    window_size=smooth_window,
                    sigma=smooth_sigma
                )

            # Update statistics
            angles = self.fiber_trajectory.get_angles()
            if angles:
                all_angles = np.concatenate(angles)
                stats_text = f"Fibers: {self.fiber_trajectory.get_num_fibers()}\n"
                stats_text += f"Slices: {len(self.fiber_trajectory.trajectories)}\n"
                stats_text += f"Angle: {all_angles.mean():.2f}° ± {all_angles.std():.2f}°\n"
                stats_text += f"Range: [{all_angles.min():.2f}°, {all_angles.max():.2f}°]"
                self.stats_label.setText(stats_text)

        self.export_screenshot_btn.setEnabled(True)
        self.trajectory_histogram_btn.setEnabled(True)
        self.updateVisualization()

        # Update status to complete
        self.updateMainStatus("Ready")

    def setCameraPreset(self, preset_name):
        """Set the 3D camera to a preset view position."""
        if not hasattr(self, 'plotter_3d') or self.plotter_3d is None:
            return

        # PyVista camera positions: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'
        preset_map = {
            "Isometric": 'iso',
            "Front (XY)": 'xy',
            "Back (-XY)": '-xy',
            "Right (YZ)": 'yz',
            "Left (-YZ)": '-yz',
            "Top (XZ)": 'xz',
            "Bottom (-XZ)": '-xz'
        }

        if preset_name in preset_map:
            # Use view_vector for negative directions
            position = preset_map[preset_name]
            if position.startswith('-'):
                # Handle negative views using view_vector
                self.plotter_3d.view_vector(*self._get_view_vector(position))
            else:
                self.plotter_3d.camera_position = position
            self.plotter_3d.reset_camera()
            self.plotter_3d.update()

    def _get_view_vector(self, position):
        """Get view vector and up vector for camera positioning."""
        # Returns (view_direction, viewup)
        vectors = {
            '-xy': ((0, 0, -1), (0, 1, 0)),   # Looking from -Z towards +Z
            '-yz': ((-1, 0, 0), (0, 0, 1)),   # Looking from -X towards +X
            '-xz': ((0, -1, 0), (0, 0, 1)),   # Looking from -Y towards +Y
        }
        return vectors.get(position, ((0, 0, 1), (0, 1, 0)))

    def resetCamera(self):
        """Reset the 3D camera to fit all visible objects."""
        if not hasattr(self, 'plotter_3d') or self.plotter_3d is None:
            return
        self.plotter_3d.reset_camera()
        self.plotter_3d.update()

    def togglePerspective(self, state):
        """Toggle between perspective and parallel (orthographic) projection."""
        if not hasattr(self, 'plotter_3d') or self.plotter_3d is None:
            return
        if state:
            # Perspective projection
            self.plotter_3d.camera.parallel_projection = False
        else:
            # Parallel (orthographic) projection
            self.plotter_3d.camera.parallel_projection = True
        self.plotter_3d.update()

    def updateVisualization(self):
        """Update all viewport visualizations."""
        if self.fiber_trajectory is None:
            return
        self.update3DView()
        self.updateSliceViews()

    def update3DView(self):
        """Update the 3D trajectory view using PyVista."""
        # Clear previous actors
        self.plotter_3d.clear()

        # Fiber Trajectories mode
        if self.fiber_trajectory is None and not self.roi_trajectories:
            return

        cmap = self.colormap_combo.currentText()
        angle_min = self.trajectory_settings['tilt_min']
        angle_max = self.trajectory_settings['tilt_max']
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_by_fiber = self.trajectory_settings.get('color_by_fiber', False)
        color_mode = self.color_mode_combo.currentText()
        use_azimuth = "Azimuth" in color_mode

        # Line width from slider
        line_width = float(self.line_width_slider.value())

        # Saturation range settings (for HSV azimuth mode)
        sat_min = self.trajectory_settings['sat_min']
        sat_max = self.trajectory_settings['sat_max']

        # Colormap for fiber-based coloring
        fiber_cmap = plt.get_cmap('tab20')

        # Build all fiber trajectories as a single PolyData for efficiency
        all_points = []
        all_lines = []
        all_angles_data = []
        all_azimuths_data = []
        all_fiber_indices = []  # track fiber index for color_by_fiber
        point_offset = 0
        global_bounds = None
        global_fiber_idx = 0  # global fiber counter across all ROIs

        # Process all ROI trajectories
        trajectories_to_render = []
        if self.roi_trajectories:
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                fiber_traj = roi_data['trajectory']
                offset = roi_data['offset']  # (z_offset, y_offset, x_offset)
                trajectories_to_render.append((fiber_traj, offset))

                # Update global bounds
                bounds = roi_data['bounds']
                if global_bounds is None:
                    global_bounds = list(bounds)
                else:
                    global_bounds[0] = min(global_bounds[0], bounds[0])
                    global_bounds[1] = max(global_bounds[1], bounds[1])
                    global_bounds[2] = min(global_bounds[2], bounds[2])
                    global_bounds[3] = max(global_bounds[3], bounds[3])
                    global_bounds[4] = min(global_bounds[4], bounds[4])
                    global_bounds[5] = max(global_bounds[5], bounds[5])
        elif self.fiber_trajectory is not None:
            trajectories_to_render.append((self.fiber_trajectory, (0, 0, 0)))
            bounds = self.fiber_trajectory.bounds
            if bounds:
                global_bounds = [0, bounds[0], 0, bounds[1], 0, bounds[2]]

        for fiber_traj, offset in trajectories_to_render:
            z_offset, y_offset, x_offset = offset

            # Use per-fiber trajectories if available (variable length)
            if hasattr(fiber_traj, 'fiber_trajectories') and fiber_traj.fiber_trajectories:
                fiber_trajectories = fiber_traj.fiber_trajectories
                fiber_angles_list = fiber_traj.fiber_angles if hasattr(fiber_traj, 'fiber_angles') else None
                fiber_azimuths_list = getattr(fiber_traj, 'fiber_azimuths', None)

                for fiber_idx, traj in enumerate(fiber_trajectories):
                    if len(traj) < 2:
                        continue

                    fiber_points = []
                    fiber_angles = []
                    fiber_azimuths = []

                    for i, (z, point) in enumerate(traj):
                        x = point[0] + x_offset
                        y = point[1] + y_offset
                        z_global = z + z_offset
                        fiber_points.append([x, y, z_global])

                        if fiber_angles_list and fiber_idx < len(fiber_angles_list) and i < len(fiber_angles_list[fiber_idx]):
                            fiber_angles.append(fiber_angles_list[fiber_idx][i])
                        if fiber_azimuths_list and fiber_idx < len(fiber_azimuths_list) and i < len(fiber_azimuths_list[fiber_idx]):
                            fiber_azimuths.append(fiber_azimuths_list[fiber_idx][i])

                    n_pts = len(fiber_points)
                    if n_pts < 2:
                        continue

                    all_points.extend(fiber_points)
                    all_angles_data.extend(fiber_angles)
                    all_azimuths_data.extend(fiber_azimuths)
                    all_fiber_indices.extend([global_fiber_idx] * n_pts)  # same fiber index for all points
                    global_fiber_idx += 1

                    # Line connectivity: [n_points, idx0, idx1, idx2, ...]
                    line = [n_pts] + list(range(point_offset, point_offset + n_pts))
                    all_lines.extend(line)
                    point_offset += n_pts
            else:
                # Fallback to old slice-based format
                trajectories = fiber_traj.trajectories
                angles = fiber_traj.angles
                azimuths = getattr(fiber_traj, 'azimuths', angles)

                if len(trajectories) < 2:
                    continue

                n_fibers = len(trajectories[0][1])

                for fiber_idx in range(n_fibers):
                    fiber_points = []
                    fiber_angles = []
                    fiber_azimuths = []

                    for slice_idx, (z, slice_points) in enumerate(trajectories):
                        x = slice_points[fiber_idx, 0] + x_offset
                        y = slice_points[fiber_idx, 1] + y_offset
                        z_global = z + z_offset
                        fiber_points.append([x, y, z_global])

                        if slice_idx < len(angles):
                            fiber_angles.append(angles[slice_idx][fiber_idx])
                        if slice_idx < len(azimuths):
                            fiber_azimuths.append(azimuths[slice_idx][fiber_idx])

                    n_pts = len(fiber_points)
                    if n_pts < 2:
                        continue

                    all_points.extend(fiber_points)
                    all_angles_data.extend(fiber_angles)
                    all_azimuths_data.extend(fiber_azimuths)
                    all_fiber_indices.extend([global_fiber_idx] * n_pts)  # same fiber index for all points
                    global_fiber_idx += 1

                    # Line connectivity: [n_points, idx0, idx1, idx2, ...]
                    line = [n_pts] + list(range(point_offset, point_offset + n_pts))
                    all_lines.extend(line)
                    point_offset += n_pts

        if not all_points:
            return

        # Create PolyData
        points_array = np.array(all_points)
        lines_array = np.array(all_lines)
        poly = pv.PolyData(points_array, lines=lines_array)

        if color_by_fiber and all_fiber_indices:
            # Color each fiber with a unique color from tab20 colormap
            fiber_indices_array = np.array(all_fiber_indices)
            rgb_colors = np.zeros((len(fiber_indices_array), 3), dtype=np.uint8)
            for i, idx in enumerate(fiber_indices_array):
                r, g, b, _ = fiber_cmap(idx % 20 / 20.0)
                rgb_colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
            poly['RGB'] = rgb_colors
            self.plotter_3d.add_mesh(
                poly,
                scalars='RGB',
                rgb=True,
                line_width=line_width,
                render_lines_as_tubes=True
            )
        elif color_by_angle:
            if use_azimuth and all_azimuths_data:
                # Use HSV coloring for azimuth with saturation based on tilt angle
                azimuth_array = np.array(all_azimuths_data)
                angles_array = np.array(all_angles_data) if all_angles_data else None
                # Convert azimuth (0-360) to RGB using HSV
                # Saturation is based on tilt angle with configurable range
                rgb_colors = np.zeros((len(azimuth_array), 3), dtype=np.uint8)
                for i, az in enumerate(azimuth_array):
                    # Convert numpy azimuth to display coordinates
                    # numpy: 90° = +Y (down in display), 270° = -Y (up in display)
                    # display: 90° = up, 270° = down
                    display_az = (360 - az) % 360
                    hue = display_az / 360.0
                    if angles_array is not None and i < len(angles_array):
                        # Map tilt angle from [sat_min, sat_max] to [0, 1] saturation
                        saturation = np.clip((angles_array[i] - sat_min) / (sat_max - sat_min + 1e-6), 0.0, 1.0)
                    else:
                        saturation = 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 1.0)
                    rgb_colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
                poly['RGB'] = rgb_colors
                self.plotter_3d.add_mesh(
                    poly,
                    scalars='RGB',
                    rgb=True,
                    line_width=line_width,
                    render_lines_as_tubes=True
                )
            elif all_angles_data:
                poly['angle'] = np.array(all_angles_data)
                self.plotter_3d.add_mesh(
                    poly,
                    scalars='angle',
                    cmap=cmap,
                    clim=(angle_min, angle_max),
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    scalar_bar_args={'title': 'Angle (°)', 'n_labels': 5}
                )
            else:
                self.plotter_3d.add_mesh(
                    poly,
                    color='blue',
                    line_width=line_width,
                    render_lines_as_tubes=True
                )
        else:
            self.plotter_3d.add_mesh(
                poly,
                color='blue',
                line_width=line_width,
                render_lines_as_tubes=True
            )

        # Add domain boundary boxes for each ROI
        if self.roi_trajectories:
            roi_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
            for i, (roi_name, roi_data) in enumerate(self.roi_trajectories.items()):
                bounds = roi_data['bounds']
                z_min, z_max, y_min, y_max, x_min, x_max = bounds
                box = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
                color = roi_colors[i % len(roi_colors)]
                self.plotter_3d.add_mesh(box, style='wireframe', color=color, line_width=2, label=roi_name)
        elif global_bounds is not None:
            box = pv.Box(bounds=(global_bounds[4], global_bounds[5],
                                 global_bounds[2], global_bounds[3],
                                 global_bounds[0], global_bounds[1]))
            self.plotter_3d.add_mesh(box, style='wireframe', color='gray', line_width=1)

        self.plotter_3d.add_axes()
        self.plotter_3d.reset_camera()
        self.plotter_3d.update()

        # Show/hide color wheel legend based on azimuth mode
        self._show_color_wheel_legend(use_azimuth and color_by_angle)

    def updateSliceViews(self):
        """Update the slice views with trajectory overlay (optimized)."""
        from matplotlib.collections import EllipseCollection

        if self.fiber_trajectory is None and not self.roi_trajectories:
            return

        try:
            self._updateSliceViewsImpl()
        except Exception as e:
            import traceback
            print(f"[ERROR] updateSliceViews failed: {e}")
            traceback.print_exc()

    def _updateSliceViewsImpl(self):
        """Internal implementation of updateSliceViews."""
        from matplotlib.collections import EllipseCollection

        cmap_name = self.colormap_combo.currentText()
        angle_min = self.trajectory_settings['tilt_min']
        angle_max = self.trajectory_settings['tilt_max']
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_by_fiber = self.trajectory_settings.get('color_by_fiber', False)
        color_mode = self.color_mode_combo.currentText()
        use_azimuth = "Azimuth" in color_mode

        # Saturation range settings (for HSV azimuth mode)
        sat_min = self.trajectory_settings['sat_min']
        sat_max = self.trajectory_settings['sat_max']

        # CT overlay settings
        show_ct = self.show_ct_check.isChecked()
        ct_alpha = self.ct_opacity_slider.value() / 100.0

        # Fiber diameter circle settings
        show_fiber_diameter = self.trajectory_settings['show_fiber_diameter']

        # Get colormap for tilt angle mode
        cmap = plt.get_cmap(cmap_name)

        # Colormap for fiber-based coloring (use tab20 for distinct colors)
        fiber_cmap = plt.get_cmap('tab20')

        # Vectorized color computation for tilt angles
        def angles_to_colors(angles_arr, fiber_indices=None):
            if color_by_fiber and fiber_indices is not None:
                # Each fiber gets a unique color based on its index
                n_fibers = len(fiber_indices)
                colors = np.zeros((n_fibers, 4))
                for i, idx in enumerate(fiber_indices):
                    colors[i] = fiber_cmap(idx % 20 / 20.0)
                return colors
            if not color_by_angle:
                return np.full((len(angles_arr), 4), [0, 0, 1, 1])  # blue
            norm_angles = np.clip((angles_arr - angle_min) / (angle_max - angle_min + 1e-6), 0, 1)
            return cmap(norm_angles)

        # Vectorized color computation for azimuth (HSV)
        def azimuths_to_colors(azimuths_arr, tilts_arr, fiber_indices=None):
            if color_by_fiber and fiber_indices is not None:
                # Each fiber gets a unique color based on its index
                n_fibers = len(fiber_indices)
                colors = np.zeros((n_fibers, 3))
                for i, idx in enumerate(fiber_indices):
                    colors[i] = fiber_cmap(idx % 20 / 20.0)[:3]
                return colors
            if not color_by_angle:
                return np.full((len(azimuths_arr), 3), [0, 0, 1])  # blue
            # Convert numpy azimuth to display coordinates
            display_azimuths = (360 - azimuths_arr) % 360
            hues = display_azimuths / 360.0
            # Map tilt angle from [sat_min, sat_max] to [0, 1] saturation
            saturations = np.clip((tilts_arr - sat_min) / (sat_max - sat_min + 1e-6), 0.0, 1.0)
            # HSV to RGB conversion (vectorized)
            colors = np.zeros((len(azimuths_arr), 3))
            for i in range(len(azimuths_arr)):
                colors[i] = colorsys.hsv_to_rgb(hues[i], saturations[i], 1.0)
            return colors

        # Get global bounds from main window volume if available
        global_volume = None
        if self.main_window and hasattr(self.main_window, 'current_volume') and self.main_window.current_volume is not None:
            global_volume = self.main_window.current_volume

        z_pos = self.current_slice['z']
        y_pos = self.current_slice['y']
        x_pos = self.current_slice['x']

        # Prepare axes
        canvas_xy = self.viewport_frames[1]['canvas']
        ax_xy = canvas_xy.axes
        ax_xy.clear()
        ax_xy.set_facecolor('#2a2a2a')
        ax_xy.set_title(f'XY Slice at Z={z_pos}', color='white', fontsize=10)
        ax_xy.set_xlabel('X', color='white', fontsize=8)
        ax_xy.set_ylabel('Y', color='white', fontsize=8)
        ax_xy.tick_params(colors='white', labelsize=8)

        canvas_xz = self.viewport_frames[2]['canvas']
        ax_xz = canvas_xz.axes
        ax_xz.clear()
        ax_xz.set_facecolor('#2a2a2a')
        ax_xz.set_title(f'XZ Slice at Y={y_pos}', color='white', fontsize=10)
        ax_xz.set_xlabel('X', color='white', fontsize=8)
        ax_xz.set_ylabel('Z', color='white', fontsize=8)
        ax_xz.tick_params(colors='white', labelsize=8)

        canvas_yz = self.viewport_frames[3]['canvas']
        ax_yz = canvas_yz.axes
        ax_yz.clear()
        ax_yz.set_facecolor('#2a2a2a')
        ax_yz.set_title(f'YZ Slice at X={x_pos}', color='white', fontsize=10)
        ax_yz.set_xlabel('Y', color='white', fontsize=8)
        ax_yz.set_ylabel('Z', color='white', fontsize=8)
        ax_yz.tick_params(colors='white', labelsize=8)

        # Show CT images from global volume, volume_data, or ROI volumes
        if show_ct:
            # Try to get CT volume from various sources
            ct_volume = None
            if global_volume is not None:
                ct_volume = global_volume
            elif self.volume_data is not None:
                ct_volume = self.volume_data

            if ct_volume is not None:
                # Use global/stored volume
                if z_pos < ct_volume.shape[0]:
                    ct_slice = ct_volume[z_pos, :, :]
                    ax_xy.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                extent=[0, ct_slice.shape[1], 0, ct_slice.shape[0]])
                if y_pos < ct_volume.shape[1]:
                    ct_slice = ct_volume[:, y_pos, :]
                    ax_xz.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                extent=[0, ct_slice.shape[1], 0, ct_slice.shape[0]], aspect='auto')
                if x_pos < ct_volume.shape[2]:
                    ct_slice = ct_volume[:, :, x_pos]
                    ax_yz.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                extent=[0, ct_slice.shape[1], 0, ct_slice.shape[0]], aspect='auto')
            elif self.roi_trajectories:
                # Use ROI volumes when global volume is not available
                for roi_name, roi_data in self.roi_trajectories.items():
                    if not isinstance(roi_data, dict):
                        continue
                    roi_volume = roi_data.get('volume', None)
                    if roi_volume is None:
                        continue
                    offset = roi_data['offset']  # (z_offset, y_offset, x_offset)
                    z_offset, y_offset, x_offset = offset

                    # XY slice - check if z_pos is within this ROI
                    roi_z_pos = z_pos - z_offset
                    if 0 <= roi_z_pos < roi_volume.shape[0]:
                        ct_slice = roi_volume[roi_z_pos, :, :]
                        ax_xy.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                    extent=[x_offset, x_offset + ct_slice.shape[1],
                                           y_offset, y_offset + ct_slice.shape[0]])

                    # XZ slice - check if y_pos is within this ROI
                    roi_y_pos = y_pos - y_offset
                    if 0 <= roi_y_pos < roi_volume.shape[1]:
                        ct_slice = roi_volume[:, roi_y_pos, :]
                        ax_xz.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                    extent=[x_offset, x_offset + ct_slice.shape[1],
                                           z_offset, z_offset + ct_slice.shape[0]], aspect='auto')

                    # YZ slice - check if x_pos is within this ROI
                    roi_x_pos = x_pos - x_offset
                    if 0 <= roi_x_pos < roi_volume.shape[2]:
                        ct_slice = roi_volume[:, :, roi_x_pos]
                        ax_yz.imshow(ct_slice, cmap='gray', alpha=ct_alpha, origin='lower',
                                    extent=[y_offset, y_offset + ct_slice.shape[1],
                                           z_offset, z_offset + ct_slice.shape[0]], aspect='auto')

        tolerance = 5.0  # pixels

        # Helper to draw circles using EllipseCollection (batch rendering)
        def draw_circles_batch(ax, centers, radius, transform=None):
            if len(centers) == 0:
                return
            widths = np.full(len(centers), radius * 2)
            heights = np.full(len(centers), radius * 2)
            offsets = np.array(centers)
            ec = EllipseCollection(
                widths, heights, np.zeros(len(centers)),
                units='x', offsets=offsets, transOffset=ax.transData,
                facecolors='none', edgecolors='red', linewidths=0.5
            )
            ax.add_collection(ec)

        # Render trajectories for all ROIs
        if self.roi_trajectories:
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                fiber_traj = roi_data['trajectory']
                offset = roi_data['offset']  # (z_offset, y_offset, x_offset)
                z_offset, y_offset, x_offset = offset
                trajectories = fiber_traj.trajectories
                angles = fiber_traj.angles if fiber_traj.angles else []
                azimuths = getattr(fiber_traj, 'azimuths', None)
                if azimuths is None or len(azimuths) == 0:
                    azimuths = angles  # fallback to angles if azimuths not available

                # Get ROI bounds for range checking
                bounds = roi_data.get('bounds', None)
                if bounds:
                    roi_z_min, roi_z_max, roi_y_min, roi_y_max, roi_x_min, roi_x_max = bounds
                    roi_height = roi_y_max - roi_y_min
                    roi_width = roi_x_max - roi_x_min
                else:
                    roi_height = roi_width = 1000  # fallback

                # Get propagation axis and fiber diameter for circle rendering
                prop_axis = getattr(fiber_traj, 'propagation_axis', 2)
                fiber_diameter = getattr(fiber_traj, 'fiber_diameter', 7.0)
                radius = fiber_diameter / 2.0

                # XY Slice - find trajectory slice at z_pos (relative to ROI)
                roi_z_pos = z_pos - z_offset
                if 0 <= roi_z_pos < len(trajectories):
                    z, points = trajectories[roi_z_pos]
                    n_points = len(points)
                    # Ensure angles array has correct length (may differ when new fibers are added dynamically)
                    if roi_z_pos < len(angles) and len(angles[roi_z_pos]) == n_points:
                        slice_angles = np.array(angles[roi_z_pos])
                    else:
                        slice_angles = np.zeros(n_points)
                    fiber_indices = np.arange(n_points)  # fiber index for each point
                    # Ensure azimuths array has correct length for azimuth mode
                    if use_azimuth:
                        if roi_z_pos < len(azimuths) and len(azimuths[roi_z_pos]) == n_points:
                            slice_azimuths = np.array(azimuths[roi_z_pos])
                        else:
                            slice_azimuths = np.zeros(n_points)
                        colors = azimuths_to_colors(slice_azimuths, slice_angles, fiber_indices)
                    else:
                        colors = angles_to_colors(slice_angles, fiber_indices)
                    ax_xy.scatter(points[:, 0] + x_offset, points[:, 1] + y_offset, c=colors, s=4, alpha=0.8)
                    if show_fiber_diameter and prop_axis == 2:
                        centers = [(pt[0] + x_offset, pt[1] + y_offset) for pt in points]
                        draw_circles_batch(ax_xy, centers, radius)

                # XZ Slice - collect all points near y_pos in one pass
                roi_y_pos = y_pos - y_offset
                xz_x_all, xz_z_all, xz_angles_all, xz_azimuths_all = [], [], [], []
                xz_fiber_indices_all = []  # track fiber indices for color_by_fiber
                xz_circle_centers = []

                # Only process if y_pos is within ROI range
                if 0 <= roi_y_pos < roi_height + tolerance:
                    for slice_idx, (z, points) in enumerate(trajectories):
                        mask = np.abs(points[:, 1] - roi_y_pos) < tolerance
                        if np.any(mask):
                            matched_indices = np.where(mask)[0]
                            n_matched = len(matched_indices)
                            xz_x_all.extend(points[mask, 0] + x_offset)
                            xz_z_all.extend(np.full(n_matched, slice_idx + z_offset))
                            # Ensure angles array has correct length
                            if slice_idx < len(angles) and len(angles[slice_idx]) == len(points):
                                xz_angles_all.extend(np.array(angles[slice_idx])[mask])
                            else:
                                xz_angles_all.extend(np.zeros(n_matched))
                            xz_fiber_indices_all.extend(matched_indices)
                            # Ensure azimuths array has correct length for azimuth mode
                            if use_azimuth:
                                if slice_idx < len(azimuths) and len(azimuths[slice_idx]) == len(points):
                                    xz_azimuths_all.extend(np.array(azimuths[slice_idx])[mask])
                                else:
                                    xz_azimuths_all.extend(np.zeros(n_matched))
                            if show_fiber_diameter and prop_axis == 1:
                                xz_circle_centers.extend([(pt[0] + x_offset, slice_idx + z_offset) for pt in points[mask]])

                if xz_x_all:
                    xz_x_all = np.array(xz_x_all)
                    xz_z_all = np.array(xz_z_all)
                    xz_angles_all = np.array(xz_angles_all)
                    xz_fiber_indices_all = np.array(xz_fiber_indices_all)
                    # Ensure all arrays have the same length
                    n_points = len(xz_x_all)
                    if use_azimuth and len(xz_azimuths_all) == n_points:
                        xz_azimuths_all = np.array(xz_azimuths_all)
                        colors = azimuths_to_colors(xz_azimuths_all, xz_angles_all, xz_fiber_indices_all)
                    else:
                        colors = angles_to_colors(xz_angles_all, xz_fiber_indices_all)
                    ax_xz.scatter(xz_x_all, xz_z_all, c=colors, s=2, alpha=0.6)
                    if xz_circle_centers:
                        draw_circles_batch(ax_xz, xz_circle_centers, radius)

                # YZ Slice - collect all points near x_pos in one pass
                roi_x_pos = x_pos - x_offset
                yz_y_all, yz_z_all, yz_angles_all, yz_azimuths_all = [], [], [], []
                yz_fiber_indices_all = []  # track fiber indices for color_by_fiber
                yz_circle_centers = []
                # Only process if x_pos is within ROI range
                if 0 <= roi_x_pos < roi_width + tolerance:
                    for slice_idx, (z, points) in enumerate(trajectories):
                        mask = np.abs(points[:, 0] - roi_x_pos) < tolerance
                        if np.any(mask):
                            matched_indices = np.where(mask)[0]
                            n_matched = len(matched_indices)
                            yz_y_all.extend(points[mask, 1] + y_offset)
                            yz_z_all.extend(np.full(n_matched, slice_idx + z_offset))
                            # Ensure angles array has correct length
                            if slice_idx < len(angles) and len(angles[slice_idx]) == len(points):
                                yz_angles_all.extend(np.array(angles[slice_idx])[mask])
                            else:
                                yz_angles_all.extend(np.zeros(n_matched))
                            yz_fiber_indices_all.extend(matched_indices)
                            # Ensure azimuths array has correct length for azimuth mode
                            if use_azimuth:
                                if slice_idx < len(azimuths) and len(azimuths[slice_idx]) == len(points):
                                    yz_azimuths_all.extend(np.array(azimuths[slice_idx])[mask])
                                else:
                                    yz_azimuths_all.extend(np.zeros(n_matched))
                            if show_fiber_diameter and prop_axis == 0:
                                yz_circle_centers.extend([(pt[1] + y_offset, slice_idx + z_offset) for pt in points[mask]])

                if yz_y_all:
                    yz_y_all = np.array(yz_y_all)
                    yz_z_all = np.array(yz_z_all)
                    yz_angles_all = np.array(yz_angles_all)
                    yz_fiber_indices_all = np.array(yz_fiber_indices_all)
                    # Ensure all arrays have the same length
                    n_points = len(yz_y_all)
                    if use_azimuth and len(yz_azimuths_all) == n_points:
                        yz_azimuths_all = np.array(yz_azimuths_all)
                        colors = azimuths_to_colors(yz_azimuths_all, yz_angles_all, yz_fiber_indices_all)
                    else:
                        colors = angles_to_colors(yz_angles_all, yz_fiber_indices_all)
                    ax_yz.scatter(yz_y_all, yz_z_all, c=colors, s=2, alpha=0.6)
                    if yz_circle_centers:
                        draw_circles_batch(ax_yz, yz_circle_centers, radius)

        elif self.fiber_trajectory is not None:
            # Fall back to single trajectory rendering
            trajectories = self.fiber_trajectory.trajectories
            angles = self.fiber_trajectory.angles if self.fiber_trajectory.angles else []
            azimuths = getattr(self.fiber_trajectory, 'azimuths', None)
            if azimuths is None or len(azimuths) == 0:
                azimuths = angles  # fallback to angles if azimuths not available

            prop_axis = getattr(self.fiber_trajectory, 'propagation_axis', 2)
            fiber_diameter = getattr(self.fiber_trajectory, 'fiber_diameter', 7.0)
            radius = fiber_diameter / 2.0

            if z_pos < len(trajectories):
                z, points = trajectories[z_pos]
                n_points = len(points)
                # Ensure angles array has correct length
                if z_pos < len(angles) and len(angles[z_pos]) == n_points:
                    slice_angles = np.array(angles[z_pos])
                else:
                    slice_angles = np.zeros(n_points)
                fiber_indices = np.arange(n_points)  # fiber index for each point
                # Ensure azimuths array has correct length for azimuth mode
                if use_azimuth:
                    if z_pos < len(azimuths) and len(azimuths[z_pos]) == n_points:
                        slice_azimuths = np.array(azimuths[z_pos])
                    else:
                        slice_azimuths = np.zeros(n_points)
                    colors = azimuths_to_colors(slice_azimuths, slice_angles, fiber_indices)
                else:
                    colors = angles_to_colors(slice_angles, fiber_indices)
                ax_xy.scatter(points[:, 0], points[:, 1], c=colors, s=4, alpha=0.8)
                if show_fiber_diameter and prop_axis == 2:
                    centers = [(pt[0], pt[1]) for pt in points]
                    draw_circles_batch(ax_xy, centers, radius)

            # XZ Slice - batch collection
            xz_x_all, xz_z_all, xz_angles_all, xz_azimuths_all = [], [], [], []
            xz_fiber_indices_all = []
            xz_circle_centers = []
            for slice_idx, (z, points) in enumerate(trajectories):
                mask = np.abs(points[:, 1] - y_pos) < tolerance
                if np.any(mask):
                    matched_indices = np.where(mask)[0]
                    n_matched = len(matched_indices)
                    xz_x_all.extend(points[mask, 0])
                    xz_z_all.extend(np.full(n_matched, slice_idx))
                    # Ensure angles array has correct length
                    if slice_idx < len(angles) and len(angles[slice_idx]) == len(points):
                        xz_angles_all.extend(np.array(angles[slice_idx])[mask])
                    else:
                        xz_angles_all.extend(np.zeros(n_matched))
                    xz_fiber_indices_all.extend(matched_indices)
                    # Ensure azimuths array has correct length for azimuth mode
                    if use_azimuth:
                        if slice_idx < len(azimuths) and len(azimuths[slice_idx]) == len(points):
                            xz_azimuths_all.extend(np.array(azimuths[slice_idx])[mask])
                        else:
                            xz_azimuths_all.extend(np.zeros(n_matched))
                    if show_fiber_diameter and prop_axis == 1:
                        xz_circle_centers.extend([(pt[0], slice_idx) for pt in points[mask]])

            if xz_x_all:
                xz_x_all = np.array(xz_x_all)
                xz_z_all = np.array(xz_z_all)
                xz_angles_all = np.array(xz_angles_all)
                xz_fiber_indices_all = np.array(xz_fiber_indices_all)
                # Ensure all arrays have the same length
                n_points = len(xz_x_all)
                if use_azimuth and len(xz_azimuths_all) == n_points:
                    xz_azimuths_all = np.array(xz_azimuths_all)
                    colors = azimuths_to_colors(xz_azimuths_all, xz_angles_all, xz_fiber_indices_all)
                else:
                    colors = angles_to_colors(xz_angles_all, xz_fiber_indices_all)
                ax_xz.scatter(xz_x_all, xz_z_all, c=colors, s=2, alpha=0.6)
                if xz_circle_centers:
                    draw_circles_batch(ax_xz, xz_circle_centers, radius)

            # YZ Slice - batch collection
            yz_y_all, yz_z_all, yz_angles_all, yz_azimuths_all = [], [], [], []
            yz_fiber_indices_all = []
            yz_circle_centers = []
            for slice_idx, (z, points) in enumerate(trajectories):
                mask = np.abs(points[:, 0] - x_pos) < tolerance
                if np.any(mask):
                    matched_indices = np.where(mask)[0]
                    n_matched = len(matched_indices)
                    yz_y_all.extend(points[mask, 1])
                    yz_z_all.extend(np.full(n_matched, slice_idx))
                    # Ensure angles array has correct length
                    if slice_idx < len(angles) and len(angles[slice_idx]) == len(points):
                        yz_angles_all.extend(np.array(angles[slice_idx])[mask])
                    else:
                        yz_angles_all.extend(np.zeros(n_matched))
                    yz_fiber_indices_all.extend(matched_indices)
                    # Ensure azimuths array has correct length for azimuth mode
                    if use_azimuth:
                        if slice_idx < len(azimuths) and len(azimuths[slice_idx]) == len(points):
                            yz_azimuths_all.extend(np.array(azimuths[slice_idx])[mask])
                        else:
                            yz_azimuths_all.extend(np.zeros(n_matched))
                    if show_fiber_diameter and prop_axis == 0:
                        yz_circle_centers.extend([(pt[1], slice_idx) for pt in points[mask]])

            if yz_y_all:
                yz_y_all = np.array(yz_y_all)
                yz_z_all = np.array(yz_z_all)
                yz_angles_all = np.array(yz_angles_all)
                yz_fiber_indices_all = np.array(yz_fiber_indices_all)
                # Ensure all arrays have the same length
                n_points = len(yz_y_all)
                if use_azimuth and len(yz_azimuths_all) == n_points:
                    yz_azimuths_all = np.array(yz_azimuths_all)
                    colors = azimuths_to_colors(yz_azimuths_all, yz_angles_all, yz_fiber_indices_all)
                else:
                    colors = angles_to_colors(yz_angles_all, yz_fiber_indices_all)
                ax_yz.scatter(yz_y_all, yz_z_all, c=colors, s=2, alpha=0.6)
                if yz_circle_centers:
                    draw_circles_batch(ax_yz, yz_circle_centers, radius)

        # Set axis limits based on global volume or ROI bounds
        if global_volume is not None:
            ax_xy.set_xlim(0, global_volume.shape[2])
            ax_xy.set_ylim(0, global_volume.shape[1])
            ax_xz.set_xlim(0, global_volume.shape[2])
            ax_xz.set_ylim(0, global_volume.shape[0])
            ax_yz.set_xlim(0, global_volume.shape[1])
            ax_yz.set_ylim(0, global_volume.shape[0])
        elif self.volume_shape is not None:
            # Use stored volume shape from structure tensor
            ax_xy.set_xlim(0, self.volume_shape[2])
            ax_xy.set_ylim(0, self.volume_shape[1])
            ax_xz.set_xlim(0, self.volume_shape[2])
            ax_xz.set_ylim(0, self.volume_shape[0])
            ax_yz.set_xlim(0, self.volume_shape[1])
            ax_yz.set_ylim(0, self.volume_shape[0])
        elif self.roi_trajectories:
            # Compute bounds from ROI trajectories
            x_max, y_max, z_max = 0, 0, 0
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                bounds = roi_data.get('bounds', None)
                if bounds:
                    roi_z_min, roi_z_max, roi_y_min, roi_y_max, roi_x_min, roi_x_max = bounds
                    x_max = max(x_max, roi_x_max)
                    y_max = max(y_max, roi_y_max)
                    z_max = max(z_max, roi_z_max)
            if x_max > 0 and y_max > 0 and z_max > 0:
                ax_xy.set_xlim(0, x_max)
                ax_xy.set_ylim(0, y_max)
                ax_xz.set_xlim(0, x_max)
                ax_xz.set_ylim(0, z_max)
                ax_yz.set_xlim(0, y_max)
                ax_yz.set_ylim(0, z_max)

        ax_xy.set_aspect('equal')
        ax_xz.set_aspect('equal')
        ax_yz.set_aspect('equal')

        # Force redraw of all canvases
        canvas_xy.draw_idle()
        canvas_xz.draw_idle()
        canvas_yz.draw_idle()
        canvas_xy.flush_events()
        canvas_xz.flush_events()
        canvas_yz.flush_events()

    def openSettingsDialog(self):
        """Open fiber trajectory settings dialog."""
        dialog = FiberTrajectorySettingsDialog(self, self.trajectory_settings)
        if dialog.exec() == QDialog.Accepted:
            self.trajectory_settings = dialog.getSettings()
            # Update slice views if show_fiber_diameter changed
            self.updateSliceViews()

    def openTrajectoryHistogramDialog(self):
        """Open fiber trajectory angle histogram dialog."""
        dialog = TrajectoryHistogramDialog(self)
        if dialog.exec() == QDialog.Accepted:
            config = dialog.getConfiguration()
            self.showTrajectoryHistogramPanel(config)

    def showTrajectoryHistogramPanel(self, config):
        """Show the trajectory histogram panel with the given configuration."""
        if not self.main_window:
            QMessageBox.warning(self, "Error", "Main window not found.")
            return

        # Collect trajectory angle data
        angle_data = self._collectTrajectoryAngles(config)

        # Check if we have any data
        has_tilt = angle_data.get('tilt') is not None and len(angle_data['tilt']) > 0
        has_azimuth = angle_data.get('azimuth') is not None and len(angle_data['azimuth']) > 0

        if not has_tilt and not has_azimuth:
            QMessageBox.warning(self, "No Data", "No trajectory angle data available.\n\nPlease generate fiber trajectory first.")
            return

        # Show histogram panel in Modelling tab (right of left panel)
        self.histogram_panel.setVisible(True)
        self.content_splitter.setSizes([200, 350, 600])  # Adjust splitter sizes
        self.histogram_panel.plotTrajectoryHistogram(config, angle_data)

    def _collectTrajectoryAngles(self, config):
        """Collect trajectory angle data from all ROIs or single trajectory."""
        angle_data = {
            'tilt': [],
            'azimuth': [],
            'xz_projection': [],
            'yz_projection': []
        }

        collected = False

        def collect_from_trajectory(traj):
            """Helper to collect angles from a trajectory object."""
            nonlocal collected
            if not traj:
                return

            # Get tilt angles
            if hasattr(traj, 'angles') and traj.angles:
                for slice_angles in traj.angles:
                    if slice_angles is not None:
                        arr = np.array(slice_angles).flatten()
                        angle_data['tilt'].extend(arr.tolist())
                        collected = True

            # Get azimuth angles
            if hasattr(traj, 'azimuths') and traj.azimuths:
                for slice_azimuths in traj.azimuths:
                    if slice_azimuths is not None:
                        arr = np.array(slice_azimuths).flatten()
                        angle_data['azimuth'].extend(arr.tolist())

            # Calculate XZ and YZ projection angles from tilt and azimuth
            if hasattr(traj, 'angles') and traj.angles and hasattr(traj, 'azimuths') and traj.azimuths:
                for slice_idx in range(len(traj.angles)):
                    slice_tilts = traj.angles[slice_idx]
                    slice_azimuths = traj.azimuths[slice_idx] if slice_idx < len(traj.azimuths) else None

                    if slice_tilts is not None and slice_azimuths is not None:
                        tilts = np.array(slice_tilts).flatten()
                        azimuths = np.array(slice_azimuths).flatten()

                        # Ensure same length
                        min_len = min(len(tilts), len(azimuths))
                        tilts = tilts[:min_len]
                        azimuths = azimuths[:min_len]

                        # Convert to radians
                        tilt_rad = np.radians(tilts)
                        azimuth_rad = np.radians(azimuths)

                        # XZ projection angle: arctan(tan(tilt) * cos(azimuth))
                        # This is the angle seen when looking along Y axis
                        xz_proj = np.degrees(np.arctan(np.tan(tilt_rad) * np.cos(azimuth_rad)))
                        angle_data['xz_projection'].extend(xz_proj.tolist())

                        # YZ projection angle: arctan(tan(tilt) * sin(azimuth))
                        # This is the angle seen when looking along X axis
                        yz_proj = np.degrees(np.arctan(np.tan(tilt_rad) * np.sin(azimuth_rad)))
                        angle_data['yz_projection'].extend(yz_proj.tolist())

        # Collect from ROI trajectories if specified
        if config.get('rois') and self.roi_trajectories:
            for roi_name in config['rois']:
                if roi_name in self.roi_trajectories:
                    roi_data = self.roi_trajectories[roi_name]
                    traj = roi_data.get('trajectory')
                    collect_from_trajectory(traj)

        # If no specific ROIs selected but ROI trajectories exist, use all of them
        if not collected and not config.get('use_single_trajectory') and self.roi_trajectories:
            for roi_name, roi_data in self.roi_trajectories.items():
                traj = roi_data.get('trajectory')
                collect_from_trajectory(traj)

        # Use single trajectory if explicitly requested or as fallback
        if (config.get('use_single_trajectory') or not collected) and self.fiber_trajectory is not None:
            collect_from_trajectory(self.fiber_trajectory)

        # Convert to numpy arrays
        angle_data['tilt'] = np.array(angle_data['tilt']) if angle_data['tilt'] else None
        angle_data['azimuth'] = np.array(angle_data['azimuth']) if angle_data['azimuth'] else None
        angle_data['xz_projection'] = np.array(angle_data['xz_projection']) if angle_data['xz_projection'] else None
        angle_data['yz_projection'] = np.array(angle_data['yz_projection']) if angle_data['yz_projection'] else None

        return angle_data

    def showExportScreenshotDialog(self):
        """Show export screenshot dialog."""
        dialog = ExportScreenshotDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.exportScreenshot(dialog.getSettings())

    def exportScreenshot(self, settings):
        """Export screenshot with specified settings."""
        view_type = settings['view']
        bg_color = settings['background']
        include_ct = settings['include_ct']
        resolution = settings.get('resolution', (1280, 960))
        include_legend = settings.get('include_legend', True)

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        if not filename:
            return

        # Ensure extension
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'

        try:
            if view_type == '3D View':
                self._export_3d_screenshot(filename, bg_color, resolution, include_legend)
            else:
                self._export_slice_screenshot(filename, view_type, bg_color, include_ct, resolution, include_legend)
            QMessageBox.information(self, "Success", f"Screenshot saved to:\n{filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save screenshot:\n{str(e)}")

    def _export_3d_screenshot(self, filename, bg_color, resolution, include_legend=True):
        """Export 3D view screenshot."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import io

        # Set background color
        if bg_color == 'White':
            self.plotter_3d.set_background('white')
            text_color = 'black'
        elif bg_color == 'Gray':
            self.plotter_3d.set_background('#808080')
            text_color = 'white'
        else:  # Transparent
            self.plotter_3d.set_background('white')
            text_color = 'black'

        # Take screenshot with specified resolution
        screenshot_array = self.plotter_3d.screenshot(
            transparent_background=(bg_color == 'Transparent'),
            window_size=resolution,
            return_img=True
        )

        # Restore original background
        self.plotter_3d.set_background('#2a2a2a')

        # Check if we need to add color legend
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_mode = self.color_mode_combo.currentText()
        use_azimuth = "Azimuth" in color_mode

        if include_legend and color_by_angle and use_azimuth:
            # Add color wheel to the screenshot using PIL/matplotlib
            self._add_color_wheel_to_3d_screenshot(screenshot_array, filename, text_color, bg_color)
        elif include_legend and color_by_angle and not use_azimuth:
            # Add colorbar for tilt mode
            cmap_name = self.colormap_combo.currentText()
            angle_min = self.trajectory_settings['tilt_min']
            angle_max = self.trajectory_settings['tilt_max']
            self._add_colorbar_to_3d_screenshot(screenshot_array, filename, text_color, bg_color, cmap_name, angle_min, angle_max)
        else:
            # Save directly without legend
            img = Image.fromarray(screenshot_array)
            img.save(filename)

    def _export_slice_screenshot(self, filename, view_type, bg_color, include_ct, resolution, include_legend=True):
        """Export slice view screenshot."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Map view type to viewport index
        view_map = {'XY Slice': 1, 'XZ Slice': 2, 'YZ Slice': 3}
        viewport_idx = view_map.get(view_type, 1)

        # Determine background color
        transparent = False
        if bg_color == 'White':
            face_color = 'white'
            text_color = 'black'
        elif bg_color == 'Gray':
            face_color = '#808080'
            text_color = 'white'
        else:  # Transparent
            face_color = 'white'
            text_color = 'black'
            transparent = True

        # If resolution is None (Current View), save directly from canvas
        if resolution is None:
            if viewport_idx < len(self.viewport_frames):
                canvas = self.viewport_frames[viewport_idx]['canvas']
                if canvas and hasattr(canvas, 'figure'):
                    fig = canvas.figure

                    # Store original colors
                    original_facecolor = fig.get_facecolor()
                    original_ax_colors = []
                    original_ax_patch_colors = []

                    # Set background for saving
                    if transparent:
                        # For transparent, set figure background to transparent
                        fig.set_facecolor('none')
                        for ax in fig.get_axes():
                            original_ax_colors.append(ax.get_facecolor())
                            original_ax_patch_colors.append(ax.patch.get_facecolor())
                            ax.set_facecolor('none')
                            ax.patch.set_facecolor('none')
                    else:
                        fig.set_facecolor(face_color)
                        for ax in fig.get_axes():
                            original_ax_colors.append(ax.get_facecolor())
                            original_ax_patch_colors.append(ax.patch.get_facecolor())
                            ax.set_facecolor(face_color)
                            ax.patch.set_facecolor(face_color)

                    # Save without bbox_inches='tight' to preserve axis labels
                    fig.savefig(filename, facecolor=fig.get_facecolor(),
                               edgecolor='none', transparent=transparent)

                    # Restore original background
                    fig.set_facecolor(original_facecolor)
                    for ax, orig_color, orig_patch in zip(fig.get_axes(), original_ax_colors, original_ax_patch_colors):
                        ax.set_facecolor(orig_color)
                        ax.patch.set_facecolor(orig_patch)
                    canvas.draw()
                    return

        # Calculate figure size based on resolution (use 100 dpi)
        dpi = 100
        fig_width = resolution[0] / dpi
        fig_height = resolution[1] / dpi

        # Create figure with appropriate background and resolution
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=face_color, dpi=dpi)
        ax.set_facecolor(face_color if face_color != 'none' else 'white')

        # Get current slice positions
        z_pos = self.current_slice['z']
        y_pos = self.current_slice['y']
        x_pos = self.current_slice['x']

        # Get visualization settings
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_mode = self.color_mode_combo.currentText()
        use_azimuth = "Azimuth" in color_mode
        cmap_name = self.colormap_combo.currentText()
        angle_min = self.trajectory_settings['tilt_min']
        angle_max = self.trajectory_settings['tilt_max']
        sat_min = self.trajectory_settings['sat_min']
        sat_max = self.trajectory_settings['sat_max']
        show_fiber_diameter = self.trajectory_settings['show_fiber_diameter']

        cmap = plt.get_cmap(cmap_name)

        def angle_to_color(angle):
            if not color_by_angle:
                return 'blue'
            norm_angle = np.clip((angle - angle_min) / (angle_max - angle_min + 1e-6), 0, 1)
            return cmap(norm_angle)

        def azimuth_to_color(azimuth, tilt_angle=None):
            if not color_by_angle:
                return 'blue'
            # Convert numpy azimuth to display azimuth (Y-axis flip compensation)
            display_az = (360 - azimuth) % 360
            hue = display_az / 360.0
            # Use tilt angle for saturation with configurable range
            if tilt_angle is not None:
                saturation = np.clip((tilt_angle - sat_min) / (sat_max - sat_min + 1e-6), 0.0, 1.0)
            else:
                saturation = 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, 1.0)
            return (r, g, b)

        # Determine which slice to export
        if view_type == 'XY Slice':
            slice_idx = z_pos
            if include_ct and self.main_window and self.main_window.viewer and self.main_window.viewer.current_volume is not None:
                vol = self.main_window.viewer.current_volume
                if slice_idx < vol.shape[0]:
                    ax.imshow(vol[slice_idx], cmap='gray', origin='lower', alpha=0.7)

            # Plot fiber points
            self._plot_xy_slice_for_export(ax, z_pos, use_azimuth, angle_to_color, azimuth_to_color, show_fiber_diameter)
            ax.set_xlabel('X', color=text_color)
            ax.set_ylabel('Y', color=text_color)
            ax.set_title(f'XY Slice (Z={slice_idx})', color=text_color)

        elif view_type == 'XZ Slice':
            slice_idx = y_pos
            if include_ct and self.main_window and self.main_window.viewer and self.main_window.viewer.current_volume is not None:
                vol = self.main_window.viewer.current_volume
                if slice_idx < vol.shape[1]:
                    ax.imshow(vol[:, slice_idx, :].T, cmap='gray', origin='lower', aspect='auto', alpha=0.7)

            self._plot_xz_slice_for_export(ax, y_pos, use_azimuth, angle_to_color, azimuth_to_color, show_fiber_diameter)
            ax.set_xlabel('Z', color=text_color)
            ax.set_ylabel('X', color=text_color)
            ax.set_title(f'XZ Slice (Y={slice_idx})', color=text_color)

        elif view_type == 'YZ Slice':
            slice_idx = x_pos
            if include_ct and self.main_window and self.main_window.viewer and self.main_window.viewer.current_volume is not None:
                vol = self.main_window.viewer.current_volume
                if slice_idx < vol.shape[2]:
                    ax.imshow(vol[:, :, slice_idx].T, cmap='gray', origin='lower', aspect='auto', alpha=0.7)

            self._plot_yz_slice_for_export(ax, x_pos, use_azimuth, angle_to_color, azimuth_to_color, show_fiber_diameter)
            ax.set_xlabel('Z', color=text_color)
            ax.set_ylabel('Y', color=text_color)
            ax.set_title(f'YZ Slice (X={slice_idx})', color=text_color)

        ax.tick_params(colors=text_color)
        ax.set_aspect('equal')

        # Add color legend if requested
        if include_legend and color_by_angle:
            if use_azimuth:
                # Add color wheel legend for azimuth mode
                self._add_color_wheel_to_export(fig, ax, text_color)
            else:
                # Add colorbar for tilt angle mode
                self._add_colorbar_to_export(fig, ax, cmap, angle_min, angle_max, text_color)

        # Save with transparency if requested
        plt.savefig(filename, dpi=150, bbox_inches='tight',
                   transparent=(bg_color == 'Transparent'),
                   facecolor=face_color if face_color != 'none' else 'white')
        plt.close(fig)

    def _add_color_wheel_to_export(self, fig, ax, text_color):
        """Add color wheel legend to export figure for azimuth mode using the existing color wheel."""
        from matplotlib.patches import Wedge
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create inset axes for color wheel in bottom-right corner
        ax_inset = inset_axes(ax, width="15%", height="15%", loc='lower right',
                              borderpad=1.0)

        # Use the same drawing parameters as _create_color_wheel_legend
        n_seg = 72  # Number of angular segments
        n_rad = 15  # Number of radial segments for saturation
        for i_theta in range(n_seg):
            hue = (i_theta + 0.5) / n_seg
            # Flip across X-axis (negative angles) - same as viewport
            theta1 = -(i_theta + 1) * 360 / n_seg
            theta2 = -i_theta * 360 / n_seg
            for i_r in range(n_rad):
                r2 = (i_r + 1) / n_rad
                sat = (i_r + 0.5) / n_rad
                color = colorsys.hsv_to_rgb(hue, sat, 1.0)
                wedge = Wedge((0, 0), r2, theta1, theta2, width=1.0 / n_rad,
                             facecolor=color, edgecolor='none')
                ax_inset.add_patch(wedge)

        ax_inset.set_xlim(-1.4, 1.4)
        ax_inset.set_ylim(-1.4, 1.4)
        ax_inset.set_aspect('equal')
        ax_inset.axis('off')

        # Add direction labels (same as viewport)
        label_style = {'fontsize': 6, 'color': text_color, 'ha': 'center', 'va': 'center',
                       'fontweight': 'bold'}
        ax_inset.text(1.2, 0, '+X', **label_style)
        ax_inset.text(-1.2, 0, '-X', **label_style)
        ax_inset.text(0, 1.2, 'Up', **label_style)
        ax_inset.text(0, -1.2, 'Down', **label_style)

    def _add_colorbar_to_export(self, fig, ax, cmap, vmin, vmax, text_color):
        """Add colorbar to export figure for tilt angle mode."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt

        # Create a ScalarMappable for the colorbar
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Tilt Angle (°)', color=text_color)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticks(), color=text_color)
        for label in cbar.ax.get_yticklabels():
            label.set_color(text_color)

    def _add_color_wheel_to_3d_screenshot(self, screenshot_array, filename, text_color, bg_color):
        """Add color wheel legend to 3D screenshot and save using the existing color wheel pixmap."""
        from PIL import Image
        import numpy as np

        # Convert screenshot to PIL Image
        screenshot_img = Image.fromarray(screenshot_array)

        # Get the color wheel pixmap from the label
        if hasattr(self, 'color_wheel_label') and self.color_wheel_label.pixmap():
            pixmap = self.color_wheel_label.pixmap()
            # Convert QPixmap to PIL Image
            qimg = pixmap.toImage()
            width = qimg.width()
            height = qimg.height()

            # Convert QImage to numpy array
            ptr = qimg.bits()
            arr = np.array(ptr).reshape(height, width, 4)  # RGBA

            # Create PIL Image from numpy array
            wheel_img = Image.fromarray(arr, 'RGBA')

            # Resize wheel to fit (15% of screenshot width)
            wheel_size = int(screenshot_array.shape[1] * 0.15)
            wheel_img = wheel_img.resize((wheel_size, wheel_size), Image.Resampling.LANCZOS)

            # Calculate position (bottom-right corner with margin)
            margin = int(screenshot_array.shape[1] * 0.02)
            x = screenshot_array.shape[1] - wheel_size - margin
            y = screenshot_array.shape[0] - wheel_size - margin

            # Paste wheel onto screenshot (with alpha compositing)
            screenshot_img = screenshot_img.convert('RGBA')
            screenshot_img.paste(wheel_img, (x, y), wheel_img)

        # Save the final image
        if bg_color == 'Transparent':
            screenshot_img.save(filename, 'PNG')
        else:
            # Convert to RGB for non-transparent
            screenshot_img = screenshot_img.convert('RGB')
            screenshot_img.save(filename)

    def _add_colorbar_to_3d_screenshot(self, screenshot_array, filename, text_color, bg_color, cmap_name, vmin, vmax):
        """Add colorbar legend to 3D screenshot and save."""
        import matplotlib.pyplot as plt
        from PIL import Image

        # Get image dimensions
        height, width = screenshot_array.shape[:2]

        # Create figure matching screenshot size
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        # Set background
        if bg_color == 'White':
            fig.patch.set_facecolor('white')
        elif bg_color == 'Gray':
            fig.patch.set_facecolor('#808080')
        else:
            fig.patch.set_alpha(0)

        # Add screenshot as background
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(screenshot_array)
        ax.axis('off')

        # Add colorbar on right side
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Tilt Angle (°)', color=text_color, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
        for label in cbar.ax.get_yticklabels():
            label.set_color(text_color)

        # Save
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0,
                   transparent=(bg_color == 'Transparent'))
        plt.close(fig)

    def _plot_xy_slice_for_export(self, ax, z_pos, use_azimuth, angle_to_color, azimuth_to_color, show_diameter):
        """Plot XY slice fiber points for export."""
        from matplotlib.patches import Circle

        tolerance = 3
        all_trajectories = [(self.fiber_trajectory, 0, 0, 0)] if self.fiber_trajectory else []
        for roi_name, roi_data in self.roi_trajectories.items():
            if isinstance(roi_data, dict):
                offset = roi_data['offset']
                all_trajectories.append((roi_data['trajectory'], offset[2], offset[1], offset[0]))

        for fiber_traj, x_offset, y_offset, z_offset in all_trajectories:
            if fiber_traj is None:
                continue
            trajectories = fiber_traj.trajectories
            angles = fiber_traj.angles
            azimuths = getattr(fiber_traj, 'azimuths', angles)
            fiber_diameter = getattr(fiber_traj, 'fiber_diameter', 7.0)
            radius = fiber_diameter / 2.0

            roi_z_pos = z_pos - z_offset
            if 0 <= roi_z_pos < len(trajectories):
                z, points = trajectories[roi_z_pos]
                slice_angles = angles[roi_z_pos] if roi_z_pos < len(angles) else np.zeros(len(points))
                if use_azimuth and roi_z_pos < len(azimuths):
                    colors = [azimuth_to_color(az, tilt) for az, tilt in zip(azimuths[roi_z_pos], slice_angles)]
                else:
                    colors = [angle_to_color(a) for a in slice_angles]
                ax.scatter(points[:, 0] + x_offset, points[:, 1] + y_offset, c=colors, s=10, alpha=0.8)

                if show_diameter:
                    for pt in points:
                        circle = Circle((pt[0] + x_offset, pt[1] + y_offset), radius,
                                       fill=False, edgecolor='red', linewidth=0.5)
                        ax.add_patch(circle)

    def _plot_xz_slice_for_export(self, ax, y_pos, use_azimuth, angle_to_color, azimuth_to_color, show_diameter):
        """Plot XZ slice fiber points for export."""
        from matplotlib.patches import Circle
        tolerance = 3

        all_trajectories = [(self.fiber_trajectory, 0, 0, 0)] if self.fiber_trajectory else []
        for roi_name, roi_data in self.roi_trajectories.items():
            if isinstance(roi_data, dict):
                offset = roi_data['offset']
                all_trajectories.append((roi_data['trajectory'], offset[2], offset[1], offset[0]))

        for fiber_traj, x_offset, y_offset, z_offset in all_trajectories:
            if fiber_traj is None:
                continue
            trajectories = fiber_traj.trajectories
            angles = fiber_traj.angles
            azimuths = getattr(fiber_traj, 'azimuths', angles)

            roi_y_pos = y_pos - y_offset
            for slice_idx, (z, points) in enumerate(trajectories):
                mask = np.abs(points[:, 1] - roi_y_pos) < tolerance
                if np.any(mask):
                    x_coords = points[mask, 0] + x_offset
                    z_coords = np.full(np.sum(mask), z + z_offset)
                    slice_angles = angles[slice_idx] if slice_idx < len(angles) else np.zeros(len(points))
                    if use_azimuth and slice_idx < len(azimuths):
                        colors = [azimuth_to_color(az, tilt) for az, tilt in zip(np.array(azimuths[slice_idx])[mask], slice_angles[mask])]
                    else:
                        colors = [angle_to_color(a) for a in slice_angles[mask]]
                    ax.scatter(z_coords, x_coords, c=colors, s=5, alpha=0.6)

    def _plot_yz_slice_for_export(self, ax, x_pos, use_azimuth, angle_to_color, azimuth_to_color, show_diameter):
        """Plot YZ slice fiber points for export."""
        from matplotlib.patches import Circle
        tolerance = 3

        all_trajectories = [(self.fiber_trajectory, 0, 0, 0)] if self.fiber_trajectory else []
        for roi_name, roi_data in self.roi_trajectories.items():
            if isinstance(roi_data, dict):
                offset = roi_data['offset']
                all_trajectories.append((roi_data['trajectory'], offset[2], offset[1], offset[0]))

        for fiber_traj, x_offset, y_offset, z_offset in all_trajectories:
            if fiber_traj is None:
                continue
            trajectories = fiber_traj.trajectories
            angles = fiber_traj.angles
            azimuths = getattr(fiber_traj, 'azimuths', angles)

            roi_x_pos = x_pos - x_offset
            for slice_idx, (z, points) in enumerate(trajectories):
                mask = np.abs(points[:, 0] - roi_x_pos) < tolerance
                if np.any(mask):
                    y_coords = points[mask, 1] + y_offset
                    z_coords = np.full(np.sum(mask), z + z_offset)
                    slice_angles = angles[slice_idx] if slice_idx < len(angles) else np.zeros(len(points))
                    if use_azimuth and slice_idx < len(azimuths):
                        colors = [azimuth_to_color(az, tilt) for az, tilt in zip(np.array(azimuths[slice_idx])[mask], slice_angles[mask])]
                    else:
                        colors = [angle_to_color(a) for a in slice_angles[mask]]
                    ax.scatter(z_coords, y_coords, c=colors, s=5, alpha=0.6)


class AnalysisTab(QWidget):
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Ribbon toolbar with gray frame (matching visualization tab)
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0; }")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(10)

        # Analysis Operations Group
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QHBoxLayout(analysis_group)

        self.edit_roi_btn = RibbonButton("Edit\nROI")
        self.edit_roi_btn.clicked.connect(self.toggleROIEdit)
        self.edit_roi_btn.setCheckable(True)  # Make it a toggle button
        analysis_layout.addWidget(self.edit_roi_btn)

        self.compute_btn = RibbonButton("Compute\nOrientation")
        self.compute_btn.clicked.connect(self.computeOrientation)
        analysis_layout.addWidget(self.compute_btn)

        self.edit_range_btn = RibbonButton("Edit\nRange")
        self.edit_range_btn.clicked.connect(self.openRangeEditor)
        self.edit_range_btn.setEnabled(False)  # Enable after orientation computation
        analysis_layout.addWidget(self.edit_range_btn)

        self.histogram_btn = RibbonButton("Histogram")
        self.histogram_btn.clicked.connect(self.openHistogramDialog)
        self.histogram_btn.setEnabled(False)  # Enable after orientation computation
        analysis_layout.addWidget(self.histogram_btn)

        self.void_analysis_btn = RibbonButton("Void\nAnalysis")
        self.void_analysis_btn.clicked.connect(self.computeVoidAnalysis)
        analysis_layout.addWidget(self.void_analysis_btn)

        self.magnify_btn = RibbonButton("Magnify")
        self.magnify_btn.setCheckable(True)
        self.magnify_btn.clicked.connect(self.toggleMagnify)
        analysis_layout.addWidget(self.magnify_btn)

        toolbar_layout.addWidget(analysis_group)

        # Reference Vector Group
        ref_group = QGroupBox("Reference Vector")
        ref_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        ref_layout = QVBoxLayout(ref_group)

        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Z-axis", "Y-axis", "X-axis"])
        self.ref_combo.setCurrentText("Z-axis")  # Default to Z-axis
        self.ref_combo.setStyleSheet("""
            QComboBox {
                padding: 4px;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                background-color: white;
                font-size: 11px;
                min-width: 60px;
            }
            QComboBox:hover {
                border: 1px solid #005499;
                background-color: #e5f3ff;
            }
        """)
        ref_layout.addWidget(self.ref_combo)

        toolbar_layout.addWidget(ref_group)

        # Fiber Detection Group (Watershed-based)
        fiber_group = QGroupBox("Fiber Detection")
        fiber_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        fiber_layout = QHBoxLayout(fiber_group)

        self.fiber_detect_settings_btn = RibbonButton("Detection\nSettings")
        self.fiber_detect_settings_btn.clicked.connect(self.openFiberDetectionSettings)
        fiber_layout.addWidget(self.fiber_detect_settings_btn)

        self.fiber_detect_btn = RibbonButton("Detect\nFibers")
        self.fiber_detect_btn.clicked.connect(self.detectFibers)
        fiber_layout.addWidget(self.fiber_detect_btn)

        toolbar_layout.addWidget(fiber_group)

        # InSegt Group (Interactive Segmentation)
        insegt_group = QGroupBox("InSegt")
        insegt_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        insegt_layout = QHBoxLayout(insegt_group)

        self.insegt_settings_btn = RibbonButton("InSegt\nSettings")
        self.insegt_settings_btn.setToolTip("Configure InSegt processing parameters")
        self.insegt_settings_btn.clicked.connect(self.openInSegtSettings)
        insegt_layout.addWidget(self.insegt_settings_btn)

        self.insegt_labeling_btn = RibbonButton("Labeling")
        self.insegt_labeling_btn.setToolTip(
            "Open InSegt interactive labeling tool.\n"
            "Draw fiber (red) and background (green) annotations."
        )
        self.insegt_labeling_btn.clicked.connect(self.openInSegtLabeling)
        insegt_layout.addWidget(self.insegt_labeling_btn)

        self.insegt_run_btn = RibbonButton("Run")
        self.insegt_run_btn.setToolTip(
            "Apply InSegt model to detect fibers in all slices.\n"
            "Requires labeling to be completed first."
        )
        self.insegt_run_btn.clicked.connect(self.runInSegt)
        self.insegt_run_btn.setEnabled(False)  # Disabled until labeling is done
        insegt_layout.addWidget(self.insegt_run_btn)

        toolbar_layout.addWidget(insegt_group)

        toolbar_layout.addStretch()

        layout.addWidget(toolbar)
        layout.addStretch()

        # Initialize fiber detection settings (watershed-based)
        self.fiber_detection_settings = {
            'min_diameter': 5.0,
            'max_diameter': 25.0,
            'min_distance': 5,
            'show_watershed': True,
            'show_centers': True,
            'center_marker_size': 3
        }

        # InSegt settings (separate)
        self.insegt_settings = {
            'scale': 0.5,
            'sigmas': [1, 2],
            'patch_size': 9,
            'branching_factor': 5,
            'number_layers': 4,
            'training_patches': 10000
        }

        # InSegt model storage
        self.insegt_model = None
        self.insegt_labels = None
        self._insegt_scale = 0.5  # Store scale used during labeling
        self._insegt_labels_ready = False  # Flag to enable Run button

    def openFiberDetectionSettings(self):
        """Open fiber detection settings dialog"""
        dialog = FiberDetectionSettingsDialog(self, self.fiber_detection_settings)
        if dialog.exec() == QDialog.Accepted:
            self.fiber_detection_settings = dialog.getSettings()

    def detectFibers(self):
        """Detect fibers in all slices of the ROI and display results"""
        main_window = getattr(self, 'main_window', None)

        if not main_window:
            print("No main window reference")
            return

        if main_window.current_volume is None:
            QMessageBox.warning(self, "Warning", "No volume loaded - please import a volume first")
            return

        if not main_window.viewer.rois:
            QMessageBox.warning(self, "Warning", "No ROI defined - please create an ROI first")
            return

        # Get the last ROI
        roi_name = f"ROI{main_window.viewer.roi_counter}"
        if roi_name not in main_window.viewer.rois:
            # Try to find any ROI
            roi_name = list(main_window.viewer.rois.keys())[-1] if main_window.viewer.rois else None

        if roi_name is None:
            QMessageBox.warning(self, "Warning", "No valid ROI found")
            return

        roi_data = main_window.viewer.rois[roi_name]
        bounds = roi_data.get('bounds')
        if bounds is None:
            QMessageBox.warning(self, "Warning", "ROI bounds not defined")
            return

        z_min, z_max, y_min, y_max, x_min, x_max = bounds
        n_slices = z_max - z_min

        main_window.status_label.setText(f"Detecting fibers in {roi_name} ({n_slices} slices)...")
        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, n_slices)
        QApplication.processEvents()

        try:
            # Import detect_fiber_centers
            from acsc.fiber_trajectory import detect_fiber_centers

            # Check if watershed display is enabled
            show_watershed = self.fiber_detection_settings.get('show_watershed', True)

            # Detect fiber centers in all slices
            all_slice_results = {}
            total_fibers = 0
            all_diameters = []

            for i, z in enumerate(range(z_min, z_max)):
                # Update progress
                main_window.progress_bar.setValue(i)
                if i % 10 == 0:
                    main_window.status_label.setText(f"Detecting fibers: slice {i+1}/{n_slices}...")
                    QApplication.processEvents()

                # Extract the slice from ROI
                slice_image = main_window.current_volume[z, y_min:y_max, x_min:x_max]

                # Determine threshold percentile (None for Otsu)
                threshold_method = self.fiber_detection_settings.get('threshold_method', 'otsu')
                threshold_percentile = None
                if threshold_method == 'percentile':
                    threshold_percentile = self.fiber_detection_settings.get('threshold_percentile', 50.0)

                # Detect fiber centers (with labels if watershed display is enabled)
                if show_watershed:
                    centers, diameters, labels = detect_fiber_centers(
                        slice_image,
                        min_diameter=self.fiber_detection_settings['min_diameter'],
                        max_diameter=self.fiber_detection_settings['max_diameter'],
                        min_distance=self.fiber_detection_settings['min_distance'],
                        return_labels=True,
                        threshold_percentile=threshold_percentile
                    )
                else:
                    centers, diameters = detect_fiber_centers(
                        slice_image,
                        min_diameter=self.fiber_detection_settings['min_diameter'],
                        max_diameter=self.fiber_detection_settings['max_diameter'],
                        min_distance=self.fiber_detection_settings['min_distance'],
                        threshold_percentile=threshold_percentile
                    )
                    labels = None

                if len(centers) > 0:
                    all_slice_results[z] = {
                        'centers': centers,
                        'diameters': diameters,
                        'labels': labels
                    }
                    total_fibers += len(centers)
                    all_diameters.extend(diameters.tolist())

            main_window.progress_bar.setValue(n_slices)
            main_window.showProgress(False)

            if total_fibers == 0:
                QMessageBox.information(self, "Detection Result", "No fibers detected with current settings.")
                main_window.status_label.setText("No fibers detected")
                return

            # Store detection results for all slices
            main_window.viewer.fiber_detection_result = {
                'all_slices': all_slice_results,
                'roi_offset': (x_min, y_min),
                'roi_bounds': bounds,
                'settings': self.fiber_detection_settings.copy()
            }

            # Trigger visualization update
            main_window.viewer.show_fiber_detection = True
            # Force redraw of views
            main_window.viewer.renderVolume()

            # Show results
            all_diameters_arr = np.array(all_diameters)
            mean_diameter = np.mean(all_diameters_arr)
            std_diameter = np.std(all_diameters_arr)
            avg_fibers_per_slice = total_fibers / n_slices

            main_window.status_label.setText(
                f"Detected fibers in {len(all_slice_results)}/{n_slices} slices (avg: {avg_fibers_per_slice:.0f}/slice)"
            )

            QMessageBox.information(self, "Detection Complete",
                f"Fiber detection completed\n\n"
                f"Slices processed: {n_slices}\n"
                f"Slices with fibers: {len(all_slice_results)}\n"
                f"Avg fibers per slice: {avg_fibers_per_slice:.1f}\n\n"
                f"Mean diameter: {mean_diameter:.2f} px\n"
                f"Std deviation: {std_diameter:.2f} px\n"
                f"Min diameter: {np.min(all_diameters_arr):.2f} px\n"
                f"Max diameter: {np.max(all_diameters_arr):.2f} px"
            )

        except Exception as e:
            main_window.showProgress(False)
            QMessageBox.critical(self, "Detection Error", f"Failed to detect fibers:\n{str(e)}")
            main_window.status_label.setText(f"Detection failed: {str(e)}")

    def openInSegtSettings(self):
        """Open InSegt settings dialog."""
        dialog = InSegtSettingsDialog(self, self.insegt_settings)
        if dialog.exec() == QDialog.Accepted:
            self.insegt_settings = dialog.getSettings()

    def openInSegtLabeling(self):
        """Open InSegt interactive labeling tool for the current slice.

        Launches InSegt in a subprocess to avoid Qt conflicts.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        main_window = getattr(self, 'main_window', None)

        if not main_window:
            QMessageBox.warning(self, "Error", "No main window reference")
            return

        if main_window.current_volume is None:
            QMessageBox.warning(self, "Error", "No volume loaded. Please import a volume first.")
            return

        # Get ROI bounds
        rois = main_window.viewer.rois if main_window.viewer else {}
        bounds = None
        for name, roi_data in rois.items():
            if name != '_no_roi' and roi_data.get('bounds') is not None:
                bounds = roi_data['bounds']
                break

        if bounds is None:
            # Use full slice
            y_min, y_max = 0, main_window.current_volume.shape[1]
            x_min, x_max = 0, main_window.current_volume.shape[2]
        else:
            _, _, y_min, y_max, x_min, x_max = bounds

        # Store ROI bounds for later use
        self._insegt_roi_bounds = (y_min, y_max, x_min, x_max)

        # Get current slice index from slider
        current_z = main_window.z_slice_slider.value() if hasattr(main_window, 'z_slice_slider') else 0
        slice_image = main_window.current_volume[current_z, y_min:y_max, x_min:x_max]

        main_window.status_label.setText(f"Launching InSegt labeling for slice {current_z} (ROI: {x_max-x_min}x{y_max-y_min})...")
        QApplication.processEvents()

        try:
            # Create temp directory for communication
            self._insegt_temp_dir = tempfile.mkdtemp(prefix="insegt_")
            temp_dir = Path(self._insegt_temp_dir)

            # Save ROI image to temp file
            image_path = temp_dir / "slice_image.npy"
            np.save(str(image_path), slice_image)

            # Get path to the runner script
            script_path = Path(__file__).parent / "insegt" / "run_insegt_gui.py"

            # Get settings from insegt_settings
            insegt_scale = self.insegt_settings.get('scale', 0.5)
            sigmas = self.insegt_settings.get('sigmas', [1, 2])
            patch_size = self.insegt_settings.get('patch_size', 9)
            branching_factor = self.insegt_settings.get('branching_factor', 5)
            number_layers = self.insegt_settings.get('number_layers', 4)
            training_patches = self.insegt_settings.get('training_patches', 10000)

            self._insegt_scale = insegt_scale

            # Build command
            cmd = [
                sys.executable,
                str(script_path),
                str(image_path),
                str(temp_dir),
                "--sigmas", ",".join(str(s) for s in sigmas),
                "--patch-size", str(patch_size),
                "--branching-factor", str(branching_factor),
                "--number-layers", str(number_layers),
                "--training-patches", str(training_patches),
                "--scale", str(insegt_scale)
            ]

            # Launch subprocess
            self._insegt_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )

            main_window.status_label.setText(
                f"InSegt labeling opened for slice {current_z}. "
                "Close the InSegt window when done."
            )

            # Store current slice info
            self._insegt_slice_z = current_z

            # Start timer to check for completion
            self._insegt_check_timer = QTimer()
            self._insegt_check_timer.timeout.connect(self._checkInSegtProcess)
            self._insegt_check_timer.start(1000)  # Check every second

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open InSegt labeling:\n{str(e)}")
            main_window.status_label.setText(f"InSegt error: {str(e)}")

    def _checkInSegtProcess(self):
        """Check if InSegt subprocess has completed."""
        from pathlib import Path

        main_window = getattr(self, 'main_window', None)

        if not hasattr(self, '_insegt_process') or self._insegt_process is None:
            if hasattr(self, '_insegt_check_timer'):
                self._insegt_check_timer.stop()
            return

        # Check if process is still running
        poll = self._insegt_process.poll()
        if poll is None:
            # Still running
            return

        # Process finished - stop timer
        self._insegt_check_timer.stop()

        # Check status file
        temp_dir = Path(self._insegt_temp_dir)
        status_file = temp_dir / "insegt_status.txt"
        labels_file = temp_dir / "insegt_labels.npy"

        try:
            if status_file.exists():
                with open(status_file, 'r') as f:
                    lines = f.read().strip().split('\n')

                status = lines[0] if len(lines) > 0 else "unknown"

                if status == "completed":
                    labels_path = lines[1] if len(lines) > 1 else str(labels_file)

                    # Load labels
                    if Path(labels_path).exists():
                        self.insegt_labels = np.load(labels_path)
                        self._insegt_labels_ready = True

                        # Enable Run button
                        self.insegt_run_btn.setEnabled(True)

                        if main_window:
                            main_window.status_label.setText(
                                f"InSegt labeling completed. Click 'Run' to detect fibers."
                            )

                elif status == "error":
                    error_msg = lines[1] if len(lines) > 1 else "Unknown error"
                    QMessageBox.warning(self, "InSegt Error", f"InSegt process error:\n{error_msg}")
                    if main_window:
                        main_window.status_label.setText(f"InSegt error: {error_msg}")

            else:
                # No status file - check stderr
                stderr = self._insegt_process.stderr.read().decode() if self._insegt_process.stderr else ""
                if stderr:
                    print(f"InSegt stderr: {stderr}")
                if main_window:
                    main_window.status_label.setText("InSegt process ended (no status)")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing InSegt results:\n{str(e)}")
            if main_window:
                main_window.status_label.setText(f"InSegt error: {str(e)}")

        finally:
            # Cleanup
            self._insegt_process = None

            # Optionally cleanup temp directory (keep for debugging)
            # import shutil
            # shutil.rmtree(self._insegt_temp_dir, ignore_errors=True)

    def runInSegt(self):
        """Apply InSegt model to all slices using the labeled data."""
        from pathlib import Path

        main_window = getattr(self, 'main_window', None)
        if not main_window or main_window.current_volume is None:
            QMessageBox.warning(self, "Error", "No volume loaded.")
            return

        if self.insegt_labels is None or not self._insegt_labels_ready:
            QMessageBox.warning(self, "Error",
                "No labels available.\nPlease run 'Labeling' first.")
            return

        # Build model from the training slice
        process_scale = self.insegt_settings.get('scale', 0.5)
        if hasattr(self, '_insegt_scale'):
            process_scale = self._insegt_scale

        # Get ROI bounds
        rois = main_window.viewer.rois if main_window.viewer else {}
        bounds = None

        for name, roi_data in rois.items():
            if name != '_no_roi' and roi_data.get('bounds') is not None:
                bounds = roi_data['bounds']
                break

        if bounds is None:
            z_min, z_max = 0, main_window.current_volume.shape[0]
            y_min, y_max = 0, main_window.current_volume.shape[1]
            x_min, x_max = 0, main_window.current_volume.shape[2]
            bounds = (z_min, z_max, y_min, y_max, x_min, x_max)

        z_min, z_max, y_min, y_max, x_min, x_max = bounds
        n_slices = z_max - z_min

        main_window.status_label.setText(f"Building InSegt model (scale={process_scale})...")
        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, n_slices + 1)
        main_window.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            from acsc.insegt.fiber_model import FiberSegmentationModel
            from scipy.ndimage import distance_transform_edt
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            from skimage.measure import regionprops
            import acsc.insegt.models.utils as insegt_utils

            # Build model from the first (training) slice at scaled resolution
            training_z = self._insegt_slice_z if hasattr(self, '_insegt_slice_z') else z_min
            training_slice = main_window.current_volume[training_z, y_min:y_max, x_min:x_max]

            # Scale training image
            if process_scale != 1.0:
                training_slice_scaled = cv.resize(
                    training_slice,
                    None,
                    fx=process_scale,
                    fy=process_scale,
                    interpolation=cv.INTER_AREA
                )
            else:
                training_slice_scaled = training_slice

            # Create and build model
            self.insegt_model = FiberSegmentationModel(
                sigmas=[1, 2],
                patch_size=9,
                branching_factor=5,
                number_layers=4,
                training_patches=10000
            )
            self.insegt_model.build_from_image(training_slice_scaled)

            main_window.progress_bar.setValue(1)
            main_window.status_label.setText(f"Detecting fibers in {n_slices} slices...")
            QApplication.processEvents()

            # Process all slices
            all_slice_results = {}
            total_fibers = 0
            all_diameters = []

            for i, z in enumerate(range(z_min, z_max)):
                main_window.progress_bar.setValue(i + 1)
                if i % 10 == 0:
                    main_window.status_label.setText(f"InSegt detecting: slice {i+1}/{n_slices}...")
                    QApplication.processEvents()

                # Extract slice
                slice_image = main_window.current_volume[z, y_min:y_max, x_min:x_max]
                orig_shape = slice_image.shape

                # Downscale for faster processing
                if process_scale != 1.0:
                    slice_small = cv.resize(
                        slice_image,
                        None,
                        fx=process_scale,
                        fy=process_scale,
                        interpolation=cv.INTER_AREA
                    )
                else:
                    slice_small = slice_image

                # Set image and process at reduced scale
                self.insegt_model.set_image(slice_small)
                probs = self.insegt_model.process(self.insegt_labels)
                segmentation_small = insegt_utils.segment_probabilities(probs)

                # Upscale segmentation back to original size
                if process_scale != 1.0:
                    segmentation = cv.resize(
                        segmentation_small.astype(np.uint8),
                        (orig_shape[1], orig_shape[0]),
                        interpolation=cv.INTER_NEAREST
                    )
                else:
                    segmentation = segmentation_small

                # Fiber is class 1
                binary = (segmentation == 1)

                if np.sum(binary) == 0:
                    continue

                # Distance transform and watershed
                distance = distance_transform_edt(binary)
                min_distance = int(self.fiber_detection_settings['min_diameter'] / 2)

                coords = peak_local_max(
                    distance,
                    min_distance=max(min_distance, 3),
                    labels=binary,
                    exclude_border=False
                )

                if len(coords) == 0:
                    continue

                markers = np.zeros_like(binary, dtype=np.int32)
                for j, (y, x) in enumerate(coords):
                    markers[y, x] = j + 1

                labels_ws = watershed(-distance, markers, mask=binary)
                props = regionprops(labels_ws)

                centers = []
                diameters = []
                valid_labels_list = []

                for prop in props:
                    area = prop.area
                    diameter = 2 * np.sqrt(area / np.pi)

                    min_d = self.fiber_detection_settings['min_diameter']
                    max_d = self.fiber_detection_settings['max_diameter']

                    if min_d < diameter < max_d:
                        y, x = prop.centroid
                        centers.append([x, y])
                        diameters.append(diameter)
                        valid_labels_list.append(prop.label)

                if len(centers) > 0:
                    filtered_labels = np.zeros_like(labels_ws)
                    for new_label, old_label in enumerate(valid_labels_list, start=1):
                        filtered_labels[labels_ws == old_label] = new_label

                    all_slice_results[z] = {
                        'centers': np.array(centers),
                        'diameters': np.array(diameters),
                        'labels': filtered_labels
                    }
                    total_fibers += len(centers)
                    all_diameters.extend(diameters)

            main_window.progress_bar.setValue(n_slices + 1)
            main_window.showProgress(False)

            if total_fibers == 0:
                QMessageBox.information(self, "Result", "No fibers detected with InSegt model.")
                main_window.status_label.setText("No fibers detected")
                return

            # Store results
            main_window.viewer.fiber_detection_result = {
                'all_slices': all_slice_results,
                'roi_offset': (x_min, y_min),
                'roi_bounds': bounds,
                'settings': self.fiber_detection_settings.copy()
            }

            main_window.viewer.show_fiber_detection = True
            main_window.viewer.renderVolume()

            # Show results
            all_diameters_arr = np.array(all_diameters)
            mean_diameter = np.mean(all_diameters_arr)
            std_diameter = np.std(all_diameters_arr)
            avg_fibers = total_fibers / n_slices

            main_window.status_label.setText(
                f"InSegt detected {total_fibers} fibers in {len(all_slice_results)}/{n_slices} slices"
            )

            QMessageBox.information(self, "InSegt Detection Complete",
                f"Fiber detection completed\n\n"
                f"Slices processed: {n_slices}\n"
                f"Slices with fibers: {len(all_slice_results)}\n"
                f"Total fibers: {total_fibers}\n"
                f"Avg fibers per slice: {avg_fibers:.1f}\n\n"
                f"Mean diameter: {mean_diameter:.2f} px\n"
                f"Std deviation: {std_diameter:.2f} px"
            )

        except Exception as e:
            main_window.showProgress(False)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"InSegt detection failed:\n{str(e)}")
            main_window.status_label.setText(f"InSegt error: {str(e)}")

    def saveInSegtModel(self):
        """Save the current InSegt model to a file."""
        if self.insegt_model is None:
            QMessageBox.warning(self, "Error", "No InSegt model to save.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save InSegt Model", "", "InSegt Model (*.insegt);;All Files (*)"
        )

        if filepath:
            if not filepath.endswith('.insegt'):
                filepath += '.insegt'
            try:
                self.insegt_model.save(filepath)
                QMessageBox.information(self, "Success", f"Model saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")

    def toggleROIEdit(self, checked):
        """Toggle ROI editing mode"""
        main_window = getattr(self, 'main_window', None)
        if main_window and main_window.viewer:
            main_window.viewer.toggleROI(checked)

        # Update button text and disable compute button when editing
        if checked:
            self.edit_roi_btn.setText("Apply\nROI")
            self.compute_btn.setEnabled(False)
        else:
            self.edit_roi_btn.setText("Edit\nROI")
            self.compute_btn.setEnabled(True)

    def computeVoidAnalysis(self):
        """Compute void analysis for all ROIs"""
        main_window = getattr(self, 'main_window', None)

        if not main_window:
            print("No main window reference")
            return

        if main_window.current_volume is None:
            print("No volume loaded - please import a volume first")
            main_window.status_label.setText("No volume loaded - please import a volume first")
            return

        if not main_window.viewer.rois:
            print("No ROIs defined - please create at least one ROI first")
            main_window.status_label.setText("No ROIs defined - please create ROI first")
            return

        # Get threshold value from slider
        threshold = main_window.threshold_slider.value()
        main_window.viewer.void_threshold = threshold

        # Activate void analysis
        main_window.viewer.void_analysis_active = True

        # Trigger re-render to show void overlay
        main_window.viewer.renderVolume()

        main_window.status_label.setText(f"Void analysis computed with threshold={threshold}")

    def toggleMagnify(self, checked):
        """Toggle magnify/zoom mode for slice viewers"""
        main_window = getattr(self, 'main_window', None)

        if not main_window:
            print("No main window reference")
            return

        if not main_window.viewer:
            return

        if checked:
            # Enable zoom mode
            main_window.viewer.enableZoom(True)
            self.magnify_btn.setText("Reset\nZoom")
            main_window.status_label.setText("Magnify mode: Use mouse wheel to zoom, drag to pan")
        else:
            # Disable zoom mode and reset view
            main_window.viewer.enableZoom(False)
            self.magnify_btn.setText("Magnify")
            main_window.status_label.setText("Zoom reset")

    def computeOrientation(self):
        """Compute 3 types of orientation analysis"""
        # Use direct reference to main window
        main_window = getattr(self, 'main_window', None)

        if not main_window:
            print("No main window reference")
            return

        if main_window.current_volume is None:
            print("No volume loaded - please import a volume first")
            main_window.status_label.setText("No volume loaded - please import a volume first")
            return

        print(f"Volume shape: {main_window.current_volume.shape}")

        if main_window.current_volume is not None:
            volume = main_window.current_volume

            # Check if ROI is defined and extract ROI subvolume
            # For now, use the last created ROI (most recent)
            if main_window.viewer.rois:
                # Get the last ROI
                roi_name = f"ROI{main_window.viewer.roi_counter}"
                if roi_name in main_window.viewer.rois:
                    bounds = main_window.viewer.rois[roi_name]['bounds']
                    z_min, z_max, y_min, y_max, x_min, x_max = bounds
                    print(f"Using {roi_name}: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
                    volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
                    print(f"ROI volume shape: {volume.shape}")

            noise_scale = main_window.noise_scale_slider.value()

            main_window.showProgress(True)
            main_window.progress_bar.setRange(0, 3)

            try:
                # Step 1: Compute structure tensor
                main_window.progress_bar.setValue(1)
                main_window.progress_bar.setFormat("Computing structure tensor... (1/3)")
                QApplication.processEvents()

                structure_tensor = compute_structure_tensor(volume, noise_scale=noise_scale)

                # Step 2: Compute orientation without reference (theta, phi)
                main_window.progress_bar.setValue(2)
                main_window.progress_bar.setFormat("Computing orientation angles... (2/3)")
                QApplication.processEvents()

                # Use the working compute_orientation function
                theta, phi = compute_orientation(structure_tensor)

                # Step 3: Compute reference orientation and trim edges
                main_window.progress_bar.setValue(3)
                main_window.progress_bar.setFormat("Computing reference orientation and trimming... (3/3)")
                QApplication.processEvents()

                # Get selected reference vector
                ref_text = self.ref_combo.currentText()
                if ref_text == "X-axis":
                    reference_vector = [0, 0, 1]  # X-axis (depth direction in volume)
                elif ref_text == "Y-axis":
                    reference_vector = [0, 1, 0]  # Y-axis
                else:  # "Z-axis"
                    reference_vector = [1, 0, 0]  # Z-axis

                # Compute reference orientation using proper ACSC method
                reference_orientation = compute_orientation(structure_tensor, reference_vector)

                # Trim edges from all orientation volumes using noise_scale as trim width
                trim_width = noise_scale
                theta_trimmed = drop_edges_3D(trim_width, theta)
                phi_trimmed = drop_edges_3D(trim_width, phi)
                reference_trimmed = drop_edges_3D(trim_width, reference_orientation)

                # Store trimmed orientation data and trim information
                main_window.orientation_data['theta'] = theta_trimmed
                main_window.orientation_data['phi'] = phi_trimmed
                main_window.orientation_data['reference'] = reference_trimmed
                main_window.orientation_data['trim_width'] = trim_width
                main_window.orientation_data['structure_tensor'] = structure_tensor
                main_window.orientation_data['noise_scale'] = noise_scale

                main_window.showProgress(False)
                main_window.status_label.setText(f"Analysis complete (Noise scale: {noise_scale})")

                # Enable edit range and histogram buttons
                self.edit_range_btn.setEnabled(True)
                self.histogram_btn.setEnabled(True)

                # Enable histogram button in Simulation tab as well
                if hasattr(main_window, 'simulation_tab'):
                    main_window.simulation_tab.histogram_btn.setEnabled(True)

                # Pass structure tensor to Modelling tab for fiber trajectory generation
                if hasattr(main_window, 'modelling_tab'):
                    main_window.modelling_tab.setStructureTensor(structure_tensor, volume.shape, volume)

                # Store orientation data in the ROI structure
                if roi_name in main_window.viewer.rois:
                    main_window.viewer.rois[roi_name]['theta'] = theta_trimmed.astype(np.float32)
                    main_window.viewer.rois[roi_name]['phi'] = phi_trimmed.astype(np.float32)
                    main_window.viewer.rois[roi_name]['angle'] = reference_trimmed.astype(np.float32)
                    main_window.viewer.rois[roi_name]['trim_width'] = trim_width  # Store trim width for centering

                    # Dynamically add Orientation-ROI toggle to pipeline
                    main_window.viewer.addOrientationROIToggle(roi_name)

            except Exception as e:
                main_window.showProgress(False)
                main_window.status_label.setText(f"Analysis failed: {str(e)}")
                print(f"Orientation computation error: {e}")
        else:
            print("No volume loaded for orientation analysis")

    def openRangeEditor(self):
        """Open color bar range editor dialog"""
        main_window = getattr(self, 'main_window', None)
        if not main_window:
            return

        # Create and show range editor dialog
        dialog = ColorBarRangeDialog(main_window, self)
        dialog.exec()

    def openHistogramDialog(self):
        """Open histogram configuration dialog"""
        main_window = getattr(self, 'main_window', None)
        if not main_window:
            return

        # Create and show histogram dialog
        dialog = HistogramDialog(main_window, self)
        if dialog.exec() == QDialog.Accepted:
            # Show histogram panel after dialog is accepted
            main_window.showHistogramPanel(dialog.getConfiguration())

class HistogramDialog(QDialog):
    def __init__(self, main_window, analysis_tab):
        super().__init__()
        self.main_window = main_window
        self.analysis_tab = analysis_tab
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Histogram Configuration")
        self.setModal(True)
        self.resize(450, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Histogram Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Bin Settings Group
        bin_group = QGroupBox("Bin Settings")
        bin_layout = QGridLayout(bin_group)

        bin_layout.addWidget(QLabel("Number of Bins:"), 0, 0)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(10, 200)
        self.bins_spin.setValue(50)
        bin_layout.addWidget(self.bins_spin, 0, 1)

        bin_layout.addWidget(QLabel("Range Min:"), 1, 0)
        self.range_min_spin = QDoubleSpinBox()
        self.range_min_spin.setRange(-360, 360)
        self.range_min_spin.setValue(0)
        self.range_min_spin.setDecimals(1)
        bin_layout.addWidget(self.range_min_spin, 1, 1)

        bin_layout.addWidget(QLabel("Range Max:"), 2, 0)
        self.range_max_spin = QDoubleSpinBox()
        self.range_max_spin.setRange(-360, 360)
        self.range_max_spin.setValue(180)
        self.range_max_spin.setDecimals(1)
        bin_layout.addWidget(self.range_max_spin, 2, 1)

        self.auto_range_check = QCheckBox("Auto Range")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.toggled.connect(self.toggleAutoRange)
        bin_layout.addWidget(self.auto_range_check, 3, 0, 1, 2)

        layout.addWidget(bin_group)

        # ROI Selection Group
        roi_group = QGroupBox("Select ROIs")
        roi_layout = QVBoxLayout(roi_group)

        self.roi_checkboxes = {}

        # Add checkbox for each ROI that has orientation data
        if hasattr(self.main_window, 'viewer') and hasattr(self.main_window.viewer, 'rois'):
            for roi_name, roi_data in self.main_window.viewer.rois.items():
                # Only show ROIs that have orientation data
                if (roi_data.get('theta') is not None or
                    roi_data.get('phi') is not None or
                    roi_data.get('angle') is not None):
                    roi_check = QCheckBox(roi_name)
                    roi_check.setChecked(True)
                    self.roi_checkboxes[roi_name] = roi_check
                    roi_layout.addWidget(roi_check)

        # If no ROIs available, add a label
        if not self.roi_checkboxes:
            no_roi_label = QLabel("No ROIs with orientation data available")
            no_roi_label.setStyleSheet("color: gray; font-style: italic;")
            roi_layout.addWidget(no_roi_label)

        layout.addWidget(roi_group)

        # Orientation Selection Group
        orientation_group = QGroupBox("Select Orientations")
        orientation_layout = QVBoxLayout(orientation_group)

        self.ref_check = QCheckBox("Reference Orientation")
        self.theta_check = QCheckBox("X-Z Orientation")
        self.phi_check = QCheckBox("Y-Z Orientation")

        # Always enable all checkboxes - we'll check per-ROI availability
        self.ref_check.setEnabled(True)
        self.ref_check.setChecked(True)

        self.theta_check.setEnabled(True)
        self.theta_check.setChecked(True)

        self.phi_check.setEnabled(True)
        self.phi_check.setChecked(True)

        orientation_layout.addWidget(self.ref_check)
        orientation_layout.addWidget(self.theta_check)
        orientation_layout.addWidget(self.phi_check)

        layout.addWidget(orientation_group)

        # Statistical Display Options Group
        stats_group = QGroupBox("Statistical Analysis")
        stats_layout = QVBoxLayout(stats_group)

        self.show_mean_check = QCheckBox("Show Mean")
        self.show_mean_check.setChecked(True)
        self.show_deviation_check = QCheckBox("Show Standard Deviation")
        self.show_deviation_check.setChecked(True)
        self.show_cv_check = QCheckBox("Show Coefficient of Variation (CV)")
        self.show_cv_check.setChecked(True)

        stats_layout.addWidget(self.show_mean_check)
        stats_layout.addWidget(self.show_deviation_check)
        stats_layout.addWidget(self.show_cv_check)

        layout.addWidget(stats_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.show_btn = QPushButton("Show")
        self.cancel_btn = QPushButton("Cancel")

        self.show_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.show_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Initialize auto range
        self.toggleAutoRange(True)

    def toggleAutoRange(self, checked):
        """Enable/disable manual range controls"""
        self.range_min_spin.setEnabled(not checked)
        self.range_max_spin.setEnabled(not checked)

        if checked:
            # Calculate auto range from selected orientations
            self.updateAutoRange()

    def updateAutoRange(self):
        """Calculate and set auto range based on selected ROIs and orientations"""
        min_val = float('inf')
        max_val = float('-inf')

        # Iterate through selected ROIs
        for roi_name, roi_check in self.roi_checkboxes.items():
            if not roi_check.isChecked():
                continue

            if hasattr(self.main_window, 'viewer') and roi_name in self.main_window.viewer.rois:
                roi_data = self.main_window.viewer.rois[roi_name]

                if self.ref_check.isChecked():
                    data = roi_data.get('angle')
                    if data is not None:
                        min_val = min(min_val, np.nanmin(data))
                        max_val = max(max_val, np.nanmax(data))

                if self.theta_check.isChecked():
                    data = roi_data.get('theta')
                    if data is not None:
                        min_val = min(min_val, np.nanmin(data))
                        max_val = max(max_val, np.nanmax(data))

                if self.phi_check.isChecked():
                    data = roi_data.get('phi')
                    if data is not None:
                        min_val = min(min_val, np.nanmin(data))
                        max_val = max(max_val, np.nanmax(data))

        if min_val != float('inf') and max_val != float('-inf'):
            self.range_min_spin.setValue(min_val)
            self.range_max_spin.setValue(max_val)

    def getConfiguration(self):
        """Return the histogram configuration"""
        # Get selected ROIs
        selected_rois = []
        for roi_name, roi_check in self.roi_checkboxes.items():
            if roi_check.isChecked():
                selected_rois.append(roi_name)

        config = {
            'bins': self.bins_spin.value(),
            'range': (self.range_min_spin.value(), self.range_max_spin.value()),
            'auto_range': self.auto_range_check.isChecked(),
            'rois': selected_rois,
            'orientations': {
                'reference': self.ref_check.isChecked() and self.ref_check.isEnabled(),
                'theta': self.theta_check.isChecked() and self.theta_check.isEnabled(),
                'phi': self.phi_check.isChecked() and self.phi_check.isEnabled()
            },
            'statistics': {
                'mean': self.show_mean_check.isChecked(),
                'std': self.show_deviation_check.isChecked(),
                'cv': self.show_cv_check.isChecked()
            }
        }
        return config


class TrajectoryHistogramDialog(QDialog):
    """Dialog for configuring fiber trajectory angle histogram."""
    def __init__(self, visualization_tab):
        super().__init__()
        self.visualization_tab = visualization_tab
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Trajectory Angle Histogram")
        self.setModal(True)
        self.resize(400, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Fiber Trajectory Histogram Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Bin Settings Group
        bin_group = QGroupBox("Bin Settings")
        bin_layout = QGridLayout(bin_group)

        bin_layout.addWidget(QLabel("Number of Bins:"), 0, 0)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(10, 200)
        self.bins_spin.setValue(50)
        bin_layout.addWidget(self.bins_spin, 0, 1)

        bin_layout.addWidget(QLabel("Range Min (°):"), 1, 0)
        self.range_min_spin = QDoubleSpinBox()
        self.range_min_spin.setRange(-180, 180)
        self.range_min_spin.setValue(0)
        self.range_min_spin.setDecimals(1)
        bin_layout.addWidget(self.range_min_spin, 1, 1)

        bin_layout.addWidget(QLabel("Range Max (°):"), 2, 0)
        self.range_max_spin = QDoubleSpinBox()
        self.range_max_spin.setRange(-180, 180)
        self.range_max_spin.setValue(30)
        self.range_max_spin.setDecimals(1)
        bin_layout.addWidget(self.range_max_spin, 2, 1)

        self.auto_range_check = QCheckBox("Auto Range")
        self.auto_range_check.setChecked(True)
        self.auto_range_check.toggled.connect(self.toggleAutoRange)
        bin_layout.addWidget(self.auto_range_check, 3, 0, 1, 2)

        layout.addWidget(bin_group)

        # ROI Selection Group
        roi_group = QGroupBox("Select ROIs")
        roi_layout = QVBoxLayout(roi_group)

        self.roi_checkboxes = {}

        # Add checkbox for each ROI that has trajectory data
        if hasattr(self.visualization_tab, 'roi_trajectories'):
            for roi_name in self.visualization_tab.roi_trajectories.keys():
                roi_check = QCheckBox(roi_name)
                roi_check.setChecked(True)
                self.roi_checkboxes[roi_name] = roi_check
                roi_layout.addWidget(roi_check)

        # If no ROI trajectories but has single trajectory
        if not self.roi_checkboxes and self.visualization_tab.fiber_trajectory is not None:
            single_check = QCheckBox("Current Trajectory")
            single_check.setChecked(True)
            self.roi_checkboxes['_single'] = single_check
            roi_layout.addWidget(single_check)

        # If no trajectories available
        if not self.roi_checkboxes:
            no_traj_label = QLabel("No trajectory data available")
            no_traj_label.setStyleSheet("color: gray; font-style: italic;")
            roi_layout.addWidget(no_traj_label)

        layout.addWidget(roi_group)

        # Angle Type Selection Group
        angle_group = QGroupBox("Select Angle Types")
        angle_layout = QVBoxLayout(angle_group)

        self.tilt_check = QCheckBox("Tilt Angle (from fiber axis)")
        self.tilt_check.setChecked(True)
        angle_layout.addWidget(self.tilt_check)

        self.azimuth_check = QCheckBox("Azimuth Angle (in cross-section)")
        self.azimuth_check.setChecked(False)
        angle_layout.addWidget(self.azimuth_check)

        self.xz_projection_check = QCheckBox("XZ Projection Angle")
        self.xz_projection_check.setChecked(False)
        angle_layout.addWidget(self.xz_projection_check)

        self.yz_projection_check = QCheckBox("YZ Projection Angle")
        self.yz_projection_check.setChecked(False)
        angle_layout.addWidget(self.yz_projection_check)

        layout.addWidget(angle_group)

        # Statistical Display Options Group
        stats_group = QGroupBox("Statistical Analysis")
        stats_layout = QVBoxLayout(stats_group)

        self.show_mean_check = QCheckBox("Show Mean")
        self.show_mean_check.setChecked(True)
        self.show_deviation_check = QCheckBox("Show Standard Deviation")
        self.show_deviation_check.setChecked(True)
        self.show_cv_check = QCheckBox("Show Coefficient of Variation (CV)")
        self.show_cv_check.setChecked(True)

        stats_layout.addWidget(self.show_mean_check)
        stats_layout.addWidget(self.show_deviation_check)
        stats_layout.addWidget(self.show_cv_check)

        layout.addWidget(stats_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.show_btn = QPushButton("Show")
        self.cancel_btn = QPushButton("Cancel")

        self.show_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.show_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Initialize auto range
        self.toggleAutoRange(True)

    def toggleAutoRange(self, checked):
        """Enable/disable manual range controls"""
        self.range_min_spin.setEnabled(not checked)
        self.range_max_spin.setEnabled(not checked)

    def getConfiguration(self):
        """Return the histogram configuration"""
        # Get selected ROIs
        selected_rois = []
        use_single = False
        for roi_name, roi_check in self.roi_checkboxes.items():
            if roi_check.isChecked():
                if roi_name == '_single':
                    use_single = True
                else:
                    selected_rois.append(roi_name)

        config = {
            'bins': self.bins_spin.value(),
            'range': (self.range_min_spin.value(), self.range_max_spin.value()),
            'auto_range': self.auto_range_check.isChecked(),
            'rois': selected_rois if selected_rois else None,
            'use_single_trajectory': use_single,
            'angles': {
                'tilt': self.tilt_check.isChecked(),
                'azimuth': self.azimuth_check.isChecked(),
                'xz_projection': self.xz_projection_check.isChecked(),
                'yz_projection': self.yz_projection_check.isChecked()
            },
            'statistics': {
                'mean': self.show_mean_check.isChecked(),
                'std': self.show_deviation_check.isChecked(),
                'cv': self.show_cv_check.isChecked()
            }
        }
        return config


class ColorBarRangeDialog(QDialog):
    def __init__(self, main_window, analysis_tab):
        super().__init__()
        self.main_window = main_window
        self.analysis_tab = analysis_tab
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Colormap & Range Editor")
        self.setModal(True)
        self.resize(450, 450)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Colormap & Range Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Intensity/Volume Range Group
        intensity_group = QGroupBox("Intensity (Base Volume)")
        intensity_layout = QGridLayout(intensity_group)

        self.intensity_min_edit = QDoubleSpinBox()
        self.intensity_max_edit = QDoubleSpinBox()
        self.intensity_auto_check = QCheckBox("Auto Range")

        # Set range for intensity controls
        self.intensity_min_edit.setRange(-999999, 999999)
        self.intensity_max_edit.setRange(-999999, 999999)
        self.intensity_min_edit.setDecimals(2)
        self.intensity_max_edit.setDecimals(2)

        intensity_layout.addWidget(QLabel("Min:"), 0, 0)
        intensity_layout.addWidget(self.intensity_min_edit, 0, 1)
        intensity_layout.addWidget(QLabel("Max:"), 1, 0)
        intensity_layout.addWidget(self.intensity_max_edit, 1, 1)
        intensity_layout.addWidget(self.intensity_auto_check, 2, 0, 1, 2)

        layout.addWidget(intensity_group)

        # Orientation Range Group
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QGridLayout(orientation_group)

        self.orientation_min_edit = QDoubleSpinBox()
        self.orientation_max_edit = QDoubleSpinBox()
        self.orientation_auto_check = QCheckBox("Auto Range")

        # Set range for orientation controls
        self.orientation_min_edit.setRange(-999999, 999999)
        self.orientation_max_edit.setRange(-999999, 999999)
        self.orientation_min_edit.setDecimals(2)
        self.orientation_max_edit.setDecimals(2)

        orientation_layout.addWidget(QLabel("Min:"), 0, 0)
        orientation_layout.addWidget(self.orientation_min_edit, 0, 1)
        orientation_layout.addWidget(QLabel("Max:"), 1, 0)
        orientation_layout.addWidget(self.orientation_max_edit, 1, 1)
        orientation_layout.addWidget(self.orientation_auto_check, 2, 0, 1, 2)

        layout.addWidget(orientation_group)

        # Orientation Colormap Selection Group
        colormap_group = QGroupBox("Orientation Colormap")
        colormap_layout = QFormLayout(colormap_group)

        colormap_options = ["hot", "cool", "jet", "rainbow", "viridis", "plasma",
                           "inferno", "magma", "coolwarm", "RdYlBu", "Spectral", "hsv"]

        # Single colormap selector for orientation histogram colorbar
        self.orientation_colormap_combo = QComboBox()
        self.orientation_colormap_combo.addItems(colormap_options)
        self.orientation_colormap_combo.setCurrentText(self.main_window.viewer.orientation_colormap)
        colormap_layout.addRow("Colormap:", self.orientation_colormap_combo)

        layout.addWidget(colormap_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset")
        self.close_btn = QPushButton("Close")

        self.apply_btn.clicked.connect(self.applyRanges)
        self.reset_btn.clicked.connect(self.resetRanges)
        self.close_btn.clicked.connect(self.accept)

        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Connect auto range checkboxes BEFORE initialization so toggle is triggered
        self.intensity_auto_check.toggled.connect(self.toggleIntensityAuto)
        self.orientation_auto_check.toggled.connect(self.toggleOrientationAuto)

        # Initialize values (will trigger toggle functions via signals)
        self.loadCurrentRanges()

    def loadCurrentRanges(self):
        """Load current color bar range values"""
        # Set default values
        if self.main_window.current_volume is not None:
            vol_min = float(np.min(self.main_window.current_volume))
            vol_max = float(np.max(self.main_window.current_volume))
            self.intensity_min_edit.setValue(vol_min)
            self.intensity_max_edit.setValue(vol_max)

        # Set orientation ranges if available (use first available orientation data)
        if hasattr(self.main_window, 'orientation_data') and self.main_window.orientation_data:
            active_orientation = None
            # Try to get any available orientation data
            for key in ['reference', 'theta', 'phi']:
                if key in self.main_window.orientation_data and self.main_window.orientation_data[key] is not None:
                    active_orientation = self.main_window.orientation_data[key]
                    break

            if active_orientation is not None:
                ori_min = float(np.nanmin(active_orientation))
                ori_max = float(np.nanmax(active_orientation))
                self.orientation_min_edit.setValue(ori_min)
                self.orientation_max_edit.setValue(ori_max)

        # Load saved auto range settings if available
        if hasattr(self.main_window, 'colorbar_ranges'):
            ranges = self.main_window.colorbar_ranges
            self.intensity_auto_check.setChecked(ranges.get('intensity_auto', True))
            self.orientation_auto_check.setChecked(ranges.get('orientation_auto', True))
        else:
            # Set auto range as default
            self.intensity_auto_check.setChecked(True)
            self.orientation_auto_check.setChecked(True)

    def toggleIntensityAuto(self, checked):
        """Toggle intensity auto range - disable/enable spinboxes"""
        self.intensity_min_edit.setEnabled(not checked)
        self.intensity_max_edit.setEnabled(not checked)

    def toggleOrientationAuto(self, checked):
        """Toggle orientation auto range - disable/enable spinboxes"""
        self.orientation_min_edit.setEnabled(not checked)
        self.orientation_max_edit.setEnabled(not checked)

    def applyRanges(self):
        """Apply the range settings to color bars"""
        # Store range settings in main window
        if not hasattr(self.main_window, 'colorbar_ranges'):
            self.main_window.colorbar_ranges = {}

        # Intensity ranges
        if self.intensity_auto_check.isChecked():
            self.main_window.colorbar_ranges['intensity_auto'] = True
        else:
            self.main_window.colorbar_ranges['intensity_auto'] = False
            self.main_window.colorbar_ranges['intensity_min'] = self.intensity_min_edit.value()
            self.main_window.colorbar_ranges['intensity_max'] = self.intensity_max_edit.value()

        # Orientation ranges
        if self.orientation_auto_check.isChecked():
            self.main_window.colorbar_ranges['orientation_auto'] = True
        else:
            self.main_window.colorbar_ranges['orientation_auto'] = False
            self.main_window.colorbar_ranges['orientation_min'] = self.orientation_min_edit.value()
            self.main_window.colorbar_ranges['orientation_max'] = self.orientation_max_edit.value()

        # Apply colormap selection
        self.main_window.viewer.orientation_colormap = self.orientation_colormap_combo.currentText()

        # Apply ranges directly to viewer
        ranges = self.main_window.colorbar_ranges
        if ranges.get('intensity_auto', True):
            self.main_window.viewer.intensity_range = None
        else:
            self.main_window.viewer.intensity_range = [ranges.get('intensity_min', 0), ranges.get('intensity_max', 255)]

        if ranges.get('orientation_auto', True):
            self.main_window.viewer.orientation_range = None
        else:
            self.main_window.viewer.orientation_range = [ranges.get('orientation_min', 0), ranges.get('orientation_max', 180)]

        # Update the current display
        self.main_window.viewer.renderVolume()

        self.main_window.status_label.setText("Color bar ranges and colormap updated")

    def resetRanges(self):
        """Reset ranges to auto"""
        self.intensity_auto_check.setChecked(True)
        self.orientation_auto_check.setChecked(True)
        self.loadCurrentRanges()


class ExportDialog(QDialog):
    """Dialog for exporting simulation results to Excel"""
    def __init__(self, parent=None, simulation_data=None):
        super().__init__(parent)
        self.setWindowTitle("Export to Excel")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.simulation_data = simulation_data or []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Data Selection Group
        data_group = QGroupBox("Select Data to Export")
        data_layout = QVBoxLayout(data_group)

        self.export_ss_curve = QCheckBox("Stress-Strain Curves")
        self.export_ss_curve.setChecked(True)
        data_layout.addWidget(self.export_ss_curve)

        self.export_results = QCheckBox("Results Summary (Strength, Strain)")
        self.export_results.setChecked(True)
        data_layout.addWidget(self.export_results)

        self.export_params = QCheckBox("Material Parameters")
        self.export_params.setChecked(True)
        data_layout.addWidget(self.export_params)

        layout.addWidget(data_group)

        # Case Selection Group
        case_group = QGroupBox("Select Cases to Export")
        case_layout = QVBoxLayout(case_group)

        self.case_checkboxes = []
        if self.simulation_data:
            for i, data in enumerate(self.simulation_data):
                case_name = data.get('case_name', f'Case {i+1}')
                checkbox = QCheckBox(case_name)
                checkbox.setChecked(True)
                self.case_checkboxes.append(checkbox)
                case_layout.addWidget(checkbox)
        else:
            no_data_label = QLabel("No simulation data available")
            no_data_label.setStyleSheet("color: gray; font-style: italic;")
            case_layout.addWidget(no_data_label)

        # Select All / Deselect All buttons
        if self.simulation_data:
            btn_layout = QHBoxLayout()
            select_all_btn = QPushButton("Select All")
            select_all_btn.clicked.connect(self.selectAllCases)
            btn_layout.addWidget(select_all_btn)

            deselect_all_btn = QPushButton("Deselect All")
            deselect_all_btn.clicked.connect(self.deselectAllCases)
            btn_layout.addWidget(deselect_all_btn)
            btn_layout.addStretch()
            case_layout.addLayout(btn_layout)

        layout.addWidget(case_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.export_btn = QPushButton("Export...")
        self.export_btn.clicked.connect(self.doExport)
        self.export_btn.setEnabled(bool(self.simulation_data))
        self.export_btn.setDefault(True)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    def selectAllCases(self):
        for cb in self.case_checkboxes:
            cb.setChecked(True)

    def deselectAllCases(self):
        for cb in self.case_checkboxes:
            cb.setChecked(False)

    def doExport(self):
        """Export data to Excel file"""
        # Get selected cases
        selected_indices = [i for i, cb in enumerate(self.case_checkboxes) if cb.isChecked()]
        if not selected_indices:
            QMessageBox.warning(self, "No Cases Selected", "Please select at least one case to export.")
            return

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel File", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return

        if not file_path.endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            import pandas as pd

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Export Results Summary
                if self.export_results.isChecked():
                    results_data = []
                    for i in selected_indices:
                        data = self.simulation_data[i]
                        results_data.append({
                            'Case Name': data.get('case_name', f'Case {i+1}'),
                            'Compressive Strength (MPa)': data.get('strength', 0),
                            'Ultimate Strain': data.get('strain', 0),
                            'Mode': data.get('mode', 'N/A')
                        })
                    df_results = pd.DataFrame(results_data)
                    df_results.to_excel(writer, sheet_name='Results', index=False)

                # Export Material Parameters
                if self.export_params.isChecked():
                    params_data = []
                    for i in selected_indices:
                        data = self.simulation_data[i]
                        params = data.get('material_params', {})
                        settings = data.get('settings', {})
                        params_data.append({
                            'Case Name': data.get('case_name', f'Case {i+1}'),
                            # Material Parameters
                            'E1 (MPa)': params.get('E1', 0),
                            'E2 (MPa)': params.get('E2', 0),
                            'Poisson Ratio': params.get('nu', 0),
                            'G (MPa)': params.get('G', 0),
                            'tau_y (MPa)': params.get('tau_y', 0),
                            'K': params.get('K', 0),
                            'n': params.get('n', 0),
                            # Fiber Orientation
                            'Initial Misalignment (°)': params.get('initial_misalignment', 'N/A'),
                            'Std Deviation (°)': params.get('std_deviation', 'N/A'),
                            # Simulation Settings
                            'Max Shear Stress (MPa)': settings.get('maximum_shear_stress', 'N/A'),
                            'Shear Stress Step (MPa)': settings.get('shear_stress_step_size', 'N/A'),
                            'Max Axial Strain': settings.get('maximum_axial_strain', 'N/A'),
                            'Max Fiber Misalignment (°)': settings.get('maximum_fiber_misalignment', 'N/A'),
                            'Fiber Misalignment Step (°)': settings.get('fiber_misalignment_step_size', 'N/A'),
                            # Strain Correction
                            'Kink Width (mm)': settings.get('kink_width') if settings.get('kink_width') is not None else 'N/A',
                            'Gauge Length (mm)': settings.get('gauge_length') if settings.get('gauge_length') is not None else 'N/A'
                        })
                    df_params = pd.DataFrame(params_data)
                    df_params.to_excel(writer, sheet_name='Parameters', index=False)

                # Export Stress-Strain Curves
                if self.export_ss_curve.isChecked():
                    ss_data = {}
                    max_len = 0
                    for i in selected_indices:
                        data = self.simulation_data[i]
                        case_name = data.get('case_name', f'Case {i+1}')
                        strain_array = data.get('strain_array', [])
                        stress_array = data.get('stress_array', [])
                        ss_data[f'{case_name}_Strain'] = strain_array
                        ss_data[f'{case_name}_Stress(MPa)'] = stress_array
                        max_len = max(max_len, len(strain_array))

                    # Pad shorter arrays with NaN
                    import numpy as np
                    for key in ss_data:
                        arr = ss_data[key]
                        if len(arr) < max_len:
                            ss_data[key] = np.pad(arr, (0, max_len - len(arr)),
                                                   mode='constant', constant_values=np.nan)

                    df_ss = pd.DataFrame(ss_data)
                    df_ss.to_excel(writer, sheet_name='Stress-Strain', index=False)

            QMessageBox.information(self, "Export Complete", f"Data exported successfully to:\n{file_path}")
            self.accept()

        except ImportError:
            QMessageBox.critical(self, "Error", "pandas and openpyxl are required for Excel export.\nInstall with: pip install pandas openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")


class FiberTrajectorySettingsDialog(QDialog):
    """Dialog for fiber trajectory generation settings"""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Fiber Trajectory Settings")
        self.setModal(True)
        self.setMinimumWidth(400)

        # Default settings
        self.settings = settings or {
            'fiber_diameter': 12.0,
            'volume_fraction': 0.5,
            'propagation_axis': 'Z (default)',
            'integration_method': 'RK4',
            'tilt_min': 0.0,
            'tilt_max': 20.0,
            'sat_min': 0.0,
            'sat_max': 20.0,
            'relax': True,
            'color_by_angle': True,
            'color_by_fiber': False,
            'show_fiber_diameter': False,
            'resample': False,
            'resample_interval': 20,
            'use_detected_centers': False,
            'detection_interval': 1,
            'max_matching_distance': 10.0,
            'add_new_fibers': False,
            'new_fiber_interval': 10,
            'smooth_trajectories': True,
            'smooth_method': 'gaussian',
            'smooth_sigma': 1.0,
            'smooth_window': 5
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Fiber Parameters Group
        fiber_group = QGroupBox("Fiber Parameters")
        fiber_layout = QFormLayout(fiber_group)

        self.fiber_diameter_spin = QDoubleSpinBox()
        self.fiber_diameter_spin.setRange(1.0, 50.0)
        self.fiber_diameter_spin.setValue(self.settings['fiber_diameter'])
        self.fiber_diameter_spin.setSuffix(" px")
        fiber_layout.addRow("Fiber Diameter:", self.fiber_diameter_spin)

        self.volume_fraction_spin = QDoubleSpinBox()
        self.volume_fraction_spin.setRange(0.1, 0.9)
        self.volume_fraction_spin.setValue(self.settings['volume_fraction'])
        self.volume_fraction_spin.setSingleStep(0.1)
        fiber_layout.addRow("Volume Fraction:", self.volume_fraction_spin)

        self.prop_axis_combo = QComboBox()
        self.prop_axis_combo.addItems(["Z (default)", "Y", "X"])
        self.prop_axis_combo.setCurrentText(self.settings['propagation_axis'])
        fiber_layout.addRow("Propagation Axis:", self.prop_axis_combo)

        self.integration_method_combo = QComboBox()
        self.integration_method_combo.addItems(["Euler", "RK4"])
        self.integration_method_combo.setCurrentText(self.settings.get('integration_method', 'RK4'))
        self.integration_method_combo.setToolTip(
            "Euler: 1st-order, fast but less accurate for curved trajectories\n"
            "RK4: 4th-order Runge-Kutta, slower but more accurate"
        )
        fiber_layout.addRow("Integration Method:", self.integration_method_combo)

        layout.addWidget(fiber_group)

        # Color Mapping Group
        color_group = QGroupBox("Color Mapping")
        color_layout = QFormLayout(color_group)

        # Tilt Range
        tilt_widget = QWidget()
        tilt_layout = QHBoxLayout(tilt_widget)
        tilt_layout.setContentsMargins(0, 0, 0, 0)
        self.tilt_min_spin = QDoubleSpinBox()
        self.tilt_min_spin.setRange(0, 90)
        self.tilt_min_spin.setValue(self.settings['tilt_min'])
        self.tilt_min_spin.setSuffix("°")
        tilt_layout.addWidget(self.tilt_min_spin)
        tilt_layout.addWidget(QLabel("-"))
        self.tilt_max_spin = QDoubleSpinBox()
        self.tilt_max_spin.setRange(0, 90)
        self.tilt_max_spin.setValue(self.settings['tilt_max'])
        self.tilt_max_spin.setSuffix("°")
        tilt_layout.addWidget(self.tilt_max_spin)
        color_layout.addRow("Tilt Range:", tilt_widget)

        # Saturation Range
        sat_widget = QWidget()
        sat_layout = QHBoxLayout(sat_widget)
        sat_layout.setContentsMargins(0, 0, 0, 0)
        self.sat_min_spin = QDoubleSpinBox()
        self.sat_min_spin.setRange(0, 90)
        self.sat_min_spin.setValue(self.settings['sat_min'])
        self.sat_min_spin.setSuffix("°")
        sat_layout.addWidget(self.sat_min_spin)
        sat_layout.addWidget(QLabel("-"))
        self.sat_max_spin = QDoubleSpinBox()
        self.sat_max_spin.setRange(0, 90)
        self.sat_max_spin.setValue(self.settings['sat_max'])
        self.sat_max_spin.setSuffix("°")
        sat_layout.addWidget(self.sat_max_spin)
        color_layout.addRow("Saturation Range:", sat_widget)

        layout.addWidget(color_group)

        # Options Group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.relax_check = QCheckBox("Maintain Fiber Distance")
        self.relax_check.setChecked(self.settings['relax'])
        options_layout.addWidget(self.relax_check)

        self.color_by_angle_check = QCheckBox("Color by Angle")
        self.color_by_angle_check.setChecked(self.settings['color_by_angle'])
        self.color_by_angle_check.toggled.connect(self._onColorModeChanged)
        options_layout.addWidget(self.color_by_angle_check)

        self.color_by_fiber_check = QCheckBox("Color by Fiber (each fiber has unique color)")
        self.color_by_fiber_check.setChecked(self.settings.get('color_by_fiber', False))
        self.color_by_fiber_check.setToolTip(
            "Assign a unique color to each fiber trajectory.\n"
            "Useful for tracking individual fibers across slices."
        )
        self.color_by_fiber_check.toggled.connect(self._onColorModeChanged)
        options_layout.addWidget(self.color_by_fiber_check)

        self.show_fiber_diameter_check = QCheckBox("Show Fiber Diameter in Slice Views")
        self.show_fiber_diameter_check.setChecked(self.settings['show_fiber_diameter'])
        options_layout.addWidget(self.show_fiber_diameter_check)

        layout.addWidget(options_group)

        # Resample Group
        resample_group = QGroupBox("Resampling")
        resample_layout = QFormLayout(resample_group)

        self.resample_check = QCheckBox("Enable Resampling")
        self.resample_check.setChecked(self.settings['resample'])
        resample_layout.addRow(self.resample_check)

        self.resample_interval_spin = QSpinBox()
        self.resample_interval_spin.setRange(5, 100)
        self.resample_interval_spin.setValue(self.settings['resample_interval'])
        self.resample_interval_spin.setSuffix(" slices")
        resample_layout.addRow("Interval:", self.resample_interval_spin)

        layout.addWidget(resample_group)

        # Image-based Tracking Group
        tracking_group = QGroupBox("Image-based Tracking")
        tracking_layout = QFormLayout(tracking_group)

        self.use_detected_centers_check = QCheckBox("Use Detected Fiber Centers")
        self.use_detected_centers_check.setChecked(self.settings.get('use_detected_centers', False))
        self.use_detected_centers_check.setToolTip(
            "Use fiber centers detected in Analysis tab as initial positions.\n"
            "Trajectory tracking will match predicted positions to detected centers."
        )
        self.use_detected_centers_check.toggled.connect(self._onDetectedCentersToggled)
        tracking_layout.addRow(self.use_detected_centers_check)

        self.detection_interval_spin = QSpinBox()
        self.detection_interval_spin.setRange(1, 20)
        self.detection_interval_spin.setValue(self.settings.get('detection_interval', 1))
        self.detection_interval_spin.setSuffix(" slices")
        self.detection_interval_spin.setToolTip(
            "Interval between nearest-neighbor matching to detected centers.\n"
            "1 = match every slice, higher = faster but less accurate"
        )
        tracking_layout.addRow("Detection Interval:", self.detection_interval_spin)

        self.max_matching_distance_spin = QDoubleSpinBox()
        self.max_matching_distance_spin.setRange(1.0, 50.0)
        self.max_matching_distance_spin.setValue(self.settings.get('max_matching_distance', 10.0))
        self.max_matching_distance_spin.setSuffix(" px")
        self.max_matching_distance_spin.setToolTip(
            "Maximum distance for matching predicted position to detected center.\n"
            "If no center is within this distance, prediction is kept."
        )
        tracking_layout.addRow("Max Matching Distance:", self.max_matching_distance_spin)

        self.add_new_fibers_check = QCheckBox("Add New Fibers from Detection")
        self.add_new_fibers_check.setChecked(self.settings.get('add_new_fibers', False))
        self.add_new_fibers_check.setToolTip(
            "Add unmatched detected centers as new fiber trajectories.\n"
            "Useful for tracking fibers that enter the domain from boundaries."
        )
        self.add_new_fibers_check.toggled.connect(self._onAddNewFibersToggled)
        tracking_layout.addRow(self.add_new_fibers_check)

        self.new_fiber_interval_spin = QSpinBox()
        self.new_fiber_interval_spin.setRange(1, 50)
        self.new_fiber_interval_spin.setValue(self.settings.get('new_fiber_interval', 10))
        self.new_fiber_interval_spin.setSuffix(" slices")
        self.new_fiber_interval_spin.setToolTip(
            "Interval at which to check for new fibers from unmatched detections.\n"
            "Lower = more frequent checks, higher = less overhead"
        )
        tracking_layout.addRow("New Fiber Interval:", self.new_fiber_interval_spin)

        layout.addWidget(tracking_group)

        # Store reference to resample group for enabling/disabling
        self.resample_group = resample_group

        # Initially update UI based on detected centers option
        self._onDetectedCentersToggled(self.use_detected_centers_check.isChecked())

        # Trajectory Smoothing Group
        smooth_group = QGroupBox("Trajectory Smoothing")
        smooth_layout = QFormLayout(smooth_group)

        self.smooth_check = QCheckBox("Apply Smoothing")
        self.smooth_check.setChecked(self.settings.get('smooth_trajectories', True))
        self.smooth_check.setToolTip(
            "Apply smoothing to reduce trajectory oscillation.\n"
            "Recommended when using image-based tracking."
        )
        self.smooth_check.toggled.connect(self._onSmoothToggled)
        smooth_layout.addRow(self.smooth_check)

        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItems(["gaussian", "moving_average"])
        self.smooth_method_combo.setCurrentText(self.settings.get('smooth_method', 'gaussian'))
        self.smooth_method_combo.setToolTip(
            "gaussian: Smooth using Gaussian filter (sigma parameter)\n"
            "moving_average: Simple moving average (window size parameter)"
        )
        self.smooth_method_combo.currentTextChanged.connect(self._onSmoothMethodChanged)
        smooth_layout.addRow("Method:", self.smooth_method_combo)

        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.5, 10.0)
        self.smooth_sigma_spin.setValue(self.settings.get('smooth_sigma', 1.0))
        self.smooth_sigma_spin.setSingleStep(0.5)
        self.smooth_sigma_spin.setToolTip("Gaussian filter sigma (larger = more smoothing)")
        smooth_layout.addRow("Sigma:", self.smooth_sigma_spin)

        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(3, 21)
        self.smooth_window_spin.setSingleStep(2)
        self.smooth_window_spin.setValue(self.settings.get('smooth_window', 5))
        self.smooth_window_spin.setToolTip("Moving average window size (odd number, larger = more smoothing)")
        smooth_layout.addRow("Window Size:", self.smooth_window_spin)

        # Initially update UI based on current settings
        self._onSmoothToggled(self.smooth_check.isChecked())
        self._onSmoothMethodChanged(self.smooth_method_combo.currentText())

        layout.addWidget(smooth_group)

        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def _onDetectedCentersToggled(self, checked):
        """Enable/disable options based on whether detected centers are used.

        When using detected fiber centers:
        - Volume Fraction is disabled (fiber count comes from detection)
        - Integration Method is disabled (always uses RK4 with detection)
        - Resampling is disabled (not applicable with detection-based tracking)
        - Detection interval and max matching distance are enabled
        """
        # Enable/disable tracking-specific options
        self.detection_interval_spin.setEnabled(checked)
        self.max_matching_distance_spin.setEnabled(checked)
        self.add_new_fibers_check.setEnabled(checked)
        self.new_fiber_interval_spin.setEnabled(checked and self.add_new_fibers_check.isChecked())

        # Disable options that don't apply when using detected centers
        self.volume_fraction_spin.setEnabled(not checked)
        self.integration_method_combo.setEnabled(not checked)
        self.resample_group.setEnabled(not checked)

    def _onAddNewFibersToggled(self, checked):
        """Enable/disable new fiber interval based on add_new_fibers checkbox."""
        self.new_fiber_interval_spin.setEnabled(checked and self.use_detected_centers_check.isChecked())

    def _onColorModeChanged(self, checked):
        """Handle mutual exclusivity between color_by_angle and color_by_fiber."""
        sender = self.sender()
        if sender == self.color_by_angle_check and checked:
            # If color_by_angle is checked, uncheck color_by_fiber
            self.color_by_fiber_check.blockSignals(True)
            self.color_by_fiber_check.setChecked(False)
            self.color_by_fiber_check.blockSignals(False)
        elif sender == self.color_by_fiber_check and checked:
            # If color_by_fiber is checked, uncheck color_by_angle
            self.color_by_angle_check.blockSignals(True)
            self.color_by_angle_check.setChecked(False)
            self.color_by_angle_check.blockSignals(False)

    def _onSmoothToggled(self, checked):
        """Enable/disable smoothing options based on checkbox state"""
        self.smooth_method_combo.setEnabled(checked)
        self.smooth_sigma_spin.setEnabled(checked)
        self.smooth_window_spin.setEnabled(checked)
        if checked:
            self._onSmoothMethodChanged(self.smooth_method_combo.currentText())

    def _onSmoothMethodChanged(self, method):
        """Show/hide relevant smoothing parameters based on method"""
        if not self.smooth_check.isChecked():
            return
        if method == 'gaussian':
            self.smooth_sigma_spin.setEnabled(True)
            self.smooth_window_spin.setEnabled(False)
        else:  # moving_average
            self.smooth_sigma_spin.setEnabled(False)
            self.smooth_window_spin.setEnabled(True)

    def getSettings(self):
        """Return the current settings"""
        return {
            'fiber_diameter': self.fiber_diameter_spin.value(),
            'volume_fraction': self.volume_fraction_spin.value(),
            'propagation_axis': self.prop_axis_combo.currentText(),
            'integration_method': self.integration_method_combo.currentText(),
            'tilt_min': self.tilt_min_spin.value(),
            'tilt_max': self.tilt_max_spin.value(),
            'sat_min': self.sat_min_spin.value(),
            'sat_max': self.sat_max_spin.value(),
            'relax': self.relax_check.isChecked(),
            'color_by_angle': self.color_by_angle_check.isChecked(),
            'color_by_fiber': self.color_by_fiber_check.isChecked(),
            'show_fiber_diameter': self.show_fiber_diameter_check.isChecked(),
            'resample': self.resample_check.isChecked(),
            'resample_interval': self.resample_interval_spin.value(),
            'use_detected_centers': self.use_detected_centers_check.isChecked(),
            'detection_interval': self.detection_interval_spin.value(),
            'max_matching_distance': self.max_matching_distance_spin.value(),
            'add_new_fibers': self.add_new_fibers_check.isChecked(),
            'new_fiber_interval': self.new_fiber_interval_spin.value(),
            'smooth_trajectories': self.smooth_check.isChecked(),
            'smooth_method': self.smooth_method_combo.currentText(),
            'smooth_sigma': self.smooth_sigma_spin.value(),
            'smooth_window': self.smooth_window_spin.value()
        }


class InSegtSettingsDialog(QDialog):
    """Dialog for InSegt (Interactive Segmentation) settings."""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("InSegt Settings")
        self.setModal(True)
        self.setMinimumWidth(350)

        self.settings = settings or {
            'scale': 0.5,
            'sigmas': [1, 2],
            'patch_size': 9,
            'branching_factor': 5,
            'number_layers': 4,
            'training_patches': 10000
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Processing Scale
        scale_group = QGroupBox("Processing Scale")
        scale_layout = QFormLayout(scale_group)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["0.25 (fastest)", "0.5 (balanced)", "0.75", "1.0 (full resolution)"])
        scale_val = self.settings.get('scale', 0.5)
        if scale_val <= 0.25:
            self.scale_combo.setCurrentIndex(0)
        elif scale_val <= 0.5:
            self.scale_combo.setCurrentIndex(1)
        elif scale_val <= 0.75:
            self.scale_combo.setCurrentIndex(2)
        else:
            self.scale_combo.setCurrentIndex(3)
        self.scale_combo.setToolTip(
            "Image scale for processing:\n"
            "0.25 = ~16x faster (lower accuracy)\n"
            "0.5 = ~4x faster (good balance)\n"
            "1.0 = full resolution (slowest, best accuracy)"
        )
        scale_layout.addRow("Scale:", self.scale_combo)

        layout.addWidget(scale_group)

        # Model Parameters
        model_group = QGroupBox("Model Parameters")
        model_layout = QFormLayout(model_group)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(5, 15)
        self.patch_size_spin.setSingleStep(2)
        self.patch_size_spin.setValue(self.settings.get('patch_size', 9))
        self.patch_size_spin.setToolTip("Patch size for KM-tree (odd number)")
        model_layout.addRow("Patch Size:", self.patch_size_spin)

        self.branching_factor_spin = QSpinBox()
        self.branching_factor_spin.setRange(2, 10)
        self.branching_factor_spin.setValue(self.settings.get('branching_factor', 5))
        self.branching_factor_spin.setToolTip("Branching factor for KM-tree")
        model_layout.addRow("Branching Factor:", self.branching_factor_spin)

        self.number_layers_spin = QSpinBox()
        self.number_layers_spin.setRange(2, 8)
        self.number_layers_spin.setValue(self.settings.get('number_layers', 4))
        self.number_layers_spin.setToolTip("Number of layers in KM-tree")
        model_layout.addRow("Number of Layers:", self.number_layers_spin)

        self.training_patches_spin = QSpinBox()
        self.training_patches_spin.setRange(1000, 50000)
        self.training_patches_spin.setSingleStep(1000)
        self.training_patches_spin.setValue(self.settings.get('training_patches', 10000))
        self.training_patches_spin.setToolTip("Number of training patches for KM-tree")
        model_layout.addRow("Training Patches:", self.training_patches_spin)

        layout.addWidget(model_group)

        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def getSettings(self):
        """Return the current settings."""
        scale_text = self.scale_combo.currentText()
        if "0.25" in scale_text:
            scale = 0.25
        elif "0.5" in scale_text:
            scale = 0.5
        elif "0.75" in scale_text:
            scale = 0.75
        else:
            scale = 1.0

        return {
            'scale': scale,
            'sigmas': [1, 2],  # Fixed for now
            'patch_size': self.patch_size_spin.value(),
            'branching_factor': self.branching_factor_spin.value(),
            'number_layers': self.number_layers_spin.value(),
            'training_patches': self.training_patches_spin.value()
        }


class FiberDetectionSettingsDialog(QDialog):
    """Dialog for fiber detection settings in Analysis tab"""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Fiber Detection Settings")
        self.setModal(True)
        self.setMinimumWidth(400)

        # Default settings
        self.settings = settings or {
            'min_diameter': 5.0,
            'max_diameter': 25.0,
            'min_distance': 5,
            'threshold_method': 'otsu',
            'threshold_percentile': 50.0,
            'show_watershed': True,
            'show_centers': True,
            'center_marker_size': 3
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Detection Parameters Group
        detect_group = QGroupBox("Detection Parameters")
        detect_layout = QFormLayout(detect_group)

        # Diameter range
        diameter_widget = QWidget()
        diameter_layout = QHBoxLayout(diameter_widget)
        diameter_layout.setContentsMargins(0, 0, 0, 0)

        self.min_diameter_spin = QDoubleSpinBox()
        self.min_diameter_spin.setRange(1.0, 50.0)
        self.min_diameter_spin.setValue(self.settings['min_diameter'])
        self.min_diameter_spin.setSuffix(" px")
        self.min_diameter_spin.setToolTip("Minimum fiber diameter to detect")
        diameter_layout.addWidget(self.min_diameter_spin)

        diameter_layout.addWidget(QLabel("-"))

        self.max_diameter_spin = QDoubleSpinBox()
        self.max_diameter_spin.setRange(1.0, 100.0)
        self.max_diameter_spin.setValue(self.settings['max_diameter'])
        self.max_diameter_spin.setSuffix(" px")
        self.max_diameter_spin.setToolTip("Maximum fiber diameter to detect")
        diameter_layout.addWidget(self.max_diameter_spin)

        detect_layout.addRow("Diameter Range:", diameter_widget)

        self.min_distance_spin = QSpinBox()
        self.min_distance_spin.setRange(1, 50)
        self.min_distance_spin.setValue(self.settings['min_distance'])
        self.min_distance_spin.setSuffix(" px")
        self.min_distance_spin.setToolTip("Minimum distance between fiber centers")
        detect_layout.addRow("Min Peak Distance:", self.min_distance_spin)

        layout.addWidget(detect_group)

        # Threshold Group
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QFormLayout(threshold_group)

        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems(["otsu", "percentile"])
        self.threshold_method_combo.setCurrentText(self.settings.get('threshold_method', 'otsu'))
        self.threshold_method_combo.setToolTip(
            "otsu: Automatic threshold using Otsu's method\n"
            "percentile: Manual threshold based on pixel intensity percentile"
        )
        self.threshold_method_combo.currentTextChanged.connect(self._onThresholdMethodChanged)
        threshold_layout.addRow("Method:", self.threshold_method_combo)

        self.threshold_percentile_spin = QDoubleSpinBox()
        self.threshold_percentile_spin.setRange(1.0, 99.0)
        self.threshold_percentile_spin.setValue(self.settings.get('threshold_percentile', 50.0))
        self.threshold_percentile_spin.setSuffix(" %")
        self.threshold_percentile_spin.setToolTip(
            "Percentile value for thresholding.\n"
            "Higher values = stricter threshold (fewer fibers detected)\n"
            "Lower values = looser threshold (more fibers detected)"
        )
        threshold_layout.addRow("Percentile:", self.threshold_percentile_spin)

        # Initialize percentile spin state
        self._onThresholdMethodChanged(self.threshold_method_combo.currentText())

        layout.addWidget(threshold_group)

        # Visualization Group
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)

        self.show_watershed_check = QCheckBox("Show Watershed Regions")
        self.show_watershed_check.setChecked(self.settings['show_watershed'])
        self.show_watershed_check.setToolTip("Display colored watershed segmentation regions")
        viz_layout.addRow(self.show_watershed_check)

        self.show_centers_check = QCheckBox("Show Fiber Centers")
        self.show_centers_check.setChecked(self.settings['show_centers'])
        self.show_centers_check.setToolTip("Display detected fiber center points")
        viz_layout.addRow(self.show_centers_check)

        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(self.settings['center_marker_size'])
        self.marker_size_spin.setSuffix(" px")
        self.marker_size_spin.setToolTip("Size of center marker points")
        viz_layout.addRow("Marker Size:", self.marker_size_spin)

        layout.addWidget(viz_group)

        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def _onThresholdMethodChanged(self, method):
        """Enable/disable percentile spin based on threshold method"""
        self.threshold_percentile_spin.setEnabled(method == 'percentile')

    def getSettings(self):
        """Return the current settings"""
        return {
            'min_diameter': self.min_diameter_spin.value(),
            'max_diameter': self.max_diameter_spin.value(),
            'min_distance': self.min_distance_spin.value(),
            'threshold_method': self.threshold_method_combo.currentText(),
            'threshold_percentile': self.threshold_percentile_spin.value(),
            'show_watershed': self.show_watershed_check.isChecked(),
            'show_centers': self.show_centers_check.isChecked(),
            'center_marker_size': self.marker_size_spin.value()
        }


class SimulationSettingsDialog(QDialog):
    """Dialog for advanced simulation settings"""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.setModal(True)
        self.setMinimumWidth(350)

        # Default settings
        self.settings = settings or {
            'maximum_shear_stress': 100.0,
            'shear_stress_step_size': 0.1,
            'maximum_axial_strain': 0.02,
            'maximum_fiber_misalignment': 20.0,
            'fiber_misalignment_step_size': 0.1,
            'kink_width': None,
            'gauge_length': None
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Shear Stress Group
        shear_group = QGroupBox("Shear Stress Parameters")
        shear_layout = QFormLayout(shear_group)

        self.max_shear_spin = QDoubleSpinBox()
        self.max_shear_spin.setRange(10, 500)
        self.max_shear_spin.setValue(self.settings['maximum_shear_stress'])
        self.max_shear_spin.setSuffix(" MPa")
        self.max_shear_spin.setDecimals(1)
        shear_layout.addRow("Maximum Shear Stress:", self.max_shear_spin)

        self.shear_step_spin = QDoubleSpinBox()
        self.shear_step_spin.setRange(0.01, 1.0)
        self.shear_step_spin.setValue(self.settings['shear_stress_step_size'])
        self.shear_step_spin.setSuffix(" MPa")
        self.shear_step_spin.setDecimals(2)
        self.shear_step_spin.setSingleStep(0.01)
        shear_layout.addRow("Step Size:", self.shear_step_spin)

        layout.addWidget(shear_group)

        # Axial Strain Group
        strain_group = QGroupBox("Axial Strain Parameters")
        strain_layout = QFormLayout(strain_group)

        self.max_strain_spin = QDoubleSpinBox()
        self.max_strain_spin.setRange(0.001, 0.1)
        self.max_strain_spin.setValue(self.settings['maximum_axial_strain'])
        self.max_strain_spin.setDecimals(4)
        self.max_strain_spin.setSingleStep(0.001)
        strain_layout.addRow("Maximum Axial Strain:", self.max_strain_spin)

        layout.addWidget(strain_group)

        # Fiber Misalignment Group
        misalign_group = QGroupBox("Fiber Misalignment Parameters")
        misalign_layout = QFormLayout(misalign_group)

        self.max_misalign_spin = QDoubleSpinBox()
        self.max_misalign_spin.setRange(5, 90)
        self.max_misalign_spin.setValue(self.settings['maximum_fiber_misalignment'])
        self.max_misalign_spin.setSuffix(" °")
        self.max_misalign_spin.setDecimals(1)
        misalign_layout.addRow("Maximum Misalignment:", self.max_misalign_spin)

        self.misalign_step_spin = QDoubleSpinBox()
        self.misalign_step_spin.setRange(0.01, 1.0)
        self.misalign_step_spin.setValue(self.settings['fiber_misalignment_step_size'])
        self.misalign_step_spin.setSuffix(" °")
        self.misalign_step_spin.setDecimals(2)
        self.misalign_step_spin.setSingleStep(0.01)
        misalign_layout.addRow("Step Size:", self.misalign_step_spin)

        layout.addWidget(misalign_group)

        # Strain Correction Group (Kink Width / Gauge Length)
        correction_group = QGroupBox("Strain Correction (Optional)")
        correction_layout = QFormLayout(correction_group)

        self.use_correction_check = QCheckBox("Enable strain correction")
        self.use_correction_check.setChecked(self.settings.get('kink_width') is not None)
        self.use_correction_check.stateChanged.connect(self.onCorrectionToggled)
        correction_layout.addRow(self.use_correction_check)

        self.kink_width_spin = QDoubleSpinBox()
        self.kink_width_spin.setRange(0.001, 100)
        self.kink_width_spin.setValue(self.settings.get('kink_width') or 0.1)  # Default 100 μm
        self.kink_width_spin.setSuffix(" mm")
        self.kink_width_spin.setDecimals(3)
        self.kink_width_spin.setSingleStep(0.1)
        correction_layout.addRow("Kink Width (w_k):", self.kink_width_spin)

        self.gauge_length_spin = QDoubleSpinBox()
        self.gauge_length_spin.setRange(0.1, 1000)
        self.gauge_length_spin.setValue(self.settings.get('gauge_length') or 10.0)
        self.gauge_length_spin.setSuffix(" mm")
        self.gauge_length_spin.setDecimals(2)
        self.gauge_length_spin.setSingleStep(1.0)
        correction_layout.addRow("Gauge Length (L_g):", self.gauge_length_spin)

        # Set initial enabled state based on checkbox
        is_checked = self.use_correction_check.isChecked()
        self.kink_width_spin.setEnabled(is_checked)
        self.gauge_length_spin.setEnabled(is_checked)

        layout.addWidget(correction_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.resetToDefaults)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)

    def onCorrectionToggled(self, state):
        """Toggle strain correction inputs"""
        enabled = (state == Qt.Checked.value) if hasattr(Qt.Checked, 'value') else (state == 2)
        self.kink_width_spin.setEnabled(enabled)
        self.gauge_length_spin.setEnabled(enabled)

    def resetToDefaults(self):
        """Reset all values to defaults"""
        self.max_shear_spin.setValue(100.0)
        self.shear_step_spin.setValue(0.1)
        self.max_strain_spin.setValue(0.02)
        self.max_misalign_spin.setValue(20.0)
        self.misalign_step_spin.setValue(0.1)
        self.use_correction_check.setChecked(False)
        self.kink_width_spin.setValue(0.1)  # 100 μm
        self.gauge_length_spin.setValue(10.0)

    def getSettings(self):
        """Return current settings"""
        use_correction = self.use_correction_check.isChecked()
        return {
            'maximum_shear_stress': self.max_shear_spin.value(),
            'shear_stress_step_size': self.shear_step_spin.value(),
            'maximum_axial_strain': self.max_strain_spin.value(),
            'maximum_fiber_misalignment': self.max_misalign_spin.value(),
            'fiber_misalignment_step_size': self.misalign_step_spin.value(),
            'kink_width': self.kink_width_spin.value() if use_correction else None,
            'gauge_length': self.gauge_length_spin.value() if use_correction else None
        }


class SimulationTab(QWidget):
    """Toolbar portion of Simulation tab (displayed in tab area)"""
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.main_window = None
        # Store simulation settings
        self.simulation_settings = {
            'maximum_shear_stress': 100.0,
            'shear_stress_step_size': 0.1,
            'maximum_axial_strain': 0.02,
            'maximum_fiber_misalignment': 20.0,
            'fiber_misalignment_step_size': 0.1,
            'kink_width': None,
            'gauge_length': None
        }
        # Store simulation results for export
        self.simulation_results = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Ribbon toolbar with gray frame (matching other tabs)
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #f0f0f0; border-bottom: 1px solid #d0d0d0; }")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(10)

        # Settings Group (first)
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)

        self.settings_btn = RibbonButton("Settings")
        self.settings_btn.clicked.connect(self.openSettingsDialog)
        settings_layout.addWidget(self.settings_btn)

        toolbar_layout.addWidget(settings_group)

        # Simulation Group
        sim_group = QGroupBox("Simulation")
        sim_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        sim_layout = QHBoxLayout(sim_group)

        self.run_btn = RibbonButton("Run\nSimulation")
        self.run_btn.clicked.connect(self.runSimulation)
        sim_layout.addWidget(self.run_btn)

        self.clear_btn = RibbonButton("Clear\nGraph")
        self.clear_btn.clicked.connect(self.clearGraph)
        sim_layout.addWidget(self.clear_btn)

        toolbar_layout.addWidget(sim_group)

        # Analysis Group (Histogram)
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QVBoxLayout(analysis_group)

        self.histogram_btn = RibbonButton("Histogram")
        self.histogram_btn.clicked.connect(self.openHistogramDialog)
        self.histogram_btn.setEnabled(False)  # Enable after orientation data is available
        analysis_layout.addWidget(self.histogram_btn)

        toolbar_layout.addWidget(analysis_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QVBoxLayout(export_group)

        self.export_btn = RibbonButton("Export\nXLSX")
        self.export_btn.clicked.connect(self.openExportDialog)
        export_layout.addWidget(self.export_btn)

        toolbar_layout.addWidget(export_group)
        toolbar_layout.addStretch()

        layout.addWidget(toolbar)
        layout.addStretch()

    def setViewer(self, viewer):
        """Set the viewer reference (for future use)"""
        self.viewer = viewer

    def setMainWindow(self, main_window):
        """Set the main window reference"""
        self.main_window = main_window

    def openHistogramDialog(self):
        """Open histogram configuration dialog"""
        if not self.main_window:
            return

        # Get analysis tab for orientation data
        analysis_tab = self.main_window.analysis_tab

        # Create and show histogram dialog
        dialog = HistogramDialog(self.main_window, analysis_tab)
        if dialog.exec() == QDialog.Accepted:
            # Show histogram panel in Simulation content panel
            config = dialog.getConfiguration()
            sim_content = self.main_window.simulation_content
            sim_content.histogram_panel.setVisible(True)
            # Pass viewer.rois if ROIs are selected, otherwise pass orientation_data
            if config.get('rois'):
                sim_content.histogram_panel.plotHistogramMultiROI(config, self.main_window.viewer.rois)
            else:
                sim_content.histogram_panel.plotHistogram(config, self.main_window.orientation_data)

    def openSettingsDialog(self):
        """Open simulation settings dialog"""
        dialog = SimulationSettingsDialog(self, self.simulation_settings)
        if dialog.exec() == QDialog.Accepted:
            self.simulation_settings = dialog.getSettings()

    def clearGraph(self):
        """Clear the stress-strain graph and simulation results"""
        if self.main_window:
            self.main_window.simulation_content.clearGraph()
            self.simulation_results.clear()
            self.main_window.status_label.setText("Graph cleared")

    def openExportDialog(self):
        """Open export dialog"""
        dialog = ExportDialog(self, self.simulation_results)
        dialog.exec()

    def runSimulation(self):
        """Run compression strength simulation"""
        if not self.main_window:
            return

        from acsc.simulation import estimate_compression_strength, estimate_compression_strength_from_profile, MaterialParams

        sim_content = self.main_window.simulation_content

        # Get material parameters from UI
        material = MaterialParams(
            longitudinal_modulus=sim_content.e1_spin.value(),
            transverse_modulus=sim_content.e2_spin.value(),
            poisson_ratio=sim_content.nu_spin.value(),
            shear_modulus=sim_content.g_spin.value(),
            tau_y=sim_content.tau_y_spin.value(),
            K=sim_content.k_spin.value(),
            n=sim_content.n_spin.value()
        )

        # Get simulation settings
        settings = self.simulation_settings

        # Check if using 3D orientation data
        use_3d_orientation = sim_content.use_3d_orientation_check.isChecked()

        self.main_window.status_label.setText("Running simulation...")
        QApplication.processEvents()

        try:
            if use_3d_orientation:
                # Use 3D orientation data from analysis
                orientation_data = self.main_window.orientation_data
                if orientation_data['reference'] is None:
                    raise ValueError("No 3D orientation data available. Please run orientation analysis first.")

                # Run simulation with measured orientation profile
                strength, strain, stress_curve, strain_array = estimate_compression_strength_from_profile(
                    orientation_profile=orientation_data['reference'],
                    material_params=material,
                    maximum_shear_stress=settings['maximum_shear_stress'],
                    shear_stress_step_size=settings['shear_stress_step_size'],
                    maximum_axial_strain=settings['maximum_axial_strain'],
                    maximum_fiber_misalignment=settings['maximum_fiber_misalignment'],
                    fiber_misalignment_step_size=settings['fiber_misalignment_step_size'],
                    kink_width=settings.get('kink_width'),
                    gauge_length=settings.get('gauge_length')
                )
            else:
                # Use manual input with Gaussian distribution
                initial_misalignment = sim_content.initial_misalignment_spin.value()
                std_deviation = sim_content.std_deviation_spin.value()

                strength, strain, stress_curve, strain_array = estimate_compression_strength(
                    initial_misalignment=initial_misalignment,
                    standard_deviation=std_deviation,
                    material_params=material,
                    maximum_shear_stress=settings['maximum_shear_stress'],
                    shear_stress_step_size=settings['shear_stress_step_size'],
                    maximum_axial_strain=settings['maximum_axial_strain'],
                    maximum_fiber_misalignment=settings['maximum_fiber_misalignment'],
                    fiber_misalignment_step_size=settings['fiber_misalignment_step_size'],
                    kink_width=settings.get('kink_width'),
                    gauge_length=settings.get('gauge_length')
                )

            # Update results
            sim_content.updateResults(strength, strain)

            # Get case name and plot stress-strain curve
            case_name = sim_content.case_name_edit.text().strip()
            if not case_name:
                case_name = f"Case {len(sim_content.ax.lines) + 1}"
            sim_content.plotStressStrain(strain_array, stress_curve, case_name)

            mode = "3D data" if use_3d_orientation else "Gaussian"

            # Store simulation result for export
            result_data = {
                'case_name': case_name,
                'strength': strength,
                'strain': strain,
                'stress_array': stress_curve,
                'strain_array': strain_array,
                'mode': mode,
                'material_params': {
                    'E1': sim_content.e1_spin.value(),
                    'E2': sim_content.e2_spin.value(),
                    'nu': sim_content.nu_spin.value(),
                    'G': sim_content.g_spin.value(),
                    'tau_y': sim_content.tau_y_spin.value(),
                    'K': sim_content.k_spin.value(),
                    'n': sim_content.n_spin.value(),
                    'initial_misalignment': sim_content.initial_misalignment_spin.value() if not use_3d_orientation else 'N/A',
                    'std_deviation': sim_content.std_deviation_spin.value() if not use_3d_orientation else 'N/A'
                },
                'settings': {
                    'maximum_shear_stress': settings['maximum_shear_stress'],
                    'shear_stress_step_size': settings['shear_stress_step_size'],
                    'maximum_axial_strain': settings['maximum_axial_strain'],
                    'maximum_fiber_misalignment': settings['maximum_fiber_misalignment'],
                    'fiber_misalignment_step_size': settings['fiber_misalignment_step_size'],
                    'kink_width': settings.get('kink_width'),
                    'gauge_length': settings.get('gauge_length')
                }
            }
            self.simulation_results.append(result_data)

            self.main_window.status_label.setText(
                f"Simulation complete ({mode}): Strength = {strength:.2f} MPa, Strain = {strain:.4f}"
            )

        except ValueError as e:
            self.main_window.status_label.setText(f"Simulation error: {str(e)}")
            QMessageBox.warning(self.main_window, "Simulation Error", str(e))
        except Exception as e:
            self.main_window.status_label.setText(f"Simulation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Simulation failed:\n{str(e)}")


class SimulationContentPanel(QWidget):
    """Main content panel for Simulation (displayed below tabs)"""
    def __init__(self):
        super().__init__()
        self.material_presets = {}
        self.loadMaterialPresets()
        self.initUI()

    def loadMaterialPresets(self):
        """Load material presets from JSON file"""
        import json
        preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'material_params.json')
        try:
            with open(preset_path, 'r') as f:
                self.material_presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load material presets: {e}")
            self.material_presets = {}

    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Left panel - Material Parameters
        left_panel = QWidget()
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Material Parameters Group
        group_style = """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

        # Case Name Group
        case_group = QGroupBox("Case")
        case_group.setStyleSheet(group_style)
        case_layout = QFormLayout(case_group)
        case_layout.setSpacing(8)
        case_layout.setContentsMargins(10, 20, 10, 10)

        self.case_name_edit = QLineEdit()
        self.case_name_edit.setPlaceholderText("Enter case name...")
        self.case_name_edit.setText("Case 1")
        case_layout.addRow("Name:", self.case_name_edit)

        left_layout.addWidget(case_group)

        material_group = QGroupBox("Material Parameters")
        material_group.setStyleSheet(group_style)
        material_layout = QFormLayout(material_group)
        material_layout.setSpacing(8)
        material_layout.setContentsMargins(10, 20, 10, 10)

        # Preset Selection
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom")
        for name in self.material_presets.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self.onPresetChanged)
        material_layout.addRow("Preset:", self.preset_combo)

        # Longitudinal Modulus (E1)
        self.e1_spin = QDoubleSpinBox()
        self.e1_spin.setRange(0, 500000)
        self.e1_spin.setValue(0)
        self.e1_spin.setSuffix(" MPa")
        self.e1_spin.setDecimals(0)
        material_layout.addRow("Longitudinal Modulus (E₁):", self.e1_spin)

        # Transverse Modulus (E2)
        self.e2_spin = QDoubleSpinBox()
        self.e2_spin.setRange(0, 100000)
        self.e2_spin.setValue(0)
        self.e2_spin.setSuffix(" MPa")
        self.e2_spin.setDecimals(0)
        material_layout.addRow("Transverse Modulus (E₂):", self.e2_spin)

        # Poisson's Ratio (nu)
        self.nu_spin = QDoubleSpinBox()
        self.nu_spin.setRange(0, 0.5)
        self.nu_spin.setValue(0)
        self.nu_spin.setDecimals(2)
        self.nu_spin.setSingleStep(0.01)
        material_layout.addRow("Poisson's Ratio (ν):", self.nu_spin)

        # Shear Modulus (G)
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(0, 50000)
        self.g_spin.setValue(0)
        self.g_spin.setSuffix(" MPa")
        self.g_spin.setDecimals(0)
        material_layout.addRow("Shear Modulus (G):", self.g_spin)

        left_layout.addWidget(material_group)

        # Plasticity Parameters Group
        plasticity_group = QGroupBox("Plasticity Parameters")
        plasticity_group.setStyleSheet(group_style)
        plasticity_layout = QFormLayout(plasticity_group)
        plasticity_layout.setSpacing(8)
        plasticity_layout.setContentsMargins(10, 20, 10, 10)

        # Yield Stress (tau_y)
        self.tau_y_spin = QDoubleSpinBox()
        self.tau_y_spin.setRange(0, 500)
        self.tau_y_spin.setValue(0)
        self.tau_y_spin.setSuffix(" MPa")
        self.tau_y_spin.setDecimals(1)
        plasticity_layout.addRow("Yield Stress (τᵧ):", self.tau_y_spin)

        # Hardening Coefficient (K)
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(0, 10)
        self.k_spin.setValue(0)
        self.k_spin.setDecimals(3)
        self.k_spin.setSingleStep(0.01)
        plasticity_layout.addRow("Hardening Coeff. (K):", self.k_spin)

        # Hardening Exponent (n)
        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0, 10)
        self.n_spin.setValue(0)
        self.n_spin.setDecimals(1)
        self.n_spin.setSingleStep(0.1)
        plasticity_layout.addRow("Hardening Exponent (n):", self.n_spin)

        left_layout.addWidget(plasticity_group)

        # Fiber Orientation Group
        orientation_group = QGroupBox("Fiber Orientation")
        orientation_group.setStyleSheet(group_style)
        orientation_layout = QFormLayout(orientation_group)
        orientation_layout.setSpacing(8)
        orientation_layout.setContentsMargins(10, 20, 10, 10)

        # Use 3D Orientation toggle
        self.use_3d_orientation_check = QCheckBox("Use 3D Orientation Data")
        self.use_3d_orientation_check.setChecked(False)
        self.use_3d_orientation_check.stateChanged.connect(self.onUse3DOrientationChanged)
        orientation_layout.addRow(self.use_3d_orientation_check)

        # Initial Misalignment
        self.initial_misalignment_spin = QDoubleSpinBox()
        self.initial_misalignment_spin.setRange(0, 90)
        self.initial_misalignment_spin.setValue(0)
        self.initial_misalignment_spin.setSuffix(" °")
        self.initial_misalignment_spin.setDecimals(2)
        self.initial_misalignment_spin.setSingleStep(0.1)
        orientation_layout.addRow("Initial Misalignment:", self.initial_misalignment_spin)

        # Standard Deviation
        self.std_deviation_spin = QDoubleSpinBox()
        self.std_deviation_spin.setRange(0, 30)
        self.std_deviation_spin.setValue(0)
        self.std_deviation_spin.setSuffix(" °")
        self.std_deviation_spin.setDecimals(2)
        self.std_deviation_spin.setSingleStep(0.1)
        orientation_layout.addRow("Standard Deviation:", self.std_deviation_spin)

        left_layout.addWidget(orientation_group)

        # Results Group
        results_group = QGroupBox("Results")
        results_group.setStyleSheet(group_style)
        results_layout = QFormLayout(results_group)
        results_layout.setSpacing(8)
        results_layout.setContentsMargins(10, 20, 10, 10)

        self.strength_label = QLabel("--")
        self.strength_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        results_layout.addRow("Compressive Strength:", self.strength_label)

        self.strain_label = QLabel("--")
        self.strain_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        results_layout.addRow("Ultimate Strain:", self.strain_label)

        left_layout.addWidget(results_group)
        left_layout.addStretch()

        layout.addWidget(left_panel)

        # Histogram Panel (initially hidden, shown when Histogram button is clicked)
        self.histogram_panel = HistogramPanel()
        self.histogram_panel.setVisible(False)
        layout.addWidget(self.histogram_panel)

        # Right panel - Stress-Strain Graph Display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure for stress-strain curve
        self.figure = Figure(figsize=(6, 5), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Strain')
        self.ax.set_ylabel('Stress (MPa)')
        self.ax.set_title('Stress-Strain Curve')
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()

        right_layout.addWidget(self.canvas)

        layout.addWidget(right_panel, 1)  # Right panel expands

    def updateResults(self, strength, strain):
        """Update the results labels"""
        self.strength_label.setText(f"{strength:.2f} MPa")
        self.strain_label.setText(f"{strain:.4f}")

    def plotStressStrain(self, strain_array, stress_curve, case_name=None):
        """Add stress-strain curve to the plot with case name as legend"""
        # Plot with case name as label
        label = case_name if case_name else f"Case {len(self.ax.lines) + 1}"
        self.ax.plot(strain_array, stress_curve, linewidth=2, label=label)
        self.ax.set_xlabel('Strain')
        self.ax.set_ylabel('Stress (MPa)')
        self.ax.set_title('Stress-Strain Curve')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best')
        self.figure.tight_layout()
        self.canvas.draw()

    def clearGraph(self):
        """Clear all curves from the graph"""
        self.ax.clear()
        self.ax.set_xlabel('Strain')
        self.ax.set_ylabel('Stress (MPa)')
        self.ax.set_title('Stress-Strain Curve')
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def onPresetChanged(self, preset_name):
        """Apply material preset values"""
        if preset_name == "Custom":
            return

        if preset_name in self.material_presets:
            params = self.material_presets[preset_name]
            # Block signals to prevent triggering preset change back to Custom
            self.e1_spin.blockSignals(True)
            self.e2_spin.blockSignals(True)
            self.nu_spin.blockSignals(True)
            self.g_spin.blockSignals(True)
            self.tau_y_spin.blockSignals(True)
            self.k_spin.blockSignals(True)
            self.n_spin.blockSignals(True)

            # Set values
            self.e1_spin.setValue(params.get('longitudinal_modulus', 150000))
            self.e2_spin.setValue(params.get('transverse_modulus', 10000))
            self.nu_spin.setValue(params.get('poisson_ratio', 0.3))
            self.g_spin.setValue(params.get('shear_modulus', 5000))
            self.tau_y_spin.setValue(params.get('tau_y', 50))
            self.k_spin.setValue(params.get('K', 0.1))
            self.n_spin.setValue(params.get('n', 2.0))

            # Unblock signals
            self.e1_spin.blockSignals(False)
            self.e2_spin.blockSignals(False)
            self.nu_spin.blockSignals(False)
            self.g_spin.blockSignals(False)
            self.tau_y_spin.blockSignals(False)
            self.k_spin.blockSignals(False)
            self.n_spin.blockSignals(False)

    def onUse3DOrientationChanged(self, state):
        """Toggle between manual input and 3D orientation data"""
        use_3d = (state == Qt.Checked.value) if hasattr(Qt.Checked, 'value') else (state == 2)
        # Disable/enable manual input fields
        self.initial_misalignment_spin.setEnabled(not use_3d)
        self.std_deviation_spin.setEnabled(not use_3d)


class ACSCMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_volume = None
        self.orientation_data = {
            'reference': None,
            'theta': None,
            'phi': None
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ACSC - Axial Compressive Strength Calculator")

        # Set application icon
        import os
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'acsc_logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
        """)

        # Central widget with full-width tabs at top
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Full-width tab widget at the top
        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(200)

        # Create tabs
        self.volume_tab = VolumeTab()
        self.tabs.addTab(self.volume_tab, "Volume")

        # Analysis tab
        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

        # Modelling tab (fiber trajectory)
        self.modelling_tab = VisualizationTab()
        self.modelling_tab.setMainWindow(self)
        self.tabs.addTab(self.modelling_tab, "Modelling")

        # Simulation tab - last tab
        self.simulation_tab = SimulationTab()
        self.simulation_tab.setMainWindow(self)
        self.tabs.addTab(self.simulation_tab, "Simulation")

        main_layout.addWidget(self.tabs)

        # Simulation content panel (shown when Simulation tab is selected)
        self.simulation_content = SimulationContentPanel()
        self.simulation_content.setVisible(False)
        main_layout.addWidget(self.simulation_content)

        # Create horizontal splitter for viewer and left slider panel
        self.content_splitter = QSplitter(Qt.Horizontal)

        # Left slider panel (only sliders)
        slider_panel = QWidget()
        slider_panel.setMaximumWidth(250)
        slider_panel.setMinimumWidth(200)
        slider_layout = QVBoxLayout(slider_panel)
        slider_layout.setContentsMargins(10, 10, 10, 10)

        # Slice control sliders
        slice_group = QGroupBox("Slice Controls")
        slice_layout = QGridLayout(slice_group)

        slice_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_slice_slider = QSlider(Qt.Horizontal)
        self.x_slice_slider.setEnabled(False)
        self.x_slice_slider.valueChanged.connect(self.updateSlices)
        slice_layout.addWidget(self.x_slice_slider, 0, 1)
        self.x_slice_spin = QSpinBox()
        self.x_slice_spin.setEnabled(False)
        self.x_slice_spin.setMaximumWidth(80)
        self.x_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)
        slice_layout.addWidget(self.x_slice_spin, 0, 2)

        slice_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_slice_slider = QSlider(Qt.Horizontal)
        self.y_slice_slider.setEnabled(False)
        self.y_slice_slider.valueChanged.connect(self.updateSlices)
        slice_layout.addWidget(self.y_slice_slider, 1, 1)
        self.y_slice_spin = QSpinBox()
        self.y_slice_spin.setEnabled(False)
        self.y_slice_spin.setMaximumWidth(80)
        self.y_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)
        slice_layout.addWidget(self.y_slice_spin, 1, 2)

        slice_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_slice_slider = QSlider(Qt.Horizontal)
        self.z_slice_slider.setEnabled(False)
        self.z_slice_slider.valueChanged.connect(self.updateSlices)
        slice_layout.addWidget(self.z_slice_slider, 2, 1)
        self.z_slice_spin = QSpinBox()
        self.z_slice_spin.setEnabled(False)
        self.z_slice_spin.setMaximumWidth(80)
        self.z_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)
        slice_layout.addWidget(self.z_slice_spin, 2, 2)

        slider_layout.addWidget(slice_group)

        # Noise-scale control for Analysis
        noise_group = QGroupBox("Analysis")
        noise_layout = QVBoxLayout(noise_group)

        noise_layout.addWidget(QLabel("Noise Scale:"))
        self.noise_scale_slider = QSlider(Qt.Horizontal)
        self.noise_scale_slider.setRange(1, 20)
        self.noise_scale_slider.setValue(10)
        self.noise_scale_slider.valueChanged.connect(self.updateNoiseScale)
        noise_layout.addWidget(self.noise_scale_slider)

        self.noise_scale_label = QLabel("10")
        self.noise_scale_label.setAlignment(Qt.AlignCenter)
        noise_layout.addWidget(self.noise_scale_label)

        self.noise_group = noise_group
        self.noise_group.setVisible(False)  # Initially hidden
        slider_layout.addWidget(noise_group)

        slider_layout.addStretch()

        self.content_splitter.addWidget(slider_panel)

        # Histogram Panel (initially hidden)
        self.histogram_panel = HistogramPanel()
        self.histogram_panel.setVisible(False)
        self.content_splitter.addWidget(self.histogram_panel)

        # 2D Slice Viewer (lightweight matplotlib-based viewer)
        self.viewer = Viewer2D(parent_window=self)
        self.content_splitter.addWidget(self.viewer)

        # Connect visualization controls to viewer
        self.volume_tab.connectViewer(self.viewer)

        # Connect analysis tab to viewer and main window
        self.analysis_tab.viewer = self.viewer
        self.analysis_tab.main_window = self

        # Connect simulation tab to viewer
        self.simulation_tab.viewer = self.viewer

        # Connect viewer to main window for custom ranges
        self.viewer.main_window = self

        # Add Void Analysis and Pipeline Panel to slider layout (before the stretch)
        # We need to insert it before the stretch, so remove stretch first, add controls, then re-add stretch
        slider_layout = slider_panel.layout()
        # Remove the last item (stretch)
        last_item = slider_layout.takeAt(slider_layout.count() - 1)

        # Add Void Analysis group
        void_group = QGroupBox("Void Analysis")
        void_layout = QVBoxLayout(void_group)
        void_layout.setSpacing(8)

        # Threshold slider
        threshold_label = QLabel("Threshold:")
        void_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(self.viewer.void_threshold)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(25)
        self.threshold_slider.valueChanged.connect(self.onThresholdChanged)
        void_layout.addWidget(self.threshold_slider)

        # Threshold value display
        self.threshold_value_label = QLabel(f"Value: {self.viewer.void_threshold}")
        self.threshold_value_label.setAlignment(Qt.AlignCenter)
        void_layout.addWidget(self.threshold_value_label)

        # Reset button
        reset_btn = QPushButton("Reset Threshold")
        reset_btn.clicked.connect(self.onResetThreshold)
        void_layout.addWidget(reset_btn)

        slider_layout.addWidget(void_group)

        # Add Pipeline group
        pipeline_group = QGroupBox("Pipeline")
        pipeline_layout_inner = QVBoxLayout(pipeline_group)
        pipeline_layout_inner.setSpacing(5)

        # Add orientation container from viewer (intensity always shown)
        pipeline_layout_inner.addWidget(self.viewer.orientation_container)

        slider_layout.addWidget(pipeline_group)

        # Re-add the stretch at the end
        if last_item:
            slider_layout.addItem(last_item)

        # Connect tab changes to update controls
        self.tabs.currentChanged.connect(self.onTabChanged)

        # Set splitter proportions and add separator line
        self.content_splitter.setStretchFactor(0, 0)  # Slider panel fixed size
        self.content_splitter.setStretchFactor(1, 1)  # Viewer takes remaining space

        # Style the splitter to show a visible separator line
        self.content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                width: 1px;
                margin: 0px;
                padding: 0px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
        """)

        main_layout.addWidget(self.content_splitter)

        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                font-size: 12px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().setStyleSheet("QStatusBar { border-top: 1px solid #d0d0d0; }")

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def updateNoiseScale(self):
        """Update noise scale value for analysis"""
        value = self.noise_scale_slider.value()
        self.noise_scale_label.setText(str(value))

    def onThresholdChanged(self, value):
        """Handle threshold slider change - only update label"""
        self.threshold_value_label.setText(f"Value: {value}")

    def onResetThreshold(self):
        """Reset threshold to default value"""
        default_threshold = 50
        self.threshold_slider.setValue(default_threshold)
        # valueChanged signal will trigger onThresholdChanged

    def onTabChanged(self, index):
        """Handle tab change to show/hide appropriate controls"""
        # Tab indices: 0=Volume, 1=Analysis, 2=Modelling, 3=Simulation
        if index == 2:  # Modelling tab (fiber trajectory)
            # Hide slicer view, Modelling tab has its own viewport
            self.content_splitter.setVisible(False)
            self.simulation_content.setVisible(False)
            self.noise_group.setVisible(False)
            # Remove height limit for Modelling tab (full window)
            self.tabs.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        elif index == 3:  # Simulation tab
            # Hide slicer view, show simulation content
            self.content_splitter.setVisible(False)
            self.simulation_content.setVisible(True)
            self.noise_group.setVisible(False)
            self.tabs.setMaximumHeight(200)
        else:
            # Show slicer view, hide simulation content
            self.content_splitter.setVisible(True)
            self.simulation_content.setVisible(False)
            # Show noise group only for Analysis tab
            self.noise_group.setVisible(index == 1)
            self.tabs.setMaximumHeight(200)


    def updateSlices(self):
        """Update viewer slices from main window sliders"""
        # Update spin boxes to match sliders
        self.x_slice_spin.valueChanged.disconnect()
        self.y_slice_spin.valueChanged.disconnect()
        self.z_slice_spin.valueChanged.disconnect()

        self.x_slice_spin.setValue(self.x_slice_slider.value())
        self.y_slice_spin.setValue(self.y_slice_slider.value())
        self.z_slice_spin.setValue(self.z_slice_slider.value())

        self.x_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)
        self.y_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)
        self.z_slice_spin.valueChanged.connect(self.updateSlicesFromSpin)

        # Update viewer
        if self.viewer:
            self.viewer.updateSlices(
                self.x_slice_slider.value(),
                self.y_slice_slider.value(),
                self.z_slice_slider.value()
            )

    def updateSlicesFromSpin(self):
        """Update sliders when spin boxes change"""
        # Temporarily disconnect slider signals to avoid loops
        self.x_slice_slider.valueChanged.disconnect()
        self.y_slice_slider.valueChanged.disconnect()
        self.z_slice_slider.valueChanged.disconnect()

        # Update sliders from spin boxes
        self.x_slice_slider.setValue(self.x_slice_spin.value())
        self.y_slice_slider.setValue(self.y_slice_spin.value())
        self.z_slice_slider.setValue(self.z_slice_spin.value())

        # Reconnect signals
        self.x_slice_slider.valueChanged.connect(self.updateSlices)
        self.y_slice_slider.valueChanged.connect(self.updateSlices)
        self.z_slice_slider.valueChanged.connect(self.updateSlices)

        # Update viewer
        self.updateSlices()

    def updateControlsForRenderMode(self, method):
        """Enable/disable controls based on render mode"""
        # Reset all controls
        self.x_slice_slider.setEnabled(False)
        self.y_slice_slider.setEnabled(False)
        self.z_slice_slider.setEnabled(False)
        self.x_slice_spin.setEnabled(False)
        self.y_slice_spin.setEnabled(False)
        self.z_slice_spin.setEnabled(False)

        # Enable appropriate controls
        if method == "Slices":
            self.x_slice_slider.setEnabled(True)
            self.y_slice_slider.setEnabled(True)
            self.z_slice_slider.setEnabled(True)
            self.x_slice_spin.setEnabled(True)
            self.y_slice_spin.setEnabled(True)
            self.z_slice_spin.setEnabled(True)

# Color bar methods removed - PyVista handles color bar automatically

    def setVolume(self, volume):
        self.current_volume = volume
        self.viewer.setVolume(volume)

        # Update main window control ranges
        if volume is not None:
            # Update slice slider ranges
            self.x_slice_slider.setRange(0, volume.shape[2] - 1)
            self.x_slice_slider.setValue(volume.shape[2] // 2)
            self.x_slice_spin.setRange(0, volume.shape[2] - 1)
            self.x_slice_spin.setValue(volume.shape[2] // 2)

            self.y_slice_slider.setRange(0, volume.shape[1] - 1)
            self.y_slice_slider.setValue(volume.shape[1] // 2)
            self.y_slice_spin.setRange(0, volume.shape[1] - 1)
            self.y_slice_spin.setValue(volume.shape[1] // 2)

            self.z_slice_slider.setRange(0, volume.shape[0] - 1)
            self.z_slice_slider.setValue(volume.shape[0] // 2)
            self.z_slice_spin.setRange(0, volume.shape[0] - 1)
            self.z_slice_spin.setValue(volume.shape[0] // 2)

            # Enable slice controls for 2D viewer (always shows slices)
            self.x_slice_slider.setEnabled(True)
            self.y_slice_slider.setEnabled(True)
            self.z_slice_slider.setEnabled(True)
            self.x_slice_spin.setEnabled(True)
            self.y_slice_spin.setEnabled(True)
            self.z_slice_spin.setEnabled(True)

        # Switch to visualization tab
        self.tabs.setCurrentWidget(self.volume_tab)

    def updateStatus(self, message):
        self.status_label.setText(message)

    def updateProgress(self, value):
        self.progress_bar.setValue(value)

    def showProgress(self, show):
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setValue(0)

    def showHistogramPanel(self, config):
        """Show the histogram panel with the given configuration"""
        self.histogram_panel.setVisible(True)
        # Pass viewer.rois if ROIs are selected, otherwise pass orientation_data
        if config.get('rois'):
            self.histogram_panel.plotHistogramMultiROI(config, self.viewer.rois)
        else:
            self.histogram_panel.plotHistogram(config, self.orientation_data)

    def hideHistogramPanel(self):
        """Hide the histogram panel"""
        self.histogram_panel.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = ACSCMainWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()