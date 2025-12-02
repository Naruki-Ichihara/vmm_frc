import sys
import os
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
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QAction, QIcon
import cv2 as cv
from acsc.io import import_image_sequence, trim_image
from acsc.analysis import compute_structure_tensor, compute_orientation, drop_edges_3D, _orientation_function, _orientation_function_reference
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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

        left_layout.addWidget(QLabel("Initial Number:"), 3, 0)
        self.initial_number_spin = QSpinBox()
        self.initial_number_spin.setRange(0, 9999)
        self.initial_number_spin.setValue(0)
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
                image_files.extend(sorted(path.glob(f"*{ext.upper()}")))

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


class VisualizationTab(QWidget):
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
        self.ref_combo.setCurrentText("X-axis")  # Default to X-axis (depth direction)
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
        toolbar_layout.addStretch()

        layout.addWidget(toolbar)
        layout.addStretch()

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

                main_window.showProgress(False)
                main_window.status_label.setText(f"Analysis complete (Noise scale: {noise_scale})")

                # Enable edit range and histogram buttons
                self.edit_range_btn.setEnabled(True)
                self.histogram_btn.setEnabled(True)

                # Enable histogram button in Simulation tab as well
                if hasattr(main_window, 'simulation_tab'):
                    main_window.simulation_tab.histogram_btn.setEnabled(True)

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
        self.kink_width_spin.setValue(self.settings.get('kink_width') or 0.5)
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
        self.kink_width_spin.setValue(0.5)
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

        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)

        self.settings_btn = RibbonButton("Settings")
        self.settings_btn.clicked.connect(self.openSettingsDialog)
        settings_layout.addWidget(self.settings_btn)

        toolbar_layout.addWidget(settings_group)

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
        self.visualization_tab = VisualizationTab()
        self.tabs.addTab(self.visualization_tab, "Visualization")

        # Analysis tab
        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

        # Simulation tab
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
        self.visualization_tab.connectViewer(self.viewer)

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
        if index == 2:  # Simulation tab (index 2)
            # Hide slicer view, show simulation content
            self.content_splitter.setVisible(False)
            self.simulation_content.setVisible(True)
            self.noise_group.setVisible(False)
        else:
            # Show slicer view, hide simulation content
            self.content_splitter.setVisible(True)
            self.simulation_content.setVisible(False)
            # Show noise group only for Analysis tab
            self.noise_group.setVisible(index == 1)


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
        self.tabs.setCurrentWidget(self.visualization_tab)

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