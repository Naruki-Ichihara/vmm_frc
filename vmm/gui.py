import sys
import os
from pathlib import Path
import numpy as np
from vmm.theme import COLORS
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QSpinBox, QComboBox, QFileDialog, QGroupBox,
                               QGridLayout, QFormLayout, QTextEdit, QProgressBar, QMessageBox,
                               QCheckBox, QRadioButton, QSlider, QDoubleSpinBox, QStackedWidget,
                               QButtonGroup, QFrame, QScrollArea, QTabBar,
                               QSizePolicy, QSplitter, QDialog)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QPixmap
import cv2 as cv
from vmm.io import import_image_sequence, trim_image
from vmm.analysis import compute_structure_tensor, compute_orientation, drop_edges_3D
from vmm.adjustment import ImageAdjuster, AdjustmentSettings, export_adjustment_settings
from vmm.logger import get_logger
import matplotlib

logger = get_logger()
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
        from vmm.theme import get_toolbar_style
        toolbar = QFrame()
        toolbar.setStyleSheet(get_toolbar_style())
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

        self.crop_check = QCheckBox("Enable Cropping")
        process_layout.addWidget(self.crop_check)

        toolbar_layout.addWidget(process_group)

        # Normalization Group
        norm_group = QGroupBox("Intensity Normalization")
        norm_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        norm_layout = QVBoxLayout(norm_group)

        self.normalize_check = QCheckBox("Enable Normalization")
        self.normalize_check.setToolTip("Correct inter-slice brightness variations")
        norm_layout.addWidget(self.normalize_check)

        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(["mean", "histogram", "minmax"])
        self.normalize_combo.setToolTip(
            "mean: Match slice means to global mean\n"
            "histogram: Match histograms to first slice\n"
            "minmax: Normalize using global min/max"
        )
        self.normalize_combo.setEnabled(False)
        norm_layout.addWidget(self.normalize_combo)

        self.normalize_check.toggled.connect(self.normalize_combo.setEnabled)

        toolbar_layout.addWidget(norm_group)

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
        from vmm.theme import get_import_button_style
        self.import_btn = QPushButton("Import")
        self.import_btn.setMinimumHeight(35)
        self.import_btn.setMinimumWidth(80)
        self.import_btn.setStyleSheet(get_import_button_style())
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
            'crop_enabled': self.crop_check.isChecked(),
            'crop_start': self.crop_start_input.text(),
            'crop_end': self.crop_end_input.text(),
            'normalize_enabled': self.normalize_check.isChecked(),
            'normalize_method': self.normalize_combo.currentText()
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

            # Determine normalization method
            normalize_method = None
            if self.params.get('normalize_enabled', False):
                normalize_method = self.params.get('normalize_method', 'mean')
                self.status.emit(f"Importing {self.params['num_images']} images with {normalize_method} normalization...")
            else:
                self.status.emit(f"Importing {self.params['num_images']} images...")
            self.progress.emit(25)

            # Import image sequence (automatically converts to grayscale uint8)
            volume = import_image_sequence(
                path_template=self.params['path_template'],
                number_of_images=self.params['num_images'],
                number_of_digits=self.params['num_digits'],
                format=self.params['format'],
                initial_number=self.params['initial_number'],
                process=process_func,
                normalize=normalize_method
            )

            self.progress.emit(100)
            self.status.emit(f"Import complete! Volume shape: {volume.shape}")
            self.finished.emit(volume)

        except Exception as e:
            self.error.emit(str(e))


class ExportVTPDialog(QDialog):
    """Dialog for configuring VTP export options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export VTP Options")
        self.setMinimumWidth(380)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Scalar arrays to export
        arrays_group = QGroupBox("Scalar Arrays")
        arrays_layout = QVBoxLayout(arrays_group)

        self.xz_check = QCheckBox("XZ_Orientation (X-Z plane angle)")
        self.xz_check.setChecked(True)
        arrays_layout.addWidget(self.xz_check)

        self.yz_check = QCheckBox("YZ_Orientation (Y-Z plane angle)")
        self.yz_check.setChecked(True)
        arrays_layout.addWidget(self.yz_check)

        self.fiber_id_check = QCheckBox("FiberID (unique fiber identifier)")
        self.fiber_id_check.setChecked(True)
        arrays_layout.addWidget(self.fiber_id_check)

        layout.addWidget(arrays_group)

        # Azimuth options (cyclic colormap)
        azimuth_group = QGroupBox("Azimuth (for cyclic colormap)")
        azimuth_layout = QVBoxLayout(azimuth_group)

        self.azimuth_check = QCheckBox("Azimuth (0° to 360°, cyclic)")
        self.azimuth_check.setChecked(True)
        self.azimuth_check.setToolTip(
            "True azimuth angle in 0°-360° range.\n"
            "Use with HSV or cyclic colormap in Paraview.\n"
            "0°/360° = same direction (cyclic)"
        )
        azimuth_layout.addWidget(self.azimuth_check)

        self.azimuth_norm_check = QCheckBox("Azimuth_Normalized (0 to 1, for Hue)")
        self.azimuth_norm_check.setChecked(True)
        self.azimuth_norm_check.setToolTip(
            "Azimuth normalized to 0-1 range.\n"
            "Use directly as Hue in HSV colormap:\n"
            "  0° → 0.0 (Red)\n"
            "  120° → 0.33 (Green)\n"
            "  240° → 0.67 (Blue)\n"
            "  360° → 1.0 (Red, cyclic)"
        )
        azimuth_layout.addWidget(self.azimuth_norm_check)

        layout.addWidget(azimuth_group)

        # HSV to RGB color options
        rgb_group = QGroupBox("Pre-rendered RGB Color")
        rgb_layout = QVBoxLayout(rgb_group)

        self.rgb_check = QCheckBox("Azimuth_RGB (HSV→RGB converted color)")
        self.rgb_check.setChecked(True)
        self.rgb_check.setToolTip(
            "Pre-computed RGB color from HSV:\n"
            "  H = Azimuth (0°-360° → color wheel)\n"
            "  S = Tilt angle (Z-axis deviation)\n"
            "  V = 1.0 (full brightness)\n\n"
            "Use directly in Paraview without colormap setup."
        )
        rgb_layout.addWidget(self.rgb_check)

        layout.addWidget(rgb_group)

        # Usage info
        info_group = QGroupBox("Usage in Paraview")
        info_layout = QVBoxLayout(info_group)
        info_label = QLabel(
            "For pre-rendered RGB color:\n"
            "1. Open VTP → Apply 'Tube' filter\n"
            "2. Color by 'Azimuth_RGB'\n"
            "3. Turn off 'Map Scalars'\n\n"
            "HSV mapping:\n"
            "  Hue = Azimuth direction\n"
            "  Saturation = Tilt magnitude"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 11px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)

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

    def getSettings(self):
        """Return the current dialog settings."""
        return {
            'export_xz': self.xz_check.isChecked(),
            'export_yz': self.yz_check.isChecked(),
            'export_azimuth': self.azimuth_check.isChecked(),
            'export_azimuth_norm': self.azimuth_norm_check.isChecked(),
            'export_fiber_id': self.fiber_id_check.isChecked(),
            'export_rgb': self.rgb_check.isChecked()
        }


class RibbonButton(QPushButton):
    def __init__(self, text, icon_name=None):
        super().__init__()
        from vmm.theme import get_ribbon_button_style
        self.setText(text)
        self.setMinimumSize(55, 45)
        self.setMaximumSize(80, 45)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setStyleSheet(get_ribbon_button_style())

class RibbonComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        from vmm.theme import get_ribbon_combobox_style
        self.setMinimumWidth(120)
        self.setStyleSheet(get_ribbon_combobox_style())

class HistogramPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(3, 4), facecolor=COLORS['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(150, 200)
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

        # Get density option
        use_density = config.get('use_density', False)

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
                                          alpha=0.7, edgecolor='black',
                                          density=use_density)

                ax.set_title('Reference Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Density' if use_density else 'Frequency', fontsize=10)
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
                                          alpha=0.7, edgecolor='black',
                                          density=use_density)

                ax.set_title('X-Z Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Density' if use_density else 'Frequency', fontsize=10)
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
                                          alpha=0.7, edgecolor='black',
                                          density=use_density)

                ax.set_title('Y-Z Orientation', fontsize=12)
                ax.set_xlabel('Angle (degrees)', fontsize=10)
                ax.set_ylabel('Density' if use_density else 'Frequency', fontsize=10)
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

        # Check if any data was plotted
        if not self.histogram_data:
            self.stats_text.setText("No orientation data available.\n\nPlease run orientation analysis first.")

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
        roi_colors = COLORS['roi_colors']

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

        # Get density option
        use_density = config.get('use_density', False)

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
                                          label=roi_name, histtype='stepfilled',
                                          density=use_density)

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
            ax.set_ylabel('Density' if use_density else 'Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            if max_count > 0:
                ax.set_ylim(0, max_count * 1.1)  # Add 10% headroom

            # Add legend with ROI names and colors
            if len(selected_rois) > 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # Update statistics display
        self.stats_text.setText(stats_text)

        # Check if any data was plotted
        if not self.histogram_data:
            self.stats_text.setText("No orientation data available for selected ROIs.\n\nPlease run orientation analysis first.")

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

        # Plot X-Z orientation histogram (formerly tilt)
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

        # Plot Y-Z orientation histogram (formerly azimuth)
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
                        hist_range = (-90, 90)  # Y-Z orientation range

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
                    cv_val = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0

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
            QMessageBox.warning(self, "No Data", "No histogram data to export.\nPlease plot a histogram first.")
            return

        # Debug: Check data validity
        valid_data = False
        for key, hist_data in self.histogram_data.items():
            if hist_data.get('data') is not None and len(hist_data.get('data', [])) > 0:
                valid_data = True
                break

        if not valid_data:
            QMessageBox.warning(self, "No Data", "Histogram data is empty.\nPlease ensure orientation data exists for the selected ROI.")
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
                writer.writerow(['Generated from VMM-FRC Analysis'])
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
        self.figure = Figure(figsize=(3, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(150, 200)
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
            config['angles'].get('true_azimuth', False) and angle_data.get('true_azimuth') is not None
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

        # Plot X-Z orientation histogram (formerly tilt)
        if config['angles'].get('tilt', False) and angle_data.get('tilt') is not None:
            plot_angle_histogram(angle_data['tilt'], 'X-Z Orientation', 'green', 'X-Z')

        # Plot Y-Z orientation histogram (formerly azimuth)
        if config['angles'].get('azimuth', False) and angle_data.get('azimuth') is not None:
            plot_angle_histogram(angle_data['azimuth'], 'Y-Z Orientation', 'orange', 'Y-Z')

        # Plot True Azimuth histogram
        if config['angles'].get('true_azimuth', False) and angle_data.get('true_azimuth') is not None:
            plot_angle_histogram(angle_data['true_azimuth'], 'Azimuth', 'purple', 'Azimuth')

        self.stats_text.setText(stats_text)
        self.figure.tight_layout()
        self.canvas.draw()

    def hidePanel(self):
        """Hide the histogram panel."""
        self.setVisible(False)

    def exportToCSV(self):
        """Export histogram data and statistics to CSV."""
        if not self.histogram_data:
            QMessageBox.warning(self, "No Data", "No histogram data to export.\nPlease plot a histogram first.")
            return

        # Check data validity
        valid_data = False
        for key, hist_data in self.histogram_data.items():
            if hist_data.get('data') is not None and len(hist_data.get('data', [])) > 0:
                valid_data = True
                break

        if not valid_data:
            QMessageBox.warning(self, "No Data", "Histogram data is empty.\nPlease ensure trajectory data exists.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Histogram Data", "",
            "CSV Files (*.csv);;All Files (*)"
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
                writer.writerow(['Generated from VMM-FRC Fiber Trajectory Analysis'])
                writer.writerow([])

                # Write configuration info
                if hasattr(self, 'current_config') and self.current_config:
                    config = self.current_config
                    writer.writerow(['Configuration:'])
                    if 'bins' in config:
                        writer.writerow(['Bins:', config['bins']])
                    if 'range' in config:
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

        # ROI (Region of Interest) for orientation computation
        self.roi_enabled = False
        self.roi_mode = 'rectangle'  # 'rectangle' or 'polygon'
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

        # Volume fraction visualization
        self.vf_map = None
        self.vf_roi_bounds = None
        self.vf_polygon_mask = None  # Polygon mask for VF overlay
        self.show_vf_overlay = False

        # Void visualization
        self.void_mask = None
        self.void_roi_bounds = None
        self.show_void_overlay = False

        # Image adjustment
        self.adjuster = ImageAdjuster()

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
        self.figure_xy = Figure(figsize=(3, 3), facecolor=COLORS['bg_primary'])
        # Add extra space on left and bottom for axis arrows
        self.figure_xy.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_xy = FigureCanvas(self.figure_xy)
        self.canvas_xy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_xy.setMinimumSize(100, 100)
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
        self.figure_xz = Figure(figsize=(3, 3), facecolor=COLORS['bg_primary'])
        # Add extra space on left and bottom for axis arrows
        self.figure_xz.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_xz = FigureCanvas(self.figure_xz)
        self.canvas_xz.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_xz.setMinimumSize(100, 100)
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
        self.figure_yz = Figure(figsize=(3, 3), facecolor=COLORS['bg_primary'])
        # Add extra space on left and bottom for axis arrows
        self.figure_yz.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.95)
        self.canvas_yz = FigureCanvas(self.figure_yz)
        self.canvas_yz.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_yz.setMinimumSize(100, 100)
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
        self.figure_hist = Figure(figsize=(6, 2), facecolor=COLORS['bg_primary'])
        self.canvas_hist = FigureCanvas(self.figure_hist)
        self.canvas_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_hist.setMinimumSize(200, 80)

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

        # Store original volume for adjustment
        if volume is not None:
            self.adjuster.set_original_volume(volume)
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

    def setROIMode(self, mode):
        """Set the ROI selection mode ('rectangle' or 'polygon')"""
        self.roi_mode = mode.lower() if mode else 'rectangle'

    def resetAllROIs(self):
        """Reset all ROIs and clear related data."""
        # Clear ROI selectors
        if self.roi_selector_xy is not None:
            self.roi_selector_xy.set_active(False)
            self.roi_selector_xy = None
        if self.roi_selector_xz is not None:
            self.roi_selector_xz.set_active(False)
            self.roi_selector_xz = None
        if self.roi_selector_yz is not None:
            self.roi_selector_yz.set_active(False)
            self.roi_selector_yz = None

        # Clear ROI data
        self.rois = {}
        self.roi_counter = 0
        self.current_roi_name = None
        self.roi_enabled = False

        # Re-render
        self.renderVolume()

    def getROIPolygonMask(self, roi_name, shape_2d):
        """Generate a 2D polygon mask for the given ROI.

        Args:
            roi_name: Name of the ROI
            shape_2d: Tuple (height, width) for the mask shape

        Returns:
            numpy array of bool with True inside the polygon/rectangle
        """
        from skimage.draw import polygon as draw_polygon

        if roi_name not in self.rois:
            return np.ones(shape_2d, dtype=bool)

        roi_data = self.rois[roi_name]
        bounds = roi_data['bounds']
        polygon_xy = roi_data.get('polygon_xy')

        z_min, z_max, y_min, y_max, x_min, x_max = bounds

        # Create mask for the bounding box region
        mask = np.zeros(shape_2d, dtype=bool)

        if polygon_xy is not None and len(polygon_xy) >= 3:
            # Use polygon mask
            # polygon_xy contains (x, y) coordinates
            poly_x = np.array([v[0] for v in polygon_xy])
            poly_y = np.array([v[1] for v in polygon_xy])

            # Generate polygon mask using skimage
            rr, cc = draw_polygon(poly_y, poly_x, shape=shape_2d)
            mask[rr, cc] = True
        else:
            # Use rectangular mask
            mask[y_min:y_max, x_min:x_max] = True

        return mask

    def getROIPolygonMask3D(self, roi_name):
        """Generate a 3D mask for the given ROI (polygon extruded along Z).

        Args:
            roi_name: Name of the ROI

        Returns:
            numpy array of bool with shape matching the ROI bounds
        """
        if roi_name not in self.rois or self.current_volume is None:
            return None

        roi_data = self.rois[roi_name]
        bounds = roi_data['bounds']
        z_min, z_max, y_min, y_max, x_min, x_max = bounds

        # Get 2D mask for XY slice within the bounding box
        roi_height = y_max - y_min
        roi_width = x_max - x_min
        n_slices = z_max - z_min

        polygon_xy = roi_data.get('polygon_xy')

        if polygon_xy is not None and len(polygon_xy) >= 3:
            from skimage.draw import polygon as draw_polygon

            # Translate polygon to ROI-local coordinates
            poly_x = np.array([v[0] - x_min for v in polygon_xy])
            poly_y = np.array([v[1] - y_min for v in polygon_xy])

            # Create 2D mask in local coordinates
            mask_2d = np.zeros((roi_height, roi_width), dtype=bool)
            rr, cc = draw_polygon(poly_y, poly_x, shape=(roi_height, roi_width))
            mask_2d[rr, cc] = True

            # Extrude to 3D
            mask_3d = np.zeros((n_slices, roi_height, roi_width), dtype=bool)
            for z in range(n_slices):
                mask_3d[z] = mask_2d
        else:
            # Full rectangular region
            mask_3d = np.ones((n_slices, roi_height, roi_width), dtype=bool)

        return mask_3d

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
        from matplotlib.widgets import RectangleSelector, PolygonSelector

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

        if self.roi_mode == 'polygon':
            # Polygon selection mode
            self.roi_selector_xy = PolygonSelector(
                self.ax_xy,
                self._onPolygonSelectXY,
                useblit=True,
                props=dict(color=color, linestyle='-', linewidth=2, alpha=0.5)
            )
            # For polygon mode, only XY selector is used (2D polygon on current Z slice)
            self.roi_selector_xz = None
            self.roi_selector_yz = None
        else:
            # Rectangle selection mode (default)
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

    def _onPolygonSelectXY(self, vertices):
        """Handle polygon selection in XY view"""
        if self.current_roi_name is None or self.current_roi_name not in self.rois:
            return

        if len(vertices) < 3:
            return

        # Store polygon vertices in ROI data
        self.rois[self.current_roi_name]['polygon_xy'] = vertices

        # Calculate bounding box from polygon
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Clamp to volume bounds
        if self.current_volume is not None:
            volume_shape = self.current_volume.shape
            x_min = max(0, x_min)
            x_max = min(volume_shape[2], x_max)
            y_min = max(0, y_min)
            y_max = min(volume_shape[1], y_max)

        # Update bounds (keep Z unchanged)
        bounds = self.rois[self.current_roi_name]['bounds']
        z_min, z_max = bounds[0], bounds[1]
        self.rois[self.current_roi_name]['bounds'] = [z_min, z_max, y_min, y_max, x_min, x_max]

        self.renderVolume()

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
        """Draw colored overlay shapes for all ROIs on all three views"""
        from matplotlib.patches import Rectangle, Polygon

        # Draw each ROI
        for roi_name, roi_data in self.rois.items():
            bounds = roi_data['bounds']
            color = roi_data['color']
            z_min, z_max, y_min, y_max, x_min, x_max = bounds

            # Check if this ROI has polygon vertices
            polygon_xy = roi_data.get('polygon_xy')

            # XY view - draw polygon if available, otherwise rectangle
            if polygon_xy is not None and len(polygon_xy) >= 3:
                # Draw polygon
                poly_patch = Polygon(polygon_xy, fill=False, edgecolor=color,
                                    linewidth=2, linestyle='--', closed=True)
                self.ax_xy.add_patch(poly_patch)
                # Add label at first vertex
                self.ax_xy.text(polygon_xy[0][0] + 5, polygon_xy[0][1] + 5, roi_name,
                               color=color, fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                # Draw rectangle
                rect_xy = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    fill=False, edgecolor=color, linewidth=2, linestyle='--')
                self.ax_xy.add_patch(rect_xy)
                self.ax_xy.text(x_min + 5, y_min + 5, roi_name, color=color, fontsize=9,
                               fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # XZ view - always rectangle (polygon is 2D on XY plane)
            rect_xz = Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            self.ax_xz.add_patch(rect_xz)
            self.ax_xz.text(x_min + 5, z_min + 5, roi_name, color=color, fontsize=9,
                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # YZ view - always rectangle (polygon is 2D on XY plane)
            rect_yz = Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
            self.ax_yz.add_patch(rect_yz)
            self.ax_yz.text(y_min + 5, z_min + 5, roi_name, color=color, fontsize=9,
                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

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

    def _renderVfOverlay(self):
        """Render volume fraction overlay on all slice views.

        Handles polygon masks by using NaN values (which are transparent in imshow).
        """
        if not self.show_vf_overlay or self.vf_map is None or self.vf_roi_bounds is None:
            return

        z_min, z_max, y_min, y_max, x_min, x_max = self.vf_roi_bounds

        # Get Vf slice for XY view (only if current Z slice is within ROI)
        if z_min <= self.slice_z < z_max:
            vf_z_idx = self.slice_z - z_min
            if 0 <= vf_z_idx < self.vf_map.shape[0]:
                vf_slice_xy = self.vf_map[vf_z_idx, :, :]
                # Use masked array to handle NaN values (from polygon mask)
                vf_slice_xy_masked = np.ma.masked_invalid(vf_slice_xy)
                extent_xy = [x_min, x_max, y_min, y_max]
                self.ax_xy.imshow(vf_slice_xy_masked, cmap='jet', origin='lower',
                                 extent=extent_xy, alpha=0.5, zorder=2,
                                 vmin=0, vmax=1, interpolation='bilinear')

        # XZ view (Y slice) - always show using center Y of ROI if outside bounds
        if y_min <= self.slice_y < y_max:
            vf_y_idx = self.slice_y - y_min
        else:
            # Use center Y slice of ROI when outside bounds
            vf_y_idx = (y_max - y_min) // 2
        if 0 <= vf_y_idx < self.vf_map.shape[1]:
            vf_slice_xz = self.vf_map[:, vf_y_idx, :]
            vf_slice_xz_masked = np.ma.masked_invalid(vf_slice_xz)
            extent_xz = [x_min, x_max, z_min, z_max]
            self.ax_xz.imshow(vf_slice_xz_masked, cmap='jet', origin='lower',
                             extent=extent_xz, alpha=0.5, zorder=2,
                             vmin=0, vmax=1, interpolation='bilinear')

        # YZ view (X slice) - always show using center X of ROI if outside bounds
        if x_min <= self.slice_x < x_max:
            vf_x_idx = self.slice_x - x_min
        else:
            # Use center X slice of ROI when outside bounds
            vf_x_idx = (x_max - x_min) // 2
        if 0 <= vf_x_idx < self.vf_map.shape[2]:
            vf_slice_yz = self.vf_map[:, :, vf_x_idx]
            vf_slice_yz_masked = np.ma.masked_invalid(vf_slice_yz)
            extent_yz = [y_min, y_max, z_min, z_max]
            self.ax_yz.imshow(vf_slice_yz_masked, cmap='jet', origin='lower',
                             extent=extent_yz, alpha=0.5, zorder=2,
                             vmin=0, vmax=1, interpolation='bilinear')

    def _renderVoidOverlay(self):
        """Render void overlay on all slice views.

        Shows void regions in red with transparency.
        void_mask shape: (z_size, y_size, x_size) - indices within ROI
        void_roi_bounds: [z_min, z_max, y_min, y_max, x_min, x_max] - coordinates in full volume
        """
        if not self.show_void_overlay or self.void_mask is None or self.void_roi_bounds is None:
            return

        z_min, z_max, y_min, y_max, x_min, x_max = self.void_roi_bounds

        # Create red colormap for voids
        from matplotlib.colors import ListedColormap
        red_cmap = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # transparent and red

        # XY view (Z slice) - void_mask[z, y, x]
        if z_min <= self.slice_z < z_max:
            void_z_idx = self.slice_z - z_min
            if 0 <= void_z_idx < self.void_mask.shape[0]:
                void_slice_xy = self.void_mask[void_z_idx, :, :].astype(np.float32)
                # extent: [x_left, x_right, y_bottom, y_top]
                extent_xy = [x_min, x_max, y_max, y_min]  # y_max first for correct orientation
                self.ax_xy.imshow(void_slice_xy, cmap=red_cmap, origin='upper',
                                 extent=extent_xy, alpha=0.6, zorder=3,
                                 vmin=0, vmax=1, interpolation='nearest')

        # XZ view (Y slice) - void_mask[:, y, :]
        if y_min <= self.slice_y < y_max:
            void_y_idx = self.slice_y - y_min
            if 0 <= void_y_idx < self.void_mask.shape[1]:
                void_slice_xz = self.void_mask[:, void_y_idx, :].astype(np.float32)
                # extent: [x_left, x_right, z_bottom, z_top]
                extent_xz = [x_min, x_max, z_max, z_min]  # z_max first for correct orientation
                self.ax_xz.imshow(void_slice_xz, cmap=red_cmap, origin='upper',
                                 extent=extent_xz, alpha=0.6, zorder=3,
                                 vmin=0, vmax=1, interpolation='nearest')

        # YZ view (X slice) - void_mask[:, :, x]
        if x_min <= self.slice_x < x_max:
            void_x_idx = self.slice_x - x_min
            if 0 <= void_x_idx < self.void_mask.shape[2]:
                void_slice_yz = self.void_mask[:, :, void_x_idx].astype(np.float32)
                # extent: [y_left, y_right, z_bottom, z_top]
                extent_yz = [y_min, y_max, z_max, z_min]  # z_max first for correct orientation
                self.ax_yz.imshow(void_slice_yz, cmap=red_cmap, origin='upper',
                                 extent=extent_yz, alpha=0.6, zorder=3,
                                 vmin=0, vmax=1, interpolation='nearest')

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
        colors = {'x': COLORS['axis_x'], 'y': COLORS['axis_y'], 'z': COLORS['axis_z']}

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

        # Render orientation overlays if active
        self._renderOrientationOverlay()

        # Render fiber detection results if available
        self._renderFiberDetection()

        # Render Vf overlay if enabled
        self._renderVfOverlay()

        # Render void overlay if enabled
        self._renderVoidOverlay()

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
            # Apply image adjustments if any
            if not self.adjuster.settings.is_default():
                slice_xy = self.adjuster.apply_to_slice(slice_xy)
            im_xy = self.ax_xy.imshow(slice_xy, cmap=self.colormap, origin='lower',
                                       vmin=vmin, vmax=vmax, aspect='equal')
            self._setSquareAspect(self.ax_xy, slice_xy.shape)
            self.ax_xy.set_title(f'XY Plane (Z={self.slice_z})', fontsize=10, fontweight='bold')
            self.ax_xy.axis('off')
            # Draw axis arrows
            self._drawAxisArrows(self.ax_xy, 'xy', slice_xy.shape)

            # Render fiber detection results if available
            self._renderFiberDetection()

            # Render volume fraction overlay if available
            self._renderVfOverlay()

            self.figure_xy.tight_layout()

            # Render XZ plane (Y slice)
            slice_xz = volume[:, self.slice_y, :]
            # Apply image adjustments if any
            if not self.adjuster.settings.is_default():
                slice_xz = self.adjuster.apply_to_slice(slice_xz)
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
            # Apply image adjustments if any
            if not self.adjuster.settings.is_default():
                slice_yz = self.adjuster.apply_to_slice(slice_yz)
            im_yz = self.ax_yz.imshow(slice_yz, cmap=self.colormap, origin='lower',
                                       vmin=vmin, vmax=vmax, aspect='equal')
            self._setSquareAspect(self.ax_yz, slice_yz.shape)
            self.ax_yz.set_title(f'YZ Plane (X={self.slice_x})', fontsize=10, fontweight='bold')
            self.ax_yz.axis('off')
            # Draw axis arrows
            self._drawAxisArrows(self.ax_yz, 'yz', slice_yz.shape)
            self.figure_yz.tight_layout()

            # Check if orientation overlay should be shown
            self._renderOrientationOverlay()

            # Render Vf overlay if enabled
            self._renderVfOverlay()

            # Render void overlay if enabled
            self._renderVoidOverlay()

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
                roi_colors = COLORS['roi_colors']

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
            if isinstance(widget, VMMMainWindow):
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

    def export3D(self, filename, iso_value):
        """Export functionality removed - 2D viewer only"""
        QMessageBox.warning(None, "Not Available",
                            "3D export is not available in 2D slice viewer mode.\n"
                            "This feature requires the 3D PyVista viewer.")


class VolumeTab(QWidget):
    """Tab for volume data visualization and import - ribbon is now managed by MainWindow."""
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.main_window = None
        self.initUI()

    def initUI(self):
        # Ribbon is now managed by MainWindow's ribbon_stack
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

    def updateColormap(self, colormap=None):
        """Update colormap - colormap param or from MainWindow's combo"""
        if self.viewer:
            if colormap is None and hasattr(self, 'main_window') and hasattr(self.main_window, 'colormap_combo'):
                colormap = self.main_window.colormap_combo.currentText()
            if colormap:
                self.viewer.setColormap(colormap)

# Color bar colormap sync removed - PyVista handles color bar automatically

# Control methods removed - now handled by main window

    def resetView(self):
        if self.viewer:
            self.viewer.resetCamera()

    def export3D(self):
        """Export CT volume to VTK format for Paraview."""
        import pyvista as pv

        if not self.viewer or self.viewer.current_volume is None:
            QMessageBox.warning(self, "No Data", "No volume loaded to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export CT Volume", "",
            "VTK ImageData (*.vti);;Legacy VTK (*.vtk);;All Files (*)"
        )
        if not filename:
            return

        if not filename.lower().endswith(('.vti', '.vtk')):
            filename += '.vti'

        try:
            volume = self.viewer.current_volume

            # Transpose from (Z, Y, X) to (X, Y, Z) for VTK coordinate system
            # This matches the fiber trajectory VTP export coordinate system
            volume_vtk = np.transpose(volume, (2, 1, 0))

            # Create VTK ImageData
            grid = pv.ImageData()
            grid.dimensions = np.array(volume_vtk.shape) + 1  # (X+1, Y+1, Z+1)
            grid.spacing = (1, 1, 1)

            # Add volume data as cell data (Fortran order for VTK)
            grid.cell_data['CTValue'] = volume_vtk.flatten(order='F')

            grid.save(filename)

            QMessageBox.information(
                self, "Export Successful",
                f"CT volume exported to:\n{filename}\n\n"
                f"Volume shape: {volume.shape}\n"
                f"Value range: [{volume.min()}, {volume.max()}]\n\n"
                "Data field: CTValue"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

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
            'tilt_min': -20.0,
            'tilt_max': 20.0,
            'relax': True,
            'color_by_angle': True,
            'show_fiber_diameter': False,
            'resample': False,
            'resample_interval': 20
        }
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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
        self.color_mode_combo.addItems(["X-Z Orientation", "Y-Z Orientation", "Azimuth", "Azimuth (saturation)"])
        self.color_mode_combo.currentTextChanged.connect(self.updateVisualization)
        self.color_mode_combo.setToolTip(
            "X-Z: Angle in X-Z plane (projection)\n"
            "Y-Z: Angle in Y-Z plane (projection)\n"
            "Azimuth: True azimuth (0°-360°, uniform saturation)\n"
            "Azimuth (saturation): Azimuth with tilt-based saturation"
        )
        colormap_layout.addWidget(self.color_mode_combo)

        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["coolwarm", "viridis", "jet", "rainbow", "plasma", "turbo", "hsv"])
        self.colormap_combo.currentTextChanged.connect(self.updateVisualization)
        colormap_layout.addWidget(self.colormap_combo)

        left_layout.addWidget(colormap_group)

        # ROI Visibility Group
        roi_visibility_group = QGroupBox("ROI Visibility")
        self.roi_visibility_layout = QVBoxLayout(roi_visibility_group)
        self.roi_visibility_layout.setSpacing(2)
        self.roi_checkboxes = {}  # Store checkboxes for each ROI
        self.roi_visibility_label = QLabel("No ROIs")
        self.roi_visibility_layout.addWidget(self.roi_visibility_label)
        left_layout.addWidget(roi_visibility_group)

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

        from vmm.theme import get_viewer_frame_style, get_viewer_title_style
        for i, title in enumerate(viewport_titles):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box | QFrame.Plain)
            frame.setStyleSheet(get_viewer_frame_style())
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(2, 2, 2, 2)
            frame_layout.setSpacing(0)

            # Title label
            title_label = QLabel(title)
            title_label.setStyleSheet(get_viewer_title_style())
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setMaximumHeight(20)
            frame_layout.addWidget(title_label)

            if i == 0:
                # 3D View: Use PyVista QtInteractor
                self.plotter_3d = QtInteractor(frame)
                self.plotter_3d.set_background(COLORS['chart_bg'])
                self.plotter_3d.add_axes()
                frame_layout.addWidget(self.plotter_3d.interactor)
                canvas = self.plotter_3d
            else:
                # Slice views: Use Matplotlib canvas
                fig = Figure(figsize=(4, 4), facecolor=COLORS['chart_bg'])
                fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                ax = fig.add_subplot(111)
                ax.set_facecolor(COLORS['chart_bg'])
                ax.tick_params(colors=COLORS['text_white'], labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(COLORS['chart_spine'])
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

    def updateMainStatus(self, message: str):
        """Update the main window status bar with a message."""
        if self.main_window and hasattr(self.main_window, 'status_label'):
            self.main_window.status_label.setText(message)
            QApplication.processEvents()

    def _getVfMapFromAnalysisTab(self):
        """Get Vf map and bounds from analysis tab if available.

        Returns:
            Tuple of (vf_map, vf_roi_bounds) or (None, None) if not available.
        """
        if self.main_window and hasattr(self.main_window, 'analysis_tab'):
            analysis_tab = self.main_window.analysis_tab
            if hasattr(analysis_tab, 'vf_map') and analysis_tab.vf_map is not None:
                vf_map = analysis_tab.vf_map
                vf_roi_bounds = getattr(analysis_tab, 'vf_roi_bounds', None)
                if vf_roi_bounds is not None:
                    import numpy as np
                    self.updateMainStatus(
                        f"Using Vf map: shape={vf_map.shape}, "
                        f"range=[{vf_map.min():.4f}, {vf_map.max():.4f}], "
                        f"mean={vf_map.mean():.4f}"
                    )
                    return vf_map, vf_roi_bounds
        return None, None

    def setMainWindow(self, main_window):
        """Set reference to main window for accessing shared data."""
        self.main_window = main_window

    def setStructureTensor(self, structure_tensor, volume_shape, volume_data=None):
        """Set structure tensor data for trajectory generation."""
        self.structure_tensor = structure_tensor
        self.volume_shape = volume_shape
        self.volume_data = volume_data
        # Enable generate button in MainWindow's ribbon
        if self.main_window and hasattr(self.main_window, 'generate_btn'):
            self.main_window.generate_btn.setEnabled(True)

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
        # ROI selection removed - all ROIs are automatically used
        pass

    def _updateSliceSlidersToTrajectoryBounds(self):
        """Update slice sliders to center of trajectory bounds."""
        # Get actual volume dimensions from main window if available
        main_window = getattr(self, 'main_window', None)
        if main_window and main_window.current_volume is not None:
            vol_shape = main_window.current_volume.shape
            x_min, x_max = 0, vol_shape[2]
            y_min, y_max = 0, vol_shape[1]
            z_min, z_max = 0, vol_shape[0]
        else:
            x_min, x_max = 0, 100
            y_min, y_max = 0, 100
            z_min, z_max = 0, 100

        if self.roi_trajectories:
            # Compute bounds from all ROI trajectories
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                bounds = roi_data.get('bounds', None)
                offset = roi_data.get('offset', (0, 0, 0))
                z_offset, y_offset, x_offset = offset
                if bounds:
                    roi_z_min, roi_z_max, roi_y_min, roi_y_max, roi_x_min, roi_x_max = bounds
                    x_max = max(x_max, roi_x_max)
                    y_max = max(y_max, roi_y_max)
                    z_max = max(z_max, roi_z_max)
                else:
                    # Compute from trajectory data
                    fiber_traj = roi_data.get('trajectory')
                    if fiber_traj and fiber_traj.trajectories:
                        z_max = max(z_max, z_offset + len(fiber_traj.trajectories))
                        for _, points in fiber_traj.trajectories:
                            if len(points) > 0:
                                x_max = max(x_max, x_offset + int(np.max(points[:, 0])))
                                y_max = max(y_max, y_offset + int(np.max(points[:, 1])))
        elif self.fiber_trajectory is not None:
            # Use single trajectory bounds
            trajectories = self.fiber_trajectory.trajectories
            if trajectories:
                z_max = len(trajectories)
                for _, points in trajectories:
                    if len(points) > 0:
                        x_max = max(x_max, int(np.max(points[:, 0])))
                        y_max = max(y_max, int(np.max(points[:, 1])))

        # Update slider ranges and center positions
        if x_max > 0:
            self.x_slice_slider.setRange(0, x_max - 1)
            self.x_slice_slider.setValue(x_max // 2)
            self.current_slice['x'] = x_max // 2
        if y_max > 0:
            self.y_slice_slider.setRange(0, y_max - 1)
            self.y_slice_slider.setValue(y_max // 2)
            self.current_slice['y'] = y_max // 2
        if z_max > 0:
            self.z_slice_slider.setRange(0, z_max - 1)
            self.z_slice_slider.setValue(z_max // 2)
            self.current_slice['z'] = z_max // 2

    def onROISelectionChanged(self):
        """Handle ROI selection change - regenerate trajectories for selected ROIs."""
        self.updateVisualization()

    def getSelectedROIs(self):
        """Get list of all ROI names (all ROIs are automatically selected)."""
        if self.main_window and hasattr(self.main_window, 'viewer'):
            rois = self.main_window.viewer.rois if hasattr(self.main_window.viewer, 'rois') else {}
            return list(rois.keys())
        return []

    def generateTrajectory(self):
        """Generate fiber trajectory for selected ROIs."""
        from vmm.fiber_trajectory import create_fiber_distribution, FiberTrajectory, detect_fiber_centers
        from vmm.analysis import compute_structure_tensor

        # Show ROI selection dialog
        if self.main_window and hasattr(self.main_window, 'viewer'):
            rois = self.main_window.viewer.rois if hasattr(self.main_window.viewer, 'rois') else {}
            if rois:
                dialog = ROISelectDialog(self.main_window, rois,
                                         title="Select ROIs for Trajectory Generation",
                                         button_text="Generate")
                if dialog.exec() != QDialog.Accepted:
                    return  # User cancelled
                selected_rois = dialog.getSelectedROIs()
            else:
                selected_rois = []
        else:
            selected_rois = []

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

        # Fibers are assumed to be aligned along Z-axis
        # (reference_vector selection removed - always Z-axis propagation)

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

                # Get polygon mask for this ROI (in ROI-local coordinates)
                polygon_mask_3d = self.main_window.viewer.getROIPolygonMask3D(roi_name)

                # Get volume data for this ROI
                if self.main_window.current_volume is not None:
                    roi_volume = self.main_window.current_volume[z_min:z_max, y_min:y_max, x_min:x_max]
                    roi_shape = roi_volume.shape

                    # ALWAYS use structure tensor from Analysis tab - do not compute locally
                    global_st = self.main_window.orientation_data.get('structure_tensor')
                    cached_roi_name = self.main_window.orientation_data.get('roi_name')

                    if global_st is None:
                        # No structure tensor available - show error
                        QMessageBox.critical(
                            self,
                            "Orientation Analysis Required",
                            f"Structure tensor not found.\n\n"
                            f"Please compute orientation in the Analysis tab for ROI '{roi_name}' first.\n\n"
                            f"Steps:\n"
                            f"1. Go to Analysis tab\n"
                            f"2. Select '{roi_name}' from ROI list\n"
                            f"3. Click 'Compute Orientation'"
                        )
                        return

                    # Check if the cached structure tensor is for the same ROI
                    if cached_roi_name != roi_name:
                        QMessageBox.critical(
                            self,
                            "ROI Mismatch",
                            f"Cached orientation data is for ROI '{cached_roi_name}', but you selected '{roi_name}'.\n\n"
                            f"Please compute orientation in the Analysis tab for ROI '{roi_name}' first."
                        )
                        return

                    # Use the cached structure tensor (already in ROI coordinates)
                    self.updateMainStatus(f"[{roi_name}] Using structure tensor from Analysis tab...")
                    print(f"[INFO] Reusing structure tensor from Analysis tab for ROI '{roi_name}'")
                    print(f"[DEBUG] Structure tensor shape: {global_st.shape}")
                    roi_structure_tensor = global_st

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
                            # Get Vf map from analysis tab if available
                            vf_map, vf_roi_bounds = self._getVfMapFromAnalysisTab()

                            # Get 2D polygon mask for initial sampling plane
                            polygon_mask_2d = None
                            if polygon_mask_3d is not None:
                                # Use the first slice of the 3D mask as 2D mask
                                polygon_mask_2d = polygon_mask_3d[0]

                            # Fall back to Poisson disk sampling
                            fiber_traj.initialize(
                                shape=roi_shape,
                                fiber_diameter=fiber_diameter,
                                fiber_volume_fraction=volume_fraction,
                                scale=1.0,
                                seed=42 + hash(roi_name) % 1000,
                                vf_map=vf_map,
                                vf_roi_bounds=vf_roi_bounds,
                                polygon_mask=polygon_mask_2d
                            )
                        else:
                            self.updateMainStatus(f"[{roi_name}] Using {len(initial_centers)} detected fiber centers...")

                            # Initialize with detected centers
                            # Centers are in ROI-local coordinates already
                            fiber_traj.bounds = roi_shape

                            # Store polygon mask
                            if polygon_mask_3d is not None:
                                fiber_traj.polygon_mask = polygon_mask_3d[0]  # 2D mask
                            else:
                                fiber_traj.polygon_mask = None

                            # Propagation is always along Z-axis
                            fiber_traj.propagation_axis = 2  # Z-axis

                            # Set fiber diameter from detection settings
                            diameters = all_slices[initial_slice]['diameters']
                            fiber_traj.fiber_diameter = np.mean(diameters) if len(diameters) > 0 else fiber_diameter

                            # Filter initial centers by polygon mask if available
                            if polygon_mask_3d is not None:
                                polygon_mask_2d = polygon_mask_3d[0]
                                inside_polygon = np.zeros(len(initial_centers), dtype=bool)
                                mask_h, mask_w = polygon_mask_2d.shape
                                for i, (x, y) in enumerate(initial_centers):
                                    ix, iy = int(round(x)), int(round(y))
                                    if 0 <= iy < mask_h and 0 <= ix < mask_w:
                                        inside_polygon[i] = polygon_mask_2d[iy, ix]
                                centers_before = len(initial_centers)
                                initial_centers = initial_centers[inside_polygon]
                                diameters = diameters[inside_polygon]
                                self.updateMainStatus(f"[{roi_name}] Polygon filter: {centers_before} -> {len(initial_centers)} centers")

                            if len(initial_centers) == 0:
                                self.updateMainStatus(f"[{roi_name}] No centers inside polygon, skipping...")
                                continue

                            # Exclude fibers near the boundary BEFORE initializing trajectories
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
                            n_excluded = np.sum(near_boundary)
                            if n_excluded > 0:
                                logger.info(f"[{roi_name}] Excluded {n_excluded} fibers near boundary (margin={boundary_margin:.1f}px)")

                            # Keep only non-boundary fibers
                            valid_fibers = ~near_boundary
                            initial_centers = initial_centers[valid_fibers]
                            diameters = diameters[valid_fibers]

                            if len(initial_centers) == 0:
                                self.updateMainStatus(f"[{roi_name}] No valid centers after boundary exclusion, skipping...")
                                continue

                            # Initialize points
                            fiber_traj.points = initial_centers
                            fiber_traj.trajectories = [(0, initial_centers.copy())]
                            fiber_traj.angles = [np.zeros(len(initial_centers))]
                            fiber_traj.azimuths = [np.zeros(len(initial_centers))]

                            # Initialize per-fiber trajectory data (only for valid fibers)
                            n_fibers = len(initial_centers)
                            fiber_traj.fiber_trajectories = [[(0, initial_centers[i].copy())] for i in range(n_fibers)]
                            fiber_traj.fiber_angles = [[0.0] for _ in range(n_fibers)]
                            fiber_traj.fiber_azimuths = [[0.0] for _ in range(n_fibers)]
                            fiber_traj.fiber_azimuth_angles = [[0.0] for _ in range(n_fibers)]
                            # All fibers are active (boundary fibers already excluded)
                            fiber_traj.active_fibers = np.ones(n_fibers, dtype=bool)
                    else:
                        # Use Poisson disk sampling (original behavior)
                        self.updateMainStatus(f"[{roi_name}] Creating fiber distribution...")
                        # Get Vf map from analysis tab if available
                        vf_map, vf_roi_bounds = self._getVfMapFromAnalysisTab()

                        # Get 2D polygon mask for initial sampling plane
                        polygon_mask_2d = None
                        if polygon_mask_3d is not None:
                            polygon_mask_2d = polygon_mask_3d[0]

                        fiber_traj.initialize(
                            shape=roi_shape,
                            fiber_diameter=fiber_diameter,
                            fiber_volume_fraction=volume_fraction,
                            scale=1.0,
                            seed=42 + hash(roi_name) % 1000,
                            vf_map=vf_map,
                            vf_roi_bounds=vf_roi_bounds,
                            polygon_mask=polygon_mask_2d
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

            # Get Vf map from analysis tab if available
            vf_map, vf_roi_bounds = self._getVfMapFromAnalysisTab()

            # Use Poisson disk sampling
            self.fiber_trajectory.initialize(
                shape=self.volume_shape,
                fiber_diameter=fiber_diameter,
                fiber_volume_fraction=volume_fraction,
                scale=1.0,
                seed=42,
                vf_map=vf_map,
                vf_roi_bounds=vf_roi_bounds
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

        # Enable export buttons in MainWindow's ribbon
        if self.main_window:
            if hasattr(self.main_window, 'export_traj_vtk_btn'):
                self.main_window.export_traj_vtk_btn.setEnabled(True)
            if hasattr(self.main_window, 'trajectory_histogram_btn'):
                self.main_window.trajectory_histogram_btn.setEnabled(True)

        # Update slice sliders to center of trajectory bounds before visualization
        self._updateSliceSlidersToTrajectoryBounds()

        # Update ROI visibility checkboxes
        self._updateROIVisibilityCheckboxes()

        self.updateVisualization()

        # Update status to complete
        self.updateMainStatus("Ready")

    def _updateROIVisibilityCheckboxes(self):
        """Update the ROI visibility checkboxes based on available ROI trajectories."""
        # Clear existing checkboxes
        for checkbox in self.roi_checkboxes.values():
            self.roi_visibility_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.roi_checkboxes.clear()

        # Hide "No ROIs" label if we have ROIs
        if self.roi_trajectories:
            self.roi_visibility_label.hide()

            # Create checkbox for each ROI
            for roi_name in self.roi_trajectories.keys():
                checkbox = QCheckBox(roi_name)
                checkbox.setChecked(True)  # Default to visible
                checkbox.toggled.connect(self.updateVisualization)
                self.roi_visibility_layout.addWidget(checkbox)
                self.roi_checkboxes[roi_name] = checkbox
        else:
            self.roi_visibility_label.show()
            self.roi_visibility_label.setText("No ROIs")

    def _getVisibleROIs(self):
        """Get list of ROI names that are currently visible (checked)."""
        visible_rois = []
        for roi_name, checkbox in self.roi_checkboxes.items():
            if checkbox.isChecked():
                visible_rois.append(roi_name)
        return visible_rois

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
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_by_fiber = self.trajectory_settings.get('color_by_fiber', False)
        color_mode = self.color_mode_combo.currentText()

        # Determine which angle data to use based on color mode
        # "X-Z Orientation" -> use angles (X-Z projection)
        # "Y-Z Orientation" -> use azimuths (Y-Z projection)
        # "Azimuth" -> use azimuth_angles (true azimuth, uniform saturation)
        # "Azimuth (saturation)" -> use azimuth_angles with tilt-based saturation
        use_xz = "X-Z" in color_mode
        use_yz = "Y-Z" in color_mode
        use_true_azimuth = "Azimuth" in color_mode
        use_tilt_saturation = color_mode == "Azimuth (saturation)"

        # Set appropriate angle range based on color mode
        # Azimuth uses HSV cyclic colormap (0-360)
        # X-Z/Y-Z Orientation use linear colormap with user settings or defaults
        if use_true_azimuth:
            angle_min = 0.0
            angle_max = 360.0
        else:
            # Use user settings for X-Z/Y-Z orientation
            angle_min = self.trajectory_settings['tilt_min']
            angle_max = self.trajectory_settings['tilt_max']

        # Line width from slider
        line_width = float(self.line_width_slider.value())

        # Colormap for fiber-based coloring
        fiber_cmap = plt.get_cmap('tab20')

        # Build all fiber trajectories as a single PolyData for efficiency
        all_points = []
        all_lines = []
        all_angles_data = []  # X-Z orientation angles
        all_azimuths_data = []  # Y-Z orientation angles
        all_azimuth_angles_data = []  # True azimuth angles (-180 to 180)
        all_fiber_indices = []  # track fiber index for color_by_fiber
        point_offset = 0
        global_bounds = None
        global_fiber_idx = 0  # global fiber counter across all ROIs

        # Process all ROI trajectories (only visible ones)
        trajectories_to_render = []
        visible_rois = self._getVisibleROIs()
        if self.roi_trajectories:
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                # Skip if ROI is not visible
                if roi_name not in visible_rois:
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
                fiber_azimuth_angles_list = getattr(fiber_traj, 'fiber_azimuth_angles', None)

                for fiber_idx, traj in enumerate(fiber_trajectories):
                    if len(traj) < 2:
                        continue

                    # Get angle arrays for this fiber
                    fiber_angles_arr = None
                    fiber_azimuths_arr = None
                    fiber_azimuth_angles_arr = None

                    if fiber_angles_list and fiber_idx < len(fiber_angles_list):
                        fiber_angles_arr = fiber_angles_list[fiber_idx]
                    if fiber_azimuths_list and fiber_idx < len(fiber_azimuths_list):
                        fiber_azimuths_arr = fiber_azimuths_list[fiber_idx]
                    if fiber_azimuth_angles_list and fiber_idx < len(fiber_azimuth_angles_list):
                        fiber_azimuth_angles_arr = fiber_azimuth_angles_list[fiber_idx]

                    fiber_points = []
                    fiber_angles = []
                    fiber_azimuths = []
                    fiber_azimuth_angles = []

                    for i, (z, point) in enumerate(traj):
                        x = point[0] + x_offset
                        y = point[1] + y_offset
                        z_global = z + z_offset
                        fiber_points.append([x, y, z_global])

                        # Get angles - use available data, otherwise use previous value or 0
                        if fiber_angles_arr and i < len(fiber_angles_arr):
                            fiber_angles.append(fiber_angles_arr[i])
                        elif fiber_angles:
                            fiber_angles.append(fiber_angles[-1])  # Use previous value
                        else:
                            fiber_angles.append(0.0)

                        if fiber_azimuths_arr and i < len(fiber_azimuths_arr):
                            fiber_azimuths.append(fiber_azimuths_arr[i])
                        elif fiber_azimuths:
                            fiber_azimuths.append(fiber_azimuths[-1])  # Use previous value
                        else:
                            fiber_azimuths.append(0.0)

                        if fiber_azimuth_angles_arr and i < len(fiber_azimuth_angles_arr):
                            fiber_azimuth_angles.append(fiber_azimuth_angles_arr[i])
                        elif fiber_azimuth_angles:
                            fiber_azimuth_angles.append(fiber_azimuth_angles[-1])  # Use previous value
                        else:
                            fiber_azimuth_angles.append(0.0)

                    n_pts = len(fiber_points)
                    if n_pts < 2:
                        continue

                    all_points.extend(fiber_points)
                    all_angles_data.extend(fiber_angles)
                    all_azimuths_data.extend(fiber_azimuths)
                    all_azimuth_angles_data.extend(fiber_azimuth_angles)
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
                azimuth_angles = getattr(fiber_traj, 'azimuth_angles', azimuths)

                if len(trajectories) < 2:
                    continue

                n_fibers = len(trajectories[0][1])

                for fiber_idx in range(n_fibers):
                    fiber_points = []
                    fiber_angles = []
                    fiber_azimuths = []
                    fiber_azimuth_angles = []

                    for slice_idx, (z, slice_points) in enumerate(trajectories):
                        x = slice_points[fiber_idx, 0] + x_offset
                        y = slice_points[fiber_idx, 1] + y_offset
                        z_global = z + z_offset
                        fiber_points.append([x, y, z_global])

                        # Get angles - use available data, otherwise use previous value or 0
                        if slice_idx < len(angles) and fiber_idx < len(angles[slice_idx]):
                            fiber_angles.append(angles[slice_idx][fiber_idx])
                        elif fiber_angles:
                            fiber_angles.append(fiber_angles[-1])
                        else:
                            fiber_angles.append(0.0)
                        if slice_idx < len(azimuths) and fiber_idx < len(azimuths[slice_idx]):
                            fiber_azimuths.append(azimuths[slice_idx][fiber_idx])
                        elif fiber_azimuths:
                            fiber_azimuths.append(fiber_azimuths[-1])
                        else:
                            fiber_azimuths.append(0.0)
                        if slice_idx < len(azimuth_angles) and fiber_idx < len(azimuth_angles[slice_idx]):
                            fiber_azimuth_angles.append(azimuth_angles[slice_idx][fiber_idx])
                        elif fiber_azimuth_angles:
                            fiber_azimuth_angles.append(fiber_azimuth_angles[-1])
                        else:
                            fiber_azimuth_angles.append(0.0)

                    n_pts = len(fiber_points)
                    if n_pts < 2:
                        continue

                    all_points.extend(fiber_points)
                    all_angles_data.extend(fiber_angles)
                    all_azimuths_data.extend(fiber_azimuths)
                    all_azimuth_angles_data.extend(fiber_azimuth_angles)
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
            if use_true_azimuth and all_azimuth_angles_data:
                from matplotlib.colors import hsv_to_rgb
                # True azimuth mode: convert -180~180 to 0~360 for cyclic colormap
                azimuth_arr = np.array(all_azimuth_angles_data)
                azimuth_360 = np.where(azimuth_arr < 0, azimuth_arr + 360, azimuth_arr)

                if use_tilt_saturation:
                    # Use tilt-based saturation with RGB colors
                    tilt_arr = np.array(all_angles_data)
                    sat_min = self.trajectory_settings.get('saturation_min', 0.0)
                    sat_max = self.trajectory_settings.get('saturation_max', 45.0)
                    sat_range = sat_max - sat_min if sat_max > sat_min else 1.0
                    saturation = np.clip((np.abs(tilt_arr) - sat_min) / sat_range, 0, 1)
                    # Build HSV and convert to RGB
                    n_pts = len(azimuth_360)
                    hsv = np.zeros((n_pts, 3))
                    hsv[:, 0] = azimuth_360 / 360.0
                    hsv[:, 1] = saturation
                    hsv[:, 2] = 1.0
                    rgb = hsv_to_rgb(hsv)
                    poly['rgb'] = (rgb * 255).astype(np.uint8)
                    self.plotter_3d.add_mesh(
                        poly,
                        scalars='rgb',
                        rgb=True,
                        line_width=line_width,
                        render_lines_as_tubes=True
                    )
                else:
                    # Uniform saturation - use HSV colormap directly
                    poly['angle'] = azimuth_360
                    self.plotter_3d.add_mesh(
                        poly,
                        scalars='angle',
                        cmap='hsv',
                        clim=(angle_min, angle_max),
                        line_width=line_width,
                        render_lines_as_tubes=True,
                        scalar_bar_args={'title': 'Azimuth (°)', 'n_labels': 5}
                    )
            elif use_yz and all_azimuths_data:
                # Y-Z orientation mode: use linear colormap
                poly['angle'] = np.array(all_azimuths_data)
                self.plotter_3d.add_mesh(
                    poly,
                    scalars='angle',
                    cmap=cmap,
                    clim=(angle_min, angle_max),
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    scalar_bar_args={'title': 'Y-Z Angle (°)', 'n_labels': 5}
                )
            elif use_xz and all_angles_data:
                # X-Z orientation mode: use linear colormap
                poly['angle'] = np.array(all_angles_data)
                self.plotter_3d.add_mesh(
                    poly,
                    scalars='angle',
                    cmap=cmap,
                    clim=(angle_min, angle_max),
                    line_width=line_width,
                    render_lines_as_tubes=True,
                    scalar_bar_args={'title': 'X-Z Angle (°)', 'n_labels': 5}
                )
            elif all_angles_data:
                # Default: X-Z orientation
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

        # Add domain boundary boxes for each visible ROI
        if self.roi_trajectories:
            roi_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
            for i, (roi_name, roi_data) in enumerate(self.roi_trajectories.items()):
                # Skip if ROI is not visible
                if roi_name not in visible_rois:
                    continue
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
        color_by_angle = self.trajectory_settings['color_by_angle']
        color_by_fiber = self.trajectory_settings.get('color_by_fiber', False)
        color_mode = self.color_mode_combo.currentText()
        # Determine which angle data to use based on color mode
        # "X-Z Orientation" -> use angles (X-Z projection)
        # "Y-Z Orientation" -> use azimuths (Y-Z projection)
        # "Azimuth" -> use azimuth_angles (true azimuth, uniform saturation)
        # "Azimuth (saturation)" -> use azimuth_angles with tilt-based saturation
        use_xz = "X-Z" in color_mode
        use_yz = "Y-Z" in color_mode
        use_true_azimuth = "Azimuth" in color_mode
        use_tilt_saturation = color_mode == "Azimuth (saturation)"

        # Set appropriate angle range based on color mode
        # Azimuth uses HSV cyclic colormap (0-360)
        if use_true_azimuth:
            angle_min = 0.0
            angle_max = 360.0
        else:
            angle_min = self.trajectory_settings['tilt_min']
            angle_max = self.trajectory_settings['tilt_max']

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

        # Vectorized color computation for Y-Z orientation (uses angles_to_colors with azimuths_arr)
        def yz_angles_to_colors(azimuths_arr, fiber_indices=None):
            if color_by_fiber and fiber_indices is not None:
                n_fibers = len(fiber_indices)
                colors = np.zeros((n_fibers, 4))
                for i, idx in enumerate(fiber_indices):
                    colors[i] = fiber_cmap(idx % 20 / 20.0)
                return colors
            if not color_by_angle:
                return np.full((len(azimuths_arr), 4), [0, 0, 1, 1])  # blue
            norm_angles = np.clip((azimuths_arr - angle_min) / (angle_max - angle_min + 1e-6), 0, 1)
            return cmap(norm_angles)

        # Vectorized color computation for true azimuth (HSV cyclic colormap, 0-360)
        def true_azimuth_to_colors(azimuth_angles_arr, tilts_arr, fiber_indices=None):
            from matplotlib.colors import hsv_to_rgb
            if color_by_fiber and fiber_indices is not None:
                n_fibers = len(fiber_indices)
                colors = np.zeros((n_fibers, 4))
                for i, idx in enumerate(fiber_indices):
                    colors[i] = fiber_cmap(idx % 20 / 20.0)
                return colors
            if not color_by_angle:
                return np.full((len(azimuth_angles_arr), 4), [0, 0, 1, 1])  # blue
            # Convert -180~180 to 0~360 for cyclic colormap
            azimuth_360 = np.where(azimuth_angles_arr < 0, azimuth_angles_arr + 360, azimuth_angles_arr)
            # Saturation: uniform or tilt-based
            if use_tilt_saturation:
                sat_min = self.trajectory_settings.get('saturation_min', 0.0)
                sat_max = self.trajectory_settings.get('saturation_max', 45.0)
                sat_range = sat_max - sat_min if sat_max > sat_min else 1.0
                saturation = np.clip((np.abs(tilts_arr) - sat_min) / sat_range, 0, 1)
            else:
                saturation = np.ones(len(azimuth_360))  # Uniform saturation
            # Build HSV array
            n_points = len(azimuth_360)
            hsv = np.zeros((n_points, 3))
            hsv[:, 0] = azimuth_360 / 360.0  # Hue
            hsv[:, 1] = saturation            # Saturation
            hsv[:, 2] = 1.0                   # Value
            rgb = hsv_to_rgb(hsv)
            # Add alpha channel
            colors = np.ones((n_points, 4))
            colors[:, :3] = rgb
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
        ax_xy.set_facecolor(COLORS['chart_bg'])
        ax_xy.set_title(f'XY Slice at Z={z_pos}', color=COLORS['text_white'], fontsize=10)
        ax_xy.set_xlabel('X', color=COLORS['text_white'], fontsize=8)
        ax_xy.set_ylabel('Y', color=COLORS['text_white'], fontsize=8)
        ax_xy.tick_params(colors=COLORS['text_white'], labelsize=8)

        canvas_xz = self.viewport_frames[2]['canvas']
        ax_xz = canvas_xz.axes
        ax_xz.clear()
        ax_xz.set_facecolor(COLORS['chart_bg'])
        ax_xz.set_title(f'XZ Slice at Y={y_pos}', color=COLORS['text_white'], fontsize=10)
        ax_xz.set_xlabel('X', color=COLORS['text_white'], fontsize=8)
        ax_xz.set_ylabel('Z', color=COLORS['text_white'], fontsize=8)
        ax_xz.tick_params(colors=COLORS['text_white'], labelsize=8)

        canvas_yz = self.viewport_frames[3]['canvas']
        ax_yz = canvas_yz.axes
        ax_yz.clear()
        ax_yz.set_facecolor(COLORS['chart_bg'])
        ax_yz.set_title(f'YZ Slice at X={x_pos}', color=COLORS['text_white'], fontsize=10)
        ax_yz.set_xlabel('Y', color=COLORS['text_white'], fontsize=8)
        ax_yz.set_ylabel('Z', color=COLORS['text_white'], fontsize=8)
        ax_yz.tick_params(colors=COLORS['text_white'], labelsize=8)

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
                # Use ROI volumes when global volume is not available (only visible ROIs)
                visible_rois_ct = self._getVisibleROIs()
                for roi_name, roi_data in self.roi_trajectories.items():
                    if not isinstance(roi_data, dict):
                        continue
                    # Skip if ROI is not visible
                    if roi_name not in visible_rois_ct:
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

        # Render trajectories for all visible ROIs
        visible_rois = self._getVisibleROIs()
        if self.roi_trajectories:
            for roi_name, roi_data in self.roi_trajectories.items():
                if not isinstance(roi_data, dict):
                    continue
                # Skip if ROI is not visible
                if roi_name not in visible_rois:
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
                    roi_depth = roi_z_max - roi_z_min
                else:
                    # Use trajectory length as fallback for depth
                    roi_depth = len(trajectories)
                    roi_height = roi_width = 1000  # fallback for width/height

                # Get propagation axis and fiber diameter for circle rendering
                prop_axis = getattr(fiber_traj, 'propagation_axis', 2)
                fiber_diameter = getattr(fiber_traj, 'fiber_diameter', 7.0)
                radius = fiber_diameter / 2.0

                # Helper function to get angle data with proper padding for resampled fibers
                def get_padded_angles_roi(angle_arr, slice_idx, n_pts):
                    if slice_idx < len(angle_arr):
                        arr = np.array(angle_arr[slice_idx])
                        if len(arr) == n_pts:
                            return arr
                        elif len(arr) < n_pts:
                            # Pad with zeros for new fibers added by resampling
                            return np.concatenate([arr, np.zeros(n_pts - len(arr))])
                        else:
                            # Truncate if somehow longer
                            return arr[:n_pts]
                    return np.zeros(n_pts)

                # Helper function for masked angle access
                def get_slice_angles_masked_roi(angle_arr, slice_idx, n_pts, mask):
                    padded = get_padded_angles_roi(angle_arr, slice_idx, n_pts)
                    return padded[mask]

                # XY Slice - find trajectory slice at z_pos (relative to ROI)
                roi_z_pos = z_pos - z_offset
                num_slices = len(trajectories)
                # Check against actual trajectory length (primary constraint)
                if 0 <= roi_z_pos < num_slices:
                    z, points = trajectories[roi_z_pos]
                    n_points = len(points)

                    slice_angles = get_padded_angles_roi(angles, roi_z_pos, n_points)
                    fiber_indices = np.arange(n_points)  # fiber index for each point

                    # Get azimuths for Y-Z or true azimuth mode
                    if use_yz or use_true_azimuth:
                        slice_azimuths = get_padded_angles_roi(azimuths, roi_z_pos, n_points)
                        if use_true_azimuth:
                            # Get true azimuth angles if available
                            azimuth_angles_data = getattr(fiber_traj, 'azimuth_angles', None)
                            if azimuth_angles_data:
                                slice_azimuth_angles = get_padded_angles_roi(azimuth_angles_data, roi_z_pos, n_points)
                            else:
                                slice_azimuth_angles = slice_azimuths  # fallback to azimuths
                            colors = true_azimuth_to_colors(slice_azimuth_angles, slice_angles, fiber_indices)
                        else:
                            colors = yz_angles_to_colors(slice_azimuths, fiber_indices)
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

                # Always process - let the mask filter out points that don't match
                # This ensures we don't miss points due to incorrect bounds calculation
                if roi_y_pos >= -tolerance:
                    for slice_idx, (z, points) in enumerate(trajectories):
                        mask = np.abs(points[:, 1] - roi_y_pos) < tolerance
                        if np.any(mask):
                            matched_indices = np.where(mask)[0]
                            n_matched = len(matched_indices)
                            n_pts = len(points)
                            xz_x_all.extend(points[mask, 0] + x_offset)
                            xz_z_all.extend(np.full(n_matched, slice_idx + z_offset))
                            # Get angles with proper padding for resampled fibers
                            xz_angles_all.extend(get_slice_angles_masked_roi(angles, slice_idx, n_pts, mask))
                            xz_fiber_indices_all.extend(matched_indices)
                            # Get azimuths for Y-Z or true azimuth mode
                            if use_yz or use_true_azimuth:
                                xz_azimuths_all.extend(get_slice_angles_masked_roi(azimuths, slice_idx, n_pts, mask))
                            if show_fiber_diameter and prop_axis == 1:
                                xz_circle_centers.extend([(pt[0] + x_offset, slice_idx + z_offset) for pt in points[mask]])

                if xz_x_all:
                    xz_x_all = np.array(xz_x_all)
                    xz_z_all = np.array(xz_z_all)
                    xz_angles_all = np.array(xz_angles_all)
                    xz_fiber_indices_all = np.array(xz_fiber_indices_all)
                    # Ensure all arrays have the same length
                    n_points = len(xz_x_all)
                    if (use_yz or use_true_azimuth) and len(xz_azimuths_all) == n_points:
                        xz_azimuths_all = np.array(xz_azimuths_all)
                        if use_true_azimuth:
                            colors = true_azimuth_to_colors(xz_azimuths_all, xz_angles_all, xz_fiber_indices_all)
                        else:
                            colors = yz_angles_to_colors(xz_azimuths_all, xz_fiber_indices_all)
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
                # Always process - let the mask filter out points that don't match
                if roi_x_pos >= -tolerance:
                    for slice_idx, (z, points) in enumerate(trajectories):
                        mask = np.abs(points[:, 0] - roi_x_pos) < tolerance
                        if np.any(mask):
                            matched_indices = np.where(mask)[0]
                            n_matched = len(matched_indices)
                            n_pts = len(points)
                            yz_y_all.extend(points[mask, 1] + y_offset)
                            yz_z_all.extend(np.full(n_matched, slice_idx + z_offset))
                            # Get angles with proper padding for resampled fibers
                            yz_angles_all.extend(get_slice_angles_masked_roi(angles, slice_idx, n_pts, mask))
                            yz_fiber_indices_all.extend(matched_indices)
                            # Get azimuths for Y-Z or true azimuth mode
                            if use_yz or use_true_azimuth:
                                yz_azimuths_all.extend(get_slice_angles_masked_roi(azimuths, slice_idx, n_pts, mask))
                            if show_fiber_diameter and prop_axis == 0:
                                yz_circle_centers.extend([(pt[1] + y_offset, slice_idx + z_offset) for pt in points[mask]])

                if yz_y_all:
                    yz_y_all = np.array(yz_y_all)
                    yz_z_all = np.array(yz_z_all)
                    yz_angles_all = np.array(yz_angles_all)
                    yz_fiber_indices_all = np.array(yz_fiber_indices_all)
                    # Ensure all arrays have the same length
                    n_points = len(yz_y_all)
                    if (use_yz or use_true_azimuth) and len(yz_azimuths_all) == n_points:
                        yz_azimuths_all = np.array(yz_azimuths_all)
                        if use_true_azimuth:
                            colors = true_azimuth_to_colors(yz_azimuths_all, yz_angles_all, yz_fiber_indices_all)
                        else:
                            colors = yz_angles_to_colors(yz_azimuths_all, yz_fiber_indices_all)
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

            # Helper function to get angle data with proper padding for resampled fibers
            def get_padded_angles(angle_arr, slice_idx, n_pts):
                if slice_idx < len(angle_arr):
                    arr = np.array(angle_arr[slice_idx])
                    if len(arr) == n_pts:
                        return arr
                    elif len(arr) < n_pts:
                        # Pad with zeros for new fibers added by resampling
                        return np.concatenate([arr, np.zeros(n_pts - len(arr))])
                    else:
                        # Truncate if somehow longer
                        return arr[:n_pts]
                return np.zeros(n_pts)

            # Helper function to get padded angles for XZ/YZ slices with mask
            def get_slice_angles_masked(angle_arr, slice_idx, n_pts, mask):
                padded = get_padded_angles(angle_arr, slice_idx, n_pts)
                return padded[mask]

            if z_pos < len(trajectories):
                z, points = trajectories[z_pos]
                n_points = len(points)

                slice_angles = get_padded_angles(angles, z_pos, n_points)
                fiber_indices = np.arange(n_points)  # fiber index for each point

                # Get azimuths for Y-Z or true azimuth mode
                if use_yz or use_true_azimuth:
                    slice_azimuths = get_padded_angles(azimuths, z_pos, n_points)
                    if use_true_azimuth:
                        azimuth_angles = getattr(self.fiber_trajectory, 'azimuth_angles', None)
                        if azimuth_angles:
                            slice_azimuth_angles = get_padded_angles(azimuth_angles, z_pos, n_points)
                        else:
                            slice_azimuth_angles = slice_azimuths
                        colors = true_azimuth_to_colors(slice_azimuth_angles, slice_angles, fiber_indices)
                    else:
                        colors = yz_angles_to_colors(slice_azimuths, fiber_indices)
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
                    n_pts = len(points)
                    xz_x_all.extend(points[mask, 0])
                    xz_z_all.extend(np.full(n_matched, slice_idx))
                    # Get angles with proper padding for resampled fibers
                    xz_angles_all.extend(get_slice_angles_masked(angles, slice_idx, n_pts, mask))
                    xz_fiber_indices_all.extend(matched_indices)
                    # Get azimuths for Y-Z or true azimuth mode
                    if use_yz or use_true_azimuth:
                        xz_azimuths_all.extend(get_slice_angles_masked(azimuths, slice_idx, n_pts, mask))
                    if show_fiber_diameter and prop_axis == 1:
                        xz_circle_centers.extend([(pt[0], slice_idx) for pt in points[mask]])

            if xz_x_all:
                xz_x_all = np.array(xz_x_all)
                xz_z_all = np.array(xz_z_all)
                xz_angles_all = np.array(xz_angles_all)
                xz_fiber_indices_all = np.array(xz_fiber_indices_all)
                # Ensure all arrays have the same length
                n_points = len(xz_x_all)
                if (use_yz or use_true_azimuth) and len(xz_azimuths_all) == n_points:
                    xz_azimuths_all = np.array(xz_azimuths_all)
                    if use_true_azimuth:
                        colors = true_azimuth_to_colors(xz_azimuths_all, xz_angles_all, xz_fiber_indices_all)
                    else:
                        colors = yz_angles_to_colors(xz_azimuths_all, xz_fiber_indices_all)
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
                    n_pts = len(points)
                    yz_y_all.extend(points[mask, 1])
                    yz_z_all.extend(np.full(n_matched, slice_idx))
                    # Get angles with proper padding for resampled fibers
                    yz_angles_all.extend(get_slice_angles_masked(angles, slice_idx, n_pts, mask))
                    yz_fiber_indices_all.extend(matched_indices)
                    # Get azimuths for Y-Z or true azimuth mode
                    if use_yz or use_true_azimuth:
                        yz_azimuths_all.extend(get_slice_angles_masked(azimuths, slice_idx, n_pts, mask))
                    if show_fiber_diameter and prop_axis == 0:
                        yz_circle_centers.extend([(pt[1], slice_idx) for pt in points[mask]])

            if yz_y_all:
                yz_y_all = np.array(yz_y_all)
                yz_z_all = np.array(yz_z_all)
                yz_angles_all = np.array(yz_angles_all)
                yz_fiber_indices_all = np.array(yz_fiber_indices_all)
                # Ensure all arrays have the same length
                n_points = len(yz_y_all)
                if (use_yz or use_true_azimuth) and len(yz_azimuths_all) == n_points:
                    yz_azimuths_all = np.array(yz_azimuths_all)
                    if use_true_azimuth:
                        colors = true_azimuth_to_colors(yz_azimuths_all, yz_angles_all, yz_fiber_indices_all)
                    else:
                        colors = yz_angles_to_colors(yz_azimuths_all, yz_fiber_indices_all)
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
                offset = roi_data.get('offset', (0, 0, 0))
                z_offset, y_offset, x_offset = offset
                if bounds:
                    roi_z_min, roi_z_max, roi_y_min, roi_y_max, roi_x_min, roi_x_max = bounds
                    x_max = max(x_max, roi_x_max)
                    y_max = max(y_max, roi_y_max)
                    z_max = max(z_max, roi_z_max)
                else:
                    # Compute bounds from trajectory data when bounds not available
                    fiber_traj = roi_data.get('trajectory')
                    if fiber_traj and fiber_traj.trajectories:
                        trajectories = fiber_traj.trajectories
                        z_max = max(z_max, z_offset + len(trajectories))
                        for _, points in trajectories:
                            if len(points) > 0:
                                x_max = max(x_max, x_offset + np.max(points[:, 0]))
                                y_max = max(y_max, y_offset + np.max(points[:, 1]))
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
            'tilt': [],           # X-Z orientation
            'azimuth': [],        # Y-Z orientation
            'true_azimuth': []    # True azimuth: arctan2(v_d1, v_d0) without sign correction
        }

        collected = False

        def collect_from_trajectory(traj):
            """Helper to collect angles from a trajectory object."""
            nonlocal collected
            if not traj:
                return

            # Get X-Z orientation angles (tilt)
            if hasattr(traj, 'angles') and traj.angles:
                for slice_angles in traj.angles:
                    if slice_angles is not None:
                        arr = np.array(slice_angles).flatten()
                        angle_data['tilt'].extend(arr.tolist())
                        collected = True

            # Get Y-Z orientation angles (azimuth)
            if hasattr(traj, 'azimuths') and traj.azimuths:
                for slice_azimuths in traj.azimuths:
                    if slice_azimuths is not None:
                        arr = np.array(slice_azimuths).flatten()
                        angle_data['azimuth'].extend(arr.tolist())

            # Get true azimuth angles
            if hasattr(traj, 'azimuth_angles') and traj.azimuth_angles:
                for slice_azimuth_angles in traj.azimuth_angles:
                    if slice_azimuth_angles is not None:
                        arr = np.array(slice_azimuth_angles).flatten()
                        angle_data['true_azimuth'].extend(arr.tolist())

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
        angle_data['true_azimuth'] = np.array(angle_data['true_azimuth']) if angle_data['true_azimuth'] else None

        return angle_data

    def exportTrajectoryToVTK(self):
        """Export fiber trajectories to VTK format for Paraview visualization."""
        import pyvista as pv

        if self.fiber_trajectory is None and not self.roi_trajectories:
            QMessageBox.warning(self, "No Data", "No fiber trajectory data to export.")
            return

        # Show export options dialog
        dialog = ExportVTPDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        export_settings = dialog.getSettings()

        # Get save filename
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Fiber Trajectory", "",
            "VTK PolyData (*.vtp);;Legacy VTK (*.vtk);;All Files (*)"
        )
        if not filename:
            return

        # Ensure extension
        if not filename.lower().endswith(('.vtp', '.vtk')):
            filename += '.vtp'

        try:
            # Build polydata from trajectories
            all_points = []
            all_lines = []
            all_xz_angles = []      # X-Z Orientation (degrees)
            all_yz_angles = []      # Y-Z Orientation (degrees)
            all_azimuth_angles = [] # True Azimuth (-180 to 180 degrees)
            all_fiber_ids = []
            point_offset = 0
            global_fiber_idx = 0

            # Process all ROI trajectories
            trajectories_to_process = []
            if self.roi_trajectories:
                for roi_name, roi_data in self.roi_trajectories.items():
                    if not isinstance(roi_data, dict):
                        continue
                    fiber_traj = roi_data['trajectory']
                    offset = roi_data['offset']  # (z_offset, y_offset, x_offset)
                    trajectories_to_process.append((fiber_traj, offset, roi_name))
            elif self.fiber_trajectory is not None:
                trajectories_to_process.append((self.fiber_trajectory, (0, 0, 0), "main"))

            for fiber_traj, offset, roi_name in trajectories_to_process:
                z_offset, y_offset, x_offset = offset

                # Prefer fiber_trajectories (per-fiber data) over trajectories (per-slice data)
                per_fiber_trajs = getattr(fiber_traj, 'fiber_trajectories', None)
                per_fiber_angles = getattr(fiber_traj, 'fiber_angles', None)        # X-Z orientation
                per_fiber_azimuths = getattr(fiber_traj, 'fiber_azimuths', None)    # Y-Z orientation
                per_fiber_azimuth_angles = getattr(fiber_traj, 'fiber_azimuth_angles', None)  # True azimuth

                if per_fiber_trajs and len(per_fiber_trajs) > 0:
                    # Use per-fiber trajectory data (more accurate for variable-length trajectories)
                    skipped_count = 0
                    for fiber_idx, traj in enumerate(per_fiber_trajs):
                        if len(traj) < 2:
                            skipped_count += 1
                            continue  # Skip fibers with less than 2 points

                        fiber_points = []
                        fiber_xz_list = []
                        fiber_yz_list = []
                        fiber_azimuth_list = []

                        for pt_idx, (z, pt) in enumerate(traj):
                            # Coordinate mapping for VTP export:
                            # Internal: pt[0]=X (column), pt[1]=Y (row), z=Z (slice)
                            # VTP: Keep same coordinate system (X, Y, Z)
                            fiber_points.append([
                                pt[0] + x_offset,  # X = column
                                pt[1] + y_offset,  # Y = row
                                z + z_offset       # Z = slice index
                            ])

                            # Get X-Z orientation angle
                            if per_fiber_angles and fiber_idx < len(per_fiber_angles) and pt_idx < len(per_fiber_angles[fiber_idx]):
                                fiber_xz_list.append(per_fiber_angles[fiber_idx][pt_idx])
                            else:
                                fiber_xz_list.append(0.0)

                            # Get Y-Z orientation angle
                            if per_fiber_azimuths and fiber_idx < len(per_fiber_azimuths) and pt_idx < len(per_fiber_azimuths[fiber_idx]):
                                fiber_yz_list.append(per_fiber_azimuths[fiber_idx][pt_idx])
                            else:
                                fiber_yz_list.append(0.0)

                            # Get true azimuth angle (-180 to 180 degrees)
                            if per_fiber_azimuth_angles and fiber_idx < len(per_fiber_azimuth_angles) and pt_idx < len(per_fiber_azimuth_angles[fiber_idx]):
                                fiber_azimuth_list.append(per_fiber_azimuth_angles[fiber_idx][pt_idx])
                            else:
                                fiber_azimuth_list.append(0.0)

                        if len(fiber_points) > 1:
                            # Add points
                            n_pts = len(fiber_points)
                            all_points.extend(fiber_points)
                            all_xz_angles.extend(fiber_xz_list)
                            all_yz_angles.extend(fiber_yz_list)
                            all_azimuth_angles.extend(fiber_azimuth_list)
                            all_fiber_ids.extend([global_fiber_idx] * n_pts)

                            # Add line connectivity
                            line = [n_pts] + list(range(point_offset, point_offset + n_pts))
                            all_lines.extend(line)
                            point_offset += n_pts
                            global_fiber_idx += 1

                    if skipped_count > 0:
                        logger.debug(f"[{roi_name}] Skipped {skipped_count} fibers with <2 points (total: {len(per_fiber_trajs)})")
                else:
                    # Fallback to slice-based trajectories data
                    trajectories = fiber_traj.trajectories
                    angles = fiber_traj.angles if fiber_traj.angles else []           # X-Z orientation
                    azimuths = getattr(fiber_traj, 'azimuths', None)                   # Y-Z orientation
                    azimuth_angles = getattr(fiber_traj, 'azimuth_angles', None)       # True azimuth
                    if azimuths is None:
                        azimuths = []
                    if azimuth_angles is None:
                        azimuth_angles = []

                    if not trajectories:
                        continue

                    # Get number of fibers from first slice
                    n_fibers = len(trajectories[0][1]) if trajectories else 0

                    # Build fiber lines (each fiber is a polyline through all slices)
                    for fiber_idx in range(n_fibers):
                        fiber_points = []
                        fiber_xz_list = []
                        fiber_yz_list = []
                        fiber_azimuth_list = []

                        for slice_idx, (z, points) in enumerate(trajectories):
                            if fiber_idx < len(points):
                                pt = points[fiber_idx]
                                # Coordinate mapping for VTP export:
                                # Internal: pt[0]=X (column), pt[1]=Y (row), z=Z (slice)
                                # VTP: Keep same coordinate system (X, Y, Z)
                                fiber_points.append([
                                    pt[0] + x_offset,  # X = column
                                    pt[1] + y_offset,  # Y = row
                                    z + z_offset       # Z = slice index
                                ])

                                # Get X-Z orientation angle
                                if slice_idx < len(angles) and fiber_idx < len(angles[slice_idx]):
                                    fiber_xz_list.append(angles[slice_idx][fiber_idx])
                                else:
                                    fiber_xz_list.append(0.0)

                                # Get Y-Z orientation angle
                                if slice_idx < len(azimuths) and fiber_idx < len(azimuths[slice_idx]):
                                    fiber_yz_list.append(azimuths[slice_idx][fiber_idx])
                                else:
                                    fiber_yz_list.append(0.0)

                                # Get true azimuth angle (-180 to 180 degrees)
                                if slice_idx < len(azimuth_angles) and fiber_idx < len(azimuth_angles[slice_idx]):
                                    fiber_azimuth_list.append(azimuth_angles[slice_idx][fiber_idx])
                                else:
                                    fiber_azimuth_list.append(0.0)

                        if len(fiber_points) > 1:
                            # Add points
                            n_pts = len(fiber_points)
                            all_points.extend(fiber_points)
                            all_xz_angles.extend(fiber_xz_list)
                            all_yz_angles.extend(fiber_yz_list)
                            all_azimuth_angles.extend(fiber_azimuth_list)
                            all_fiber_ids.extend([global_fiber_idx] * n_pts)

                            # Add line connectivity
                            line = [n_pts] + list(range(point_offset, point_offset + n_pts))
                            all_lines.extend(line)
                            point_offset += n_pts
                            global_fiber_idx += 1

            if not all_points:
                QMessageBox.warning(self, "No Data", "No valid fiber trajectory data to export.")
                return

            # Create PyVista PolyData
            points_array = np.array(all_points)
            lines_array = np.array(all_lines)

            polydata = pv.PolyData()
            polydata.points = points_array
            polydata.lines = lines_array

            # Add scalar arrays based on export settings
            exported_arrays = []

            if export_settings['export_xz']:
                polydata.point_data['XZ_Orientation'] = np.array(all_xz_angles)
                exported_arrays.append("XZ_Orientation: X-Z plane angle (degrees)")

            if export_settings['export_yz']:
                polydata.point_data['YZ_Orientation'] = np.array(all_yz_angles)
                exported_arrays.append("YZ_Orientation: Y-Z plane angle (degrees)")

            if export_settings['export_fiber_id']:
                polydata.point_data['FiberID'] = np.array(all_fiber_ids)
                exported_arrays.append("FiberID: Unique fiber identifier")

            # Convert azimuth from -180~180 to 0~360 for cyclic colormap
            azimuth_arr = np.array(all_azimuth_angles)
            azimuth_360 = np.where(azimuth_arr < 0, azimuth_arr + 360, azimuth_arr)

            if export_settings['export_azimuth']:
                polydata.point_data['Azimuth'] = azimuth_360
                exported_arrays.append("Azimuth: True azimuth (0° to 360°, cyclic)")

            if export_settings['export_azimuth_norm']:
                # Normalize to 0-1 for direct use as Hue
                azimuth_normalized = azimuth_360 / 360.0
                polydata.point_data['Azimuth_Normalized'] = azimuth_normalized
                exported_arrays.append("Azimuth_Normalized: Azimuth/360 (0-1, for Hue)")

            # HSV to RGB conversion for pre-rendered color
            # Saturation = Tilt angle (XZ_Orientation) normalized to 0-1
            if export_settings.get('export_rgb', False):
                from matplotlib.colors import hsv_to_rgb

                # Use XZ_Orientation (tilt angle) as saturation
                xz_arr = np.array(all_xz_angles)
                tilt_abs = np.abs(xz_arr)
                # Use saturation range from settings
                sat_min = self.trajectory_settings.get('saturation_min', 0.0)
                sat_max = self.trajectory_settings.get('saturation_max', 45.0)
                sat_range = sat_max - sat_min if sat_max > sat_min else 1.0
                saturation_values = np.clip((tilt_abs - sat_min) / sat_range, 0, 1)

                # Build HSV array: H=azimuth, S=tilt, V=1.0
                n_points = len(azimuth_360)
                hsv = np.zeros((n_points, 3))
                hsv[:, 0] = azimuth_360 / 360.0  # Hue (0-1)
                hsv[:, 1] = saturation_values     # Saturation based on tilt
                hsv[:, 2] = 1.0                   # Value (brightness)

                # Convert to RGB (0-255 for VTK)
                rgb = hsv_to_rgb(hsv)
                rgb_uint8 = (rgb * 255).astype(np.uint8)

                polydata.point_data['Azimuth_RGB'] = rgb_uint8
                exported_arrays.append("Azimuth_RGB: HSV color (H=Azimuth, S=Tilt)")

            # Save to file
            polydata.save(filename)

            arrays_str = "\n".join(f"- {a}" for a in exported_arrays)

            # Count total and exported fibers
            total_fiber_count = 0
            for fiber_traj, offset, roi_name in trajectories_to_process:
                per_fiber_trajs = getattr(fiber_traj, 'fiber_trajectories', None)
                if per_fiber_trajs:
                    total_fiber_count += len(per_fiber_trajs)

            QMessageBox.information(
                self, "Export Successful",
                f"Fiber trajectories exported to:\n{filename}\n\n"
                f"Total fibers in memory: {total_fiber_count}\n"
                f"Exported fibers (≥2 points): {global_fiber_idx}\n"
                f"Skipped fibers (<2 points): {total_fiber_count - global_fiber_idx}\n"
                f"Total points: {len(all_points)}\n\n"
                f"Exported scalar arrays:\n{arrays_str}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")



class AnalysisTab(QWidget):
    """Analysis tab - ribbon is now managed by MainWindow's ribbon_stack"""
    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer
        self.main_window = None
        self.initUI()

    def initUI(self):
        # Ribbon is now managed by MainWindow's ribbon_stack
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

        # Vf (Volume Fraction) settings
        self.vf_settings = {
            'window_size': 50,
            'use_gaussian': False,
            'gaussian_sigma': 10.0,
        }

        # Vf result storage
        self.vf_map = None
        self.vf_segmentation = None
        self.vf_polygon_mask = None  # Polygon mask for Vf overlay

        # Segmentation volume (stored from Fiber Detection or InSegt)
        self.segmentation_volume = None
        self.segmentation_roi_bounds = None
        self.segmentation_polygon_mask = None  # Polygon mask for Vf calculation

        # Void analysis settings
        self.void_analysis_settings = {
            'method': 'otsu',
            'manual_threshold': 128,
            'invert': True,
            'min_size': 0,
            'closing_size': 0,
            'compute_statistics': True,
            'compute_local_vf': False,
            'local_window_size': 50,
        }

        # Void analysis results
        self.void_mask = None
        self.void_statistics = None
        self.void_local_fraction = None

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

        # Show ROI selection dialog
        dialog = ROISelectDialog(main_window, main_window.viewer.rois,
                                 title="Select ROI for Fiber Detection",
                                 button_text="Detect")
        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled

        selected_rois = dialog.getSelectedROIs()
        if not selected_rois:
            QMessageBox.warning(self, "Warning", "No ROI selected")
            return

        # Use first selected ROI for fiber detection
        roi_name = selected_rois[0]
        if len(selected_rois) > 1:
            QMessageBox.information(self, "Info",
                f"Multiple ROIs selected. Using {roi_name} for fiber detection.")

        roi_data = main_window.viewer.rois[roi_name]
        bounds = roi_data.get('bounds')
        if bounds is None:
            QMessageBox.warning(self, "Warning", "ROI bounds not defined")
            return

        z_min, z_max, y_min, y_max, x_min, x_max = bounds
        n_slices = z_max - z_min

        # Get polygon mask if available (in ROI-local coordinates)
        polygon_mask_3d = main_window.viewer.getROIPolygonMask3D(roi_name)

        main_window.status_label.setText(f"Detecting fibers in {roi_name} ({n_slices} slices)...")
        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, n_slices)
        QApplication.processEvents()

        try:
            # Import detect_fiber_centers
            from vmm.fiber_trajectory import detect_fiber_centers

            # Check if watershed display is enabled
            show_watershed = self.fiber_detection_settings.get('show_watershed', True)

            # Detect fiber centers in all slices
            all_slice_results = {}
            all_segmentation_labels = {}  # Store labels separately for Vf calculation
            total_fibers = 0
            all_diameters = []

            for i, z in enumerate(range(z_min, z_max)):
                # Update progress
                main_window.progress_bar.setValue(i)
                if i % 10 == 0:
                    main_window.status_label.setText(f"Detecting fibers: slice {i+1}/{n_slices}...")
                    QApplication.processEvents()

                # Extract the slice from ROI
                slice_image = main_window.current_volume[z, y_min:y_max, x_min:x_max].copy()

                # Apply polygon mask if available
                if polygon_mask_3d is not None:
                    slice_mask = polygon_mask_3d[i]
                    # Set pixels outside polygon to 0 (background)
                    slice_image[~slice_mask] = 0

                # Determine threshold percentile (None for Otsu)
                threshold_method = self.fiber_detection_settings.get('threshold_method', 'otsu')
                threshold_percentile = None
                if threshold_method == 'percentile':
                    threshold_percentile = self.fiber_detection_settings.get('threshold_percentile', 50.0)

                # Always get labels for Vf calculation, regardless of visualization setting
                centers, diameters, labels = detect_fiber_centers(
                    slice_image,
                    min_diameter=self.fiber_detection_settings['min_diameter'],
                    max_diameter=self.fiber_detection_settings['max_diameter'],
                    min_distance=self.fiber_detection_settings['min_distance'],
                    return_labels=True,
                    threshold_percentile=threshold_percentile
                )

                # Always store labels for Vf calculation
                all_segmentation_labels[z] = labels

                if len(centers) > 0:
                    all_slice_results[z] = {
                        'centers': centers,
                        'diameters': diameters,
                        'labels': labels if show_watershed else None  # Only store for visualization if enabled
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

            # Build and store segmentation volume for Vf calculation
            # Use all_segmentation_labels which always has labels regardless of visualization setting
            seg_volume = np.zeros((n_slices, y_max - y_min, x_max - x_min), dtype=np.uint8)
            for z, labels in all_segmentation_labels.items():
                if labels is not None:
                    seg_slice = (labels > 0).astype(np.uint8)
                    # Apply polygon mask to segmentation
                    if polygon_mask_3d is not None:
                        seg_slice[~polygon_mask_3d[z - z_min]] = 0
                    seg_volume[z - z_min] = seg_slice

            self.segmentation_volume = seg_volume
            self.segmentation_roi_bounds = bounds
            self.segmentation_polygon_mask = polygon_mask_3d  # Store for Vf calculation

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

        Opens InSegt in the same process as a modal-like window.
        """
        import cv2
        from pathlib import Path

        main_window = getattr(self, 'main_window', None)

        if not main_window:
            QMessageBox.warning(self, "Error", "No main window reference")
            return

        if main_window.current_volume is None:
            QMessageBox.warning(self, "Error", "No volume loaded. Please import a volume first.")
            return

        # Get ROI bounds using ROI selection dialog
        rois = main_window.viewer.rois if main_window.viewer else {}
        bounds = None

        if rois:
            # Show ROI selection dialog
            dialog = ROISelectDialog(main_window, rois,
                                     title="Select ROI for InSegt Labeling",
                                     button_text="Open Labeling")
            if dialog.exec() != QDialog.Accepted:
                return  # User cancelled

            selected_rois = dialog.getSelectedROIs()
            if selected_rois:
                # Use first selected ROI
                roi_name = selected_rois[0]
                if len(selected_rois) > 1:
                    QMessageBox.information(self, "Info",
                        f"Multiple ROIs selected. Using {roi_name} for InSegt labeling.")
                roi_data = rois.get(roi_name, {})
                bounds = roi_data.get('bounds')

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
            # Get settings from insegt_settings
            insegt_scale = self.insegt_settings.get('scale', 0.5)
            sigmas = self.insegt_settings.get('sigmas', [1, 2])
            patch_size = self.insegt_settings.get('patch_size', 9)
            branching_factor = self.insegt_settings.get('branching_factor', 5)
            number_layers = self.insegt_settings.get('number_layers', 4)
            training_patches = self.insegt_settings.get('training_patches', 10000)

            self._insegt_scale = insegt_scale
            self._insegt_slice_z = current_z

            # Convert to uint8 if needed
            if slice_image.dtype == np.uint16:
                image_uint8 = (slice_image / 256).astype(np.uint8)
            elif slice_image.dtype != np.uint8:
                if slice_image.max() > 0:
                    image_uint8 = ((slice_image - slice_image.min()) / (slice_image.max() - slice_image.min()) * 255).astype(np.uint8)
                else:
                    image_uint8 = np.zeros_like(slice_image, dtype=np.uint8)
            else:
                image_uint8 = slice_image

            # Downscale image for faster processing
            if insegt_scale != 1.0:
                image_scaled = cv2.resize(image_uint8, None, fx=insegt_scale, fy=insegt_scale, interpolation=cv2.INTER_AREA)
            else:
                image_scaled = image_uint8

            # Import InSegt components
            from vmm.insegt.fiber_model import FiberSegmentationModel
            from vmm.insegt.annotators.dual_panel_annotator import DualPanelAnnotator

            # Build model
            main_window.status_label.setText("Building InSegt model...")
            QApplication.processEvents()

            model = FiberSegmentationModel(
                sigmas=sigmas,
                patch_size=patch_size,
                branching_factor=branching_factor,
                number_layers=number_layers,
                training_patches=training_patches
            )
            model.build_from_image(image_scaled)
            model.set_image(image_scaled)

            # Store model for later use
            self._insegt_model = model

            # Create annotator window
            self._insegt_annotator = DualPanelAnnotator(image_scaled, model)
            self._insegt_annotator.setWindowTitle(f"InSegt Labeling - Slice {current_z}")
            self._insegt_annotator.setWindowFlags(
                self._insegt_annotator.windowFlags() | Qt.WindowStaysOnTopHint
            )

            # Connect close event to capture labels
            original_close = self._insegt_annotator.closeEvent
            def on_close(event):
                self._onInSegtClosed()
                original_close(event)
            self._insegt_annotator.closeEvent = on_close

            # Show window
            self._insegt_annotator.show()
            self._insegt_annotator.raise_()
            self._insegt_annotator.activateWindow()

            main_window.status_label.setText(
                f"InSegt labeling opened for slice {current_z}. "
                "Close the InSegt window when done."
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open InSegt labeling:\n{str(e)}")
            main_window.status_label.setText(f"InSegt error: {str(e)}")

    def _onInSegtClosed(self):
        """Handle InSegt window being closed - capture labels."""
        main_window = getattr(self, 'main_window', None)

        try:
            if hasattr(self, '_insegt_annotator') and self._insegt_annotator is not None:
                # Get labels from annotator
                self.insegt_labels = self._insegt_annotator.getLabels()
                self._insegt_labels_ready = True

                # Enable Run button in MainWindow's ribbon
                if main_window and hasattr(main_window, 'insegt_run_btn'):
                    main_window.insegt_run_btn.setEnabled(True)

                if main_window:
                    main_window.status_label.setText(
                        f"InSegt labeling completed. Click 'Run' to detect fibers."
                    )

                print(f"InSegt labels captured: shape={self.insegt_labels.shape}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Error capturing InSegt labels:\n{str(e)}")
            if main_window:
                main_window.status_label.setText(f"InSegt error: {str(e)}")

        finally:
            # Cleanup annotator reference
            self._insegt_annotator = None

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

        # Get ROI bounds using ROI selection dialog
        rois = main_window.viewer.rois if main_window.viewer else {}
        bounds = None

        if rois:
            # Show ROI selection dialog
            dialog = ROISelectDialog(main_window, rois,
                                     title="Select ROI for InSegt Segmentation",
                                     button_text="Run InSegt")
            if dialog.exec() != QDialog.Accepted:
                return  # User cancelled

            selected_rois = dialog.getSelectedROIs()
            if selected_rois:
                # Use first selected ROI
                roi_name = selected_rois[0]
                if len(selected_rois) > 1:
                    QMessageBox.information(self, "Info",
                        f"Multiple ROIs selected. Using {roi_name} for InSegt segmentation.")
                roi_data = rois.get(roi_name, {})
                bounds = roi_data.get('bounds')

        # Get polygon mask if ROI was selected
        polygon_mask_3d = None
        selected_roi_name = None
        if bounds is None:
            z_min, z_max = 0, main_window.current_volume.shape[0]
            y_min, y_max = 0, main_window.current_volume.shape[1]
            x_min, x_max = 0, main_window.current_volume.shape[2]
            bounds = (z_min, z_max, y_min, y_max, x_min, x_max)
        else:
            # Get polygon mask for the selected ROI
            if selected_rois:
                selected_roi_name = selected_rois[0]
                polygon_mask_3d = main_window.viewer.getROIPolygonMask3D(selected_roi_name)

        z_min, z_max, y_min, y_max, x_min, x_max = bounds
        n_slices = z_max - z_min

        main_window.status_label.setText(f"Building InSegt model (scale={process_scale})...")
        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, n_slices + 1)
        main_window.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            from scipy.ndimage import distance_transform_edt
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            from skimage.measure import regionprops
            import vmm.insegt.models.utils as insegt_utils

            # Use the model built during labeling (stored in _insegt_model)
            if not hasattr(self, '_insegt_model') or self._insegt_model is None:
                QMessageBox.warning(self, "Error",
                    "No InSegt model available. Please run Labeling first.")
                main_window.showProgress(False)
                return

            self.insegt_model = self._insegt_model

            main_window.progress_bar.setValue(1)
            main_window.status_label.setText(f"Detecting fibers in {n_slices} slices...")
            QApplication.processEvents()

            # Process all slices
            all_slice_results = {}
            total_fibers = 0
            all_diameters = []

            # Store InSegt multi-class segmentation (1=fiber, 2=matrix, 3=void)
            insegt_segmentation_volume = np.zeros((n_slices, y_max - y_min, x_max - x_min), dtype=np.uint8)

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

                # Store full multi-class segmentation for void analysis
                # Labels: 1=fiber, 2=matrix, 3=void
                seg_slice = segmentation.copy()
                if polygon_mask_3d is not None:
                    seg_slice[~polygon_mask_3d[i]] = 0
                insegt_segmentation_volume[i] = seg_slice

                # Fiber is class 1
                binary = (segmentation == 1)

                # Apply polygon mask if available
                if polygon_mask_3d is not None:
                    slice_mask = polygon_mask_3d[i]
                    binary = binary & slice_mask

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

            # Always store InSegt multi-class segmentation volume for void analysis
            # Labels: 0=background, 1=fiber, 2=matrix, 3=void
            # This preserves all three classes from InSegt annotation
            self.segmentation_volume = insegt_segmentation_volume
            self.segmentation_roi_bounds = bounds
            self.segmentation_polygon_mask = polygon_mask_3d  # Store for Vf calculation

            if total_fibers == 0:
                # No fibers detected, but segmentation is still available for void analysis
                QMessageBox.information(
                    self, "Result",
                    "No fibers detected with InSegt model.\n\n"
                    "Segmentation results are still available for void analysis."
                )
                main_window.status_label.setText("No fibers detected (segmentation saved)")
                return

            # Store fiber detection results
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

    def openVfSettings(self):
        """Open volume fraction settings dialog."""
        dialog = VfSettingsDialog(self, self.vf_settings)
        if dialog.exec() == QDialog.Accepted:
            self.vf_settings = dialog.getSettings()

    def computeVf(self):
        """Compute local fiber volume fraction from existing segmentation."""
        from vmm.segment import (
            estimate_local_vf,
            estimate_vf_distribution,
        )

        main_window = getattr(self, 'main_window', None)
        if not main_window or main_window.current_volume is None:
            QMessageBox.warning(self, "Error", "No volume loaded.")
            return

        # Check if segmentation is available
        if self.segmentation_volume is None or self.segmentation_roi_bounds is None:
            QMessageBox.warning(self, "Error",
                "No segmentation available.\n\n"
                "Please run 'Detect Fibers' or 'InSegt Run' first.")
            return

        # Show ROI selection dialog to confirm which ROI to use
        rois = main_window.viewer.rois if main_window.viewer else {}
        selected_roi_name = None
        polygon_mask_3d = self.segmentation_polygon_mask  # Use stored mask from segmentation

        if rois:
            dialog = ROISelectDialog(main_window, rois,
                                     title="Select ROI for Volume Fraction",
                                     button_text="Compute Vf")
            if dialog.exec() != QDialog.Accepted:
                return  # User cancelled

            selected_rois = dialog.getSelectedROIs()
            if selected_rois:
                selected_roi_name = selected_rois[0]
                # Get polygon mask for the selected ROI
                polygon_mask_3d = main_window.viewer.getROIPolygonMask3D(selected_roi_name)

        segmentation = self.segmentation_volume
        z_min, z_max, y_min, y_max, x_min, x_max = self.segmentation_roi_bounds
        n_slices = z_max - z_min

        main_window.status_label.setText(f"Computing Vf for {n_slices} slices...")
        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, 100)
        main_window.progress_bar.setValue(20)
        QApplication.processEvents()

        try:
            # Apply polygon mask to segmentation if available
            if polygon_mask_3d is not None:
                segmentation = segmentation.copy()
                segmentation[~polygon_mask_3d] = 0

            # Compute local Vf
            main_window.status_label.setText("Computing local Vf distribution...")
            QApplication.processEvents()

            window_size = self.vf_settings.get('window_size', 50)
            use_gaussian = self.vf_settings.get('use_gaussian', False)
            gaussian_sigma = self.vf_settings.get('gaussian_sigma', 10.0) if use_gaussian else None

            vf_map = estimate_local_vf(
                segmentation,
                fiber_label=1,
                void_label=3,  # Exclude void from Vf calculation
                window_size=window_size,
                gaussian_sigma=gaussian_sigma
            )

            # Apply polygon mask to vf_map (set outside to NaN)
            if polygon_mask_3d is not None:
                vf_map = vf_map.astype(np.float32)
                vf_map[~polygon_mask_3d] = np.nan

            main_window.progress_bar.setValue(70)
            QApplication.processEvents()

            # Compute statistics (only on valid pixels)
            if polygon_mask_3d is not None:
                valid_segmentation = segmentation[polygon_mask_3d]
                # Create a masked array for statistics
                hist, bin_edges, stats = estimate_vf_distribution(
                    segmentation,
                    fiber_label=1,
                    void_label=3,  # Exclude void from Vf calculation
                    window_size=window_size,
                    gaussian_sigma=gaussian_sigma
                )
                # Recalculate global_vf using only polygon region (excluding void)
                fiber_pixels = np.sum(segmentation[polygon_mask_3d] == 1)
                void_pixels = np.sum(segmentation[polygon_mask_3d] == 3)
                total_pixels = np.sum(polygon_mask_3d)
                valid_pixels = total_pixels - void_pixels
                stats['global_vf'] = fiber_pixels / valid_pixels if valid_pixels > 0 else 0
                stats['void_fraction'] = void_pixels / total_pixels if total_pixels > 0 else 0
            else:
                hist, bin_edges, stats = estimate_vf_distribution(
                    segmentation,
                    fiber_label=1,
                    void_label=3,  # Exclude void from Vf calculation
                    window_size=window_size,
                    gaussian_sigma=gaussian_sigma
                )

            main_window.progress_bar.setValue(100)
            main_window.showProgress(False)

            # Store results
            self.vf_map = vf_map
            self.vf_segmentation = segmentation
            self.vf_stats = stats
            self.vf_roi_bounds = (z_min, z_max, y_min, y_max, x_min, x_max)
            self.vf_polygon_mask = polygon_mask_3d  # Store polygon mask for overlay

            # Store in viewer for display
            main_window.viewer.vf_map = vf_map
            main_window.viewer.vf_roi_bounds = self.vf_roi_bounds
            main_window.viewer.vf_polygon_mask = polygon_mask_3d  # Store for overlay rendering
            main_window.viewer.show_vf_overlay = True

            # Add Vf toggle to pipeline
            if hasattr(main_window, 'addVfToggle'):
                main_window.addVfToggle()

            main_window.viewer.renderVolume()

            # Show results dialog
            smoothing_info = f"Gaussian sigma: {gaussian_sigma} px" if use_gaussian else "Box averaging"
            void_info = f"\n  Void fraction: {stats.get('void_fraction', 0)*100:.2f}%" if stats.get('void_fraction', 0) > 0 else ""
            msg = (
                f"Volume Fraction Analysis Complete\n\n"
                f"Window size: {window_size} px\n"
                f"Smoothing: {smoothing_info}\n\n"
                f"Results:\n"
                f"  Global Vf: {stats['global_vf']*100:.1f}%\n"
                f"  Mean local Vf: {stats['mean']*100:.1f}%\n"
                f"  Std deviation: {stats['std']*100:.1f}%\n"
                f"  Min Vf: {stats['min']*100:.1f}%\n"
                f"  Max Vf: {stats['max']*100:.1f}%{void_info}"
            )

            void_status = f", void={stats.get('void_fraction', 0)*100:.2f}%" if stats.get('void_fraction', 0) > 0 else ""
            main_window.status_label.setText(
                f"Vf computed: mean={stats['mean']*100:.1f}%, global={stats['global_vf']*100:.1f}%{void_status}"
            )

            # Enable Vf export button in MainWindow's ribbon
            if hasattr(main_window, 'export_vf_btn'):
                main_window.export_vf_btn.setEnabled(True)

            QMessageBox.information(self, "Volume Fraction Results", msg)

        except Exception as e:
            main_window.showProgress(False)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Vf computation failed:\n{str(e)}")
            main_window.status_label.setText(f"Vf error: {str(e)}")

    def toggleROIEdit(self, checked):
        """Toggle ROI editing mode"""
        main_window = getattr(self, 'main_window', None)
        if main_window and main_window.viewer:
            main_window.viewer.toggleROI(checked)

        # Update button text and disable compute button when editing (in MainWindow's ribbon)
        if main_window:
            if checked:
                if hasattr(main_window, 'edit_roi_btn'):
                    main_window.edit_roi_btn.setText("Apply\nROI")
                if hasattr(main_window, 'compute_btn'):
                    main_window.compute_btn.setEnabled(False)
            else:
                if hasattr(main_window, 'edit_roi_btn'):
                    main_window.edit_roi_btn.setText("Edit\nROI")
                if hasattr(main_window, 'compute_btn'):
                    main_window.compute_btn.setEnabled(True)

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
            if hasattr(main_window, 'magnify_btn'):
                main_window.magnify_btn.setText("Reset\nZoom")
            main_window.status_label.setText("Magnify mode: Use mouse wheel to zoom, drag to pan")
        else:
            # Disable zoom mode and reset view
            main_window.viewer.enableZoom(False)
            if hasattr(main_window, 'magnify_btn'):
                main_window.magnify_btn.setText("Magnify")
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

        # Show ROI selection dialog if ROIs exist
        selected_rois = []
        if main_window.viewer.rois:
            dialog = ROISelectDialog(main_window, main_window.viewer.rois,
                                     title="Select ROIs for Orientation Computation",
                                     button_text="Compute")
            if dialog.exec() != QDialog.Accepted:
                return  # User cancelled
            selected_rois = dialog.getSelectedROIs()

        if main_window.current_volume is not None:
            base_volume = main_window.current_volume

            # If ROIs selected, compute for each ROI
            if selected_rois:
                for roi_name in selected_rois:
                    if roi_name in main_window.viewer.rois:
                        bounds = main_window.viewer.rois[roi_name]['bounds']
                        z_min, z_max, y_min, y_max, x_min, x_max = bounds
                        logger.info(f"Computing orientation for {roi_name}: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
                        volume = base_volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()

                        # Get polygon mask if available
                        polygon_mask_3d = main_window.viewer.getROIPolygonMask3D(roi_name)

                        logger.info(f"ROI volume shape: {volume.shape}")
                        self._computeOrientationForROI(main_window, volume, roi_name, polygon_mask_3d)
            else:
                # No ROIs - compute for entire volume
                logger.info("No ROIs selected - computing for entire volume")
                volume = base_volume
                roi_name = None
                self._computeOrientationForROI(main_window, volume, roi_name, None)

    def _computeOrientationForROI(self, main_window, volume, roi_name, polygon_mask_3d=None):
        """Compute orientation for a single ROI or entire volume

        Args:
            main_window: Reference to main window
            volume: 3D volume data
            roi_name: Name of the ROI (or None for entire volume)
            polygon_mask_3d: Optional 3D polygon mask (True inside polygon)
        """
        noise_scale = main_window.noise_scale_slider.value()

        main_window.showProgress(True)
        main_window.progress_bar.setRange(0, 3)

        try:
            # Step 1: Compute structure tensor
            main_window.progress_bar.setValue(1)
            roi_label = f" for {roi_name}" if roi_name else ""
            main_window.progress_bar.setFormat(f"Computing structure tensor{roi_label}... (1/3)")
            QApplication.processEvents()

            structure_tensor = compute_structure_tensor(volume, noise_scale=noise_scale)

            # Step 2: Compute orientation without reference (theta, phi)
            main_window.progress_bar.setValue(2)
            main_window.progress_bar.setFormat(f"Computing orientation angles{roi_label}... (2/3)")
            QApplication.processEvents()

            # Use the working compute_orientation function
            theta, phi = compute_orientation(structure_tensor)

            # Step 3: Compute reference orientation and trim edges
            main_window.progress_bar.setValue(3)
            main_window.progress_bar.setFormat(f"Computing reference orientation{roi_label}... (3/3)")
            QApplication.processEvents()

            # Reference vector is always Z-axis (fibers assumed to be aligned along Z direction)
            reference_vector = [1, 0, 0]  # Z-axis

            # Compute reference orientation using proper VMM-FRC method
            reference_orientation = compute_orientation(structure_tensor, reference_vector)

            # Trim edges from all orientation volumes using noise_scale as trim width
            trim_width = noise_scale
            theta_trimmed = drop_edges_3D(trim_width, theta)
            phi_trimmed = drop_edges_3D(trim_width, phi)
            reference_trimmed = drop_edges_3D(trim_width, reference_orientation)

            # Apply polygon mask if available (set values outside polygon to NaN)
            if polygon_mask_3d is not None:
                # Trim the mask to match the trimmed orientation data
                mask_trimmed = polygon_mask_3d[trim_width:-trim_width, trim_width:-trim_width, trim_width:-trim_width]
                theta_trimmed = np.where(mask_trimmed, theta_trimmed, np.nan)
                phi_trimmed = np.where(mask_trimmed, phi_trimmed, np.nan)
                reference_trimmed = np.where(mask_trimmed, reference_trimmed, np.nan)

            # Initialize orientation_data if None (e.g., after Reset All)
            if main_window.orientation_data is None:
                main_window.orientation_data = {}

            # Store trimmed orientation data and trim information
            main_window.orientation_data['theta'] = theta_trimmed
            main_window.orientation_data['phi'] = phi_trimmed
            main_window.orientation_data['reference'] = reference_trimmed
            main_window.orientation_data['trim_width'] = trim_width
            main_window.orientation_data['structure_tensor'] = structure_tensor
            main_window.orientation_data['noise_scale'] = noise_scale
            main_window.orientation_data['roi_name'] = roi_name  # Store which ROI this data is for

            main_window.showProgress(False)
            main_window.status_label.setText(f"Analysis complete{roi_label} (Noise scale: {noise_scale})")

            # Enable edit range and histogram buttons in MainWindow's ribbon
            if hasattr(main_window, 'edit_range_btn'):
                main_window.edit_range_btn.setEnabled(True)
            if hasattr(main_window, 'histogram_btn'):
                main_window.histogram_btn.setEnabled(True)

            # Enable export buttons in MainWindow's ribbon
            if hasattr(main_window, 'export_orientation_btn'):
                main_window.export_orientation_btn.setEnabled(True)

            # Enable histogram button in Simulation tab as well
            if hasattr(main_window, 'sim_histogram_btn'):
                main_window.sim_histogram_btn.setEnabled(True)

            # Pass structure tensor to Modelling tab for fiber trajectory generation
            if hasattr(main_window, 'modelling_tab'):
                main_window.modelling_tab.setStructureTensor(structure_tensor, volume.shape, volume)

            # Store orientation data in the ROI structure
            if roi_name and roi_name in main_window.viewer.rois:
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
            import traceback
            traceback.print_exc()

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

    def exportOrientationToVTK(self):
        """Export orientation volume data to VTK format for Paraview."""
        import pyvista as pv

        main_window = getattr(self, 'main_window', None)
        if not main_window:
            QMessageBox.warning(self, "Error", "No main window reference.")
            return

        orientation_data = main_window.orientation_data
        if not orientation_data or 'theta' not in orientation_data:
            QMessageBox.warning(self, "No Data", "No orientation data available.\nPlease compute orientation first.")
            return

        # Show selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Orientation Data")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        # Description
        desc_label = QLabel("Select orientation data to export:")
        layout.addWidget(desc_label)

        # Checkboxes for each orientation type
        theta_check = QCheckBox("Theta (Azimuthal angle)")
        theta_check.setChecked(True)
        layout.addWidget(theta_check)

        phi_check = QCheckBox("Phi (Elevation angle)")
        phi_check.setChecked(True)
        layout.addWidget(phi_check)

        reference_check = QCheckBox("Reference (Angle from reference axis)")
        reference_check.setChecked(True)
        reference_check.setEnabled(orientation_data.get('reference') is not None)
        layout.addWidget(reference_check)

        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Export")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        if dialog.exec() != QDialog.Accepted:
            return

        # Check if at least one is selected
        if not (theta_check.isChecked() or phi_check.isChecked() or reference_check.isChecked()):
            QMessageBox.warning(self, "No Selection", "Please select at least one orientation type to export.")
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Orientation Volume", "",
            "VTK ImageData (*.vti);;Legacy VTK (*.vtk);;All Files (*)"
        )
        if not filename:
            return

        if not filename.lower().endswith(('.vti', '.vtk')):
            filename += '.vti'

        try:
            theta = orientation_data['theta']
            phi = orientation_data['phi']
            reference = orientation_data.get('reference', None)

            # Transpose from (Z, Y, X) to (X, Y, Z) for VTK coordinate system
            # This matches the fiber trajectory VTP export coordinate system
            theta_vtk = np.transpose(theta, (2, 1, 0))
            phi_vtk = np.transpose(phi, (2, 1, 0))
            if reference is not None:
                reference_vtk = np.transpose(reference, (2, 1, 0))

            # Create VTK ImageData (uniform grid)
            grid = pv.ImageData()
            grid.dimensions = np.array(theta_vtk.shape) + 1  # VTK needs n+1 for cell data
            grid.spacing = (1, 1, 1)

            exported_fields = []

            # Add selected scalar arrays
            if theta_check.isChecked():
                grid.cell_data['Theta'] = theta_vtk.flatten(order='F')
                exported_fields.append("Theta (Azimuthal angle)")

            if phi_check.isChecked():
                grid.cell_data['Phi'] = phi_vtk.flatten(order='F')
                exported_fields.append("Phi (Elevation angle)")

            if reference_check.isChecked() and reference is not None:
                grid.cell_data['Reference'] = reference_vtk.flatten(order='F')
                exported_fields.append("Reference (Angle from reference axis)")

            grid.save(filename)

            fields_str = "\n".join(f"- {f}" for f in exported_fields)
            QMessageBox.information(
                self, "Export Successful",
                f"Orientation volume exported to:\n{filename}\n\n"
                f"Volume shape: {theta.shape}\n\n"
                f"Exported fields:\n{fields_str}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def exportVfMapToVTK(self):
        """Export fiber volume fraction map to VTK format."""
        import pyvista as pv

        if not hasattr(self, 'vf_map') or self.vf_map is None:
            QMessageBox.warning(self, "No Data", "No Vf map available.\nPlease compute Vf first.")
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Vf Map", "",
            "VTK ImageData (*.vti);;Legacy VTK (*.vtk);;All Files (*)"
        )
        if not filename:
            return

        if not filename.lower().endswith(('.vti', '.vtk')):
            filename += '.vti'

        try:
            vf_map = self.vf_map

            # Transpose from (Z, Y, X) to (X, Y, Z) for VTK coordinate system
            # This matches the fiber trajectory VTP export coordinate system
            vf_map_vtk = np.transpose(vf_map, (2, 1, 0))

            # Create VTK ImageData
            grid = pv.ImageData()
            grid.dimensions = np.array(vf_map_vtk.shape) + 1
            grid.spacing = (1, 1, 1)

            # Add Vf as cell data
            grid.cell_data['VolumeFraction'] = vf_map_vtk.flatten(order='F')

            # Also add percentage for convenience
            grid.cell_data['Vf_Percent'] = (vf_map_vtk * 100).flatten(order='F')

            grid.save(filename)

            stats = self.vf_stats if hasattr(self, 'vf_stats') else {}

            QMessageBox.information(
                self, "Export Successful",
                f"Vf map exported to:\n{filename}\n\n"
                f"Volume shape: {vf_map.shape}\n"
                f"Vf range: [{vf_map.min()*100:.1f}%, {vf_map.max()*100:.1f}%]\n\n"
                "Available data in Paraview:\n"
                "- VolumeFraction: Fiber volume fraction (0-1)\n"
                "- Vf_Percent: Fiber volume fraction (%)"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def openVoidSettings(self):
        """Open void analysis settings dialog."""
        # Check if InSegt results are available
        insegt_available = self.segmentation_volume is not None

        dialog = VoidAnalysisSettingsDialog(
            self,
            self.void_analysis_settings,
            insegt_available=insegt_available
        )
        if dialog.exec() == QDialog.Accepted:
            self.void_analysis_settings = dialog.getSettings()

    def runVoidAnalysis(self):
        """Run void analysis on volume data."""
        from vmm.analysis import (
            segment_voids_otsu, segment_voids_from_insegt,
            compute_void_statistics, compute_local_void_fraction
        )

        main_window = getattr(self, 'main_window', None)

        if not main_window:
            print("No main window reference")
            return

        if main_window.current_volume is None:
            QMessageBox.warning(self, "Warning", "No volume loaded - please import a volume first")
            return

        method = self.void_analysis_settings.get('method', 'otsu')

        # For InSegt method, check if segmentation is available
        if method == 'insegt' and self.segmentation_volume is None:
            QMessageBox.warning(
                self, "Warning",
                "InSegt segmentation results not available.\n"
                "Please run InSegt segmentation first, or use Otsu/Manual method."
            )
            return

        # Show ROI selection dialog (optional - use Full Volume if no ROI)
        if main_window.viewer.rois:
            dialog = ROISelectDialog(
                main_window, main_window.viewer.rois,
                title="Select ROI for Void Analysis",
                button_text="Analyze"
            )
            if dialog.exec() != QDialog.Accepted:
                return  # User cancelled

            selected_rois = dialog.getSelectedROIs()
            if selected_rois:
                roi_name = selected_rois[0]
                roi_data = main_window.viewer.rois[roi_name]
                roi_bounds = roi_data['bounds']
                # bounds is [z_min, z_max, y_min, y_max, x_min, x_max]
                z_start, z_end = roi_bounds[0], roi_bounds[1]
                y_start, y_end = roi_bounds[2], roi_bounds[3]
                x_start, x_end = roi_bounds[4], roi_bounds[5]
                volume = main_window.current_volume[z_start:z_end, y_start:y_end, x_start:x_end]
            else:
                # Use full volume
                volume = main_window.current_volume
                roi_bounds = None
        else:
            # No ROIs defined, use full volume
            volume = main_window.current_volume
            roi_bounds = None

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Segment voids based on method
            if method == 'otsu':
                void_mask, threshold = segment_voids_otsu(
                    volume,
                    invert=self.void_analysis_settings.get('invert', True),
                    min_size=self.void_analysis_settings.get('min_size', 0),
                    closing_size=self.void_analysis_settings.get('closing_size', 0)
                )
                print(f"[Void Analysis] Otsu threshold: {threshold:.1f}")

            elif method == 'insegt':
                # For InSegt method, use the stored segmentation volume directly
                # The segmentation_volume is already cropped to the ROI used during InSegt
                # so we should NOT apply roi_bounds again
                seg_volume = self.segmentation_volume

                # Debug info
                print(f"[Void Analysis] InSegt segmentation volume shape: {seg_volume.shape}")
                unique_labels = np.unique(seg_volume)
                print(f"[Void Analysis] Unique labels in segmentation: {unique_labels}")
                void_count = np.sum(seg_volume == 3)
                print(f"[Void Analysis] Void label (3) voxel count: {void_count}")

                void_mask = segment_voids_from_insegt(seg_volume, void_label=3)

            else:  # manual
                threshold = self.void_analysis_settings.get('manual_threshold', 128)
                invert = self.void_analysis_settings.get('invert', True)

                if invert:
                    void_mask = volume < threshold
                else:
                    void_mask = volume > threshold

                # Apply min_size filter
                min_size = self.void_analysis_settings.get('min_size', 0)
                if min_size > 0:
                    from skimage.measure import label, regionprops
                    labeled = label(void_mask)
                    for region in regionprops(labeled):
                        if region.area < min_size:
                            void_mask[labeled == region.label] = False

            # Store void mask
            self.void_mask = void_mask

            # Pass void mask to viewer for visualization
            if roi_bounds is not None:
                void_roi_bounds = roi_bounds
            else:
                # Full volume bounds
                vol_shape = main_window.current_volume.shape
                void_roi_bounds = [0, vol_shape[0], 0, vol_shape[1], 0, vol_shape[2]]

            # For InSegt method, use the stored segmentation ROI bounds
            if method == 'insegt' and hasattr(self, 'segmentation_roi_bounds') and self.segmentation_roi_bounds is not None:
                void_roi_bounds = self.segmentation_roi_bounds

            main_window.viewer.void_mask = void_mask
            main_window.viewer.void_roi_bounds = void_roi_bounds
            main_window.viewer.show_void_overlay = True
            main_window.viewer.renderVolume()

            # Add void toggle to pipeline panel
            if hasattr(main_window, 'addVoidToggle'):
                main_window.addVoidToggle()

            # Compute statistics if requested
            if self.void_analysis_settings.get('compute_statistics', True):
                self.void_statistics = compute_void_statistics(void_mask)
            else:
                self.void_statistics = None

            # Compute local void fraction if requested
            if self.void_analysis_settings.get('compute_local_vf', False):
                window_size = self.void_analysis_settings.get('local_window_size', 50)
                self.void_local_fraction = compute_local_void_fraction(void_mask, window_size=window_size)
            else:
                self.void_local_fraction = None

            QApplication.restoreOverrideCursor()

            # Show results
            self._showVoidAnalysisResults()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Void analysis failed: {str(e)}")

    def _showVoidAnalysisResults(self):
        """Display void analysis results in a dialog."""
        if self.void_mask is None:
            return

        # Calculate basic void fraction
        void_fraction = np.mean(self.void_mask) * 100

        result_text = f"Void Fraction: {void_fraction:.2f}%\n\n"

        if self.void_statistics:
            stats = self.void_statistics
            result_text += f"Number of Voids: {stats['num_voids']}\n"
            result_text += f"Total Void Volume: {stats['total_void_volume']:.0f} voxels\n"
            result_text += f"Mean Void Size: {stats['mean_void_size']:.1f} voxels\n"
            result_text += f"Max Void Size: {stats['max_void_size']:.0f} voxels\n"
            result_text += f"Mean Sphericity: {stats['mean_sphericity']:.3f}\n"

        # Show histogram of void sizes if available
        if self.void_statistics and len(self.void_statistics.get('void_sizes', [])) > 0:
            # Create a dialog with histogram
            dialog = QDialog(self)
            dialog.setWindowTitle("Void Analysis Results")
            dialog.setMinimumSize(600, 500)

            layout = QVBoxLayout(dialog)

            # Text results
            text_label = QLabel(result_text)
            text_label.setStyleSheet("font-family: monospace;")
            layout.addWidget(text_label)

            # Histogram
            fig = Figure(figsize=(6, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            sizes = self.void_statistics['void_sizes']
            ax.hist(sizes, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Void Size (voxels)')
            ax.set_ylabel('Count')
            ax.set_title('Void Size Distribution')
            if len(sizes) > 0 and np.min(sizes) > 0:
                ax.set_xscale('log')

            fig.tight_layout()
            layout.addWidget(canvas)

            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()
        else:
            # Simple message box
            QMessageBox.information(self, "Void Analysis Results", result_text)

    def cropOrientationWithVoid(self):
        """Crop orientation data using void mask."""
        from vmm.analysis import mask_orientation_with_voids

        main_window = getattr(self, 'main_window', None)
        if not main_window:
            QMessageBox.warning(self, "Warning", "No main window reference")
            return

        # Check for orientation data - stored in orientation_data dict
        orientation_available = (
            hasattr(main_window, 'orientation_data') and
            main_window.orientation_data is not None and
            main_window.orientation_data.get('theta') is not None
        )

        # Check for void mask
        void_available = self.void_mask is not None

        # Show dialog
        dialog = CropOrientationDialog(
            self,
            void_available=void_available,
            orientation_available=orientation_available
        )

        if dialog.exec() != QDialog.Accepted:
            return

        settings = dialog.getSettings()
        dilation_pixels = settings['dilation_pixels']

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Get orientation data from orientation_data dict
            theta = main_window.orientation_data['theta']
            phi = main_window.orientation_data.get('phi', None)

            # Get void mask and align with orientation data
            void_mask = self.void_mask
            void_roi_bounds = getattr(self, 'segmentation_roi_bounds', None)

            # Debug: print shapes and bounds
            print(f"[Crop Orientation] Theta shape: {theta.shape}")
            print(f"[Crop Orientation] Void mask shape: {void_mask.shape}")
            print(f"[Crop Orientation] Void mask has voids: {np.any(void_mask)} (count: {np.sum(void_mask)})")
            print(f"[Crop Orientation] Void ROI bounds: {void_roi_bounds}")

            # If we have ROI bounds for void, we need to create aligned mask
            if void_roi_bounds is not None:
                # void_roi_bounds is in volume coordinates: [z_min, z_max, y_min, y_max, x_min, x_max]
                # void_mask shape matches the ROI size from void_roi_bounds
                # theta is already trimmed (has trim_width removed from edges)

                # Calculate the size difference (trim_width applied to theta)
                void_size = void_mask.shape  # e.g., (30, 112, 127) - original ROI size
                theta_size = theta.shape     # e.g., (10, 92, 107) - trimmed ROI size

                # Calculate trim amounts (theta is smaller due to edge trimming)
                z_trim = (void_size[0] - theta_size[0]) // 2
                y_trim = (void_size[1] - theta_size[1]) // 2
                x_trim = (void_size[2] - theta_size[2]) // 2

                print(f"[Crop Orientation] Trim amounts: z={z_trim}, y={y_trim}, x={x_trim}")

                # Extract the central region of void_mask that matches theta
                z_start = max(0, z_trim)
                z_end = z_start + theta_size[0]
                y_start = max(0, y_trim)
                y_end = y_start + theta_size[1]
                x_start = max(0, x_trim)
                x_end = x_start + theta_size[2]

                # Ensure we don't exceed void_mask bounds
                z_end = min(z_end, void_size[0])
                y_end = min(y_end, void_size[1])
                x_end = min(x_end, void_size[2])

                # Extract aligned void mask
                void_mask_aligned = void_mask[z_start:z_end, y_start:y_end, x_start:x_end].copy()

                print(f"[Crop Orientation] Extracted void mask region [{z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}]")
                print(f"[Crop Orientation] Aligned void mask shape: {void_mask_aligned.shape}")
                print(f"[Crop Orientation] Aligned void mask has voids: {np.any(void_mask_aligned)} (count: {np.sum(void_mask_aligned)})")

                # If shapes still don't match, create padded/cropped mask
                if void_mask_aligned.shape != theta.shape:
                    print(f"[Crop Orientation] Shape mismatch, adjusting mask to theta shape")
                    full_void_mask = np.zeros(theta.shape, dtype=bool)
                    # Copy what we can
                    copy_z = min(void_mask_aligned.shape[0], theta.shape[0])
                    copy_y = min(void_mask_aligned.shape[1], theta.shape[1])
                    copy_x = min(void_mask_aligned.shape[2], theta.shape[2])
                    full_void_mask[:copy_z, :copy_y, :copy_x] = void_mask_aligned[:copy_z, :copy_y, :copy_x]
                    void_mask_aligned = full_void_mask
                    print(f"[Crop Orientation] Final aligned mask has voids: {np.any(void_mask_aligned)} (count: {np.sum(void_mask_aligned)})")
            else:
                # Assume void mask is same size as orientation
                if void_mask.shape != theta.shape:
                    QMessageBox.warning(
                        self, "Warning",
                        f"Void mask shape {void_mask.shape} does not match "
                        f"orientation shape {theta.shape}."
                    )
                    QApplication.restoreOverrideCursor()
                    return
                void_mask_aligned = void_mask

            # Check if void mask has any valid data
            if void_mask_aligned.size == 0 or not np.any(void_mask_aligned):
                QMessageBox.warning(
                    self, "Warning",
                    "No void regions found in the mask.\n"
                    "Please run void analysis first or check your segmentation results."
                )
                QApplication.restoreOverrideCursor()
                return

            # Apply masking
            if phi is not None:
                masked_theta, masked_phi = mask_orientation_with_voids(
                    theta, void_mask_aligned, dilation_pixels, phi
                )
                main_window.orientation_phi = masked_phi
            else:
                masked_theta = mask_orientation_with_voids(
                    theta, void_mask_aligned, dilation_pixels
                )

            # Store masked orientation in orientation_data dict
            main_window.orientation_data['theta'] = masked_theta
            if phi is not None:
                main_window.orientation_data['phi'] = masked_phi

            # Update ROI orientation data with masking applied
            # The masked_theta is already in ROI-local coordinates (same size as ROI theta)
            rois_updated = 0
            if hasattr(main_window, 'viewer') and hasattr(main_window.viewer, 'rois'):
                for roi_name, roi_data in main_window.viewer.rois.items():
                    if 'theta' in roi_data and roi_data['theta'] is not None:
                        print(f"[Crop Orientation] ROI '{roi_name}' theta shape: {roi_data['theta'].shape}")
                        print(f"[Crop Orientation] masked_theta shape: {masked_theta.shape}")
                        # masked_theta is already cropped to ROI size, just assign directly
                        # But we need to check if shapes match
                        if roi_data['theta'].shape == masked_theta.shape:
                            roi_data['theta'] = masked_theta.astype(np.float32)
                            rois_updated += 1
                            print(f"[Crop Orientation] ROI '{roi_name}' theta updated")

                            # Also update phi if available
                            if phi is not None and 'phi' in roi_data:
                                roi_data['phi'] = masked_phi.astype(np.float32)

                            # Also update angle if present
                            if 'angle' in roi_data and roi_data['angle'] is not None:
                                if roi_data['angle'].shape == masked_theta.shape:
                                    # Apply same mask to angle data
                                    masked_angle = roi_data['angle'].copy().astype(np.float64)
                                    masked_angle[np.isnan(masked_theta)] = np.nan
                                    roi_data['angle'] = masked_angle.astype(np.float32)
                                    print(f"[Crop Orientation] ROI '{roi_name}' angle updated")
                        else:
                            print(f"[Crop Orientation] Shape mismatch for ROI '{roi_name}', skipping")

            QApplication.restoreOverrideCursor()

            # Count masked voxels
            num_masked = np.sum(np.isnan(masked_theta))
            total_voxels = masked_theta.size
            mask_percent = (num_masked / total_voxels) * 100

            # Update viewer and histogram if requested
            if settings['update_viewer']:
                main_window.viewer.renderVolume()

                # Update histogram to reflect masked data
                if hasattr(self, 'updateHistogram'):
                    self.updateHistogram()
                elif hasattr(main_window, 'analysis_tab') and hasattr(main_window.analysis_tab, 'updateHistogram'):
                    main_window.analysis_tab.updateHistogram()

            # Export if requested
            if settings['export_vtk']:
                self._exportMaskedOrientationVTK(masked_theta, phi if phi is None else masked_phi)

            result_msg = (
                f"Orientation data masked with void regions.\n\n"
                f"Dilation: {dilation_pixels} pixels\n"
                f"Masked voxels: {num_masked:,} ({mask_percent:.2f}%)\n"
                f"Valid voxels: {total_voxels - num_masked:,}"
            )
            if rois_updated > 0:
                result_msg += f"\n\nROIs updated: {rois_updated}\n(Histograms will reflect masked data)"

            QMessageBox.information(self, "Crop Orientation Complete", result_msg)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to crop orientation: {str(e)}")

    def _exportMaskedOrientationVTK(self, theta, phi=None):
        """Export masked orientation to VTK file."""
        from vmm.io import export_orientation_to_vtk

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Masked Orientation",
            "",
            "VTK Files (*.vtk);;All Files (*)"
        )

        if filepath:
            try:
                export_orientation_to_vtk(filepath, theta, phi)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Masked orientation exported to:\n{filepath}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Failed to export: {str(e)}"
                )


class CoordinateROIDialog(QDialog):
    """Dialog for creating ROI by entering coordinates"""
    def __init__(self, parent, volume_shape=None):
        super().__init__(parent)
        self.volume_shape = volume_shape  # (z, y, x)
        self.roi_bounds = None
        self.roi_name = None
        self.setWindowTitle("Create ROI by Coordinates")
        self.setModal(True)
        self.resize(400, 350)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("Enter ROI Coordinates")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # ROI Name
        name_group = QGroupBox("ROI Name")
        name_layout = QHBoxLayout(name_group)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter ROI name (e.g., ROI_1)")
        name_layout.addWidget(self.name_edit)
        layout.addWidget(name_group)

        # Get volume bounds for spinbox ranges
        z_max = self.volume_shape[0] if self.volume_shape else 1000
        y_max = self.volume_shape[1] if self.volume_shape else 1000
        x_max = self.volume_shape[2] if self.volume_shape else 1000

        # X coordinates
        x_group = QGroupBox("X Range (Width)")
        x_layout = QGridLayout(x_group)
        x_layout.addWidget(QLabel("Min:"), 0, 0)
        self.x_min_spin = QSpinBox()
        self.x_min_spin.setRange(0, x_max - 1)
        self.x_min_spin.setValue(0)
        x_layout.addWidget(self.x_min_spin, 0, 1)
        x_layout.addWidget(QLabel("Max:"), 0, 2)
        self.x_max_spin = QSpinBox()
        self.x_max_spin.setRange(1, x_max)
        self.x_max_spin.setValue(x_max)
        x_layout.addWidget(self.x_max_spin, 0, 3)
        x_layout.addWidget(QLabel(f"(0 - {x_max})"), 0, 4)
        layout.addWidget(x_group)

        # Y coordinates
        y_group = QGroupBox("Y Range (Height)")
        y_layout = QGridLayout(y_group)
        y_layout.addWidget(QLabel("Min:"), 0, 0)
        self.y_min_spin = QSpinBox()
        self.y_min_spin.setRange(0, y_max - 1)
        self.y_min_spin.setValue(0)
        y_layout.addWidget(self.y_min_spin, 0, 1)
        y_layout.addWidget(QLabel("Max:"), 0, 2)
        self.y_max_spin = QSpinBox()
        self.y_max_spin.setRange(1, y_max)
        self.y_max_spin.setValue(y_max)
        y_layout.addWidget(self.y_max_spin, 0, 3)
        y_layout.addWidget(QLabel(f"(0 - {y_max})"), 0, 4)
        layout.addWidget(y_group)

        # Z coordinates
        z_group = QGroupBox("Z Range (Depth/Slices)")
        z_layout = QGridLayout(z_group)
        z_layout.addWidget(QLabel("Min:"), 0, 0)
        self.z_min_spin = QSpinBox()
        self.z_min_spin.setRange(0, z_max - 1)
        self.z_min_spin.setValue(0)
        z_layout.addWidget(self.z_min_spin, 0, 1)
        z_layout.addWidget(QLabel("Max:"), 0, 2)
        self.z_max_spin = QSpinBox()
        self.z_max_spin.setRange(1, z_max)
        self.z_max_spin.setValue(z_max)
        z_layout.addWidget(self.z_max_spin, 0, 3)
        z_layout.addWidget(QLabel(f"(0 - {z_max})"), 0, 4)
        layout.addWidget(z_group)

        # Info label
        info_label = QLabel(
            "Note: Coordinates are in pixels. The ROI will be a rectangular region\n"
            "defined by the min/max values for each axis."
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create ROI")
        create_btn.setDefault(True)
        create_btn.clicked.connect(self._onCreate)
        create_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

    def _onCreate(self):
        """Validate and create ROI"""
        # Validate name
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a ROI name.")
            return

        # Get coordinates
        x_min = self.x_min_spin.value()
        x_max = self.x_max_spin.value()
        y_min = self.y_min_spin.value()
        y_max = self.y_max_spin.value()
        z_min = self.z_min_spin.value()
        z_max = self.z_max_spin.value()

        # Validate ranges
        if x_min >= x_max:
            QMessageBox.warning(self, "Warning", "X Min must be less than X Max.")
            return
        if y_min >= y_max:
            QMessageBox.warning(self, "Warning", "Y Min must be less than Y Max.")
            return
        if z_min >= z_max:
            QMessageBox.warning(self, "Warning", "Z Min must be less than Z Max.")
            return

        # Store results
        self.roi_name = name
        self.roi_bounds = (z_min, z_max, y_min, y_max, x_min, x_max)

        self.accept()

    def getROI(self):
        """Return the created ROI name and bounds"""
        return self.roi_name, self.roi_bounds


class ROISelectDialog(QDialog):
    """Dialog for selecting ROIs before analysis operations"""
    def __init__(self, parent, rois, title="Select ROIs", button_text="OK"):
        super().__init__(parent)
        self.rois = rois  # Dictionary of ROIs from viewer
        self.selected_rois = []
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 350)
        self._button_text = button_text
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title - extracted from window title
        title_label = QLabel("Select ROIs for the operation:")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)

        # ROI list with checkboxes
        roi_group = QGroupBox("Available ROIs")
        roi_layout = QVBoxLayout(roi_group)

        self.roi_checkboxes = {}
        if self.rois:
            for roi_name, roi_data in self.rois.items():
                checkbox = QCheckBox(roi_name)
                checkbox.setChecked(True)  # Default: all selected
                # Show ROI bounds info
                bounds = roi_data.get('bounds', [])
                if bounds:
                    z_min, z_max, y_min, y_max, x_min, x_max = bounds
                    checkbox.setToolTip(
                        f"Bounds: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]\n"
                        f"Size: {z_max-z_min} x {y_max-y_min} x {x_max-x_min}"
                    )
                # Color indicator
                color = roi_data.get('color', 'red')
                checkbox.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
                self.roi_checkboxes[roi_name] = checkbox
                roi_layout.addWidget(checkbox)
        else:
            no_roi_label = QLabel("No ROIs defined.\nPlease create ROIs using 'Edit ROI' first.")
            no_roi_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
            roi_layout.addWidget(no_roi_label)

        roi_layout.addStretch()
        layout.addWidget(roi_group)

        # Select All / Deselect All buttons
        if self.rois:
            select_layout = QHBoxLayout()
            select_all_btn = QPushButton("Select All")
            select_all_btn.clicked.connect(self._selectAll)
            select_layout.addWidget(select_all_btn)

            deselect_all_btn = QPushButton("Deselect All")
            deselect_all_btn.clicked.connect(self._deselectAll)
            select_layout.addWidget(deselect_all_btn)

            select_layout.addStretch()
            layout.addLayout(select_layout)

        # Info label
        info_label = QLabel(
            "If no ROIs are selected, the operation will be applied\n"
            "to the entire volume (may take longer)."
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px;")
        layout.addWidget(info_label)

        # Button box
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton(self._button_text)
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._onCompute)
        ok_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _selectAll(self):
        for checkbox in self.roi_checkboxes.values():
            checkbox.setChecked(True)

    def _deselectAll(self):
        for checkbox in self.roi_checkboxes.values():
            checkbox.setChecked(False)

    def _onCompute(self):
        self.selected_rois = [
            name for name, checkbox in self.roi_checkboxes.items()
            if checkbox.isChecked()
        ]
        self.accept()

    def getSelectedROIs(self):
        """Return list of selected ROI names"""
        return self.selected_rois


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
            no_roi_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
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

        # Display Options Group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        self.use_density_check = QCheckBox("Use Density (normalize to probability density)")
        self.use_density_check.setChecked(False)
        self.use_density_check.setToolTip(
            "If checked, the histogram is normalized such that the integral equals 1.\n"
            "This makes it easier to compare distributions with different sample sizes."
        )
        display_layout.addWidget(self.use_density_check)

        layout.addWidget(display_group)

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
            'use_density': self.use_density_check.isChecked(),
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
            no_traj_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
            roi_layout.addWidget(no_traj_label)

        layout.addWidget(roi_group)

        # Angle Type Selection Group
        angle_group = QGroupBox("Select Angle Types")
        angle_layout = QVBoxLayout(angle_group)

        self.tilt_check = QCheckBox("X-Z Orientation")
        self.tilt_check.setChecked(True)
        angle_layout.addWidget(self.tilt_check)

        self.azimuth_check = QCheckBox("Y-Z Orientation")
        self.azimuth_check.setChecked(False)
        angle_layout.addWidget(self.azimuth_check)

        self.true_azimuth_check = QCheckBox("Azimuth")
        self.true_azimuth_check.setChecked(False)
        angle_layout.addWidget(self.true_azimuth_check)

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
                'true_azimuth': self.true_azimuth_check.isChecked()
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
            no_data_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-style: italic;")
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
            'tilt_min': -20.0,
            'tilt_max': 20.0,
            'saturation_min': 0.0,
            'saturation_max': 45.0,
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

        # Note: Propagation axis is fixed to Z (fibers assumed along Z direction)

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
        self.tilt_min_spin.setRange(-90, 90)
        self.tilt_min_spin.setValue(self.settings['tilt_min'])
        self.tilt_min_spin.setSuffix("°")
        tilt_layout.addWidget(self.tilt_min_spin)
        tilt_layout.addWidget(QLabel("-"))
        self.tilt_max_spin = QDoubleSpinBox()
        self.tilt_max_spin.setRange(-90, 90)
        self.tilt_max_spin.setValue(self.settings['tilt_max'])
        self.tilt_max_spin.setSuffix("°")
        tilt_layout.addWidget(self.tilt_max_spin)
        color_layout.addRow("X-Z/Y-Z Range:", tilt_widget)

        # Saturation Range (for Azimuth saturation mode)
        sat_widget = QWidget()
        sat_layout = QHBoxLayout(sat_widget)
        sat_layout.setContentsMargins(0, 0, 0, 0)
        self.sat_min_spin = QDoubleSpinBox()
        self.sat_min_spin.setRange(0, 90)
        self.sat_min_spin.setValue(self.settings.get('saturation_min', 0.0))
        self.sat_min_spin.setSuffix("°")
        self.sat_min_spin.setToolTip("Tilt angle for minimum saturation (white)")
        sat_layout.addWidget(self.sat_min_spin)
        sat_layout.addWidget(QLabel("-"))
        self.sat_max_spin = QDoubleSpinBox()
        self.sat_max_spin.setRange(0, 90)
        self.sat_max_spin.setValue(self.settings.get('saturation_max', 45.0))
        self.sat_max_spin.setSuffix("°")
        self.sat_max_spin.setToolTip("Tilt angle for maximum saturation (full color)")
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
            'propagation_axis': 'Z (default)',  # Fixed: always Z-axis
            'integration_method': self.integration_method_combo.currentText(),
            'tilt_min': self.tilt_min_spin.value(),
            'tilt_max': self.tilt_max_spin.value(),
            'saturation_min': self.sat_min_spin.value(),
            'saturation_max': self.sat_max_spin.value(),
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


class VoidAnalysisSettingsDialog(QDialog):
    """Dialog for Void Analysis settings."""
    def __init__(self, parent=None, settings=None, insegt_available=False):
        super().__init__(parent)
        self.setWindowTitle("Void Analysis Settings")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.insegt_available = insegt_available

        self.settings = settings or {
            'method': 'otsu',  # 'otsu', 'insegt', 'manual'
            'manual_threshold': 128,
            'invert': True,
            'min_size': 0,
            'closing_size': 0,
            'compute_statistics': True,
            'compute_local_vf': False,
            'local_window_size': 50,
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Method selection group
        method_group = QGroupBox("Segmentation Method")
        method_layout = QVBoxLayout(method_group)

        self.method_btn_group = QButtonGroup(self)

        # Otsu method
        self.otsu_radio = QRadioButton("Otsu's Method (automatic threshold)")
        self.otsu_radio.setToolTip(
            "Automatically determines the optimal threshold to separate\n"
            "void regions from material using Otsu's algorithm."
        )
        self.method_btn_group.addButton(self.otsu_radio, 0)
        method_layout.addWidget(self.otsu_radio)

        # InSegt method
        self.insegt_radio = QRadioButton("InSegt Labels (use labeled voids)")
        self.insegt_radio.setToolTip(
            "Use void regions (label 3/Yellow) from InSegt segmentation.\n"
            "Fiber and matrix labels are treated as non-void."
        )
        self.insegt_radio.setEnabled(self.insegt_available)
        if not self.insegt_available:
            self.insegt_radio.setText("InSegt Labels (run InSegt first)")
            self.insegt_radio.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.method_btn_group.addButton(self.insegt_radio, 1)
        method_layout.addWidget(self.insegt_radio)

        # Manual threshold
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        self.manual_radio = QRadioButton("Manual Threshold:")
        self.manual_radio.setToolTip("Manually specify the threshold value.")
        self.method_btn_group.addButton(self.manual_radio, 2)
        manual_layout.addWidget(self.manual_radio)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 65535)
        self.threshold_spin.setValue(self.settings.get('manual_threshold', 128))
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.setToolTip("Threshold value (voids below this value)")
        manual_layout.addWidget(self.threshold_spin)
        manual_layout.addStretch()

        method_layout.addWidget(manual_widget)

        # Connect radio buttons to enable/disable threshold spin
        self.manual_radio.toggled.connect(self.threshold_spin.setEnabled)

        # Set initial selection
        method = self.settings.get('method', 'otsu')
        if method == 'insegt' and self.insegt_available:
            self.insegt_radio.setChecked(True)
        elif method == 'manual':
            self.manual_radio.setChecked(True)
            self.threshold_spin.setEnabled(True)
        else:
            self.otsu_radio.setChecked(True)

        layout.addWidget(method_group)

        # Otsu/Manual options group
        options_group = QGroupBox("Threshold Options")
        options_layout = QFormLayout(options_group)

        self.invert_check = QCheckBox("Voids are darker than material")
        self.invert_check.setChecked(self.settings.get('invert', True))
        self.invert_check.setToolTip(
            "Check if voids appear as dark regions in the image.\n"
            "Uncheck if voids appear bright."
        )
        options_layout.addRow("", self.invert_check)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 10000)
        self.min_size_spin.setValue(self.settings.get('min_size', 0))
        self.min_size_spin.setSuffix(" voxels")
        self.min_size_spin.setToolTip("Remove void regions smaller than this size (0 = keep all)")
        options_layout.addRow("Min Void Size:", self.min_size_spin)

        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(0, 10)
        self.closing_spin.setValue(self.settings.get('closing_size', 0))
        self.closing_spin.setToolTip("Morphological closing iterations (fills small holes, 0 = disabled)")
        options_layout.addRow("Closing Size:", self.closing_spin)

        layout.addWidget(options_group)

        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)

        self.stats_check = QCheckBox("Compute void statistics")
        self.stats_check.setChecked(self.settings.get('compute_statistics', True))
        self.stats_check.setToolTip(
            "Calculate statistics including void count, sizes,\n"
            "sphericity, and spatial distribution."
        )
        analysis_layout.addWidget(self.stats_check)

        local_vf_widget = QWidget()
        local_vf_layout = QHBoxLayout(local_vf_widget)
        local_vf_layout.setContentsMargins(0, 0, 0, 0)

        self.local_vf_check = QCheckBox("Compute local void fraction (window:")
        self.local_vf_check.setChecked(self.settings.get('compute_local_vf', False))
        self.local_vf_check.setToolTip("Create a map of local void concentration")
        local_vf_layout.addWidget(self.local_vf_check)

        self.local_window_spin = QSpinBox()
        self.local_window_spin.setRange(5, 200)
        self.local_window_spin.setValue(self.settings.get('local_window_size', 50))
        self.local_window_spin.setSuffix(" px)")
        self.local_window_spin.setEnabled(self.settings.get('compute_local_vf', False))
        local_vf_layout.addWidget(self.local_window_spin)
        local_vf_layout.addStretch()

        self.local_vf_check.toggled.connect(self.local_window_spin.setEnabled)

        analysis_layout.addWidget(local_vf_widget)

        layout.addWidget(analysis_group)

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
        if self.otsu_radio.isChecked():
            method = 'otsu'
        elif self.insegt_radio.isChecked():
            method = 'insegt'
        else:
            method = 'manual'

        return {
            'method': method,
            'manual_threshold': self.threshold_spin.value(),
            'invert': self.invert_check.isChecked(),
            'min_size': self.min_size_spin.value(),
            'closing_size': self.closing_spin.value(),
            'compute_statistics': self.stats_check.isChecked(),
            'compute_local_vf': self.local_vf_check.isChecked(),
            'local_window_size': self.local_window_spin.value(),
        }


class CropOrientationDialog(QDialog):
    """Dialog for Crop Orientation with void mask settings."""
    def __init__(self, parent=None, void_available=False, orientation_available=False):
        super().__init__(parent)
        self.setWindowTitle("Crop Orientation with Void Mask")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.void_available = void_available
        self.orientation_available = orientation_available

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Status info
        status_group = QGroupBox("Data Status")
        status_layout = QVBoxLayout(status_group)

        orientation_status = QLabel(
            "✓ Orientation data available" if self.orientation_available
            else "✗ No orientation data (run orientation analysis first)"
        )
        orientation_status.setStyleSheet(
            f"color: {'green' if self.orientation_available else COLORS['text_secondary']};"
        )
        status_layout.addWidget(orientation_status)

        void_status = QLabel(
            "✓ Void mask available" if self.void_available
            else "✗ No void mask (run void analysis first)"
        )
        void_status.setStyleSheet(
            f"color: {'green' if self.void_available else COLORS['text_secondary']};"
        )
        status_layout.addWidget(void_status)

        layout.addWidget(status_group)

        # Dilation settings
        dilation_group = QGroupBox("Void Mask Dilation")
        dilation_layout = QFormLayout(dilation_group)

        dilation_desc = QLabel(
            "Expand void regions to exclude orientation data near voids\n"
            "where measurements may be unreliable."
        )
        dilation_desc.setWordWrap(True)
        dilation_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        dilation_layout.addRow(dilation_desc)

        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(0, 50)
        self.dilation_spin.setValue(3)
        self.dilation_spin.setSuffix(" pixels")
        self.dilation_spin.setToolTip(
            "Number of pixels to expand void regions.\n"
            "Set to 0 to use exact void boundaries."
        )
        dilation_layout.addRow("Dilation amount:", self.dilation_spin)

        layout.addWidget(dilation_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)

        self.update_viewer_check = QCheckBox("Update viewer with masked orientation")
        self.update_viewer_check.setChecked(True)
        self.update_viewer_check.setToolTip("Apply masked orientation to the current viewer")
        output_layout.addWidget(self.update_viewer_check)

        self.export_check = QCheckBox("Export masked orientation to VTK file")
        self.export_check.setChecked(False)
        self.export_check.setToolTip("Save the masked orientation data to a VTK file")
        output_layout.addWidget(self.export_check)

        layout.addWidget(output_group)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.accept)
        self.apply_btn.setEnabled(self.void_available and self.orientation_available)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def getSettings(self):
        """Return the current settings."""
        return {
            'dilation_pixels': self.dilation_spin.value(),
            'update_viewer': self.update_viewer_check.isChecked(),
            'export_vtk': self.export_check.isChecked(),
        }


class VfSettingsDialog(QDialog):
    """Dialog for Fiber Volume Fraction calculation settings."""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Fraction Settings")
        self.setModal(True)
        self.setMinimumWidth(350)

        self.settings = settings or {
            'window_size': 50,
            'use_gaussian': False,
            'gaussian_sigma': 10.0,
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Computes local fiber volume fraction (Vf) from\n"
            "existing segmentation results (Fiber Detection or InSegt)."
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-style: italic;")
        layout.addWidget(info_label)

        # Local Vf Calculation Group
        vf_group = QGroupBox("Local Vf Calculation")
        vf_layout = QFormLayout(vf_group)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 500)
        self.window_spin.setValue(self.settings.get('window_size', 50))
        self.window_spin.setSuffix(" px")
        self.window_spin.setToolTip("Window size for local Vf averaging (box filter)")
        vf_layout.addRow("Window Size:", self.window_spin)

        self.use_gaussian_check = QCheckBox("Use Gaussian smoothing")
        self.use_gaussian_check.setChecked(self.settings.get('use_gaussian', False))
        self.use_gaussian_check.stateChanged.connect(self._onGaussianChanged)
        vf_layout.addRow("", self.use_gaussian_check)

        self.gaussian_spin = QDoubleSpinBox()
        self.gaussian_spin.setRange(1.0, 100.0)
        self.gaussian_spin.setValue(self.settings.get('gaussian_sigma', 10.0))
        self.gaussian_spin.setSuffix(" px")
        self.gaussian_spin.setToolTip("Gaussian sigma for smoothing (larger = smoother)")
        self.gaussian_spin.setEnabled(self.settings.get('use_gaussian', False))
        vf_layout.addRow("Gaussian Sigma:", self.gaussian_spin)

        layout.addWidget(vf_group)

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

    def _onGaussianChanged(self, state):
        """Enable/disable gaussian sigma based on checkbox."""
        # Handle both Qt.CheckState enum and int values
        if isinstance(state, int):
            self.gaussian_spin.setEnabled(state == 2)  # Qt.Checked = 2
        else:
            self.gaussian_spin.setEnabled(state == Qt.CheckState.Checked)

    def getSettings(self):
        """Return the current settings."""
        return {
            'window_size': self.window_spin.value(),
            'use_gaussian': self.use_gaussian_check.isChecked(),
            'gaussian_sigma': self.gaussian_spin.value(),
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
    def __init__(self, parent=None, settings=None, rois=None, main_window=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.rois = rois or {}
        self.main_window = main_window

        # Default settings
        self.settings = settings or {
            'maximum_shear_stress': 100.0,
            'shear_stress_step_size': 0.1,
            'maximum_axial_strain': 0.02,
            'maximum_fiber_misalignment': 20.0,
            'fiber_misalignment_step_size': 0.1,
            'kink_width': None,
            'gauge_length': None,
            'use_3d_orientation': False,
            'selected_roi': None
        }

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Orientation Data Source Group
        orientation_group = QGroupBox("Orientation Data Source")
        orientation_layout = QFormLayout(orientation_group)

        self.use_3d_orientation_check = QCheckBox("Use 3D Orientation Data")
        self.use_3d_orientation_check.setChecked(self.settings.get('use_3d_orientation', False))
        self.use_3d_orientation_check.stateChanged.connect(self.onUse3DOrientationChanged)
        orientation_layout.addRow(self.use_3d_orientation_check)

        # ROI Selection
        self.roi_combo = QComboBox()
        self.roi_combo.addItem("Global (All Data)", None)
        # Add ROIs that have orientation data
        for roi_name, roi_data in self.rois.items():
            if roi_data.get('angle') is not None or roi_data.get('theta') is not None:
                self.roi_combo.addItem(roi_name, roi_name)
        # Set current selection
        current_roi = self.settings.get('selected_roi')
        if current_roi:
            idx = self.roi_combo.findData(current_roi)
            if idx >= 0:
                self.roi_combo.setCurrentIndex(idx)
        orientation_layout.addRow("ROI:", self.roi_combo)

        # Initially set ROI combo enabled state
        self.roi_combo.setEnabled(self.use_3d_orientation_check.isChecked())

        layout.addWidget(orientation_group)

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

    def onUse3DOrientationChanged(self, state):
        """Toggle ROI selection based on 3D orientation checkbox"""
        enabled = (state == Qt.Checked.value) if hasattr(Qt.Checked, 'value') else (state == 2)
        self.roi_combo.setEnabled(enabled)

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
            'gauge_length': self.gauge_length_spin.value() if use_correction else None,
            'use_3d_orientation': self.use_3d_orientation_check.isChecked(),
            'selected_roi': self.roi_combo.currentData()
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
            'gauge_length': None,
            'use_3d_orientation': False,
            'selected_roi': None
        }
        # Store simulation results for export
        self.simulation_results = []
        self.initUI()

    def initUI(self):
        # Ribbon is now managed by MainWindow's ribbon_stack
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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
        rois = self.main_window.viewer.rois if self.main_window and self.main_window.viewer else {}
        dialog = SimulationSettingsDialog(self, self.simulation_settings, rois, self.main_window)
        if dialog.exec() == QDialog.Accepted:
            self.simulation_settings = dialog.getSettings()
            # Update UI state based on settings
            if self.main_window and hasattr(self.main_window, 'simulation_content'):
                use_3d = self.simulation_settings.get('use_3d_orientation', False)
                self.main_window.simulation_content.updateOrientationInputState(use_3d)

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

        from vmm.simulation import estimate_compression_strength, estimate_compression_strength_from_profile, MaterialParams

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

        # Check if using 3D orientation data (from settings dialog)
        use_3d_orientation = settings.get('use_3d_orientation', False)
        selected_roi = settings.get('selected_roi', None)

        self.main_window.status_label.setText("Running simulation...")
        QApplication.processEvents()

        try:
            if use_3d_orientation:
                # Get orientation data from selected ROI or global data
                if selected_roi is not None and isinstance(selected_roi, str) and selected_roi in self.main_window.viewer.rois:
                    roi_data = self.main_window.viewer.rois[selected_roi]
                    orientation_profile = roi_data.get('angle')
                    if orientation_profile is None:
                        orientation_profile = roi_data.get('reference')
                    if orientation_profile is None:
                        raise ValueError(f"No orientation data in ROI '{selected_roi}'. Please run orientation analysis first.")
                else:
                    # Use global orientation data
                    orientation_data = self.main_window.orientation_data
                    orientation_profile = orientation_data.get('reference')
                    if orientation_profile is None:
                        raise ValueError("No 3D orientation data available. Please run orientation analysis first.")

                # Run simulation with measured orientation profile
                strength, strain, stress_curve, strain_array = estimate_compression_strength_from_profile(
                    orientation_profile=orientation_profile,
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
        self.main_window = None
        self.loadMaterialPresets()
        self.initUI()

    def setMainWindow(self, main_window):
        """Set reference to main window"""
        self.main_window = main_window

    def updateOrientationInputState(self, use_3d_orientation):
        """Enable/disable manual orientation inputs based on 3D orientation setting"""
        # Disable manual inputs when using 3D orientation data
        self.initial_misalignment_spin.setEnabled(not use_3d_orientation)
        self.std_deviation_spin.setEnabled(not use_3d_orientation)

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

        # Left panel - Material Parameters (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(280)
        scroll_area.setMaximumWidth(350)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Material Parameters Group
        from vmm.theme import get_group_style
        group_style = get_group_style()

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

        # Fiber Orientation Group (for Gaussian distribution mode)
        orientation_group = QGroupBox("Fiber Orientation (Gaussian)")
        orientation_group.setStyleSheet(group_style)
        orientation_layout = QFormLayout(orientation_group)
        orientation_layout.setSpacing(8)
        orientation_layout.setContentsMargins(10, 20, 10, 10)

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
        self.strength_label.setStyleSheet(f"font-weight: bold; color: {COLORS['accent_info']};")
        results_layout.addRow("Compressive Strength:", self.strength_label)

        self.strain_label = QLabel("--")
        self.strain_label.setStyleSheet(f"font-weight: bold; color: {COLORS['accent_info']};")
        results_layout.addRow("Ultimate Strain:", self.strain_label)

        left_layout.addWidget(results_group)
        left_layout.addStretch()

        scroll_area.setWidget(left_panel)
        layout.addWidget(scroll_area)

        # Histogram Panel (initially hidden, shown when Histogram button is clicked)
        self.histogram_panel = HistogramPanel()
        self.histogram_panel.setVisible(False)
        layout.addWidget(self.histogram_panel)

        # Right panel - Stress-Strain Graph Display
        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure for stress-strain curve (always white background)
        self.figure = Figure(figsize=(4, 3), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 150)
        self.ax = self.figure.add_subplot(111)
        # Ensure white background for simulation graph regardless of theme
        self.ax.set_facecolor('white')
        self.ax.set_xlabel('Strain', color='black')
        self.ax.set_ylabel('Stress (MPa)', color='black')
        self.ax.set_title('Stress-Strain Curve', color='black')
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
        self.ax.grid(True, alpha=0.3, color='gray')
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
        # Maintain white background styling
        self.ax.set_facecolor('white')
        self.ax.set_xlabel('Strain', color='black')
        self.ax.set_ylabel('Stress (MPa)', color='black')
        self.ax.set_title('Stress-Strain Curve', color='black')
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.legend(loc='best')
        self.figure.tight_layout()
        self.canvas.draw()

    def clearGraph(self):
        """Clear all curves from the graph"""
        self.ax.clear()
        # Maintain white background styling
        self.ax.set_facecolor('white')
        self.ax.set_xlabel('Strain', color='black')
        self.ax.set_ylabel('Stress (MPa)', color='black')
        self.ax.set_title('Stress-Strain Curve', color='black')
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
        self.ax.grid(True, alpha=0.3, color='gray')
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

class VMMMainWindow(QMainWindow):
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
        self.setWindowTitle("VMM-FRC - Virtual Microstructure Modeling for Fiber Reinforced Polymer Composites")

        # Set fixed initial size to prevent canvas from expanding window
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)

        # Set application icon
        import os
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'vmm_logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Set application style
        from vmm.theme import get_main_window_style
        self.setStyleSheet(get_main_window_style())

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create tab objects first (before ribbons, as ribbons reference them)
        self.volume_tab = VolumeTab()
        self.volume_tab.main_window = self
        self.analysis_tab = AnalysisTab()
        self.analysis_tab.main_window = self
        self.modelling_tab = VisualizationTab()
        self.modelling_tab.setMainWindow(self)
        self.simulation_tab = SimulationTab()
        self.simulation_tab.setMainWindow(self)

        # Keep tabs reference for compatibility (dummy QTabWidget for setCurrentWidget)
        self.tabs = type('Tabs', (), {
            'setCurrentWidget': lambda self, w: None,
            'currentChanged': type('Signal', (), {'connect': lambda self, f: None})()
        })()

        # Tab bar only (not QTabWidget)
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Volume")
        self.tab_bar.addTab("Analysis")
        self.tab_bar.addTab("Modelling")
        self.tab_bar.addTab("Simulation")
        self.tab_bar.currentChanged.connect(self.onTabChanged)
        main_layout.addWidget(self.tab_bar)

        # Stacked widget for ribbon toolbars
        from vmm.theme import get_ribbon_stack_style
        self.ribbon_stack = QStackedWidget()
        self.ribbon_stack.setFixedHeight(100)
        self.ribbon_stack.setStyleSheet(get_ribbon_stack_style())

        # Create ribbon for each tab (Volume, Analysis, Modelling, Simulation)
        self.volume_ribbon = self._createVolumeRibbon()
        self.ribbon_stack.addWidget(self.volume_ribbon)

        self.analysis_ribbon = self._createAnalysisRibbon()
        self.ribbon_stack.addWidget(self.analysis_ribbon)

        self.modelling_ribbon = self._createModellingRibbon()
        self.ribbon_stack.addWidget(self.modelling_ribbon)

        self.simulation_ribbon = self._createSimulationRibbon()
        self.ribbon_stack.addWidget(self.simulation_ribbon)

        main_layout.addWidget(self.ribbon_stack)

        # Simulation content panel (shown when Simulation tab is selected)
        self.simulation_content = SimulationContentPanel()
        self.simulation_content.setMainWindow(self)
        self.simulation_content.setVisible(False)
        main_layout.addWidget(self.simulation_content)

        # Modelling content (VisualizationTab contains its own content)
        self.modelling_content = self.modelling_tab
        self.modelling_content.setVisible(False)
        main_layout.addWidget(self.modelling_content)

        # Create horizontal splitter for viewer and left slider panel
        self.content_splitter = QSplitter(Qt.Horizontal)

        # Left slider panel (scrollable)
        slider_scroll = QScrollArea()
        slider_scroll.setWidgetResizable(True)
        slider_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        slider_scroll.setMaximumWidth(250)
        slider_scroll.setMinimumWidth(200)
        slider_scroll.setStyleSheet("QScrollArea { border: none; }")

        slider_panel = QWidget()
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

        # Image Adjustment Group
        adjustment_group = QGroupBox("Image Adjustment")
        adjustment_layout = QGridLayout(adjustment_group)
        adjustment_layout.setSpacing(5)

        # Brightness slider
        adjustment_layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.onAdjustmentChanged)
        adjustment_layout.addWidget(self.brightness_slider, 0, 1)
        self.brightness_label = QLabel("0")
        self.brightness_label.setMinimumWidth(35)
        self.brightness_label.setAlignment(Qt.AlignRight)
        adjustment_layout.addWidget(self.brightness_label, 0, 2)

        # Contrast slider
        adjustment_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)  # 0.1 to 3.0
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.onAdjustmentChanged)
        adjustment_layout.addWidget(self.contrast_slider, 1, 1)
        self.contrast_label = QLabel("1.0")
        self.contrast_label.setMinimumWidth(35)
        self.contrast_label.setAlignment(Qt.AlignRight)
        adjustment_layout.addWidget(self.contrast_label, 1, 2)

        # Gamma slider
        adjustment_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)  # 0.1 to 3.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.onAdjustmentChanged)
        adjustment_layout.addWidget(self.gamma_slider, 2, 1)
        self.gamma_label = QLabel("1.0")
        self.gamma_label.setMinimumWidth(35)
        self.gamma_label.setAlignment(Qt.AlignRight)
        adjustment_layout.addWidget(self.gamma_label, 2, 2)

        # Sharpness slider
        adjustment_layout.addWidget(QLabel("Sharpness:"), 3, 0)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.valueChanged.connect(self.onAdjustmentChanged)
        adjustment_layout.addWidget(self.sharpness_slider, 3, 1)
        self.sharpness_label = QLabel("0")
        self.sharpness_label.setMinimumWidth(35)
        self.sharpness_label.setAlignment(Qt.AlignRight)
        adjustment_layout.addWidget(self.sharpness_label, 3, 2)

        # Invert checkbox
        self.invert_check = QCheckBox("Invert")
        self.invert_check.stateChanged.connect(self.onAdjustmentChanged)
        adjustment_layout.addWidget(self.invert_check, 4, 0, 1, 2)

        # Reset button
        self.reset_adjustment_btn = QPushButton("Reset")
        self.reset_adjustment_btn.clicked.connect(self.resetAdjustments)
        adjustment_layout.addWidget(self.reset_adjustment_btn, 4, 2)

        # Apply to Volume button
        self.apply_adjustment_btn = QPushButton("Apply to Volume")
        self.apply_adjustment_btn.setToolTip("Apply adjustments permanently to the volume data")
        self.apply_adjustment_btn.clicked.connect(self.applyAdjustmentsToVolume)
        adjustment_layout.addWidget(self.apply_adjustment_btn, 5, 0, 1, 3)

        self.adjustment_group = adjustment_group
        slider_layout.addWidget(adjustment_group)

        slider_layout.addStretch()

        slider_scroll.setWidget(slider_panel)
        self.slider_panel = slider_panel  # Save reference for later use
        self.content_splitter.addWidget(slider_scroll)

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

        # Add Pipeline Panel to slider layout (before the stretch)
        # We need to insert it before the stretch, so remove stretch first, add controls, then re-add stretch
        slider_layout = slider_panel.layout()
        # Remove the last item (stretch)
        last_item = slider_layout.takeAt(slider_layout.count() - 1)

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
        from vmm.theme import get_splitter_style, get_progress_bar_style, get_status_bar_style
        self.content_splitter.setStyleSheet(get_splitter_style())

        # Add content splitter directly to main layout
        main_layout.addWidget(self.content_splitter)

        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        main_layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().setStyleSheet(get_status_bar_style())

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # Add log viewer button to status bar
        self.log_viewer_btn = QPushButton("Show Logs")
        self.log_viewer_btn.setFixedHeight(20)
        self.log_viewer_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #4CAF50; "
            "  color: white; "
            "  border: none; "
            "  padding: 2px 8px; "
            "  border-radius: 3px; "
            "  font-size: 11px; "
            "} "
            "QPushButton:hover { "
            "  background-color: #45a049; "
            "} "
            "QPushButton:pressed { "
            "  background-color: #3d8b40; "
            "}"
        )
        self.log_viewer_btn.clicked.connect(self.showLogViewer)
        self.statusBar().addPermanentWidget(self.log_viewer_btn)

    def _createVolumeRibbon(self):
        """Create Volume tab ribbon toolbar"""
        from vmm.theme import get_ribbon_frame_style
        ribbon = QFrame()
        ribbon.setStyleSheet(get_ribbon_frame_style())
        layout = QHBoxLayout(ribbon)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # File Operations Group
        file_group = QGroupBox("File")
        file_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        file_layout = QHBoxLayout(file_group)

        self.open_btn = RibbonButton("Open\nFiles")
        self.open_btn.clicked.connect(self.openImportDialog)
        file_layout.addWidget(self.open_btn)

        layout.addWidget(file_group)

        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        appearance_layout = QVBoxLayout(appearance_group)

        appearance_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = RibbonComboBox()
        self.colormap_combo.addItems(["gray", "viridis", "jet", "coolwarm", "rainbow", "bone"])
        self.colormap_combo.currentTextChanged.connect(self.updateColormap)
        appearance_layout.addWidget(self.colormap_combo)

        layout.addWidget(appearance_group)

        # Camera Group
        camera_group = QGroupBox("Camera")
        camera_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        camera_layout = QHBoxLayout(camera_group)

        self.reset_view_btn = RibbonButton("Reset\nView")
        self.reset_view_btn.clicked.connect(self.resetView)
        camera_layout.addWidget(self.reset_view_btn)

        layout.addWidget(camera_group)

        # Switch Axis Group
        axis_group = QGroupBox("Switch Axis")
        axis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        axis_layout = QHBoxLayout(axis_group)

        self.switch_xy_btn = RibbonButton("Swap\nX↔Y")
        self.switch_xy_btn.setToolTip("Swap X and Y axes (transpose in XY plane)")
        self.switch_xy_btn.clicked.connect(lambda: self.switchAxis('xy'))
        axis_layout.addWidget(self.switch_xy_btn)

        self.switch_xz_btn = RibbonButton("Swap\nX↔Z")
        self.switch_xz_btn.setToolTip("Swap X and Z axes (rotate volume 90° around Y)")
        self.switch_xz_btn.clicked.connect(lambda: self.switchAxis('xz'))
        axis_layout.addWidget(self.switch_xz_btn)

        self.switch_yz_btn = RibbonButton("Swap\nY↔Z")
        self.switch_yz_btn.setToolTip("Swap Y and Z axes (rotate volume 90° around X)")
        self.switch_yz_btn.clicked.connect(lambda: self.switchAxis('yz'))
        axis_layout.addWidget(self.switch_yz_btn)

        self.export_seq_btn = RibbonButton("Export\nSequence")
        self.export_seq_btn.setToolTip("Export current volume as image sequence along current Z axis")
        self.export_seq_btn.clicked.connect(self.exportImageSequence)
        axis_layout.addWidget(self.export_seq_btn)

        layout.addWidget(axis_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QHBoxLayout(export_group)

        self.export_vtk_btn = RibbonButton("Export\nVTK")
        self.export_vtk_btn.setToolTip("Export CT volume to VTK format for Paraview")
        self.export_vtk_btn.clicked.connect(self.export3D)
        export_layout.addWidget(self.export_vtk_btn)

        self.export_settings_btn = RibbonButton("Export\nSettings")
        self.export_settings_btn.setToolTip("Export image adjustment settings to text file")
        self.export_settings_btn.clicked.connect(self.exportAdjustmentSettings)
        export_layout.addWidget(self.export_settings_btn)

        layout.addWidget(export_group)
        layout.addStretch()

        return ribbon

    def _createAnalysisRibbon(self):
        """Create Analysis tab ribbon toolbar"""
        from vmm.theme import get_ribbon_frame_style
        ribbon = QFrame()
        ribbon.setStyleSheet(get_ribbon_frame_style())
        layout = QHBoxLayout(ribbon)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Analysis Operations Group
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QHBoxLayout(analysis_group)

        self.edit_roi_btn = RibbonButton("Edit\nROI")
        self.edit_roi_btn.clicked.connect(self.toggleROIEdit)
        self.edit_roi_btn.setCheckable(True)
        analysis_layout.addWidget(self.edit_roi_btn)

        self.coord_roi_btn = RibbonButton("Coord\nROI")
        self.coord_roi_btn.setToolTip("Create ROI by entering coordinates")
        self.coord_roi_btn.clicked.connect(self.openCoordinateROIDialog)
        analysis_layout.addWidget(self.coord_roi_btn)

        # ROI Mode selector
        roi_mode_widget = QWidget()
        roi_mode_layout = QVBoxLayout(roi_mode_widget)
        roi_mode_layout.setContentsMargins(0, 0, 0, 0)
        roi_mode_layout.setSpacing(2)
        roi_mode_label = QLabel("ROI Mode")
        roi_mode_label.setAlignment(Qt.AlignCenter)
        roi_mode_label.setStyleSheet(f"font-size: 9px; color: {COLORS['text_secondary']};")
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["Rectangle", "Polygon"])
        self.roi_mode_combo.setStyleSheet(f"""
            QComboBox {{ padding: 2px; border: 1px solid {COLORS['border']}; border-radius: 3px;
                background-color: {COLORS['bg_input']}; font-size: 10px; min-width: 70px; }}
            QComboBox:hover {{ border: 1px solid {COLORS['accent']}; background-color: {COLORS['bg_tertiary']}; }}
        """)
        self.roi_mode_combo.currentTextChanged.connect(self.onROIModeChanged)
        roi_mode_layout.addWidget(roi_mode_label)
        roi_mode_layout.addWidget(self.roi_mode_combo)
        analysis_layout.addWidget(roi_mode_widget)

        self.compute_btn = RibbonButton("Compute\nOrientation")
        self.compute_btn.clicked.connect(self.computeOrientation)
        analysis_layout.addWidget(self.compute_btn)

        self.edit_range_btn = RibbonButton("Edit\nRange")
        self.edit_range_btn.clicked.connect(self.openRangeEditor)
        self.edit_range_btn.setEnabled(False)
        analysis_layout.addWidget(self.edit_range_btn)

        self.histogram_btn = RibbonButton("Histogram")
        self.histogram_btn.clicked.connect(self.openHistogramDialog)
        self.histogram_btn.setEnabled(False)
        analysis_layout.addWidget(self.histogram_btn)

        self.magnify_btn = RibbonButton("Magnify")
        self.magnify_btn.setCheckable(True)
        self.magnify_btn.clicked.connect(self.toggleMagnify)
        analysis_layout.addWidget(self.magnify_btn)

        self.reset_roi_btn = RibbonButton("Reset\nROI")
        self.reset_roi_btn.setToolTip("Clear all ROIs")
        self.reset_roi_btn.clicked.connect(self.resetAllROIs)
        analysis_layout.addWidget(self.reset_roi_btn)

        layout.addWidget(analysis_group)

        # (Reference Vector selection removed - always Z-axis)

        # Fiber Detection Group
        fiber_group = QGroupBox("Fiber Detection")
        fiber_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        fiber_layout = QHBoxLayout(fiber_group)

        self.fiber_detect_settings_btn = RibbonButton("Detection\nSettings")
        self.fiber_detect_settings_btn.clicked.connect(self.openFiberDetectionSettings)
        fiber_layout.addWidget(self.fiber_detect_settings_btn)

        self.fiber_detect_btn = RibbonButton("Detect\nFibers")
        self.fiber_detect_btn.clicked.connect(self.detectFibers)
        fiber_layout.addWidget(self.fiber_detect_btn)

        layout.addWidget(fiber_group)

        # InSegt Group
        insegt_group = QGroupBox("InSegt")
        insegt_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        insegt_layout = QHBoxLayout(insegt_group)

        self.insegt_settings_btn = RibbonButton("InSegt\nSettings")
        self.insegt_settings_btn.clicked.connect(self.openInSegtSettings)
        insegt_layout.addWidget(self.insegt_settings_btn)

        self.insegt_labeling_btn = RibbonButton("Labeling")
        self.insegt_labeling_btn.clicked.connect(self.openInSegtLabeling)
        insegt_layout.addWidget(self.insegt_labeling_btn)

        self.insegt_run_btn = RibbonButton("Run")
        self.insegt_run_btn.clicked.connect(self.runInSegt)
        self.insegt_run_btn.setEnabled(False)
        insegt_layout.addWidget(self.insegt_run_btn)

        layout.addWidget(insegt_group)

        # Volume Fraction Group
        vf_group = QGroupBox("Volume Fraction")
        vf_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        vf_layout_inner = QHBoxLayout(vf_group)

        self.vf_settings_btn = RibbonButton("Vf\nSettings")
        self.vf_settings_btn.clicked.connect(self.openVfSettings)
        vf_layout_inner.addWidget(self.vf_settings_btn)

        self.vf_compute_btn = RibbonButton("Compute\nVf")
        self.vf_compute_btn.clicked.connect(self.computeVf)
        vf_layout_inner.addWidget(self.vf_compute_btn)

        layout.addWidget(vf_group)

        # Void Analysis Group
        void_group = QGroupBox("Void Analysis")
        void_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        void_layout = QHBoxLayout(void_group)

        self.void_settings_btn = RibbonButton("Void\nSettings")
        self.void_settings_btn.clicked.connect(self.openVoidSettings)
        void_layout.addWidget(self.void_settings_btn)

        self.void_run_btn = RibbonButton("Analyze\nVoid")
        self.void_run_btn.clicked.connect(self.runVoidAnalysis)
        void_layout.addWidget(self.void_run_btn)

        self.crop_orientation_btn = RibbonButton("Crop\nOrientation")
        self.crop_orientation_btn.clicked.connect(self.cropOrientationWithVoid)
        self.crop_orientation_btn.setToolTip("Mask orientation data with void regions")
        void_layout.addWidget(self.crop_orientation_btn)

        layout.addWidget(void_group)

        # Reset Group
        reset_group = QGroupBox("Reset")
        reset_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        reset_layout = QHBoxLayout(reset_group)

        self.reset_all_btn = RibbonButton("Reset\nAll")
        self.reset_all_btn.clicked.connect(self.resetAllAnalysis)
        self.reset_all_btn.setToolTip("Reset all analysis results (orientation, Vf, void, fiber detection)")
        reset_layout.addWidget(self.reset_all_btn)

        layout.addWidget(reset_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QHBoxLayout(export_group)

        self.export_orientation_btn = RibbonButton("Export\nOrientation")
        self.export_orientation_btn.clicked.connect(self.exportOrientationToVTK)
        self.export_orientation_btn.setEnabled(False)
        export_layout.addWidget(self.export_orientation_btn)

        self.export_vf_btn = RibbonButton("Export\nVf Map")
        self.export_vf_btn.clicked.connect(self.exportVfMapToVTK)
        self.export_vf_btn.setEnabled(False)
        export_layout.addWidget(self.export_vf_btn)

        layout.addWidget(export_group)
        layout.addStretch()

        return ribbon

    def _createModellingRibbon(self):
        """Create Modelling tab ribbon toolbar"""
        from vmm.theme import get_ribbon_frame_style
        ribbon = QFrame()
        ribbon.setStyleSheet(get_ribbon_frame_style())
        layout = QHBoxLayout(ribbon)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)

        self.modelling_settings_btn = RibbonButton("Settings")
        self.modelling_settings_btn.clicked.connect(self.modelling_tab.openSettingsDialog)
        settings_layout.addWidget(self.modelling_settings_btn)

        layout.addWidget(settings_group)

        # Fiber Trajectory Group
        fiber_group = QGroupBox("Fiber Trajectory")
        fiber_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        fiber_layout = QHBoxLayout(fiber_group)

        self.generate_btn = RibbonButton("Generate\nTrajectory")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.modelling_tab.generateTrajectory)
        fiber_layout.addWidget(self.generate_btn)

        layout.addWidget(fiber_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QHBoxLayout(export_group)

        self.export_traj_vtk_btn = RibbonButton("Export\nVTK")
        self.export_traj_vtk_btn.setEnabled(False)
        self.export_traj_vtk_btn.clicked.connect(self.modelling_tab.exportTrajectoryToVTK)
        export_layout.addWidget(self.export_traj_vtk_btn)

        layout.addWidget(export_group)

        # Analysis Group
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QHBoxLayout(analysis_group)

        self.trajectory_histogram_btn = RibbonButton("Histogram")
        self.trajectory_histogram_btn.setEnabled(False)
        self.trajectory_histogram_btn.clicked.connect(self.modelling_tab.openTrajectoryHistogramDialog)
        analysis_layout.addWidget(self.trajectory_histogram_btn)

        layout.addWidget(analysis_group)
        layout.addStretch()

        return ribbon

    def _createSimulationRibbon(self):
        """Create Simulation tab ribbon toolbar"""
        from vmm.theme import get_ribbon_frame_style
        ribbon = QFrame()
        ribbon.setStyleSheet(get_ribbon_frame_style())
        layout = QHBoxLayout(ribbon)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)

        self.sim_settings_btn = RibbonButton("Settings")
        self.sim_settings_btn.clicked.connect(self.simulation_tab.openSettingsDialog)
        settings_layout.addWidget(self.sim_settings_btn)

        layout.addWidget(settings_group)

        # Simulation Group
        sim_group = QGroupBox("Simulation")
        sim_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        sim_layout = QHBoxLayout(sim_group)

        self.run_sim_btn = RibbonButton("Run\nSimulation")
        self.run_sim_btn.clicked.connect(self.simulation_tab.runSimulation)
        sim_layout.addWidget(self.run_sim_btn)

        self.clear_graph_btn = RibbonButton("Clear\nGraph")
        self.clear_graph_btn.clicked.connect(self.simulation_tab.clearGraph)
        sim_layout.addWidget(self.clear_graph_btn)

        layout.addWidget(sim_group)

        # Analysis Group
        analysis_group = QGroupBox("Analysis")
        analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        analysis_layout = QVBoxLayout(analysis_group)

        self.sim_histogram_btn = RibbonButton("Histogram")
        self.sim_histogram_btn.clicked.connect(self.simulation_tab.openHistogramDialog)
        self.sim_histogram_btn.setEnabled(False)
        analysis_layout.addWidget(self.sim_histogram_btn)

        layout.addWidget(analysis_group)

        # Export Group
        export_group = QGroupBox("Export")
        export_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        export_layout = QVBoxLayout(export_group)

        self.sim_export_btn = RibbonButton("Export\nXLSX")
        self.sim_export_btn.clicked.connect(self.simulation_tab.openExportDialog)
        export_layout.addWidget(self.sim_export_btn)

        layout.addWidget(export_group)
        layout.addStretch()

        return ribbon

    # Delegate methods for ribbon buttons
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
        self.setVolume(volume)

    def updateColormap(self):
        if self.viewer:
            colormap = self.colormap_combo.currentText()
            self.viewer.setColormap(colormap)

    def resetView(self):
        if self.viewer:
            self.viewer.resetCamera()

    def export3D(self):
        """Export CT volume to VTK - delegate to VolumeTab"""
        if hasattr(self, 'volume_tab') and hasattr(self.volume_tab, 'export3D'):
            if self.viewer:
                self.volume_tab.viewer = self.viewer
            self.volume_tab.export3D()

    def switchAxis(self, axis_pair):
        """Switch axes of the current volume.

        Args:
            axis_pair: 'xy', 'xz', or 'yz' indicating which axes to swap
        """
        if self.current_volume is None:
            QMessageBox.warning(self, "No Volume", "Please load a volume first.")
            return

        # Get current volume shape (Z, Y, X)
        old_shape = self.current_volume.shape

        # Perform axis swap
        if axis_pair == 'xy':
            # Swap X and Y: (Z, Y, X) -> (Z, X, Y)
            self.current_volume = np.transpose(self.current_volume, (0, 2, 1))
        elif axis_pair == 'xz':
            # Swap X and Z: (Z, Y, X) -> (X, Y, Z)
            self.current_volume = np.transpose(self.current_volume, (2, 1, 0))
        elif axis_pair == 'yz':
            # Swap Y and Z: (Z, Y, X) -> (Y, Z, X)
            self.current_volume = np.transpose(self.current_volume, (1, 0, 2))

        new_shape = self.current_volume.shape

        # Update viewer with new volume
        if self.viewer:
            self.viewer.setVolume(self.current_volume)

        # Clear analysis data since axes changed
        self.orientation_data = {}
        if hasattr(self, 'edit_range_btn'):
            self.edit_range_btn.setEnabled(False)
        if hasattr(self, 'histogram_btn'):
            self.histogram_btn.setEnabled(False)

        # Update status
        axis_names = {'xy': 'X↔Y', 'xz': 'X↔Z', 'yz': 'Y↔Z'}
        self.status_label.setText(
            f"Axes swapped ({axis_names[axis_pair]}): {old_shape} → {new_shape}"
        )

        QMessageBox.information(
            self, "Axis Switched",
            f"Axes swapped ({axis_names[axis_pair]}):\n"
            f"Old shape: {old_shape}\n"
            f"New shape: {new_shape}\n\n"
            f"Note: Analysis data has been cleared."
        )

    def exportImageSequence(self):
        """Export current volume as image sequence along current Z axis."""
        if self.current_volume is None:
            QMessageBox.warning(self, "No Volume", "Please load a volume first.")
            return

        # Get output directory and base filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Image Sequence",
            "",
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;All Files (*)"
        )

        if not filename:
            return

        # Parse filename to get base name and extension
        from pathlib import Path
        path = Path(filename)
        base_name = path.stem
        extension = path.suffix if path.suffix else '.tif'
        output_dir = path.parent

        # Get volume info
        n_slices = self.current_volume.shape[0]

        # Show progress
        self.showProgress(True)
        self.progress_bar.setMaximum(n_slices)

        try:
            import cv2

            for i in range(n_slices):
                self.progress_bar.setValue(i + 1)
                self.progress_bar.setFormat(f"Exporting slice {i+1}/{n_slices}...")
                QApplication.processEvents()

                # Get slice
                slice_data = self.current_volume[i]

                # Determine output filename with zero-padded index
                n_digits = len(str(n_slices))
                output_filename = output_dir / f"{base_name}_{str(i).zfill(n_digits)}{extension}"

                # Save slice
                if extension.lower() in ['.tif', '.tiff']:
                    cv2.imwrite(str(output_filename), slice_data)
                elif extension.lower() == '.png':
                    # Normalize to 8-bit for PNG if needed
                    if slice_data.dtype == np.uint16:
                        slice_8bit = (slice_data / 256).astype(np.uint8)
                    elif slice_data.dtype == np.float32 or slice_data.dtype == np.float64:
                        slice_8bit = ((slice_data - slice_data.min()) /
                                     (slice_data.max() - slice_data.min() + 1e-6) * 255).astype(np.uint8)
                    else:
                        slice_8bit = slice_data
                    cv2.imwrite(str(output_filename), slice_8bit)
                else:
                    cv2.imwrite(str(output_filename), slice_data)

            self.showProgress(False)
            self.status_label.setText(f"Exported {n_slices} slices to {output_dir}")

            QMessageBox.information(
                self, "Export Complete",
                f"Exported {n_slices} slices to:\n{output_dir}\n\n"
                f"Filename pattern: {base_name}_XXXX{extension}"
            )

        except Exception as e:
            self.showProgress(False)
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def toggleROIEdit(self, checked):
        """Toggle ROI editing mode - delegate to AnalysisTab"""
        if self.viewer:
            self.viewer.toggleROI(checked)
        # Update button text
        if checked:
            self.edit_roi_btn.setText("Apply\nROI")
            self.compute_btn.setEnabled(False)
        else:
            self.edit_roi_btn.setText("Edit\nROI")
            self.compute_btn.setEnabled(True)

    def onROIModeChanged(self, mode):
        if self.viewer:
            self.viewer.setROIMode(mode.lower())

    def computeOrientation(self):
        """Compute orientation - delegate to AnalysisTab"""
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'computeOrientation'):
            # Set viewer reference if needed
            if self.viewer and self.analysis_tab.viewer != self.viewer:
                self.analysis_tab.viewer = self.viewer
            self.analysis_tab.computeOrientation()

    def openRangeEditor(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openRangeEditor'):
            self.analysis_tab.openRangeEditor()

    def openHistogramDialog(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openHistogramDialog'):
            self.analysis_tab.openHistogramDialog()

    def toggleMagnify(self, checked):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'toggleMagnify'):
            if self.viewer and self.analysis_tab.viewer != self.viewer:
                self.analysis_tab.viewer = self.viewer
            self.analysis_tab.toggleMagnify(checked)

    def resetAllROIs(self):
        if self.viewer:
            self.viewer.resetAllROIs()

    def openCoordinateROIDialog(self):
        """Open dialog to create ROI by entering coordinates"""
        if self.current_volume is None:
            QMessageBox.warning(self, "Warning", "Please load a volume first.")
            return

        # Get volume shape
        volume_shape = self.current_volume.shape

        # Show coordinate input dialog
        dialog = CoordinateROIDialog(self, volume_shape)
        if dialog.exec() == QDialog.Accepted:
            roi_name, roi_bounds = dialog.getROI()
            if roi_name and roi_bounds:
                # Check if ROI name already exists
                if self.viewer and roi_name in self.viewer.rois:
                    reply = QMessageBox.question(
                        self, "ROI Exists",
                        f"ROI '{roi_name}' already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return

                # Create ROI in viewer
                if self.viewer:
                    # Generate a color for this ROI
                    color_idx = len(self.viewer.rois) % len(COLORS['roi_colors'])
                    color = COLORS['roi_colors'][color_idx]

                    # Add ROI to viewer
                    self.viewer.rois[roi_name] = {
                        'bounds': roi_bounds,
                        'color': color,
                        'polygon_xy': None,  # No polygon for coordinate-based ROI
                        'theta': None,
                        'phi': None,
                        'angle': None
                    }

                    # Update display
                    self.viewer.renderVolume()

                    # Show confirmation
                    z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
                    QMessageBox.information(
                        self, "ROI Created",
                        f"ROI '{roi_name}' created:\n"
                        f"  X: {x_min} - {x_max}\n"
                        f"  Y: {y_min} - {y_max}\n"
                        f"  Z: {z_min} - {z_max}"
                    )

    def openFiberDetectionSettings(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openFiberDetectionSettings'):
            self.analysis_tab.openFiberDetectionSettings()

    def detectFibers(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'detectFibers'):
            if self.viewer and self.analysis_tab.viewer != self.viewer:
                self.analysis_tab.viewer = self.viewer
            self.analysis_tab.detectFibers()

    def openInSegtSettings(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openInSegtSettings'):
            self.analysis_tab.openInSegtSettings()

    def openInSegtLabeling(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openInSegtLabeling'):
            self.analysis_tab.openInSegtLabeling()

    def runInSegt(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'runInSegt'):
            self.analysis_tab.runInSegt()

    def openVfSettings(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openVfSettings'):
            self.analysis_tab.openVfSettings()

    def computeVf(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'computeVf'):
            if self.viewer and self.analysis_tab.viewer != self.viewer:
                self.analysis_tab.viewer = self.viewer
            self.analysis_tab.computeVf()

    def exportOrientationToVTK(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'exportOrientationToVTK'):
            self.analysis_tab.exportOrientationToVTK()

    def exportVfMapToVTK(self):
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'exportVfMapToVTK'):
            self.analysis_tab.exportVfMapToVTK()

    def openVoidSettings(self):
        """Open void analysis settings dialog."""
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'openVoidSettings'):
            self.analysis_tab.openVoidSettings()

    def runVoidAnalysis(self):
        """Run void analysis."""
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'runVoidAnalysis'):
            self.analysis_tab.runVoidAnalysis()

    def cropOrientationWithVoid(self):
        """Crop orientation data with void mask."""
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'cropOrientationWithVoid'):
            self.analysis_tab.cropOrientationWithVoid()

    def resetAllAnalysis(self):
        """Reset all analysis results."""
        # Confirm with user
        reply = QMessageBox.question(
            self, "Reset All Analysis",
            "This will reset all analysis results including:\n\n"
            "• Orientation data (theta, phi, angle)\n"
            "• Fiber volume fraction (Vf) maps\n"
            "• Void analysis results\n"
            "• Fiber detection results\n"
            "• ROI orientation data\n"
            "• InSegt segmentation\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Reset orientation data in main window
            self.orientation_data = None
            if hasattr(self, 'orientation_theta'):
                self.orientation_theta = None
            if hasattr(self, 'orientation_phi'):
                self.orientation_phi = None

            # Reset viewer data
            if hasattr(self, 'viewer') and self.viewer:
                # Reset orientation overlays
                self.viewer.base_volume = None
                self.viewer.overlay_volume = None

                # Reset ROI orientation data
                if hasattr(self.viewer, 'rois'):
                    for roi_name, roi_data in self.viewer.rois.items():
                        roi_data['theta'] = None
                        roi_data['phi'] = None
                        roi_data['angle'] = None

                # Reset Vf data
                self.viewer.vf_map = None
                self.viewer.vf_roi_bounds = None
                self.viewer.show_vf_overlay = False

                # Reset void data
                self.viewer.void_mask = None
                self.viewer.void_roi_bounds = None
                self.viewer.show_void_overlay = False

                # Reset fiber detection
                self.viewer.show_fiber_detection = False

                # Clear orientation ROI widgets
                if hasattr(self.viewer, 'orientation_roi_widgets'):
                    self.viewer.orientation_roi_widgets.clear()

                # Re-render
                self.viewer.renderVolume()

            # Reset analysis tab data
            if hasattr(self, 'analysis_tab') and self.analysis_tab:
                # Reset segmentation
                self.analysis_tab.segmentation_volume = None
                self.analysis_tab.segmentation_roi_bounds = None
                self.analysis_tab.segmentation_polygon_mask = None

                # Reset Vf data
                self.analysis_tab.vf_map = None
                self.analysis_tab.vf_segmentation = None
                self.analysis_tab.vf_stats = None
                self.analysis_tab.vf_roi_bounds = None
                self.analysis_tab.vf_polygon_mask = None

                # Reset void data
                self.analysis_tab.void_mask = None
                self.analysis_tab.void_statistics = None
                self.analysis_tab.void_local_fraction = None

                # Reset InSegt model
                if hasattr(self.analysis_tab, '_insegt_model'):
                    self.analysis_tab._insegt_model = None
                if hasattr(self.analysis_tab, 'insegt_model'):
                    self.analysis_tab.insegt_model = None

                # Reset fiber detection settings to defaults
                self.analysis_tab.fiber_detection_settings = {
                    'min_diameter': 5,
                    'max_diameter': 20,
                    'min_distance': 8,
                    'threshold_rel': 0.3
                }

            # Reset export buttons
            if hasattr(self, 'export_orientation_btn'):
                self.export_orientation_btn.setEnabled(False)
            if hasattr(self, 'export_vf_btn'):
                self.export_vf_btn.setEnabled(False)

            # Remove dynamic toggles from pipeline panel
            if hasattr(self, '_vf_toggle_added'):
                self._vf_toggle_added = False
            if hasattr(self, '_void_toggle_added'):
                self._void_toggle_added = False

            QApplication.restoreOverrideCursor()

            # Update status
            self.status_label.setText("All analysis results have been reset.")

            QMessageBox.information(
                self, "Reset Complete",
                "All analysis results have been reset.\n\n"
                "You can now run new analysis on your data."
            )

        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to reset analysis: {str(e)}")

    def updateNoiseScale(self):
        """Update noise scale value for analysis"""
        value = self.noise_scale_slider.value()
        self.noise_scale_label.setText(str(value))

    def addVfToggle(self):
        """Add Vf overlay toggle to the pipeline panel."""
        # Check if already added
        if hasattr(self, '_vf_toggle_added') and self._vf_toggle_added:
            return

        # Get slider layout from slider_panel
        if not hasattr(self, 'slider_panel'):
            return
        slider_layout = self.slider_panel.layout()

        # Find the pipeline group
        pipeline_group = None
        for i in range(slider_layout.count()):
            item = slider_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QGroupBox) and widget.title() == "Pipeline":
                    pipeline_group = widget
                    break

        if pipeline_group is None:
            return

        # Create Vf toggle widget
        vf_widget = QWidget()
        vf_layout = QHBoxLayout(vf_widget)
        vf_layout.setContentsMargins(0, 0, 0, 0)
        vf_layout.setSpacing(5)

        self.vf_check = QCheckBox("Vf Overlay")
        self.vf_check.setChecked(True)
        self.vf_check.stateChanged.connect(self._onVfToggleChanged)
        vf_layout.addWidget(self.vf_check)

        vf_layout.addStretch()

        # Add to pipeline group layout
        pipeline_group.layout().addWidget(vf_widget)
        self._vf_toggle_added = True

    def _onVfToggleChanged(self, state):
        """Handle Vf overlay toggle change."""
        if self.viewer:
            # Handle both Qt.CheckState enum and int values
            if isinstance(state, int):
                self.viewer.show_vf_overlay = (state == 2)  # Qt.Checked = 2
            else:
                self.viewer.show_vf_overlay = (state == Qt.CheckState.Checked)
            self.viewer.renderVolume()

    def addVoidToggle(self):
        """Add Void overlay toggle to the pipeline panel."""
        # Check if already added
        if hasattr(self, '_void_toggle_added') and self._void_toggle_added:
            return

        # Get slider layout from slider_panel
        if not hasattr(self, 'slider_panel'):
            return
        slider_layout = self.slider_panel.layout()

        # Find the pipeline group
        pipeline_group = None
        for i in range(slider_layout.count()):
            item = slider_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QGroupBox) and widget.title() == "Pipeline":
                    pipeline_group = widget
                    break

        if pipeline_group is None:
            return

        # Create Void toggle widget
        void_widget = QWidget()
        void_layout = QHBoxLayout(void_widget)
        void_layout.setContentsMargins(0, 0, 0, 0)
        void_layout.setSpacing(5)

        self.void_check = QCheckBox("Void Overlay")
        self.void_check.setChecked(True)
        self.void_check.stateChanged.connect(self._onVoidToggleChanged)
        void_layout.addWidget(self.void_check)

        void_layout.addStretch()

        # Add to pipeline group layout
        pipeline_group.layout().addWidget(void_widget)
        self._void_toggle_added = True

    def _onVoidToggleChanged(self, state):
        """Handle Void overlay toggle change."""
        if self.viewer:
            # Handle both Qt.CheckState enum and int values
            if isinstance(state, int):
                self.viewer.show_void_overlay = (state == 2)  # Qt.Checked = 2
            else:
                self.viewer.show_void_overlay = (state == Qt.CheckState.Checked)
            self.viewer.renderVolume()

    def onTabChanged(self, index):
        """Handle tab change to show/hide appropriate controls"""
        # Switch ribbon stack
        self.ribbon_stack.setCurrentIndex(index)

        # Tab indices: 0=Volume, 1=Analysis, 2=Modelling, 3=Simulation
        if index == 2:  # Modelling tab (fiber trajectory)
            # Hide slicer view, Modelling tab has its own viewport
            self.content_splitter.setVisible(False)
            self.simulation_content.setVisible(False)
            self.noise_group.setVisible(False)
            self.ribbon_stack.setVisible(True)
            self.modelling_content.setVisible(True)
        elif index == 3:  # Simulation tab
            # Hide slicer view, show simulation content
            self.content_splitter.setVisible(False)
            self.simulation_content.setVisible(True)
            self.noise_group.setVisible(False)
            self.ribbon_stack.setVisible(True)
            self.modelling_content.setVisible(False)
        else:
            # Show slicer view, hide simulation content
            self.content_splitter.setVisible(True)
            self.simulation_content.setVisible(False)
            self.modelling_content.setVisible(False)
            self.ribbon_stack.setVisible(True)
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
        try:
            self.x_slice_slider.valueChanged.disconnect(self.updateSlices)
            self.y_slice_slider.valueChanged.disconnect(self.updateSlices)
            self.z_slice_slider.valueChanged.disconnect(self.updateSlices)
        except (TypeError, RuntimeError):
            pass  # Signals not connected yet

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

    def showProgress(self, show):
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")  # Reset to default percentage format

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

    def onAdjustmentChanged(self):
        """Handle changes to image adjustment sliders"""
        # Update labels
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0
        sharpness = self.sharpness_slider.value()
        invert = self.invert_check.isChecked()

        self.brightness_label.setText(f"{brightness:+d}" if brightness != 0 else "0")
        self.contrast_label.setText(f"{contrast:.2f}")
        self.gamma_label.setText(f"{gamma:.2f}")
        self.sharpness_label.setText(str(sharpness))

        # Update viewer's adjuster settings
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.adjuster.settings.brightness = float(brightness)
            self.viewer.adjuster.settings.contrast = contrast
            self.viewer.adjuster.settings.gamma = gamma
            self.viewer.adjuster.settings.sharpness = float(sharpness)
            self.viewer.adjuster.settings.invert = invert

            # Re-render with new adjustments
            self.viewer.renderVolume()

    def resetAdjustments(self):
        """Reset all image adjustments to default values"""
        # Block signals to prevent multiple renders
        self.brightness_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.gamma_slider.blockSignals(True)
        self.sharpness_slider.blockSignals(True)
        self.invert_check.blockSignals(True)

        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.sharpness_slider.setValue(0)
        self.invert_check.setChecked(False)

        # Unblock signals
        self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self.gamma_slider.blockSignals(False)
        self.sharpness_slider.blockSignals(False)
        self.invert_check.blockSignals(False)

        # Update labels
        self.brightness_label.setText("0")
        self.contrast_label.setText("1.0")
        self.gamma_label.setText("1.0")
        self.sharpness_label.setText("0")

        # Reset viewer's adjuster and re-render
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.adjuster.settings.reset()
            self.viewer.renderVolume()

    def applyAdjustmentsToVolume(self):
        """Apply current adjustments permanently to the volume data"""
        if not hasattr(self, 'viewer') or not self.viewer:
            return

        if self.viewer.current_volume is None:
            QMessageBox.warning(self, "No Volume", "No volume data loaded.")
            return

        if self.viewer.adjuster.settings.is_default():
            QMessageBox.information(self, "No Changes", "No adjustments to apply.")
            return

        reply = QMessageBox.question(
            self, "Apply Adjustments",
            "This will apply the current adjustments to the volume data in memory.\n"
            "The original image files on disk will NOT be modified.\n"
            "You can reload the files to restore the original data.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Apply adjustments to get new volume
            adjusted_volume = self.viewer.adjuster.apply_adjustments()

            # Reset adjustments
            self.resetAdjustments()

            # Set the adjusted volume as the new current/original volume
            self.setVolume(adjusted_volume)

            QMessageBox.information(self, "Success", "Adjustments applied to volume.")

    def exportAdjustmentSettings(self):
        """Export current image adjustment settings to a text file"""
        if not hasattr(self, 'viewer') or not self.viewer:
            QMessageBox.warning(self, "Error", "No viewer available.")
            return

        # Get current settings
        settings = self.viewer.adjuster.settings

        # Prepare volume info
        volume_info = None
        if self.viewer.current_volume is not None:
            vol = self.viewer.current_volume
            volume_info = {
                'Shape': f"{vol.shape}",
                'Data Type': str(vol.dtype),
                'Min Value': f"{vol.min():.2f}",
                'Max Value': f"{vol.max():.2f}"
            }

        # Get save file path
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Adjustment Settings", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if not filename:
            return

        if not filename.lower().endswith('.txt'):
            filename += '.txt'

        # Export settings
        if export_adjustment_settings(settings, filename, volume_info):
            QMessageBox.information(
                self, "Export Successful",
                f"Adjustment settings exported to:\n{filename}"
            )
        else:
            QMessageBox.critical(
                self, "Export Failed",
                "Failed to export adjustment settings."
            )

    def showLogViewer(self):
        """Show log viewer dialog"""
        from vmm.log_viewer import LogViewerDialog
        dialog = LogViewerDialog(self)
        dialog.exec()

def main():
    from vmm.splash import SplashScreen
    from vmm.theme import get_stylesheet, apply_mpl_theme

    app = QApplication(sys.argv)

    # Apply theme based on system settings
    app.setStyleSheet(get_stylesheet())
    apply_mpl_theme()

    # Show splash screen
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()

    # Create main window (this takes time)
    splash.setMessage("Initializing application...")
    QApplication.processEvents()

    window = VMMMainWindow()

    splash.setMessage("Ready!")
    QApplication.processEvents()

    # Close splash and show main window
    splash.close()
    window.showMaximized()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()