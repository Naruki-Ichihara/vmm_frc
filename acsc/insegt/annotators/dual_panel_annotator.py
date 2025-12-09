#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual Panel InSegt Annotator.

A two-panel layout annotator for fiber segmentation:
- Left panel: Original image with annotation overlay
- Right panel: Segmentation result

Colors: Cyan (fiber) and Magenta (background)
"""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import PIL.Image


class DualPanelAnnotator(QtWidgets.QWidget):
    """
    Two-panel annotator with cyan/magenta colors.
    Left: image + annotations, Right: segmentation result
    """

    # Colors: Cyan for fiber (label 1), Magenta for background (label 2)
    colors = np.array([
        [0, 0, 0],        # 0: transparent/eraser
        [0, 255, 255],    # 1: Cyan (fiber)
        [255, 0, 255],    # 2: Magenta (background)
    ], dtype=np.uint8)

    def __init__(self, image, model):
        """
        Initialize dual panel annotator.

        Args:
            image: 2D grayscale uint8 image
            model: InSegt model with process() method
        """
        super().__init__()

        self.image = image.copy()
        self.model = model
        self.liveUpdate = True

        # Image dimensions
        self.img_height, self.img_width = image.shape

        # Create pixmaps
        self.imagePix = self._grayToPixmap(image)
        self.annotationPix = QtGui.QPixmap(self.img_width, self.img_height)
        self.annotationPix.fill(QtGui.QColor(0, 0, 0, 0))
        self.segmentationPix = QtGui.QPixmap(self.img_width, self.img_height)
        self.segmentationPix.fill(QtGui.QColor(0, 0, 0, 0))
        self.cursorPix = QtGui.QPixmap(self.img_width, self.img_height)
        self.cursorPix.fill(QtGui.QColor(0, 0, 0, 0))

        # Drawing state
        self.label = 1  # 1=Cyan (fiber), 2=Magenta (background) - for cursor display
        self._drawingLabel = 1  # Label being used during current draw operation
        self.penWidth = 10
        self.lastDrawPoint = QtCore.QPoint()
        self.activelyDrawing = False

        # Display state
        self.annotationOpacity = 0.7
        self.segmentationOpacity = 0.7

        # Zoom/pan state
        self.zoomFactor = 1.0
        self.panOffset = QtCore.QPoint(0, 0)
        self.lastPanPoint = QtCore.QPoint()
        self.isPanning = False

        # Source rect (portion of image being displayed)
        self.source = QtCore.QRect(0, 0, self.img_width, self.img_height)

        # Probabilities
        self.probabilities = np.empty((0, self.img_height, self.img_width))

        # Setup UI
        self._setupUI()

        # Check model has probToSeg
        if not (hasattr(self.model, 'probToSeg') and callable(getattr(self.model, 'probToSeg'))):
            self.model.probToSeg = self._probToSeg

        self.setMouseTracking(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def _setupUI(self):
        """Setup the UI layout."""
        self.setWindowTitle("InSegt Fiber Labeling - Cyan=Fiber, Magenta=Background")

        # Calculate initial window size
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        max_width = int(screen.width() * 0.9)
        max_height = int(screen.height() * 0.85)

        # Two panels side by side with some spacing
        panel_width = min(self.img_width, (max_width - 20) // 2)
        panel_height = min(self.img_height, max_height - 50)

        # Keep aspect ratio
        scale_w = panel_width / self.img_width
        scale_h = panel_height / self.img_height
        scale = min(scale_w, scale_h)

        self.panel_width = int(self.img_width * scale)
        self.panel_height = int(self.img_height * scale)

        total_width = self.panel_width * 2 + 20  # 20px gap
        total_height = self.panel_height + 30  # Status bar

        self.resize(total_width, total_height)
        self.setMinimumSize(400, 200)

    def paintEvent(self, event):
        """Paint both panels."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Calculate panel positions
        gap = 10
        panel_w = (self.width() - gap) // 2
        panel_h = self.height() - 25

        # Left panel target rect
        left_target = QtCore.QRect(0, 0, panel_w, panel_h)
        # Right panel target rect
        right_target = QtCore.QRect(panel_w + gap, 0, panel_w, panel_h)

        # Draw left panel: image + annotations
        painter.drawPixmap(left_target, self.imagePix, self.source)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(left_target, self.annotationPix, self.source)
        painter.drawPixmap(left_target, self.cursorPix, self.source)

        # Draw right panel: image + segmentation
        painter.drawPixmap(right_target, self.imagePix, self.source)
        painter.drawPixmap(right_target, self.segmentationPix, self.source)

        # Draw panel borders
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 1))
        painter.drawRect(left_target)
        painter.drawRect(right_target)

        # Draw labels
        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(5, self.height() - 8, "Image + Annotations")
        painter.drawText(panel_w + gap + 5, self.height() - 8, "Segmentation Result")

        # Draw status
        if self.label == 0:
            label_str = "Eraser(0)"
        elif self.label == 1:
            label_str = "Fiber(1/Left)"
        else:
            label_str = "Background(2/Right)"
        status = f"Cursor: {label_str} | Pen: {self.penWidth} | Live: {'ON' if self.liveUpdate else 'OFF'}"
        painter.drawText(self.width() // 2 - 120, self.height() - 8, status)

    def _widgetToImage(self, pos):
        """Convert widget position to image coordinates."""
        gap = 10
        panel_w = (self.width() - gap) // 2
        panel_h = self.height() - 25

        # Check if in left panel
        if pos.x() < panel_w:
            # Scale from panel to source
            scale_x = self.source.width() / panel_w
            scale_y = self.source.height() / panel_h
            img_x = pos.x() * scale_x + self.source.x()
            img_y = pos.y() * scale_y + self.source.y()
            return QtCore.QPoint(int(img_x), int(img_y)), True
        return pos, False

    def mousePressEvent(self, event):
        """Handle mouse press."""
        img_pos, in_panel = self._widgetToImage(event.pos())

        if event.button() == QtCore.Qt.LeftButton and in_panel:
            # Draw with current label (selected by keyboard 0/1/2)
            self._drawingLabel = self.label
            self._drawPoint(img_pos, self._drawingLabel)
            self.lastDrawPoint = img_pos
            self.activelyDrawing = True
            self.update()
        elif event.button() == QtCore.Qt.RightButton and in_panel:
            # Right click always draws background (Magenta)
            self._drawingLabel = 2
            self._drawPoint(img_pos, self._drawingLabel)
            self.lastDrawPoint = img_pos
            self.activelyDrawing = True
            self.update()
        elif event.button() == QtCore.Qt.MiddleButton:
            # Start panning
            self.isPanning = True
            self.lastPanPoint = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        img_pos, in_panel = self._widgetToImage(event.pos())

        if self.activelyDrawing and in_panel:
            self._drawLine(self.lastDrawPoint, img_pos, self._drawingLabel)
            self.lastDrawPoint = img_pos
            self.update()
        elif self.isPanning:
            # Pan the view
            delta = event.pos() - self.lastPanPoint
            gap = 10
            panel_w = (self.width() - gap) // 2
            panel_h = self.height() - 25

            scale_x = self.source.width() / panel_w
            scale_y = self.source.height() / panel_h

            new_x = self.source.x() - int(delta.x() * scale_x)
            new_y = self.source.y() - int(delta.y() * scale_y)

            # Clamp
            new_x = max(0, min(new_x, self.img_width - self.source.width()))
            new_y = max(0, min(new_y, self.img_height - self.source.height()))

            self.source.moveTo(new_x, new_y)
            self.lastPanPoint = event.pos()
            self.update()
        elif in_panel:
            # Update cursor
            self._drawCursor(img_pos)
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self.activelyDrawing:
            self.activelyDrawing = False
            if self.liveUpdate:
                self._updateSegmentation()
            self.update()
        if self.isPanning:
            self.isPanning = False

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        img_pos, in_panel = self._widgetToImage(event.position().toPoint())
        if not in_panel:
            return

        delta = event.angleDelta().y()
        zoom_factor = 1.2 if delta > 0 else 0.8

        # Calculate new source size
        new_width = int(self.source.width() / zoom_factor)
        new_height = int(self.source.height() / zoom_factor)

        # Limit zoom
        min_size = 50
        new_width = max(min_size, min(new_width, self.img_width))
        new_height = max(min_size, min(new_height, self.img_height))

        # Keep the point under cursor fixed
        # Calculate relative position of cursor in source rect
        rel_x = (img_pos.x() - self.source.x()) / self.source.width()
        rel_y = (img_pos.y() - self.source.y()) / self.source.height()

        # New top-left
        new_x = int(img_pos.x() - rel_x * new_width)
        new_y = int(img_pos.y() - rel_y * new_height)

        # Clamp
        new_x = max(0, min(new_x, self.img_width - new_width))
        new_y = max(0, min(new_y, self.img_height - new_height))

        self.source = QtCore.QRect(new_x, new_y, new_width, new_height)
        self.update()

    def keyPressEvent(self, event):
        """Handle key press."""
        key = event.key()

        if key == QtCore.Qt.Key_1:
            self.label = 1
            self.update()
        elif key == QtCore.Qt.Key_2:
            self.label = 2
            self.update()
        elif key == QtCore.Qt.Key_0:
            self.label = 0  # Eraser
            self.update()
        elif key == QtCore.Qt.Key_Up:
            self.penWidth = min(self.penWidth + 2, 100)
            self.update()
        elif key == QtCore.Qt.Key_Down:
            self.penWidth = max(self.penWidth - 2, 2)
            self.update()
        elif key == QtCore.Qt.Key_L:
            self.liveUpdate = not self.liveUpdate
            if self.liveUpdate:
                self._updateSegmentation()
            self.update()
        elif key == QtCore.Qt.Key_Z:
            # Reset zoom
            self.source = QtCore.QRect(0, 0, self.img_width, self.img_height)
            self.update()
        elif key == QtCore.Qt.Key_H:
            self._showHelp()
        elif key in range(QtCore.Qt.Key_3, QtCore.Qt.Key_9 + 1):
            # Ignore keys 3-9
            pass

    def _drawPoint(self, pos, label=None):
        """Draw a point on annotation pixmap."""
        if label is None:
            label = self.label
        painter = QtGui.QPainter(self.annotationPix)
        if label == 0:
            # Eraser
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            color = QtGui.QColor(0, 0, 0, 0)
        else:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            c = self.colors[label]
            color = QtGui.QColor(c[0], c[1], c[2], int(255 * self.annotationOpacity))

        painter.setPen(QtGui.QPen(color, self.penWidth,
                                        QtCore.Qt.SolidLine,
                                        QtCore.Qt.RoundCap,
                                        QtCore.Qt.RoundJoin))
        painter.drawPoint(pos)
        painter.end()

    def _drawLine(self, start, end, label=None):
        """Draw a line on annotation pixmap."""
        if label is None:
            label = self.label
        painter = QtGui.QPainter(self.annotationPix)
        if label == 0:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            color = QtGui.QColor(0, 0, 0, 0)
        else:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            c = self.colors[label]
            color = QtGui.QColor(c[0], c[1], c[2], int(255 * self.annotationOpacity))

        painter.setPen(QtGui.QPen(color, self.penWidth,
                                        QtCore.Qt.SolidLine,
                                        QtCore.Qt.RoundCap,
                                        QtCore.Qt.RoundJoin))
        painter.drawLine(start, end)
        painter.end()

    def _drawCursor(self, pos):
        """Draw cursor indicator."""
        self.cursorPix.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(self.cursorPix)
        if self.label == 0:
            color = QtGui.QColor(128, 128, 128, 128)
        else:
            c = self.colors[self.label]
            color = QtGui.QColor(c[0], c[1], c[2], 128)
        painter.setPen(QtGui.QPen(color, self.penWidth,
                                        QtCore.Qt.SolidLine,
                                        QtCore.Qt.RoundCap,
                                        QtCore.Qt.RoundJoin))
        painter.drawPoint(pos)
        painter.end()

    def _updateSegmentation(self):
        """Update segmentation from current annotations."""
        # Get labels from annotation pixmap
        labels = self._pixmapToLabels(self.annotationPix)

        if labels.max() == 0:
            # No annotations yet
            self.segmentationPix.fill(QtGui.QColor(0, 0, 0, 0))
            return

        # Process through model
        self.probabilities = self.model.process(labels)

        # Convert to segmentation
        segmentation = self.model.probToSeg(self.probabilities)

        # Convert to pixmap
        self.segmentationPix = self._labelsToPixmap(segmentation, self.segmentationOpacity)

    def _pixmapToLabels(self, pixmap):
        """Convert pixmap to label array."""
        qimage = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
        # PySide6 uses bytes() instead of constBits()
        buffer = qimage.bits()
        rgba = np.frombuffer(buffer, np.uint8).reshape((pixmap.height(), pixmap.width(), 4)).copy()

        # Convert RGB to labels based on closest color
        rgb = rgba[:, :, :3].reshape(-1, 3).astype(np.int16)
        dist = np.sum(np.abs(rgb.reshape(-1, 1, 3) - self.colors.reshape(1, -1, 3).astype(np.int16)), axis=2)
        labels = np.argmin(dist, axis=1).astype(np.uint8)

        # Zero out where alpha is low (transparent)
        alpha = rgba[:, :, 3].ravel()
        labels[alpha < 50] = 0

        return labels.reshape(pixmap.height(), pixmap.width())

    def _labelsToPixmap(self, labels, opacity):
        """Convert label array to pixmap."""
        rgb = self.colors[labels, :]
        alpha = (255 * opacity * (labels > 0)).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
        rgba = np.ascontiguousarray(rgba)

        qimage = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                                    rgba.shape[1] * 4,
                                    QtGui.QImage.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(qimage.copy())

    def _grayToPixmap(self, gray):
        """Convert grayscale image to pixmap."""
        rgba = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=2)
        rgba = np.ascontiguousarray(rgba)
        qimage = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                                    rgba.shape[1] * 4,
                                    QtGui.QImage.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _probToSeg(probabilities):
        """Default probability to segmentation conversion."""
        segmentation = np.zeros(probabilities.shape[1:], dtype=np.uint8)
        if probabilities.shape[0] > 1:
            p = np.sum(probabilities, axis=0)
            np.argmax(probabilities, axis=0, out=segmentation)
            segmentation += 1
            segmentation[p == 0] = 0
        elif probabilities.shape[0] == 1:
            segmentation[probabilities[0] > 0] = 1
        return segmentation

    def _showHelp(self):
        """Show help dialog."""
        help_text = """
InSegt Fiber Labeling Tool

MOUSE:
  Left-click + drag: Draw fiber (Cyan)
  Right-click + drag: Draw background (Magenta)
  Middle-click + drag: Pan
  Scroll wheel: Zoom

KEYBOARD:
  1: Select fiber label (Cyan)
  2: Select background label (Magenta)
  0: Eraser mode
  ↑/↓: Increase/decrease pen width
  L: Toggle live update
  Z: Reset zoom
  H: Show this help
"""
        QtWidgets.QMessageBox.information(self, "Help", help_text)

    def getLabels(self):
        """Get current labels array."""
        return self._pixmapToLabels(self.annotationPix)

    def leaveEvent(self, event):
        """Clear cursor when mouse leaves."""
        self.cursorPix.fill(QtGui.QColor(0, 0, 0, 0))
        self.update()
