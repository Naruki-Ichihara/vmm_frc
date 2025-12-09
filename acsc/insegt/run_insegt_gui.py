#!/usr/bin/env python
"""
Standalone InSegt GUI runner.

This script runs InSegt annotator as a separate process to avoid Qt conflicts
with the main ACSC application.

Usage:
    python run_insegt_gui.py <image_path> <output_dir> [--model <model_path>]
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='InSegt Interactive Labeling GUI')
    parser.add_argument('image_path', help='Path to input image (numpy .npy file)')
    parser.add_argument('output_dir', help='Directory to save outputs')
    parser.add_argument('--model', help='Optional: path to existing model to load')
    parser.add_argument('--sigmas', type=str, default='1,2', help='Comma-separated sigma values')
    parser.add_argument('--patch-size', type=int, default=9, help='Patch size for KM-tree')
    parser.add_argument('--branching-factor', type=int, default=5, help='Branching factor')
    parser.add_argument('--number-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--training-patches', type=int, default=10000, help='Number of training patches')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scale for faster processing (0.5 = half size)')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image from {args.image_path}...")
    image_orig = np.load(args.image_path)
    print(f"Original image shape: {image_orig.shape}, dtype: {image_orig.dtype}")

    # Convert to uint8 if needed
    if image_orig.dtype == np.uint16:
        image_orig = (image_orig / 256).astype(np.uint8)
    elif image_orig.dtype != np.uint8:
        if image_orig.max() > 0:
            image_orig = ((image_orig - image_orig.min()) / (image_orig.max() - image_orig.min()) * 255).astype(np.uint8)
        else:
            image_orig = np.zeros_like(image_orig, dtype=np.uint8)

    # Downscale image for faster processing
    import cv2
    scale = args.scale
    if scale != 1.0:
        image = cv2.resize(image_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Scaled image shape: {image.shape} (scale={scale})")
    else:
        image = image_orig

    # Parse sigmas
    sigmas = [float(s.strip()) for s in args.sigmas.split(',')]

    # Import PySide6 and InSegt components
    print("Initializing Qt application...")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # Create Qt application FIRST
    app = QApplication(sys.argv)
    app.setApplicationName("InSegt Labeling")

    # Now import InSegt components (after QApplication is created)
    print("Loading InSegt model...")
    from acsc.insegt.fiber_model import FiberSegmentationModel
    from acsc.insegt.annotators.dual_panel_annotator import DualPanelAnnotator

    # Create or load model
    if args.model and os.path.exists(args.model):
        print(f"Loading existing model from {args.model}...")
        model = FiberSegmentationModel.load(args.model)
    else:
        print("Building new model from image...")
        model = FiberSegmentationModel(
            sigmas=sigmas,
            patch_size=args.patch_size,
            branching_factor=args.branching_factor,
            number_layers=args.number_layers,
            training_patches=args.training_patches
        )
        model.build_from_image(image)

    # Set image for processing
    model.set_image(image)
    print("Model ready!")

    # Create annotator window (dual panel: left=image+annotations, right=segmentation)
    print("Opening InSegt annotator...")
    annotator = DualPanelAnnotator(image, model)
    annotator.setWindowFlags(annotator.windowFlags() | Qt.WindowStaysOnTopHint)

    # Status file to communicate with parent process
    status_file = output_dir / "insegt_status.txt"
    labels_file = output_dir / "insegt_labels.npy"
    scale_file = output_dir / "insegt_scale.txt"

    def on_close(event):
        """Handle close event - save labels."""
        try:
            # Get labels from annotator
            labels = annotator.getLabels()

            # Save labels (at scaled resolution)
            np.save(str(labels_file), labels)
            print(f"Labels saved to {labels_file}")

            # Save scale info
            with open(scale_file, 'w') as f:
                f.write(str(scale))
            print(f"Scale ({scale}) saved to {scale_file}")

            # Write status (labels path only - model not saved externally)
            with open(status_file, 'w') as f:
                f.write(f"completed\n{labels_file}")

            print("Labeling completed. Closing window...")

        except Exception as e:
            print(f"Error saving: {e}")
            with open(status_file, 'w') as f:
                f.write(f"error\n{str(e)}")

        event.accept()

    # Override close event
    annotator.closeEvent = on_close

    # Write initial status
    with open(status_file, 'w') as f:
        f.write("running")

    # Show window
    annotator.show()
    annotator.raise_()
    annotator.activateWindow()

    print("\n" + "="*60)
    print("InSegt Labeling Tool (Dual Panel)")
    print("="*60)
    print("Left panel: Image + Annotations")
    print("Right panel: Segmentation Result")
    print("")
    print("MOUSE:")
    print("  - Left-click + drag: Draw fiber (Cyan)")
    print("  - Right-click + drag: Draw background (Magenta)")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Middle-click + drag: Pan")
    print("")
    print("KEYBOARD:")
    print("  - 1: Fiber (Cyan), 2: Background (Magenta), 0: Eraser")
    print("  - Up/Down: Change pen size")
    print("  - L: Toggle live update")
    print("  - Z: Reset zoom")
    print("  - H: Help")
    print("")
    print("Close window when done to save")
    print("="*60 + "\n")

    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
