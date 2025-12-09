"""
InSegt Fiber Segmentation Model.

This module provides a wrapper class for InSegt-based fiber segmentation
that can be saved, loaded, and applied to multiple images.
"""

import numpy as np
import pickle
from pathlib import Path

from acsc.insegt.models.kmdict import KMTree, DictionaryPropagator
from acsc.insegt.models.gaussfeat import GaussFeatureExtractor
import acsc.insegt.models.utils as utils


class FiberSegmentationModel:
    """
    A segmentation model for fiber detection using InSegt.

    This class wraps the KM-tree dictionary learning approach and can be
    saved/loaded for reuse across multiple images.
    """

    def __init__(self, sigmas=[1, 2, 4], patch_size=9, branching_factor=5,
                 number_layers=5, training_patches=30000):
        """
        Initialize the fiber segmentation model.

        Args:
            sigmas: List of sigma values for Gaussian features.
            patch_size: Patch size for KM-tree (must be odd).
            branching_factor: Branching factor for KM-tree.
            number_layers: Number of layers in KM-tree.
            training_patches: Number of training patches for KM-tree.
        """
        self.sigmas = sigmas
        self.patch_size = patch_size
        self.branching_factor = branching_factor
        self.number_layers = number_layers
        self.training_patches = training_patches

        # Model components (set after training)
        self.gauss_extractor = None
        self.kmtree = None
        self.dict_propagator = None
        self.is_trained = False

        # Training image info
        self.training_image_shape = None

    def build_from_image(self, image):
        """
        Build the KM-tree from an image.

        Args:
            image: 2D grayscale image (uint8 or uint16).
        """
        # Normalize image to float [0, 1]
        if image.dtype == np.uint8:
            img_float = utils.normalize_to_float(image)
        elif image.dtype == np.uint16:
            img_float = utils.normalize_to_float((image / 256).astype(np.uint8))
        else:
            img_float = image.astype(np.float64)
            if img_float.max() > 1:
                img_float = img_float / img_float.max()

        self.training_image_shape = image.shape

        # Create feature extractor
        self.gauss_extractor = GaussFeatureExtractor(sigmas=self.sigmas)

        # Extract features
        features = self.gauss_extractor(img_float, update_normalization=True, normalize=True)

        # Build KM-tree
        self.kmtree = KMTree(
            patch_size=self.patch_size,
            branching_factor=self.branching_factor,
            number_layers=self.number_layers,
            normalization=False
        )
        self.kmtree.build(features, self.training_patches)

        # Create dictionary propagator
        self.dict_propagator = DictionaryPropagator(
            dictionary_size=self.kmtree.tree.shape[0],
            patch_size=self.patch_size
        )

    def process(self, labels, nr_classes=None):
        """
        Process labels to produce probability maps.

        This is the method called by InSegtAnnotator.

        Args:
            labels: 2D array of labels (0=unlabeled, 1-N=classes).
            nr_classes: Number of classes (optional).

        Returns:
            Probability array of shape (n_classes, height, width).
        """
        if self.kmtree is None:
            raise RuntimeError("Model not built. Call build_from_image first.")

        if nr_classes is None:
            nr_classes = int(labels.max())

        if nr_classes == 0:
            return np.zeros((0,) + labels.shape)

        # Convert labels to one-hot encoding
        labels_onehot = utils.labels_to_onehot(labels)

        # Get assignment for current image
        # Note: We need to extract features from the current image
        # For the annotator, the image is the same as training image
        assignment = self._current_assignment

        # Propagate labels through dictionary
        self.dict_propagator.improb_to_dictprob(assignment, labels_onehot)
        probs = self.dict_propagator.dictprob_to_improb(assignment)

        return probs

    def set_image(self, image):
        """
        Set the current image for processing.

        Args:
            image: 2D grayscale image.
        """
        if self.kmtree is None:
            raise RuntimeError("Model not built. Call build_from_image first.")

        # Normalize image
        if image.dtype == np.uint8:
            img_float = utils.normalize_to_float(image)
        elif image.dtype == np.uint16:
            img_float = utils.normalize_to_float((image / 256).astype(np.uint8))
        else:
            img_float = image.astype(np.float64)
            if img_float.max() > 1:
                img_float = img_float / img_float.max()

        # Extract features using stored normalization
        features = self.gauss_extractor(img_float, update_normalization=False, normalize=True)

        # Search KM-tree
        self._current_assignment = self.kmtree.search(features)

    def segment_image(self, image, labels):
        """
        Segment an image using provided labels.

        Args:
            image: 2D grayscale image.
            labels: 2D label array from annotation.

        Returns:
            Segmentation array.
        """
        self.set_image(image)
        probs = self.process(labels)
        return utils.segment_probabilities(probs)

    def save(self, filepath):
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model.
        """
        if self.kmtree is None:
            raise RuntimeError("Model not built. Nothing to save.")

        model_data = {
            'sigmas': self.sigmas,
            'patch_size': self.patch_size,
            'branching_factor': self.branching_factor,
            'number_layers': self.number_layers,
            'training_patches': self.training_patches,
            'kmtree_tree': self.kmtree.tree,
            'kmtree_patch_size': self.kmtree.patch_size,
            'kmtree_branching_factor': self.kmtree.branching_factor,
            'kmtree_number_layers': self.kmtree.number_layers,
            'kmtree_normalization': self.kmtree.normalization,
            'gauss_normalization_means': self.gauss_extractor.normalization_means,
            'gauss_normalization_stds': self.gauss_extractor.normalization_stds,
            'gauss_normalization_count': self.gauss_extractor.normalization_count,
            'training_image_shape': self.training_image_shape,
            'dict_size': self.dict_propagator.dictionary_size if self.dict_propagator else None
        }

        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded FiberSegmentationModel.
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new model instance
        model = cls(
            sigmas=model_data['sigmas'],
            patch_size=model_data['patch_size'],
            branching_factor=model_data['branching_factor'],
            number_layers=model_data['number_layers'],
            training_patches=model_data['training_patches']
        )

        # Restore KM-tree
        model.kmtree = KMTree(
            patch_size=model_data['kmtree_patch_size'],
            branching_factor=model_data['kmtree_branching_factor'],
            number_layers=model_data['kmtree_number_layers'],
            normalization=model_data['kmtree_normalization']
        )
        model.kmtree.tree = model_data['kmtree_tree']

        # Restore Gaussian feature extractor
        model.gauss_extractor = GaussFeatureExtractor(sigmas=model_data['sigmas'])
        model.gauss_extractor.normalization_means = model_data['gauss_normalization_means']
        model.gauss_extractor.normalization_stds = model_data['gauss_normalization_stds']
        model.gauss_extractor.normalization_count = model_data['gauss_normalization_count']

        # Restore dictionary propagator
        if model_data['dict_size'] is not None:
            model.dict_propagator = DictionaryPropagator(
                dictionary_size=model_data['dict_size'],
                patch_size=model_data['patch_size']
            )

        model.training_image_shape = model_data['training_image_shape']
        model.is_trained = True

        return model


def run_insegt_annotator(image, model=None, sigmas=[1, 2, 4], patch_size=9,
                         branching_factor=5, number_layers=5, training_patches=30000):
    """
    Run the InSegt annotator GUI.

    Args:
        image: 2D grayscale image (uint8).
        model: Optional pre-trained FiberSegmentationModel.
        sigmas: Sigma values for Gaussian features (if model is None).
        patch_size: Patch size for KM-tree (if model is None).
        branching_factor: Branching factor (if model is None).
        number_layers: Number of layers (if model is None).
        training_patches: Number of training patches (if model is None).

    Returns:
        Tuple of (annotator, model) after GUI closes.
    """
    import PyQt5.QtWidgets
    from acsc.insegt.annotators.insegtannotator import InSegtAnnotator

    # Ensure image is uint8
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Create or use provided model
    if model is None:
        model = FiberSegmentationModel(
            sigmas=sigmas,
            patch_size=patch_size,
            branching_factor=branching_factor,
            number_layers=number_layers,
            training_patches=training_patches
        )
        model.build_from_image(image)

    # Set current image
    model.set_image(image)

    # Create and run annotator
    app = PyQt5.QtWidgets.QApplication.instance()
    if app is None:
        app = PyQt5.QtWidgets.QApplication([])

    annotator = InSegtAnnotator(image, model)
    annotator.show()

    # Run event loop (blocking)
    app.exec_()

    return annotator, model
