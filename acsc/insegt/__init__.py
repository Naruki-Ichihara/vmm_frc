# InSegt - Interactive Segmentation
# Core modules (no GUI dependencies)
import acsc.insegt.models.utils as utils
from acsc.insegt.models.kmdict import KMTree, DictionaryPropagator
from acsc.insegt.models.gaussfeat import GaussFeatureExtractor

# GUI modules are NOT imported here to avoid QPixmap issues
# Import them directly when needed:
#   from acsc.insegt.annotators.insegtannotator import insegt, InSegtAnnotator
