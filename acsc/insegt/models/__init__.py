# Core modules
from acsc.insegt.models.kmdict import KMTree, DictionaryPropagator
from acsc.insegt.models.gaussfeat import GaussFeatureExtractor, get_gauss_feat_im
import acsc.insegt.models.utils as utils

# Optional modules (may require additional dependencies)
try:
    from acsc.insegt.models.skbasic import sk_basic_segmentor
except ImportError:
    sk_basic_segmentor = None

try:
    from acsc.insegt.models.featsegt import gauss_features_segmentor
except ImportError:
    gauss_features_segmentor = None
