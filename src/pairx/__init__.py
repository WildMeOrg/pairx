""" Pairwise mAtching of Intermediate Representations for eXplainability."""

try:
    # placeholder for addition of setuptools_scm git tag based versioning
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

from .core import explain, pairx
from .dataset import XAIDataset, get_img_pair_from_paths
from .loaders import wildme_multispecies_miewid, toy_df

__all__ = [
    "__version__",
    "explain",
    "pairx",
    "XAIDataset",
    "get_img_pair_from_paths",
    "wildme_multispecies_miewid",
    "toy_df"
]
