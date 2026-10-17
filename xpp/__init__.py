"""XPP: cross-polarization pairing for label-free few-shot rock thin-section classification."""
from .data import (FILENAME_PATTERN, GROUP_NAMES, Images, augment, load_index, random_box, resize,  # noqa: F401
                   split_rocks, to_tensor, xpl_keys)
from .evaluation import evaluate, make_episodes  # noqa: F401
from .model import build_model, image_features, set_train  # noqa: F401
from .presets import OLD_NAMES, PRESETS  # noqa: F401

__version__ = "1.0.0"
