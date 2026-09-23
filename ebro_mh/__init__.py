"""Sequential Monte Carlo rule ensemble for binary clinical risk prediction."""

__version__ = "1.1.0"

from .metrics import classification_metrics, calibration_bins  # noqa: F401
from .methods.rule_ensemble import EBROMH  # noqa: F401
