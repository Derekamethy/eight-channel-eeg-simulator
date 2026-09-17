"""Desktop-side proof of concept for an eight-channel EEG simulator."""

from .eeg_data import EEGData, EEGAnnotation
from .synthetic_eeg import ACTIVE_CHANNELS, generate_synthetic_eeg

__all__ = ["ACTIVE_CHANNELS", "EEGAnnotation", "EEGData", "generate_synthetic_eeg"]
__version__ = "1.0.0"
