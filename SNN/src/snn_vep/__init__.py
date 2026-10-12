"""Spiking neural network experiments for the VEP EEG dataset."""

from .configs import load_config
from .training import train_one_fold

__all__ = ["load_config", "train_one_fold"]
