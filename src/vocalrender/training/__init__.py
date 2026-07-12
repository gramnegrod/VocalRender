"""
Training utilities for VoxCPM fine-tuning.

This package mirrors the training mechanics used in the minicpm-audio
tooling while relying solely on local audio-text datasets managed via
the HuggingFace ``datasets`` library.

Only core names used by ``from vocalrender.training import ...`` in training
scripts are re-exported here.  All other symbols should be imported
directly from their submodules (e.g.
``from vocalrender.training.svs_data import ...``).
"""

from .accelerator import Accelerator
from .config import SVSTrainConfig
from .tracker import TrainingTracker
from .data import (
    load_audio_text_datasets,
    HFVoxCPMDataset,
    build_dataloader,
    BatchProcessor,
)

__all__ = [
    "Accelerator",
    "SVSTrainConfig",
    "TrainingTracker",
    "load_audio_text_datasets",
    "HFVoxCPMDataset",
    "build_dataloader",
    "BatchProcessor",
]
