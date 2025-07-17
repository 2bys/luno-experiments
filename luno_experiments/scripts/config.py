"""Model configuration settings for different datasets.

This module defines the ModelConfig dictionary that maps each dataset type to its
specific model hyperparameters. The configuration includes settings for input/output
channels, Fourier modes, network width, number of layers, and dimensionality,
automatically adjusted based on whether the dataset is from APEBENCH or
advection-diffusion-reaction experiments.
"""

from luno_experiments.enums import Data
from luno_experiments.data import APEBenchScenarios, AdvDiffReactScenarios

ModelConfig = {
    k: {
        "in_channels": 13 if k in AdvDiffReactScenarios else 10,
        "out_channels": 1, 
        "modes": 12,
        "width": 18,
        "num_layers": 4,
        "dims": 1 if k in APEBenchScenarios else 2,
    }
    for k in list(Data)
}
