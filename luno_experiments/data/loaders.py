"""All data loaders for the experiments.

This module provides PyTorch Dataset and DataLoader implementations for various
experimental datasets, including advection-diffusion-reaction and APEBENCH data.
"""

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import numpy.random as rng
from loguru import logger
from luno_experiments.enums import Data, DataMode
from luno_experiments.data.apebench import APEBenchScenarios
from luno_experiments.data.adv_diff_react import AdvDiffReactScenarios

# Parameters for data loading
TIME_HISTORY = 10  # Number of time steps to use as input
TIME_FUTURE = 1    # Number of time steps to predict

class Advection(Dataset):
    """
    Dataset class for advection-diffusion-reaction data.

    This dataset loads and processes advection-diffusion-reaction trajectories,
    including initial conditions, velocity fields, and reaction terms.

    Parameters
    ----------
    data_dir : Path
        Path to the .npz file containing the dataset

    Attributes
    ----------
    trajectories : ndarray
        Array of shape (n_samples, n_timesteps, nx, ny) containing the trajectories
    velocities : ndarray
        Array of shape (n_samples, 2, nx, ny) containing velocity fields
    reaction_terms : ndarray
        Array of shape (n_samples, nx, ny) containing reaction terms
    """
    def __init__(self, data_dir: Path):
        # Set data directory
        self.data_dir = data_dir

        # Load data
        data = np.load(self.data_dir)
        self.trajectories = data["trajectories"]
        self.velocities = data["velocities"]
        self.reaction_terms = data["reactions_terms"]
        
    def subsample(self, max_num_of_samples: int):
        """
        Subsample the dataset to a maximum number of samples.

        Parameters
        ----------
        max_num_of_samples : int
            Maximum number of samples to keep
        """
        self.trajectories = self.trajectories[:max_num_of_samples]
        self.velocities = self.velocities[:max_num_of_samples]
        self.reaction_terms = self.reaction_terms[:max_num_of_samples]

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for the DataLoader.

        This function processes a batch of samples by:
        1. Sampling random start times
        2. Selecting input trajectories
        3. Selecting target trajectories
        4. Combining with velocity and reaction terms

        Parameters
        ----------
        batch : list
            List of tuples (trajectory, velocity, reaction_term)

        Returns
        -------
        tuple
            - input_traj: Array of shape (B, S1, S2, T, 13) containing input data
            - target_traj: Array of shape (B, S1, S2, T, 1) containing target data
        """
        # Sample random start time
        traj_batch, veloc_batch, react_batch = zip(*batch)
        max_start_time = traj_batch[0].shape[0] - TIME_HISTORY - TIME_FUTURE - 1
        start_times = rng.randint(0, max_start_time + 1, size=len(traj_batch))

        # Select input trajectory
        input_traj_1 = jnp.stack(
            [
                traj_batch[i][start_time : start_time + TIME_HISTORY]
                for i, start_time in enumerate(start_times)
            ],
            axis=0,
        )
        input_traj = jnp.concatenate([
            jnp.moveaxis(input_traj_1, 1, -1), # (B, T, S1, S2) -> (B, S1, S2, T)
            jnp.moveaxis(jnp.stack(veloc_batch), 1, -1),
            jnp.stack(react_batch)[..., None]
        ], axis=-1)[..., None] # (B, S1, S2, T, 13)

        # Select target trajectory
        target_traj = jnp.stack(
            [
                traj_batch[i][
                    start_time + TIME_HISTORY : start_time + TIME_HISTORY + TIME_FUTURE
                ]
                for i, start_time in enumerate(start_times)
            ],
            axis=0,
        )
        target_traj = jnp.moveaxis(target_traj, 1, -1)[..., None] # (B, T, S1, S2) -> (B, S1, S2, T, 1)

        return input_traj, target_traj

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve

        Returns
        -------
        tuple
            (trajectory, velocity, reaction_term) for the specified index
        """
        return self.trajectories[idx], self.velocities[idx], self.reaction_terms[idx]


class APEBench(Dataset):
    """
    Dataset class for APEBENCH data.

    This dataset loads and processes APEBENCH trajectories for various
    differential equation scenarios.

    Parameters
    ----------
    data_dir : Path
        Path to the .npz file containing the dataset

    Attributes
    ----------
    trajectories : ndarray
        Array of shape (n_samples, n_timesteps, 1, n_points) containing the trajectories
    """
    def __init__(self, data_dir: Path):
        # Set data directory
        self.data_dir = data_dir

        # Load data
        data = np.load(self.data_dir)
        self.trajectories = data["trajectories"]

        assert len(self.trajectories.shape) == 4, "APEBENCH data must have 4 dimensions, got {}".format(self.trajectories.shape)
        assert self.trajectories.shape[2] == 1, "APEBENCH data must have 1 channel"

    def subsample(self, max_num_of_samples: int):
        """
        Subsample the dataset to a maximum number of samples.

        Parameters
        ----------
        max_num_of_samples : int
            Maximum number of samples to keep
        """
        self.trajectories = self.trajectories[:max_num_of_samples]

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for the DataLoader.

        This function processes a batch of samples by:
        1. Sampling random start times
        2. Selecting input trajectories
        3. Selecting target trajectories

        Parameters
        ----------
        batch : list
            List of trajectory arrays

        Returns
        -------
        tuple
            - input_traj: Array of shape (B, S, T, 1) containing input data
            - target_traj: Array of shape (B, S, T, 1) containing target data
        """
        # Sample random start time
        max_start_time = batch[0].shape[0] - TIME_HISTORY - TIME_FUTURE - 1
        start_times = rng.randint(0, max_start_time + 1, size=len(batch))

        # Select input trajectory
        input_traj = jnp.stack(
            [
                batch[i][start_time : start_time + TIME_HISTORY]
                for i, start_time in enumerate(start_times)
            ],
            axis=0,
        )
        input_traj = jnp.moveaxis(input_traj, (1, 2), (-2, -1)) # (B, T, 1, S) -> (B, S, T, 1)

        # Select target trajectory
        target_traj = jnp.stack(
            [
                batch[i][
                    start_time + TIME_HISTORY : start_time + TIME_HISTORY + TIME_FUTURE
                ]
                for i, start_time in enumerate(start_times)
            ],
            axis=0,
        )
        target_traj = jnp.moveaxis(target_traj, (1, 2), (-2, -1)) # (B, T, 1, S) -> (B, S, T, 1)

        return input_traj, target_traj

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve

        Returns
        -------
        ndarray
            Trajectory array for the specified index
        """
        return self.trajectories[idx]


class APEBenchDeterministic(Dataset):
    """
    Deterministic dataset class for APEBENCH data.

    This dataset loads and processes APEBENCH trajectories for various
    differential equation scenarios.

    Parameters
    ----------
    data_dir : Path
        Path to the .npz file containing the dataset

    Attributes
    ----------
    trajectories : ndarray
        Array of shape (n_samples, n_timesteps, 1, n_points) containing the trajectories
    """
    def __init__(self, data_dir: Path):
        # Set data directory
        self.data_dir = data_dir

        # Load data
        data = np.load(self.data_dir)
        self.trajectories = data["trajectories"]
        self.num_time_steps = self.trajectories.shape[1]

        assert len(self.trajectories.shape) == 4, "APEBENCH data must have 4 dimensions, got {}".format(self.trajectories.shape)
        assert self.trajectories.shape[2] == 1, "APEBENCH data must have 1 channel"

    def subsample(self, max_num_of_samples: int):
        """
        Subsample the dataset to a maximum number of samples.

        Parameters
        ----------
        max_num_of_samples : int
            Maximum number of samples to keep
        """
        self.trajectories = self.trajectories[:max_num_of_samples]
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for the DataLoader.

        This function processes a batch of samples by:
        1. Sampling random start times
        2. Selecting input trajectories
        3. Selecting target trajectories

        Parameters
        ----------
        batch : list
            List of trajectory arrays

        Returns
        -------
        tuple
            - input_traj: Array of shape (B, S, T, 1) containing input data
            - target_traj: Array of shape (B, S, T, 1) containing target data
        """
        # Select input trajectory
        input_traj = jnp.stack(
            [
                batch[i][:TIME_HISTORY]
                for i in range(len(batch))
            ],
            axis=0,
        )
        input_traj = jnp.moveaxis(input_traj, (1, 2), (-2, -1)) # (B, T, 1, S) -> (B, S, T, 1)

        # Select target trajectory
        target_traj = jnp.stack(
            [
                batch[i][TIME_HISTORY:]
                for i in range(len(batch))
            ],
            axis=0,
        )
        target_traj = jnp.moveaxis(target_traj, (1, 2), (-2, -1)) # (B, T, 1, S) -> (B, S, T, 1)

        return input_traj, target_traj

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.trajectories) * (self.num_time_steps - TIME_HISTORY - TIME_FUTURE - 1)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve

        Returns
        -------
        ndarray
            Trajectory array for the specified index
        """
        # Get the index of the sample in the dataset
        TIME_LENGTH = TIME_HISTORY + TIME_FUTURE + 1
        sample_idx = idx // (self.num_time_steps - TIME_LENGTH)
        time_step_idx = idx % (self.num_time_steps - TIME_LENGTH)

        return self.trajectories[sample_idx][time_step_idx : time_step_idx + TIME_LENGTH]



def get_data_loader(
        folder_path: Path, 
        data_name: Data, 
        max_num_of_samples: int | str, 
        batch_size: int, 
        shuffle: bool
    ):
    """
    Create a DataLoader for the specified dataset.

    Parameters
    ----------
    folder_path : Path
        Path to the dataset directory
    data_name : Data
        Name of the dataset to load
    max_num_of_samples : int or str
        Maximum number of samples to load. If "all", loads all samples.
    batch_size : int
        Batch size for the DataLoader
    shuffle : bool
        Whether to shuffle the data

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the specified dataset

    Raises
    ------
    ValueError
        If the data_name is not supported
    """
    # Load dataset
    if data_name in APEBenchScenarios:
        dataset = APEBench(folder_path)
    elif data_name in AdvDiffReactScenarios:
        dataset = Advection(folder_path)
    else:
        msg = f"Data name {data_name} not supported"
        logger.error(msg)
        raise ValueError(msg)

    # Subsample dataset
    if max_num_of_samples != "all":
        dataset.subsample(max_num_of_samples)

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    return loader


def get_data_loaders(
        folder_path: Path | str,
        data_name: Data,
        max_num_of_samples: list[int], 
        batch_sizes: list[int], 
        shuffle_train: bool = True
    ):
    """
    Create DataLoaders for all splits (train/valid/test) of a dataset.

    Parameters
    ----------
    folder_path : Path or str
        Path to the dataset directory
    data_name : Data
        Name of the dataset to load
    max_num_of_samples : list[int]
        List of maximum sample counts for each split [train, valid, test]
    batch_sizes : list[int]
        List of batch sizes for each split [train, valid, test]
    shuffle_train : bool, optional
        Whether to shuffle the training data

    Returns
    -------
    list[DataLoader]
        List of DataLoaders for train, validation, and test splits
    """
    # Get data loaders
    loaders = [ 
        get_data_loader(
            Path(folder_path) / f"{mode}_solutions.npz",
            data_name,
            max_num_of_samples[i], 
            batch_sizes[i], 
            shuffle = mode == DataMode.TRAIN and shuffle_train
        )
        for i, mode in enumerate(list(DataMode))
    ]

    return loaders