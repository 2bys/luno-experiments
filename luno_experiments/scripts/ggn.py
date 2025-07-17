"""Gauss-Newton Matrix (GGN) computation for neural network analysis.

This module provides functionality for computing and analyzing the Gauss-Newton Matrix
(GGN) of trained FNO models. It supports both full GGN computation and low-rank
approximations using the SKERCH algorithm. The GGN is useful for understanding model
uncertainty and performing second-order optimization.

Key features:
- Full GGN computation with eigenvalue decomposition
- Low-rank GGN approximation using SKERCH
- Support for both APEBENCH and advection-diffusion-reaction models
- Flexible batch processing and memory management
"""

# Standard library imports
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

# Third-party imports
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger
from laplax.api import GGN
from laplax.util.tree import get_size
from laplax.curv.utils import LowRankTerms, get_matvec
from laplax.types import DType
from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import to_dense
from skerch import linops
from skerch.decompositions import seigh

# Local imports
from luno_experiments.enums import Data
from luno_experiments.data import AdvDiffReactScenarios
from luno_experiments.data.loaders import get_data_loaders, APEBenchDeterministic
from luno_experiments.nn.utils import load_model
from luno_experiments.scripts.config import ModelConfig
from luno_experiments.scripts.utils import str_to_bool, fix_random_seed
from luno_experiments.nn.wrapper import split_wrapper, FNOWrapper

# Enable double precision
jax.config.update("jax_enable_x64", True)

def parse_args(args: Namespace | None = None):
    p = ArgumentParser(description="GGN computation.")

    p.add_argument(
        "--data_name",
        type=Data,
        default=Data.DIFF_LIN_1,
    )

    p.add_argument(
       "--max_num_of_samples",
        type=int,
        default=25,
        help="Maximum number of samples that were used in training",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    p.add_argument(
        "--shuffle_loader",
        type=str_to_bool,
        default=False,
    )

    p.add_argument(
        "--compute_full_ggn",
        type=str_to_bool,
        default=False,
    )

    p.add_argument(
        "--max_rank",
        type=int,
        default=10,
    )

    p.add_argument(
        "--max_num_of_batches",
        type=int,
        default=None,
    )

    p.add_argument(
        "--model_seed",
        type=int,
        default=0
    )

    p.add_argument(
        "--name_tag",
        type=str,
        default="",
    )

    args = p.parse_args(args)

    # Set data path
    data_source = 'ood' if args.data_name in AdvDiffReactScenarios else 'apebench'
    args.data_folder_path = Path(f"./data/{data_source}") / args.data_name
    
    # Set model arguments
    args.model_hparams = ModelConfig[args.data_name]
    args.model_id = (
        f"d={args.data_name}_"
        f"ns={args.max_num_of_samples}_"
        f"nl={args.model_hparams.get('num_layers')}_"
        f"w={args.model_hparams.get('width')}_"
        f"m={args.model_hparams.get('modes')}"
    )

    # Random seed    
    args.seed = 42

    # Set file names
    args.low_rank_file_name = f"{args.name_tag}_low_rank_terms.pkl"

    return args


class JAXMV(linops.TorchLinOpWrapper):
    def __init__(self, matvec, shape, dtype=jnp.float32):
        self.shape = shape
        self.matvec = matvec
        self.dtype = dtype

    def __matmul__(self, x):
        x_dtype = x.dtype
        x = jnp.asarray(
            x.detach().cpu().numpy(), dtype=self.dtype
        )
        x = self.matvec(x)
        return torch.tensor(np.asarray(x), dtype=x_dtype)

    def __rmatmul__(self, x):
        return self.__matmul__(x.T)


def skerch_low_rank(
    A,
    *,
    layout=None,
    rank: int = 100,
    return_dtype: DType = jnp.float64,
    mv_jittable=True,
):
    # Setup mv product.
    matvec, size = get_matvec(A, layout=layout, jit=mv_jittable)
    op = JAXMV(matvec, (size, size))

    res = seigh(
        op, op_device="cpu", op_dtype=torch.float64, outer_dim=rank, inner_dim=rank
    )

    low_rank_result = LowRankTerms(
        U=jnp.asarray((res[0] @ res[1]).detach().cpu()),
        S=jnp.asarray(res[2].detach().cpu().numpy()),
        scalar=jnp.asarray(0.0, dtype=return_dtype),
    )
    return low_rank_result


def main(args):
    # Fix random seed
    fix_random_seed(args.seed)

    # Load data
    if args.data_name in AdvDiffReactScenarios:
        train_loader, _, _ = get_data_loaders(
            args.data_folder_path,
            args.data_name,
            [args.max_num_of_samples, "all", "all"],
            [args.batch_size] * 3,
            shuffle_train=args.shuffle_loader,
        )
    else:
        trainset = APEBenchDeterministic(
            args.data_folder_path / "train_solutions.npz"
        )
        trainset.subsample(args.max_num_of_samples)
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=args.shuffle_loader,
            num_workers=0,
            collate_fn=trainset.collate_fn,
        )
    
    # Load model
    model = load_model(
        Path("./models")/ f"{args.model_id}/s={args.model_seed}/checkpoint/state",
        args.model_hparams,
    )
    wrapper = FNOWrapper(model, dtype=jnp.float64)
    model_fn, params = split_wrapper(wrapper)

    logger.info(f"Model has {get_size(params)} parameters.")
    logger.info(f"Train loader has {len(train_loader)} batches.")

    if args.max_num_of_batches is None:
        args.max_num_of_batches = len(train_loader)

    # Initialize GGN
    logger.info("Initializing GGN.")
    
    ggn_mv = GGN(
        model_fn,
        params,
        train_loader,
        loss_fn="mse",
        has_batch=True,
    )
    

    if args.compute_full_ggn:
        flatten, unflatten = create_pytree_flattener(params)
        ggn_mv = wrap_function(
            ggn_mv, 
            input_fn=lambda x: unflatten(jnp.asarray(x, dtype=jnp.float32)),
            output_fn=lambda x: jnp.asarray(flatten(x), dtype=jnp.float32),
        )
        ggn = to_dense(ggn_mv, layout=get_size(params))
        eigen_dec = jnp.linalg.eigh(ggn)
        low_rank_terms = LowRankTerms(
            U=eigen_dec.eigenvectors,
            S=eigen_dec.eigenvalues,
            scalar=0.0,
        )

    else:
        low_rank_terms = skerch_low_rank(
            A=ggn_mv,
            layout=params,
            rank=args.max_rank,
        )
        
    # Store low rank terms
    save_path = Path("./models") / args.model_id
    with open(save_path / args.low_rank_file_name, "wb") as f:
        pickle.dump(low_rank_terms, f)

    logger.info("Ending script.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
