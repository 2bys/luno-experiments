import argparse
from luno_experiments.data import (
    AdvDiffReactScenarios,
    APEBenchScenarios,
    get_data_loaders,
)
from luno_experiments.enums import Data, Method
from luno_experiments.nn.utils import load_model, load_model_ensemble
from luno_experiments.uncertainty.calibration import calibrate
from luno_experiments.uncertainty.evaluation import evaluate, evaluate_ood, evaluate_with_samples, evaluate_full_trajectory
from pathlib import Path
import jax
from laplax.curv.utils import LowRankTerms
from luno_experiments.scripts.config import ModelConfig
from luno_experiments.uncertainty import UQMethods
from luno_experiments.scripts.utils import load_with_pickle, save_results, str_to_bool, fix_random_seed
import jax.numpy as jnp
import wandb

jax.config.update("jax_enable_x64", True)

def parse_args(
    args: argparse.Namespace | None = None
): 
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_name",
        type=Data,
        default=Data.DIFF_LIN_1,
    )

    p.add_argument(
        "--method",
        type=Method,
        default=Method.INPUT_PERTURBATIONS,
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
        "--low_rank_name",
        type=str,
        default="_low_rank_terms.pkl" 
    )

    p.add_argument(
        "--low_rank_rank",
        type=int,
        default=100,
    )

    p.add_argument(
        "--num_weight_samples",
        type=int,
        default=200,
        help="Number of weight samples to use for the method",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--trajectory_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluating full trajectories",
    )

    p.add_argument(
        "--trajectory_index",
        type=int,
        default=0,
        help="Index of the trajectory to evaluate",
    )

    p.add_argument(
        "--wandb",
        type=str_to_bool,
        default=False,
        help="Whether to log to wandb",
    )

    p.add_argument(
        "--large_run",
        type=str_to_bool,
        default=False,
        help="Whether to run a large run",
    )

    args = p.parse_args(args)

    # Set data arguments
    data_source = "apebench" if args.data_name in APEBenchScenarios else "ood"
    args.data_folder_path = Path(f"./data/{data_source}/{args.data_name}")
    
    # Set model arguments
    args.model_hparams = ModelConfig[args.data_name]    
    args.model_id = (
        f"d={args.data_name}_"
        f"ns={args.max_num_of_samples}_"
        f"nl={args.model_hparams.get('num_layers')}_"
        f"w={args.model_hparams.get('width')}_"
        f"m={args.model_hparams.get('modes')}"
    )
    args.model_seed = 0

    # Set experiment name
    args.experiment_name = f"{args.data_name}_{args.method}"

    # Additional arguments for laplace and ensemble.
    args.ensemble_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    args.low_rank_path = Path("./models") / args.model_id / args.low_rank_name

    # Set calibration arguments
    args.calibration_args = {
        "log_prior_prec_min": -5.0 if args.large_run else -3.0,
        "log_prior_prec_max": 5.0 if args.large_run else 3.0,
        "grid_size": 1000 if args.large_run else 50,
        "patience": 50 if args.large_run else 5,
    }


    return args

def evaluation(args: argparse.Namespace):
    # Fix random seed
    fix_random_seed(args.seed)

    # Setup wandb logging
    if args.wandb:
        wandb.init(
            project="luno",
            name=args.experiment_name,
            config=args,
            tags=[
                args.data_name,
                args.method,
            ],
            group=f"eval_{args.data_name}",  # Group all evaluations for the same dataset
            notes=f"Evaluation run for {args.method} on {args.data_name} with {args.max_num_of_samples} samples"
        )

    # Load data
    train_loader, valid_loader, test_loader = get_data_loaders(
        args.data_folder_path,
        args.data_name,
        [args.max_num_of_samples, "all", "all"],
        [args.batch_size] * 3,
    )

    # Load model
    model = load_model(
    Path("./models")/ f"{args.model_id}/s={args.model_seed}/checkpoint/state",
        args.model_hparams,
    )

    # Load ensemble
    if args.method == Method.ENSEMBLE:
        model_ensemble = load_model_ensemble(
            Path("./models")/ f"{args.model_id}/s={args.model_seed}/checkpoint/state",
            args.model_hparams,
            args.ensemble_seeds,
            args.model_seed,
        )
    else:
        model_ensemble = None

    # Load low rank terms
    if args.method in {Method.LUNO_LA, Method.SAMPLE_LA}:
        low_rank_terms = load_with_pickle(args.low_rank_path)
        low_rank_terms = LowRankTerms(
            U = low_rank_terms.U[:, :args.low_rank_rank],
            S = low_rank_terms.S[:args.low_rank_rank],
            scalar = 0.
        )
    else:
        low_rank_terms = None

    # Load method
    uq = UQMethods[args.method](
        model=model,
        model_ensemble=model_ensemble,
        seed=args.seed,
        low_rank_terms=low_rank_terms,
        load_calibration=False,
        load_checkpoints=False,
        num_weight_samples=args.num_weight_samples,
        checkpoint_base_path=Path("./checkpoints")/ f"{args.model_id}/{args.method}",
    )

    # Preprocess
    if uq.requires_preprocessing:
        uq.preprocess(train_loader)

    if uq.requires_calibration:
        prior_args = calibrate(uq, valid_loader, **args.calibration_args)
        uq.set_prior_args(prior_args)

    # Evaluate
    results = evaluate(uq, test_loader)
    print(jax.tree.map(jnp.mean, results))


    # # Evaluate with samples
    samples = evaluate_with_samples(uq, test_loader)

    # Evaluate OOD if applicable
    if args.data_name in AdvDiffReactScenarios:
        ood_results, ood_samples = evaluate_ood(uq, test_loader)
        trajectory_results = evaluate_full_trajectory(
            uq, 
            test_loader, 
            batch_size=args.trajectory_batch_size,
            trajectory_index=args.trajectory_index,
        )
    else: 
        ood_results = None
        ood_samples = None
        trajectory_results = None

    # Save results
    save_results(
        experiment_name=args.experiment_name,
        data_name=args.data_name,
        method_name=args.method,
        results=results,
        samples=samples,
        ood_results=ood_results,
        ood_samples=ood_samples,
        trajectory_results=trajectory_results,
        log_to_wandb=args.wandb,
    )


if __name__ == "__main__":
    args = parse_args()
    evaluation(args)
