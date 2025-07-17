import jax
from laplax.eval import evaluate_metrics_on_generator
from laplax.eval.metrics import DEFAULT_REGRESSION_METRICS, DEFAULT_REGRESSION_METRICS_DICT
from luno_experiments.uncertainty.methods._core import UQMethod
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger
from luno_experiments.data.loaders import get_data_loader
from luno_experiments.data import AdvDiffReactScenarios
import jax.numpy as jnp
import numpy as np

class TransformedDataLoader:
    def __init__(
        self, 
        dataloader: DataLoader, 
    ):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield {"input": batch[0], "target": batch[1]}
    
    def __len__(self) -> int:
        return len(self.dataloader)
    


def evaluate(uq: UQMethod, test_loader: DataLoader):
    
    # Get probabilistic predictive function
    pp = jax.jit(uq.prob_predictive())

    # Evaluate test dataset
    results = evaluate_metrics_on_generator(
        pp,
        TransformedDataLoader(test_loader),
        metrics=DEFAULT_REGRESSION_METRICS,
        has_batch=True,
        reduce=jax.numpy.concatenate
    )

    # Return results
    return results


def evaluate_with_samples(uq: UQMethod, test_loader: DataLoader, num_test_points: int = 10):
    # Get probabilistic predictive function
    pp = jax.jit(
        uq._init_prob_predictive(
            pushforward_fns=uq.sampling_pushforward_fns,
        )(prior_arguments=uq.prior_args,)
    )

    # Select test points
    batch = next(iter(test_loader))
    input = batch[0][:num_test_points]
    target = batch[1][:num_test_points]
    data = {"input": input, "target": target}

    # Get samples
    samples = jax.vmap(pp)(data["input"])
    samples["target"] = data["target"]

    # Return results
    return samples

class TrajectoryEvaluator:
    def __init__(self, dataset, trajectory_index=0, batch_size=1):
        # Get trajectory data
        traj, vels, react = dataset[trajectory_index:trajectory_index+batch_size]

        # Set data
        self.traj = jnp.moveaxis(jnp.asarray(traj), 1, -1) # (B, T, S1, S2) -> (B, S1, S2, T)
        self.vels = jnp.moveaxis(jnp.asarray(vels), 1, -1) # (B, C, S1, S2) -> (B, S1, S2, C)
        self.react = jnp.asarray(react)[..., None] # (B, S1, S2) -> (B, S1, S2, 1)
        self.batch_size = batch_size

        # Initialize state
        self.reset()

    def reset(self):
        # Set counter
        self.counter = 0

        # Initialize metrics and predictions as dictionaries of lists
        self.metrics = {"rmse": [], "chi^2": [], "nll": [], "crps": [], "time": []}
        self.predictions = {"mean": [], "std": [], "target": []}

        # Set initial trajectory state
        self.filled_traj = jnp.zeros_like(self.traj).at[..., :10].set(self.traj[..., :10])

    def next_batch(self):
        input = jnp.concatenate(
            [
                self.filled_traj[..., self.counter : self.counter + 10],
                self.vels[..., :],
                self.react[..., :],
            ],
            axis=-1,
            dtype=jnp.float64,
        )

        target = self.traj[..., self.counter + 10]
        self.counter += 1

        # Add batch dimension
        input = input[..., None]
        target = target[..., None, None]

        return {
            "input": input,
            "target": target,
        }

    def evaluate(self, method):
        self.reset()

        while self.counter < self.traj.shape[-1] - 10:
            logger.info(f"Evaluating trajectory at step {self.counter}.")
            
            # Get batch
            batch = self.next_batch()

            # Get prediction
            pred = jax.vmap(method)(batch["input"])

            # Compute metrics
            step_metrics = jax.vmap(lambda t, p: {
                "rmse": DEFAULT_REGRESSION_METRICS_DICT["rmse"](p["pred_mean"], t),
                "chi^2": DEFAULT_REGRESSION_METRICS_DICT["chi^2"](p["pred_mean"], p["pred_std"], t),
                "nll": DEFAULT_REGRESSION_METRICS_DICT["nll"](p["pred_mean"], p["pred_std"], t),
                "crps": DEFAULT_REGRESSION_METRICS_DICT["crps"](p["pred_mean"], p["pred_std"], t),
            })(batch["target"], pred)

            # Append metrics
            for metric_name, metric_value in step_metrics.items():
                self.metrics[metric_name].append(metric_value)

            # Store predictions and target
            self.predictions["mean"].append(pred["pred_mean"][..., 0, 0])
            self.predictions["std"].append(pred["pred_std"][..., 0, 0])
            self.predictions["target"].append(batch["target"][0, ..., 0, 0])

            # Update filled trajectory
            self.filled_traj = self.filled_traj.at[..., self.counter + 10].set(
                pred["pred_mean"][..., 0, 0]
            )

        # Convert lists to arrays
        self.metrics = {k: jnp.array(v) for k, v in self.metrics.items()}
        self.predictions = {k: jnp.array(v) for k, v in self.predictions.items()}

        return {"metrics": self.metrics, "predictions": self.predictions}


def evaluate_full_trajectory(uq: UQMethod, test_loader: DataLoader, batch_size: int = 1, trajectory_index: int = 0):
    """Evaluate a full trajectory using the TrajectoryEvaluator.
    
    Args:
        uq: UQMethod instance
        test_loader: DataLoader containing the test data
        trajectory_index: Index of the trajectory to evaluate
        
    Returns:
        Dictionary containing metrics and predictions for the full trajectory
    """
    # Get probabilistic predictive function
    pp = jax.jit(uq.prob_predictive())

    # Create evaluator
    evaluator = TrajectoryEvaluator(
        test_loader.dataset,
        trajectory_index=trajectory_index,
        batch_size=batch_size
    )

    # Run evaluation
    results = evaluator.evaluate(pp)
    results = jax.tree.map(np.asarray, results)

    return results



def evaluate_ood(uq: UQMethod, test_loader: DataLoader):
    # results
    results = {}
    samples = {}

    # Get names
    path = Path("./data/ood/")
    for name in list(AdvDiffReactScenarios):
        logger.info(f"Evaluating {name}...")
        data_path = path / name / "test_solutions.npz"
        loader = get_data_loader(data_path, name, "all", 1, False)
        
        # Evaluate test dataset
        results[name] = evaluate(uq, loader)
        samples[name] = evaluate_with_samples(uq, loader)

    logger.info("Finished evaluating OOD datasets")

    return results, samples
