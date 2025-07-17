from laplax.api import _make_nll_objective
from laplax.eval.calibrate import optimize_prior_prec
from loguru import logger
from luno_experiments.uncertainty.methods._core import UQMethod
from torch.utils.data import DataLoader

def calibrate(uq: UQMethod, valid_loader: DataLoader, **kwargs):
    # Get default values if not provided
    log_prior_prec_min = kwargs.get("log_prior_prec_min", -3.0)
    log_prior_prec_max = kwargs.get("log_prior_prec_max", 3.0)
    grid_size = kwargs.get("grid_size", 50)
    patience = kwargs.get("patience", 5)

    logger.debug(
        "Starting calibration on grid [{}, {}] ({} pts, pat={})",
        log_prior_prec_min,
        log_prior_prec_max,
        grid_size,
        patience,
    )

    data = next(iter(valid_loader))
    data = {"input": data[0], "target": data[1]}
    objective_fn = _make_nll_objective(
        uq._init_prob_predictive()
    )
    def objective(x):
        return objective_fn(x, data)
    
    prior_prec = optimize_prior_prec(
        objective=objective,
        log_prior_prec_min=log_prior_prec_min,
        log_prior_prec_max=log_prior_prec_max,
        grid_size=grid_size,
        patience=patience,
    )
    prior_args = {"prior_prec": prior_prec}
    logger.info("Calibration completed with prior prec {}.", prior_prec)
    
    return prior_args