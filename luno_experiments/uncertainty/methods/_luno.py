import jax
import jax.numpy as jnp
from laplax.curv.utils import LowRankTerms
from laplax.eval.utils import finalize_fns
from luno import FNOGPLastLayer
from luno.covariances.fno import CircularlySymmetricDiagonal
from luno.models.fno import FFTGrid

from luno_experiments.nn.padding import unpad_signal
from functools import partial

from linox import (
    IsotropicScalingPlusSymmetricLowRank,
    PositiveDiagonalPlusSymmetricLowRank,
    SymmetricLowRank,
    linverse,
)


def _unpad(x, ref, pad=2):
    new_shape = (*[v+2*pad for v in ref.shape[:-2]], *ref.shape[-2:])
    x = x.reshape(new_shape)
    x = unpad_signal(x, pad, list(range(ref.ndim - 2)))
    return x


def create_luno_cov(curv_est: LowRankTerms, prior_args, params=None):
    U, S, _ = jax.tree.leaves(curv_est)
    prec = prior_args.get("prior_prec", 1.0)

    # scalar prior
    if not isinstance(prec, dict):
        return linverse(IsotropicScalingPlusSymmetricLowRank(prec, U, S))

    # diagonal + low‐rank prior
    if params is None:
        raise ValueError("Params required for block‐diagonal prior")
    R, W, b = params
    diag = CircularlySymmetricDiagonal(
        prior_args["R"], prior_args["W"], prior_args["b"]
    )
    low_rank = SymmetricLowRank(U=U, S=S)
    return linverse(
        PositiveDiagonalPlusSymmetricLowRank(diagonal=diag, low_rank=low_rank)
    )


def create_luno_posterior(curv_est, wrapper):
    R, W, b = wrapper.params
    def posterior(prior_args):
        cov = create_luno_cov(curv_est, prior_args, params=(R, W, b))
        return FNOGPLastLayer(
            fno_head=wrapper._before_last_fourier_layer,
            R=R, W=W, b=b,
            weight_cov=cov,
            projection=wrapper._out_projection,
            num_output_channels=wrapper.model.out_channels,
        )
    return posterior


def luno_mean_std(results, aux, **kwargs):
    gp, grid = aux["gp"], aux["grid"]
    pred = results["map"]
    m, s = gp.mean_and_std(grid)
    results.update({
        "pred": pred,
        "pred_mean": _unpad(m, pred),
        "pred_std" : _unpad(s, pred),
    })
    return results, aux


def luno_samples(results, aux, **kwargs):
    # Get GP, grid and key
    gp, grid, key = aux["gp"], aux["grid"], aux["key"]
    pred = results["map"]
    num_samples = kwargs.get("num_samples", 10)

    # Sample from GP
    samples = gp.sample(key, grid, size=(num_samples,), dtype=grid.dtype)

    # Update results
    results.update({
        "pred": pred,
        "samples": jax.vmap(_unpad, in_axes=(0, None))(samples, pred),
    })
    return results, aux

def create_grid(inp_shape, padding=2, dtype=None):
    shape = tuple(s + 2*padding for s in inp_shape[:-2])
    return FFTGrid(shape, dtype or jnp.float64)


def set_luno_predictive(
    model_fn, 
    mean_params, 
    prior_arguments,
    posterior_fn,
    pushforward_fns,
    key,
    **kwargs,
):
    gp_op = posterior_fn(prior_arguments)
    
    
    def prob_predictive(input):
        # MAP prediction
        pred = model_fn(input, params=mean_params)

        # GP + grid setup
        gp   = gp_op(input)
        grid = create_grid(input.shape)
        aux  = {"gp": gp, "grid": grid, "key": key}
        res  = {"map": pred}

        # apply push‐through functions
        return finalize_fns(fns=pushforward_fns, results=res, aux=aux, **kwargs)

    return prob_predictive


class LUNOMixin:

    def _setup_pushforward(self, **kwargs):
        del kwargs

        self.pushforward_fns = [
            luno_mean_std,
        ]

        self.sampling_pushforward_fns = self.pushforward_fns + [luno_samples]

    def _init_prob_predictive(self, pushforward_fns: list[callable] | None = None, **kwargs):
        return partial(
            set_luno_predictive,
            model_fn=self.model_fn,
            mean_params=self.params,
            posterior_fn=self.weight_space_covariance_fn,
            pushforward_fns=pushforward_fns or self.pushforward_fns,
            key=self.key,
            **kwargs,
        )