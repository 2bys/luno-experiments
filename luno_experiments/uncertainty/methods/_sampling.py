import jax
from functools import partial
from luno_experiments.utils import default
from laplax.eval.pushforward import (
    nonlin_pred_mean,
    nonlin_pred_std,
    nonlin_samples,
    set_nonlin_pushforward,
    nonlin_setup,
)

class SamplingPredictiveMixin:
    def _setup_pushforward(self, **kwargs):
        del kwargs

        self.key = jax.random.key(self.seed)

        # Set pushforward functions with MC versions
        self.pushforward_fns = [
            nonlin_pred_mean,
            nonlin_pred_std,
        ]
        self.sampling_pushforward_fns = self.pushforward_fns + [nonlin_samples]

    def _init_prob_predictive(self, pushforward_fns: list[callable] | None = None, **kwargs):
        return partial(
            set_nonlin_pushforward,
            model_fn=self.model_fn,
            mean_params=self.params,
            posterior_fn=self.weight_space_covariance_fn,
            pushforward_fns=[nonlin_setup] + default(pushforward_fns, self.pushforward_fns),
            num_weight_samples=self.num_weight_samples,
            key=self.key,
            **kwargs,
        )