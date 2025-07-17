from laplax.curv.cov import set_posterior_fn
from laplax.enums import CurvApprox

from luno_experiments.uncertainty.methods._core import UQMethod
from luno_experiments.uncertainty.methods._luno import LUNOMixin, create_luno_posterior
from luno_experiments.uncertainty.methods._sampling import SamplingPredictiveMixin

class SAMPLE_LA(SamplingPredictiveMixin, UQMethod):
    @property
    def requires_preprocessing(self) -> bool:
        return True
    
    @property
    def requires_calibration(self) -> bool:
        return True
    
    @property
    def requires_wrapper(self) -> bool:
        return True

    def _preprocess(self, train_loader):
        del train_loader

        if self.low_rank_terms is None:
            msg = "Low rank terms are missing."
            raise ValueError(msg)

        self.weight_space_covariance_fn = set_posterior_fn(
            curv_type=CurvApprox.LANCZOS,
            curv_estimate=self.low_rank_terms,
            layout=self.params,
        )


class LUNO_LA(LUNOMixin, UQMethod):
    @property
    def requires_preprocessing(self) -> bool:
        return True
    
    @property
    def requires_calibration(self) -> bool:
        return True
    
    @property
    def requires_wrapper(self) -> bool:
        return True
    
    def _preprocess(self, train_loader):
        del train_loader

        if self.low_rank_terms is None:
            msg = "Low rank terms are missing."
            raise ValueError(msg)
        
        self.weight_space_covariance_fn = create_luno_posterior(
            self.low_rank_terms, self.wrapper
        )
