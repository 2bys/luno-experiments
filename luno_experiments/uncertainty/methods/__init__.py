from luno_experiments.uncertainty.methods._input_perturbations import InputPerturbations
from luno_experiments.enums import Method
from luno_experiments.uncertainty.methods._iso import SAMPLE_ISO, LUNO_ISO
from luno_experiments.uncertainty.methods._laplace import SAMPLE_LA, LUNO_LA
from luno_experiments.uncertainty.methods._ensemble import Ensemble

UQMethods = {
    Method.INPUT_PERTURBATIONS: InputPerturbations,
    Method.SAMPLE_ISO: SAMPLE_ISO,
    Method.LUNO_ISO: LUNO_ISO,
    Method.SAMPLE_LA: SAMPLE_LA,
    Method.LUNO_LA: LUNO_LA,
    Method.ENSEMBLE: Ensemble,
}
