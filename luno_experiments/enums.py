from enum import StrEnum

class Data(StrEnum):
    # Low data regime (APEBENCH)
    DIFF_LIN_1 = "diff_lin_1"
    DIFF_KS_CONS_1 = "diff_ks_cons_1"
    DIFF_HYP_DIFF_1 = "diff_hyp_diff_1"
    DIFF_BURGERS_1 = "diff_burgers_1"
    # OOD (ADVECTION-DIFFUSION-REACTION)
    BASE_2 = "base_2"
    FLIP_2 = "flip_2"
    POS_2 = "pos_2"
    POS_NEG_2 = "pos_neg_2"
    POS_NEG_FLIP_2 = "pos_neg_flip_2"

class DataMode(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

class Method(StrEnum):
    INPUT_PERTURBATIONS = "input_perturbations"
    ENSEMBLE = "ensemble"
    SAMPLE_ISO = "sample_iso"
    LUNO_ISO = "luno_iso"
    SAMPLE_LA = "sample_la"
    LUNO_LA = "luno_la"