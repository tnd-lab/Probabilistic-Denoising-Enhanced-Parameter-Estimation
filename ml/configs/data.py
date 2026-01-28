from utils.scales import scale_complex_matrices, resize_matrix
from utils.resize import ComplexMatrixResizer

# Load dataset
root_dir = "data/mats/freq/"
output_dir = "exps/PDNet_Y"
eval_dir = "data/evals/PDNet_Y"
p_freq_path = "data/p_freq_255.mat"

categories = [
    "P_prbs",
    "Z_PRBS_waveform_no_noise",
    "Z_PRBS_waveform",
    "Y_PRBS_waveform_no_noise",
    "Y_PRBS_waveform",
]

snrs = [
    "20",
]

transform_funcs = [
    # MinMaxScaler(feature_range=None, device="mps"),
    # ComplexMatrixResizer(method="lanczos"),
]
