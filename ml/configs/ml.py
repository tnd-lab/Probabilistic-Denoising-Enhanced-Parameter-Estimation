from models.bases import pdnet_y, dncnn_y
from models.model import DenoiseModel
from configs.radar import N_prbs, N_sym
from configs.data import output_dir
import os
import torch


batch_size = 1
num_workers = 1
pin_memory = False
device = "mps"

epochs = 2
lr = 0.001
betas = (0.5, 0.999)
train_test_split = 0.8

# Denoise model hyperparameters
input_shape = (2, N_prbs, N_sym)
hidden_channel = 16
level = 3
kls_thesh = 0.0001

depth = 10
img_channels = 2
n_filters = 64
kernel_size = 3

domain = "time"
current_model = pdnet_y.PDNet(
    input_shape=input_shape, hidden_channel=hidden_channel, level=level
)
# current_model = dncnn_y.DnCNN(
#     depth=depth,
#     img_channels=img_channels,
#     n_filters=n_filters,
#     kernel_size=kernel_size,
# ).to(device)
model = DenoiseModel(model=current_model, domain=domain).to(device)


checkpoint_path = f"{output_dir}/checkpoints/best_checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint at {checkpoint_path}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")
