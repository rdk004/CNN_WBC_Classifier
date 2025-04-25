# config.py - Configuration file for model training and inference

import torch

# ----------------------------
# Device Setup
# ----------------------------
# Priority: CUDA > MPS (Apple) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 32          # Number of images per training batch
epochs = 10              # Total training epochs
learning_rate = 0.001    # Learning rate for optimizer
num_classes = 8          # Number of output classes (WBC types)

# ----------------------------
# Image Input Dimensions
# ----------------------------
resize_x = 224           # Resize width for input images
resize_y = 224           # Resize height for input images
