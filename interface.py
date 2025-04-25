# interface.py - Interface module for WBC_Classifier_CNN training and inference

"""
This interface connects all core components of the WBC classifier project.
The grader will use these imports to train and evaluate your model.
Ensure all components work independently and follow correct naming.
"""

# ----------------------------
# Dataset and DataLoader
# ----------------------------
from dataset import WBCDataset as TheDataset
from dataset import get_dataloaders as the_dataloader

# ----------------------------
# CNN Model
# ----------------------------
from model import WBC_Classifier_CNN as TheModel

# ----------------------------
# Training Loop (Cross-validation)
# ----------------------------
from train import wbc_trainer as the_trainer

# ----------------------------
# Inference on Single Image or Batch, along with optional full statistical evaluation utility
# ----------------------------
from predict import wbc_predictor as the_predictor
from predict import predict_model # - Optional full evaluation utility;  Evaluates the best saved model (from the respective fold) on the complete test dataset

# ----------------------------
# Config Parameters
# ----------------------------
from config import batch_size as the_batch_size
from config import epochs as total_epochs
