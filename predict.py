# predict.py - Inference and Evaluation Script for WBC Classifier CNN

"""
This script supports two main functionalities:

1. wbc_predictor(input_tensor):
   - Interface-compatible function for inference.
   - Accepts a single image tensor or batch tensor.
   - Loads the best model (as determined by training).
   - Returns predicted class indices and softmax probabilities.

2. predict_model(data_dir):
   - Optional full evaluation utility.
   - Evaluates the best saved model on the complete test dataset.
   - Reports extended metrics:
       - Accuracy
       - Balanced Accuracy
       - Precision (weighted)
       - Recall (sensitivity)
       - Specificity (average)
       - F1 Score (weighted)
       - Matthews Correlation Coefficient (MCC)
       - ROC AUC (micro)
       - PRC AUC (micro)
   - Saves metrics to .txt and .npy files
   - Plots Confusion Matrix

Usage:
  from predict import wbc_predictor as the_predictor
  from predict import predict_model  # optional
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from model import WBC_Classifier_CNN
from config import device, num_classes
from dataset import get_test_loader
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create output directory for prediction metrics
os.makedirs("outputs/prediction_metrics", exist_ok=True)

def wbc_predictor(input_tensor):
    """
    Interface-compatible function to run inference on a single image or batch of images.

    Args:
        input_tensor (torch.Tensor): Tensor of shape [C, H, W] or [B, C, H, W].

    Returns:
        tuple: (predicted_class_indices, predicted_probabilities)
            - predicted_class_indices: list of integer predictions
            - predicted_probabilities: list of softmax probability vectors
    """
    # Load the best fold checkpoint dynamically
    with open("outputs/training_metrics/best_fold.txt", "r") as f:
        best_fold = int(f.read().strip())
    model_path = f"checkpoints/best_model_fold{best_fold}.pt"

    # Initialize and load model
    model = WBC_Classifier_CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # If input is a single image, add batch dimension
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)  # Get softmax probabilities
        preds = torch.argmax(probs, dim=1)  # Get predicted classes

    return preds.cpu().tolist(), probs.cpu().tolist()

def predict_model(data_dir=None):
    """
    Loads the best model, evaluates it on the test dataset, and saves metrics and plots.

    Args:
        data_dir (str): Path to the dataset folder containing 'Testing' subfolder.
    """
    logger.info("\n=== Running Full Model Evaluation ===")

    # Load test data loader
    test_loader = get_test_loader(test_dir=None if data_dir is None else os.path.join(data_dir, "Testing"))
    class_names = test_loader.dataset.classes
    logger.info(f"Detected Class Names: {class_names}")

    # Read best model fold
    with open("outputs/training_metrics/best_fold.txt", "r") as f:
        best_fold = int(f.read().strip())
    model_path = f"checkpoints/best_model_fold{best_fold}.pt"
    logger.info(f"Loading best model from: {model_path}")

    # Load model
    model = WBC_Classifier_CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    # Inference loop on test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Convert results to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_pred_bin = label_binarize(y_pred, classes=list(range(num_classes)))

    # Calculate basic evaluation metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Manually calculate specificity
    specificity = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    avg_specificity = np.mean(specificity)

    # ROC AUC and PRC AUC (micro-averaged)
    roc_auc = roc_auc_score(y_bin, y_pred_bin)
    prc_auc = average_precision_score(y_bin, y_pred_bin)

    # Log metrics to console
    logger.info("\n*** Model Performance on Test Set ***")
    logger.info(f"Accuracy:               {accuracy:.4f}")
    logger.info(f"Balanced Accuracy:      {balanced_acc:.4f}")
    logger.info(f"Precision (Weighted):   {precision:.4f}")
    logger.info(f"Recall (Sensitivity):   {recall:.4f}")
    logger.info(f"Specificity (Avg):      {avg_specificity:.4f}")
    logger.info(f"F1 Score (Weighted):    {f1:.4f}")
    logger.info(f"Matthews Corr. Coef:    {mcc:.4f}")
    logger.info(f"ROC AUC (micro):        {roc_auc:.4f}")
    logger.info(f"PRC AUC (micro):        {prc_auc:.4f}")
    logger.info("\nClassification Report:\n")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Save metrics to file
    np.save("outputs/prediction_metrics/test_metrics.npy", {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': avg_specificity,
        'f1_score': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc
    })

    # Save text file for easy reading
    with open("outputs/prediction_metrics/test_metrics.txt", "w") as f:
        f.write(f"Accuracy:               {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy:      {balanced_acc:.4f}\n")
        f.write(f"Precision (Weighted):   {precision:.4f}\n")
        f.write(f"Recall (Sensitivity):   {recall:.4f}\n")
        f.write(f"Specificity (Avg):      {avg_specificity:.4f}\n")
        f.write(f"F1 Score (Weighted):    {f1:.4f}\n")
        f.write(f"Matthews Corr. Coef:    {mcc:.4f}\n")
        f.write(f"ROC AUC (micro):        {roc_auc:.4f}\n")
        f.write(f"PRC AUC (micro):        {prc_auc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("outputs/prediction_metrics/confusion_matrix.png")
    plt.close()

    logger.info("Confusion matrix plot saved.")
