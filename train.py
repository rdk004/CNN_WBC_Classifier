# train.py - Training Script for WBC Classifier CNN

"""
This training script implements 5-Fold Stratified Cross-Validation for an 8-class
White Blood Cell (WBC) image classifier using a custom CNN architecture defined in model.py.
It performs the following:

- Loads WBC image dataset using a custom Dataset and DataLoader interface.
- Trains a CNN model using CrossEntropyLoss and the Adam optimizer.
- Tracks metrics per epoch including Accuracy, Loss, ROC AUC, and PRC AUC.
- Saves the best model checkpoint for each fold based on validation ROC AUC.
- Logs training information and generates performance plots:
    - ROC AUC and PRC AUC across folds
    - Accuracy and Loss for the best performing fold
- Outputs saved to: outputs/training_metrics/ and checkpoints/

This script is structured to work with an evaluation interface that imports:
  from train import wbc_trainer as the_trainer
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

from dataset import get_dataloaders
from model import WBC_Classifier_CNN
from config import device, batch_size, learning_rate, epochs, num_classes

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure necessary output directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs/training_metrics", exist_ok=True)

def wbc_trainer(data_dir=None):
    """
    Main training function used by the grading interface.

    Trains a WBC classification model using 5-Fold Stratified Cross-Validation.
    Records accuracy, loss, ROC AUC, and PRC AUC for training and validation sets.
    Saves best model per fold and plots training metrics.

    Args:
        data_dir (str): Optional path to root directory containing Training folder.
                        If None, directory will be auto-detected.
    """
    # Load folds from dataset
    dataloaders = get_dataloaders(train_dir=data_dir, test_dir=None, n_splits=5, verbose=True)
    folds = dataloaders['folds']

    fold_metrics = []  # Store per-fold results
    best_fold_auc = 0
    best_fold = -1

    # Cross-validation loop
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        logger.info(f"\n========== Fold {fold_idx+1}/5 ==========")

        # Initialize model and training components
        model = WBC_Classifier_CNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Track best AUC and metrics for this fold
        best_val_auc = 0
        best_val_acc = 0
        best_fold_epoch = 0

        # Metric trackers
        fold_train_roc_auc, fold_val_roc_auc = [], []
        fold_train_prc_auc, fold_val_prc_auc = [], []
        fold_train_acc, fold_val_acc = [], []
        fold_train_loss, fold_val_loss = [], []

        # Epoch-wise training loop
        for epoch in range(epochs):
            model.train()
            total_train, correct_train, running_loss = 0, 0, 0.0
            y_train_true, y_train_probs = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update training stats
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

                # Store softmax probabilities for AUC/PRC
                y_train_true.extend(labels.cpu().numpy())
                y_train_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

            # Compute training metrics
            train_acc = correct_train / total_train
            y_train_bin = label_binarize(y_train_true, classes=list(range(num_classes)))
            train_probs = np.array(y_train_probs)
            fpr, tpr, _ = roc_curve(y_train_bin.ravel(), train_probs.ravel())
            precision, recall, _ = precision_recall_curve(y_train_bin.ravel(), train_probs.ravel())
            train_roc_auc = auc(fpr, tpr)
            train_prc_auc = average_precision_score(y_train_bin, train_probs)

            # Validation pass
            model.eval()
            total_val, correct_val, val_loss = 0, 0, 0.0
            y_val_true, y_val_probs = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)
                    y_val_true.extend(labels.cpu().numpy())
                    y_val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            # Compute validation metrics
            val_acc = correct_val / total_val
            y_val_bin = label_binarize(y_val_true, classes=list(range(num_classes)))
            val_probs = np.array(y_val_probs)
            fpr, tpr, _ = roc_curve(y_val_bin.ravel(), val_probs.ravel())
            precision, recall, _ = precision_recall_curve(y_val_bin.ravel(), val_probs.ravel())
            val_roc_auc = auc(fpr, tpr)
            val_prc_auc = average_precision_score(y_val_bin, val_probs)

            # Log this epoch's performance
            logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={running_loss/len(train_loader):.4f}, Train ROC AUC={train_roc_auc:.4f}, Train PRC AUC={train_prc_auc:.4f} | Val Acc={val_acc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val ROC AUC={val_roc_auc:.4f}, Val PRC AUC={val_prc_auc:.4f}")

            # Store metrics
            fold_train_acc.append(train_acc)
            fold_val_acc.append(val_acc)
            fold_train_loss.append(running_loss / len(train_loader))
            fold_val_loss.append(val_loss / len(val_loader))
            fold_train_roc_auc.append(train_roc_auc)
            fold_val_roc_auc.append(val_roc_auc)
            fold_train_prc_auc.append(train_prc_auc)
            fold_val_prc_auc.append(val_prc_auc)

            # Save best model for this fold
            if val_roc_auc > best_val_auc:
                best_val_auc = val_roc_auc
                best_val_acc = val_acc
                best_fold_epoch = epoch
                torch.save(model.state_dict(), f"checkpoints/best_model_fold{fold_idx+1}.pt")

        # Store fold-wide metrics
        fold_metrics.append({
            'train_acc': fold_train_acc,
            'val_acc': fold_val_acc,
            'train_loss': fold_train_loss,
            'val_loss': fold_val_loss,
            'train_roc_auc': fold_train_roc_auc,
            'val_roc_auc': fold_val_roc_auc,
            'train_prc_auc': fold_train_prc_auc,
            'val_prc_auc': fold_val_prc_auc,
            'best_epoch': best_fold_epoch,
            'best_auc': best_val_auc,
            'best_acc': best_val_acc
        })

        if best_val_auc > best_fold_auc:
            best_fold_auc = best_val_auc
            best_fold = fold_idx

    # Final summary
    logger.info("\n========= Summary of Best Model =========")
    logger.info(f"Best Fold: {best_fold+1}, Best Epoch: {fold_metrics[best_fold]['best_epoch']+1}, Best Val ROC AUC: {best_fold_auc:.4f}")

    # Save results
    np.save("outputs/training_metrics/fold_metrics.npy", fold_metrics)
    with open("outputs/training_metrics/fold_metrics.txt", "w") as f:
        for idx, metrics in enumerate(fold_metrics):
            f.write(f"Fold {idx+1} - Best Val ROC AUC: {metrics['best_auc']:.4f}\n")
    with open("outputs/training_metrics/best_fold.txt", "w") as f:
        f.write(str(best_fold + 1))

    # Plot metrics
    plt.figure(figsize=(10, 5))
    for i, metrics in enumerate(fold_metrics):
        plt.plot(metrics['val_roc_auc'], label=f"Fold {i+1}")
    plt.title("Validation ROC AUC per Epoch for Each Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC AUC")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("outputs/training_metrics/foldwise_val_roc_auc_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, metrics in enumerate(fold_metrics):
        plt.plot(metrics['val_prc_auc'], label=f"Fold {i+1}")
    plt.title("Validation PRC AUC per Epoch for Each Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation PRC AUC")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("outputs/training_metrics/foldwise_val_prc_auc_plot.png")
    plt.close()

    # Plot best fold's accuracy and loss
    best = fold_metrics[best_fold]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(best['train_acc'], label='Train Accuracy')
    ax[0].plot(best['val_acc'], label='Val Accuracy')
    ax[0].set_title(f"Accuracy (Best Fold {best_fold+1})")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(best['train_loss'], label='Train Loss')
    ax[1].plot(best['val_loss'], label='Val Loss')
    ax[1].set_title(f"Loss (Best Fold {best_fold+1})")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(f"outputs/training_metrics/best_fold_{best_fold+1}_accuracy_loss.png")
    plt.close()
