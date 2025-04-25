# dataset.py

"""
Handles dataset preparation for training and testing.
- Applies augmentations for training.
- Applies only resizing and normalization for validation and test sets.
- Prepares 5-Fold Stratified Cross-Validation DataLoaders.
- Logs class distributions.
- Auto-detects dataset paths if not explicitly provided.
Compatible with `interface.py` grading structure.
"""

import os
import numpy as np
from collections import Counter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import logging
import torch
from pathlib import Path

from config import batch_size, resize_x, resize_y

# Setup logger for recording messages
logger = logging.getLogger(__name__)

def set_global_seed(seed=42):
    """
    Sets random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WBCDataset(datasets.ImageFolder):
    """
    Custom dataset class extending torchvision's ImageFolder.
    Automatically assigns labels based on folder names sorted alphabetically.

    Args:
        root (str): Root directory path.
        transform (callable, optional): Transformations to apply to images.
    """
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

def auto_detect_data_paths(base_folder_name="actual_training_testing_data"):
    """
    Auto-detects training and testing directories based on project structure.

    Args:
        base_folder_name (str): Root folder containing 'Training' and 'Testing' subfolders.

    Returns:
        tuple: (training_path, testing_path)
    """
    root = Path("project_rishabh_kulkarni").resolve()
    matches = list(root.rglob(base_folder_name))

    if not matches:
        raise FileNotFoundError(f"Folder '{base_folder_name}' not found in {root} tree.")

    base_dir = matches[0]
    training_path = base_dir / "Training"
    testing_path = base_dir / "Testing"

    if not training_path.exists():
        raise FileNotFoundError(f"'Training' folder not found in {base_dir}")
    if not testing_path.exists():
        raise FileNotFoundError(f"'Testing' folder not found in {base_dir}")

    logger.info(f"Training path detected: {training_path}")
    logger.info(f"Testing path detected: {testing_path}")

    return str(training_path), str(testing_path)

def get_train_transforms():
    """
    Returns training transformations including augmentations.

    Returns:
        torchvision.transforms.Compose: Training transformations.
    """
    return transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_test_transforms():
    """
    Returns test/validation transformations (only resize and normalization).

    Returns:
        torchvision.transforms.Compose: Test/validation transformations.
    """
    return transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_folds_loaders(train_dir=None, n_splits=5, verbose=True):
    """
    Creates stratified K-Fold training and validation DataLoaders.
    I have chosen a value of k=5 (n_splits=5) which is standard for KFold CV.

    Args:
        train_dir (str): Directory path containing training images.
        n_splits (int): Number of folds.
        verbose (bool): Whether to print class distribution.

    Returns:
        list: List of (train_loader, val_loader) tuples.
    """
    set_global_seed(42)
    if train_dir is None:
        train_dir, _ = auto_detect_data_paths()

    dataset = WBCDataset(root=train_dir, transform=get_train_transforms())
    targets = dataset.targets

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        train_labels = [targets[i] for i in train_idx]
        class_counts = Counter(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[targets[i]] for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        logger.info(f"Fold {fold_idx+1} created with {len(train_subset)} training and {len(val_subset)} validation samples")
        folds.append((train_loader, val_loader))

    if verbose:
        class_distribution(targets, dataset.classes, dataset_name="Training Set (Full)")

    return folds

def get_test_loader(test_dir=None):
    """
    Creates a DataLoader for the test dataset.

    Args:
        test_dir (str): Directory path containing test images.

    Returns:
        DataLoader: Test DataLoader.
    """
    set_global_seed(42)
    if test_dir is None:
        _, test_dir = auto_detect_data_paths()

    test_dataset = WBCDataset(root=test_dir, transform=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_distribution(test_dataset.targets, test_dataset.classes, dataset_name="Testing Set")
    return test_loader

def get_dataloaders(train_dir=None, test_dir=None, n_splits=5, verbose=True):
    """
    Wrapper function to return training folds and test DataLoader.

    Args:
        train_dir (str): Directory path containing training images.
        test_dir (str): Directory path containing test images.
        n_splits (int): Number of folds.
        verbose (bool): Whether to print class distribution.

    Returns:
        dict: {'folds': folds_list, 'test': test_loader}
    """
    folds = get_folds_loaders(train_dir=train_dir, n_splits=n_splits, verbose=verbose)
    test_loader = get_test_loader(test_dir=test_dir)
    return {'folds': folds, 'test': test_loader}

def class_distribution(targets, class_names, dataset_name="Dataset"):
    """
    Logs the class distribution of a given dataset.

    Args:
        targets (list): List of label indices.
        class_names (list): List of class names corresponding to indices.
        dataset_name (str): Dataset label for logging.
    """
    counts = Counter(targets)
    logger.info(f"\nClass distribution in {dataset_name}:")
    for class_idx, count in sorted(counts.items()):
        logger.info(f"  {class_names[class_idx]:25}: {count} images")
    logger.info(f"Total images in {dataset_name}: {len(targets)}")
