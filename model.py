import torch
import torch.nn as nn
import torch.nn.functional as F

class WBC_Classifier_CNN(nn.Module):
    """
    WBC_Classifier_CNN

    A Convolutional Neural Network (CNN) model designed for 8-class classification of White Blood Cells (WBCs).
    
    Architecture Overview:
    - Input size: (3, 224, 224)
    - 4 convolutional blocks: Conv2D → BatchNorm → ReLU → MaxPooling
    - Flatten layer
    - 2 fully connected (dense) layers with BatchNorm, Dropout(0.5), and ReLU activations
    - Final output layer with 8 neurons (one for each WBC class)

    Regularization: Dropout applied after each dense layer to prevent overfitting.
    """

    def __init__(self, num_classes=8):
        super(WBC_Classifier_CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Max pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers after flattening
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjusted for 224x224 input
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)  # Final output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 + BN + ReLU + Pool

        # Flatten for fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Pass through dense layers with dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))

        # Final output layer
        x = self.fc3(x)

        return x