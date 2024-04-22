import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score

class cs21b033(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Softmax(dim =1)
        self.fc1 = nn.Linear(28*28*1, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.m(x)
        return x

    def get_model():
      model = cs21b033()
      return model
