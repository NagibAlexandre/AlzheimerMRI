import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# Dispositivo (CPU ou GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def get_model():
    """
    Retorna uma ResNet18 pronta para classificação nas classes definidas.
    Inicialmente congela todas as camadas, mantendo apenas a FC ativa para warmup.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))

    return model.to(DEVICE)
