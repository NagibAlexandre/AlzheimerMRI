"""task.py — Federated Learning integrado à CNN ResNet18"""

import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# --- Configurações globais ---
# Diretório de dados absoluto, calculado a partir da localização deste arquivo
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Codigo
DATA_PROCESSED = os.path.join(ROOT_DIR, "dataset", "Processed")
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
IMG_SIZE = 224
BATCH = 32


# ============================================================
# Modelo (sua CNN adaptada: ResNet18 com fine-tuning parcial)
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Carrega ResNet18 com pesos ImageNet
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Congela todos os parâmetros inicialmente
        for param in self.model.parameters():
            param.requires_grad = False

        # Substitui a camada final (FC) para 4 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(CLASSES))

        # Descongela parte da rede (layer3 + layer4 + fc)
        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# ============================================================
# Funções auxiliares para dados
# ============================================================
def load_data(partition_id: int, num_partitions: int):
    """
    Carrega uma partição IID do dataset Processed para simular clientes locais.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carrega dataset completo (treino + val + test)
    train_path = os.path.join(DATA_PROCESSED, "train")
    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Dataset de treino não encontrado em: {train_path!r}.\n"
            "Verifique se o caminho está correto e se os dados foram processados."
        )

    full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

    # Divide de forma IID entre clientes
    partition_size = len(full_dataset) // num_partitions
    start = partition_id * partition_size
    end = start + partition_size if partition_id < num_partitions - 1 else len(full_dataset)
    subset_indices = list(range(start, end))
    partition = torch.utils.data.Subset(full_dataset, subset_indices)

    # Divide partição em treino/val (80/20)
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_dataset, val_dataset = random_split(partition, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=2)

    return trainloader, valloader


# ============================================================
# Funções de treino e teste compatíveis com Flower
# ============================================================
def train(net, trainloader, epochs, device):
    """Treina o modelo localmente (cliente)."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5)

    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(trainloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def test(net, valloader, device):
    """Avalia o modelo localmente (cliente)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in valloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_total / len(valloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


# ============================================================
# Utilitários para comunicação FL
# ============================================================
def get_weights(net):
    """Extrai pesos do modelo como lista de arrays numpy."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Define pesos no modelo a partir de lista numpy (vindo do servidor FL)."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
