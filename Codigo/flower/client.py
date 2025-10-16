import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import flwr as fl

# --- Config ---
DATA_PROCESSED = "Codigo/dataset/Processed/"
IMG_SIZE = 224
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", DEVICE)
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 5
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Datasets ---
train_ds = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"train"), transform_train)
val_ds   = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"val"), transform_test)
test_ds  = datasets.ImageFolder(os.path.join(DATA_PROCESSED,"test"), transform_test)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# --- Modelo ---
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

# --- Cliente Flower ---
class CnnClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)

        # --- Fase 1: Treinar apenas FC ---
        for epoch in range(WARMUP_EPOCHS):
            model.train()
            total, correct = 0, 0
            loop = tqdm(train_loader, desc=f"Warmup Epoch {epoch+1}/{WARMUP_EPOCHS}", unit="batch")
            for imgs, labels in loop:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix(loss=loss.item(), acc=correct/total)

        # --- Fase 2: Fine-tuning layer3+layer4 ---
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
        optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

        for epoch in range(FINETUNE_EPOCHS):
            model.train()
            total, correct = 0, 0
            loop = tqdm(train_loader, desc=f"FineTune Epoch {epoch+1}/{FINETUNE_EPOCHS}", unit="batch")
            for imgs, labels in loop:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer_ft.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix(loss=loss.item(), acc=correct/total)

        torch.save(model.state_dict(), "Codigo/model.pth")

        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        model.eval()
        total, correct = 0, 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return float(1 - correct / total), len(val_loader.dataset), {}

    def final_evaluation(self):
        """ Avaliação final com barra de progresso e matriz de confusão """
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            loop = tqdm(test_loader, desc="Final Evaluating", unit="batch")
            for imgs, labels in loop:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == "__main__":
    client = CnnClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    client.final_evaluation()
