import os
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import flwr as fl

# --- Config ---
DATA_PROCESSED = "Codigo/dataset/Processed/"
IMG_SIZE = 224
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=3)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=3)

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
        start_time = time.time()  # tempo de execução começa aqui
        self.set_parameters(parameters)
        history_acc = []

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
            history_acc.append(correct/total)

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
            history_acc.append(correct/total)

        end_time = time.time()
        exec_time = end_time - start_time
        avg_train_acc = np.mean(history_acc)

        print("\n--- RELATÓRIO LOCAL (Cliente) ---")
        print(f"Tempo de execução total: {exec_time:.2f} segundos")
        print(f"Acurácia média de treino: {avg_train_acc*100:.2f}%")

        torch.save(model.state_dict(), "Codigo/model.pth")
        return self.get_parameters(), len(train_loader.dataset), {"train_acc": avg_train_acc, "exec_time": exec_time}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        model.eval()
        total, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(val_loader)
        acc = correct / total
        print(f"[Val] Loss(distributed): {avg_loss:.4f} | Accuracy(distributed): {acc*100:.2f}%")
        return float(1 - acc), len(val_loader.dataset), {"loss": avg_loss, "acc": acc}

    def final_evaluation(self):
        """ Avaliação final com barra de progresso e métricas """
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

        # --- MÉTRICAS ---
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Especificidade e taxas
        tn = np.diag(cm).sum() - np.diag(cm)  # TN por classe
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        specificity = np.mean(tn / (tn + fp + 1e-10))
        fpr = np.mean(fp / (fp + tn + 1e-10))
        fnr = np.mean(fn / (fn + tp + 1e-10))

        # --- Relatório ---
        print("\n--- RELATÓRIO FINAL ---")
        print(f"Acurácia (Accuracy): {acc*100:.2f}%")
        print(f"Precisão (Precision): {precision*100:.2f}%")
        print(f"Recall / Sensibilidade: {recall*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"Especificidade: {specificity*100:.2f}%")
        print(f"Taxa de Falsos Positivos (FPR): {fpr*100:.2f}%")
        print(f"Taxa de Falsos Negativos (FNR): {fnr*100:.2f}%")

        # --- Matriz de Confusão ---
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
