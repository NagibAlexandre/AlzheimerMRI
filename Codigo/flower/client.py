import os
import sys
import time
import json
import argparse
from pathlib import Path
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib
if "--run" in sys.argv:
    matplotlib.use('Agg')

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
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "dataset" / "Processed"
RESULTS_DIR = BASE_DIR / "resultados"
MODEL_PATH = BASE_DIR / "model.pth"
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
train_ds = datasets.ImageFolder(os.path.join(DATA_PROCESSED, "train"), transform_train)
val_ds   = datasets.ImageFolder(os.path.join(DATA_PROCESSED, "val"), transform_test)
test_ds  = datasets.ImageFolder(os.path.join(DATA_PROCESSED, "test"), transform_test)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=3)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=3)
test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=3)

# --- Model ---
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
        start_time = time.time()  # execution timer starts here
        self.set_parameters(parameters)
        history_acc = []

        # --- Phase 1: Train only FC ---
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

        # --- Phase 2: Fine-tuning layer3+layer4 ---
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

        print("\n--- LOCAL REPORT (Client) ---")
        print(f"Total execution time: {exec_time:.2f} seconds")
        print(f"Average training accuracy: {avg_train_acc*100:.2f}%")

        torch.save(model.state_dict(), MODEL_PATH)
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

    def final_evaluation(self, run_id=None, client_id=None, show_plots=True):
        """ Final evaluation with progress bar and computed metrics.

        Returns a dictionary with all computed metrics.
        """
        print("\n[CLIENT] Running final evaluation...")
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

        # --- METRICS ---
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Calculating specificity, FPR, and FNR from confusion matrix
        tn = np.diag(cm).sum() - np.diag(cm)  # TN by class
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        specificity = np.mean(tn / (tn + fp + 1e-10))
        fpr = np.mean(fp / (fp + tn + 1e-10))
        fnr = np.mean(fn / (fn + tp + 1e-10))

        metrics = {
            "accuracy":    float(acc),
            "precision":   float(precision),
            "recall":      float(recall),
            "f1":          float(f1),
            "specificity": float(specificity),
            "fpr":         float(fpr),
            "fnr":         float(fnr),
        }

        # --- Report ---
        print("\n==============================")
        print("--- FINAL REPORT ---")
        print("==============================")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall / Sensitivity: {recall*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"Specificity: {specificity*100:.2f}%")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
        print(f"False Negative Rate (FNR): {fnr*100:.2f}%")

        # --- Save Confusion Matrix ---
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cm_suffix = f"_run{run_id}_client{client_id}" if run_id is not None else ""
        cm_path = RESULTS_DIR / f"confusion_matrix{cm_suffix}.png"

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {cm_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

        # --- Save metrics to JSON when running in automated mode ---
        if run_id is not None:
            metrics_path = RESULTS_DIR / f"metricas_cliente_{client_id}_run_{run_id}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"✓ Metrics saved to: {metrics_path}")

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client for Federated Learning")
    parser.add_argument("--run", type=int, default=None,
                        help="Run ID (used to save numbered metrics)")
    parser.add_argument("--client-id", type=int, default=0,
                        help="Client ID (0, 1, 2, ...)")
    args = parser.parse_args()

    print(f"\n[CLIENT {args.client_id}] Starting Flower client...")
    client = CnnClient()
    show = args.run is None  # Mostrar plots apenas em modo interativo
    try:
        fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    except Exception as e:
        print(f"\n[WARNING] Federated Learning stopped: {e}")
    finally:
        print("\n[CLIENT] Federated Learning finished. Generating final evaluation...")
        client.final_evaluation(run_id=args.run, client_id=args.client_id, show_plots=show)
        print("\n[CLIENT] Process completed!")
