import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODEL_PATH = "model.pth"
TEST_FOLDER = "dataset/Processed/test"
BATCH_SIZE = 32

NUM_STREAMS = 4

#--------- TRANSFORMAÇÕES ---------
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    """Carrega o modelo treinado"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    return model

def load_complete_dataset():
    print("Carregando dataset completo na memória...")
    
    dataset = datasets.ImageFolder(TEST_FOLDER, transform=transform_test)
    total_images = len(dataset)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Carregando batches", total=len(loader)):
            all_images.append(images)
            all_labels.append(labels)
    
    all_images_tensor = torch.cat(all_images)
    all_labels_tensor = torch.cat(all_labels)
    
    print(f"Dataset carregado: {len(all_images_tensor)} imagens")
    return all_images_tensor, all_labels_tensor, total_images

def inferir():
    print(f"=== INFERÊNCIA COM {NUM_STREAMS} CUDA STREAMS ===\n")
    print(f"Dispositivo: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    if not torch.cuda.is_available():
        print("\nAVISO: CUDA não disponível. Executando em CPU sem streams.\n")
    
    print("Carregando modelo...")
    model = load_model(MODEL_PATH, DEVICE)
    
    print("Carregando dataset...")
    all_images, all_labels, total_images = load_complete_dataset()
    
    streams = []
    if torch.cuda.is_available():
        streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
        print(f"\n{NUM_STREAMS} CUDA Streams criados")
    
    all_preds = []
    all_probs = []
    
    num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\nDistribuição de batches:")
    print(f"   Total de batches: {num_batches}")
    print(f"   Batches por stream: ~{num_batches // NUM_STREAMS}")
    print(f"   Imagens por batch: {BATCH_SIZE}")
    
    print(f"\nIniciando inferência com streams...")
    start_time = time.time()
    
    with torch.no_grad():
        batch_idx = 0
        
        pbar = tqdm(total=num_batches, desc="Processando batches")
        
        while batch_idx < num_batches:
            batch_results = []
            
            for stream_id in range(NUM_STREAMS):
                current_batch = batch_idx + stream_id
                
                if current_batch >= num_batches:
                    break
                
                start_idx = current_batch * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, total_images)
                
                if torch.cuda.is_available() and streams:
                    with torch.cuda.stream(streams[stream_id]):
                        batch_images = all_images[start_idx:end_idx].to(
                            DEVICE, 
                            non_blocking=True
                        )
                        
                        outputs = model(batch_images)
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predictions = outputs.max(1)
                        
                        preds_cpu = predictions.cpu()
                        probs_cpu = probabilities.cpu()
                        
                        batch_results.append({
                            'preds': preds_cpu,
                            'probs': probs_cpu
                        })
                else:
                    batch_images = all_images[start_idx:end_idx].to(DEVICE)
                    outputs = model(batch_images)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = outputs.max(1)
                    
                    batch_results.append({
                        'preds': predictions.cpu(),
                        'probs': probabilities.cpu()
                    })
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            for result in batch_results:
                all_preds.append(result['preds'])
                all_probs.append(result['probs'])
            
            batch_idx += NUM_STREAMS
            pbar.update(min(NUM_STREAMS, num_batches - batch_idx + NUM_STREAMS))
        
        pbar.close()
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels_combined = all_labels.numpy()
    
    print(f"\n{'='*60}")
    print(f"INFERÊNCIA CONCLUÍDA")
    print(f"Tempo total: {inference_time:.2f} segundos")
    print(f"Throughput: {total_images/inference_time:.2f} imagens/segundo")
    print(f"{'='*60}\n")
    
    return total_images, inference_time, all_preds, all_labels_combined, all_probs

def calcular_metricas(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    
    specificity = np.mean(tn / (tn + fp + 1e-10))
    fpr = np.mean(fp / (fp + tn + 1e-10))
    fnr = np.mean(fn / (fn + tp + 1e-10))
    
    return cm, acc, precision, recall, f1, specificity, fpr, fnr

def salvar_graficos(cm, all_labels, all_preds):
    os.makedirs("Codigo", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Quantidade'})
    plt.xlabel("Predição", fontsize=12, fontweight='bold')
    plt.ylabel("Real", fontsize=12, fontweight='bold')
    plt.title(f"Matriz de Confusão - {NUM_STREAMS} CUDA Streams", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Codigo/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Proporção'})
    plt.xlabel("Predição", fontsize=12, fontweight='bold')
    plt.ylabel("Real", fontsize=12, fontweight='bold')
    plt.title("Matriz de Confusão Normalizada", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Codigo/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    metrics_per_class = []
    for i in range(len(CLASSES)):
        prec = precision_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        f1_c = f1_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        metrics_per_class.append([prec, rec, f1_c])
    
    metrics_per_class = np.array(metrics_per_class)
    x = np.arange(len(CLASSES))
    width = 0.25
    
    plt.bar(x - width, metrics_per_class[:, 0], width, label='Precisão', color='steelblue', alpha=0.8)
    plt.bar(x, metrics_per_class[:, 1], width, label='Recall', color='darkorange', alpha=0.8)
    plt.bar(x + width, metrics_per_class[:, 2], width, label='F1-Score', color='green', alpha=0.8)
    
    plt.xlabel('Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Métricas por Classe', fontsize=14, fontweight='bold')
    plt.xticks(x, CLASSES, rotation=15, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Codigo/metrics_per_class.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficos salvos na pasta 'Codigo/'")

def imprimir_relatorios(all_labels, all_preds, cm, acc, precision, recall, f1, specificity, fpr, fnr):    
    print(f"{'='*60}")
    print("RELATÓRIO DE MÉTRICAS")
    print(f"{'='*60}")
    
    print(f"\nMÉTRICAS GERAIS:")
    print(f"Acurácia (Accuracy):           {acc*100:.2f}%")
    print(f"Precisão (Precision):          {precision*100:.2f}%")
    print(f"Recall (Sensibilidade):        {recall*100:.2f}%")
    print(f"F1-Score:                      {f1*100:.2f}%")
    print(f"Especificidade:                {specificity*100:.2f}%")
    print(f"Taxa Falsos Positivos (FPR):   {fpr*100:.2f}%")
    print(f"Taxa Falsos Negativos (FNR):   {fnr*100:.2f}%")
    
    print(f"\nMÉTRICAS POR CLASSE:")
    print("-" * 50)
    for i, class_name in enumerate(CLASSES):
        class_precision = precision_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        class_recall = recall_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        
        print(f"{class_name:15} | Precisão: {class_precision*100:6.2f}% | Recall: {class_recall*100:6.2f}% | F1: {class_f1*100:6.2f}%")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT (sklearn)")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

if __name__ == '__main__':
    total_images, inference_time, all_preds, all_labels, all_probs = inferir()
    
    cm, acc, precision, recall, f1, specificity, fpr, fnr = calcular_metricas(all_labels, all_preds)
    
    imprimir_relatorios(all_labels, all_preds, cm, acc, precision, recall, f1, specificity, fpr, fnr)
    
    salvar_graficos(cm, all_labels, all_preds)
    
    correct_predictions = (all_preds == all_labels).sum()
    incorrect_predictions = len(all_labels) - correct_predictions
    
    print(f"\n{'='*60}")
    print("ESTATÍSTICAS FINAIS")
    print(f"{'='*60}")
    print(f"Total de imagens:        {total_images}")
    print(f"Predições corretas:      {correct_predictions} ({acc*100:.2f}%)")
    print(f"Predições incorretas:    {incorrect_predictions} ({(1-acc)*100:.2f}%)")
    print(f"Tempo total:             {inference_time:.2f}s")
    print(f"Throughput:              {total_images/inference_time:.2f} imgs/s")
    print(f"CUDA Streams utilizados: {NUM_STREAMS}")
    print(f"Batch size:              {BATCH_SIZE}")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        print(f"GPU utilizada:            {torch.cuda.get_device_name()}")
        print(f"Memória GPU alocada:      {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"{'='*60}")