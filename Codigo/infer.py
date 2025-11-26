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
import pandas as pd
import json

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODEL_PATH = "model.pth"
TEST_FOLDER = "dataset/Processed/test"

NUM_STREAMS = 4

# Parâmetros para iteração
BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
NUM_WORKERS_LIST = [2, 4, 8]
NUM_THREADS_LIST = [1, 2, 4, 8]

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

def load_complete_dataset(batch_size, num_workers):
    """Carrega dataset completo na memória"""
    print(f"Carregando dataset (batch={batch_size}, workers={num_workers})...")
    
    dataset = datasets.ImageFolder(TEST_FOLDER, transform=transform_test)
    total_images = len(dataset)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Carregando batches", total=len(loader), leave=False):
            all_images.append(images)
            all_labels.append(labels)
    
    all_images_tensor = torch.cat(all_images)
    all_labels_tensor = torch.cat(all_labels)
    
    print(f"Dataset carregado: {len(all_images_tensor)} imagens")
    return all_images_tensor, all_labels_tensor, total_images

def inferir(batch_size, num_workers, num_threads):
    """Realiza inferência com parâmetros específicos - VERSÃO COM STREAMS PARALELAS"""
    
    # Configurar número de threads
    torch.set_num_threads(num_threads)
    
    print(f"\n{'='*70}")
    print(f"CONFIGURAÇÃO: Batch={batch_size} | Workers={num_workers} | Threads={num_threads}")
    print(f"{'='*70}")
    
    if not torch.cuda.is_available():
        print("AVISO: CUDA não disponível. Executando em CPU.")
    
    # Carrega modelo
    model = load_model(MODEL_PATH, DEVICE)
    
    # Carrega dataset
    all_images, all_labels, total_images = load_complete_dataset(batch_size, num_workers)
    
    # Cria streams
    streams = []
    if torch.cuda.is_available():
        streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
    
    all_preds = []
    all_probs = []
    
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"Iniciando inferência com {NUM_STREAMS} streams paralelas...")
    start_time = time.time()
    
    with torch.no_grad():
        batch_idx = 0
        pbar = tqdm(total=num_batches, desc="Processando", leave=False)
        
        while batch_idx < num_batches:
            batch_outputs = []  # Armazena outputs na GPU
            
            # FASE 1: AGENDAR OPERAÇÕES EM PARALELO (não bloqueia)
            for stream_id in range(NUM_STREAMS):
                current_batch = batch_idx + stream_id
                
                if current_batch >= num_batches:
                    break
                
                start_idx = current_batch * batch_size
                end_idx = min(start_idx + batch_size, total_images)
                
                if torch.cuda.is_available() and streams:
                    # Cada stream executa INDEPENDENTEMENTE
                    with torch.cuda.stream(streams[stream_id]):
                        batch_images = all_images[start_idx:end_idx].to(
                            DEVICE, 
                            non_blocking=True  # Crucial para paralelismo!
                        )
                        
                        outputs = model(batch_images)
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predictions = outputs.max(1)
                        
                        # Mantém na GPU - NÃO transfere para CPU ainda
                        batch_outputs.append({
                            'preds': predictions,
                            'probs': probabilities,
                            'stream': streams[stream_id]
                        })
                else:
                    # Fallback CPU
                    batch_images = all_images[start_idx:end_idx].to(DEVICE)
                    outputs = model(batch_images)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = outputs.max(1)
                    
                    batch_outputs.append({
                        'preds': predictions,
                        'probs': probabilities,
                        'stream': None
                    })
            
            # FASE 2: SINCRONIZAR (aguarda TODAS as streams terminarem)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # FASE 3: TRANSFERIR RESULTADOS PARA CPU
            for output_dict in batch_outputs:
                all_preds.append(output_dict['preds'].cpu())
                all_probs.append(output_dict['probs'].cpu())
            
            batch_idx += NUM_STREAMS
            pbar.update(min(NUM_STREAMS, num_batches - batch_idx + NUM_STREAMS))
        
        pbar.close()
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels_combined = all_labels.numpy()
    
    # Calcula métricas
    acc = accuracy_score(all_labels_combined, all_preds)
    
    print(f"Tempo: {inference_time:.2f}s | Throughput: {total_images/inference_time:.2f} imgs/s | Acc: {acc*100:.2f}%")
    
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_threads': num_threads,
        'total_images': total_images,
        'inference_time': inference_time,
        'throughput': total_images/inference_time,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels_combined,
        'probabilities': all_probs
    }

def calcular_metricas(all_labels, all_preds):
    """Calcula todas as métricas de avaliação"""
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

def salvar_resultados(results_list):
    """Salva resultados em CSV e JSON"""
    os.makedirs("Codigo/results", exist_ok=True)
    
    # Cria DataFrame com resultados
    df_results = pd.DataFrame([
        {
            'batch_size': r['batch_size'],
            'num_workers': r['num_workers'],
            'num_threads': r['num_threads'],
            'inference_time': r['inference_time'],
            'throughput': r['throughput'],
            'accuracy': r['accuracy']
        }
        for r in results_list
    ])
    
    # Salva CSV
    df_results.to_csv('Codigo/results/grid_search_results.csv', index=False)
    print("\nResultados salvos em: Codigo/results/grid_search_results.csv")
    
    # Salva JSON
    with open('Codigo/results/grid_search_results.json', 'w') as f:
        json.dump([
            {k: v for k, v in r.items() if k not in ['predictions', 'labels', 'probabilities']}
            for r in results_list
        ], f, indent=2)
    
    return df_results

def plotar_analises(df_results):
    """Cria gráficos de análise dos resultados"""
    os.makedirs("Codigo/results", exist_ok=True)
    
    # 1. Throughput por Batch Size (agrupado por workers e threads)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for num_workers in NUM_WORKERS_LIST:
        for num_threads in NUM_THREADS_LIST:
            data = df_results[
                (df_results['num_workers'] == num_workers) & 
                (df_results['num_threads'] == num_threads)
            ]
            axes[0].plot(data['batch_size'], data['throughput'], 
                        marker='o', label=f'W={num_workers}, T={num_threads}')
    
    axes[0].set_xlabel('Batch Size', fontweight='bold')
    axes[0].set_ylabel('Throughput (imgs/s)', fontweight='bold')
    axes[0].set_title('Throughput vs Batch Size', fontweight='bold')
    axes[0].set_xscale('log', base=2)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Tempo de Inferência por Batch Size
    for num_workers in NUM_WORKERS_LIST:
        for num_threads in NUM_THREADS_LIST:
            data = df_results[
                (df_results['num_workers'] == num_workers) & 
                (df_results['num_threads'] == num_threads)
            ]
            axes[1].plot(data['batch_size'], data['inference_time'], 
                        marker='o', label=f'W={num_workers}, T={num_threads}')
    
    axes[1].set_xlabel('Batch Size', fontweight='bold')
    axes[1].set_ylabel('Tempo de Inferência (s)', fontweight='bold')
    axes[1].set_title('Tempo vs Batch Size', fontweight='bold')
    axes[1].set_xscale('log', base=2)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Codigo/results/throughput_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap de Throughput
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, num_threads in enumerate(NUM_THREADS_LIST):
        pivot = df_results[df_results['num_threads'] == num_threads].pivot(
            index='num_workers', 
            columns='batch_size', 
            values='throughput'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[idx], cbar_kws={'label': 'Throughput (imgs/s)'})
        axes[idx].set_title(f'Throughput - {num_threads} Thread(s)', fontweight='bold')
        axes[idx].set_xlabel('Batch Size', fontweight='bold')
        axes[idx].set_ylabel('Num Workers', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Codigo/results/heatmap_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficos de análise salvos em: Codigo/results/")

def encontrar_melhor_configuracao(df_results):
    """Encontra e exibe a melhor configuração"""
    print(f"\n{'='*70}")
    print("MELHORES CONFIGURAÇÕES")
    print(f"{'='*70}")
    
    # Melhor throughput
    best_throughput = df_results.loc[df_results['throughput'].idxmax()]
    print(f"\nMelhor Throughput: {best_throughput['throughput']:.2f} imgs/s")
    print(f"  Batch Size: {int(best_throughput['batch_size'])}")
    print(f"  Num Workers: {int(best_throughput['num_workers'])}")
    print(f"  Num Threads: {int(best_throughput['num_threads'])}")
    print(f"  Tempo: {best_throughput['inference_time']:.2f}s")
    
    # Menor tempo
    best_time = df_results.loc[df_results['inference_time'].idxmin()]
    print(f"\nMenor Tempo: {best_time['inference_time']:.2f}s")
    print(f"  Batch Size: {int(best_time['batch_size'])}")
    print(f"  Num Workers: {int(best_time['num_workers'])}")
    print(f"  Num Threads: {int(best_time['num_threads'])}")
    print(f"  Throughput: {best_time['throughput']:.2f} imgs/s")
    
    print(f"\n{'='*70}")

def salvar_metricas_melhor_config(best_result):
    """Salva métricas detalhadas da melhor configuração"""
    os.makedirs("Codigo/results", exist_ok=True)
    
    all_labels = best_result['labels']
    all_preds = best_result['predictions']
    
    cm, acc, precision, recall, f1, specificity, fpr, fnr = calcular_metricas(all_labels, all_preds)
    
    # Salva matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Quantidade'})
    plt.xlabel("Predição", fontsize=12, fontweight='bold')
    plt.ylabel("Real", fontsize=12, fontweight='bold')
    plt.title(f"Matriz de Confusão - Melhor Configuração\nBatch={best_result['batch_size']}, Workers={best_result['num_workers']}, Threads={best_result['num_threads']}", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Codigo/results/best_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salva relatório
    with open('Codigo/results/best_config_metrics.txt', 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"MÉTRICAS DA MELHOR CONFIGURAÇÃO\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuração:\n")
        f.write(f"  Batch Size: {best_result['batch_size']}\n")
        f.write(f"  Num Workers: {best_result['num_workers']}\n")
        f.write(f"  Num Threads: {best_result['num_threads']}\n")
        f.write(f"  Tempo: {best_result['inference_time']:.2f}s\n")
        f.write(f"  Throughput: {best_result['throughput']:.2f} imgs/s\n\n")
        f.write(f"Métricas:\n")
        f.write(f"  Acurácia: {acc*100:.2f}%\n")
        f.write(f"  Precisão: {precision*100:.2f}%\n")
        f.write(f"  Recall: {recall*100:.2f}%\n")
        f.write(f"  F1-Score: {f1*100:.2f}%\n")
        f.write(f"  Especificidade: {specificity*100:.2f}%\n")
        f.write(f"\n{classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)}\n")
    
    print("\nMétricas detalhadas salvas em: Codigo/results/best_config_metrics.txt")

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"GRID SEARCH - OTIMIZAÇÃO DE PARÂMETROS (STREAMS PARALELAS)")
    print(f"{'='*70}")
    print(f"Batch Sizes: {BATCH_SIZES}")
    print(f"Num Workers: {NUM_WORKERS_LIST}")
    print(f"Num Threads: {NUM_THREADS_LIST}")
    print(f"Num Streams: {NUM_STREAMS}")
    print(f"Total de combinações: {len(BATCH_SIZES) * len(NUM_WORKERS_LIST) * len(NUM_THREADS_LIST)}")
    print(f"{'='*70}\n")
    
    results_list = []
    total_combinations = len(BATCH_SIZES) * len(NUM_WORKERS_LIST) * len(NUM_THREADS_LIST)
    current = 0
    
    for num_workers in NUM_WORKERS_LIST:
        for num_threads in NUM_THREADS_LIST:
            for batch_size in BATCH_SIZES:
                current += 1
                print(f"\n[{current}/{total_combinations}] Testando configuração...")
                
                try:
                    result = inferir(batch_size, num_workers, num_threads)
                    results_list.append(result)
                except Exception as e:
                    print(f"ERRO na configuração (batch={batch_size}, workers={num_workers}, threads={num_threads}): {e}")
                    continue
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH CONCLUÍDO - {len(results_list)} configurações testadas")
    print(f"{'='*70}\n")
    
    # Salva e analisa resultados
    df_results = salvar_resultados(results_list)
    plotar_analises(df_results)
    encontrar_melhor_configuracao(df_results)
    
    # Salva métricas detalhadas da melhor configuração
    best_idx = df_results['throughput'].idxmax()
    best_result = results_list[best_idx]
    salvar_metricas_melhor_config(best_result)
    
    print(f"\n{'='*70}")
    print("ANÁLISE COMPLETA!")
    print(f"{'='*70}")
    print("Arquivos gerados:")
    print("  - Codigo/results/grid_search_results.csv")
    print("  - Codigo/results/grid_search_results.json")
    print("  - Codigo/results/throughput_analysis.png")
    print("  - Codigo/results/heatmap_throughput.png")
    print("  - Codigo/results/best_confusion_matrix.png")
    print("  - Codigo/results/best_config_metrics.txt")
    print(f"{'='*70}\n")