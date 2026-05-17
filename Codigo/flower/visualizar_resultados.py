"""
Script to visualize Federated Learning results
Reads saved files from training and displays plots and metrics
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RESULTADOS_DIR = "../resultados"

def carregar_metricas():
    """Load metrics from the JSON file"""
    metrics_path = os.path.join(RESULTADOS_DIR, "metricas_fedlearning.json")
    
    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def exibir_metricas(metrics):
    """Display metrics as text"""
    print("\n" + "="*60)
    print("FINAL REPORT - FEDERATED LEARNING")
    print("="*60)
    print(f"Total training time: {metrics['tempo_total']:.2f} seconds")
    print(f"Mean time per round: {metrics['tempo_medio_rodada']:.2f} seconds")
    print(f"Mean Global Accuracy: {metrics['acuracia_global_media']*100:.2f}%")
    print(f"Mean Local Accuracy (clients): {metrics['acuracia_local_media']*100:.2f}%")
    print(f"Mean Global Loss: {metrics['loss_global_medio']:.4f}")
    print("="*60 + "\n")

def exibir_imagens():
    """Display the saved plots and confusion matrix images"""
    imagens = [
        ("metricas_graficos.png", "Federated Learning Performance Plots"),
        ("confusion_matrix.png", "Confusion Matrix - Final Evaluation")
    ]
    
    for nome_arquivo, titulo in imagens:
        caminho = os.path.join(RESULTADOS_DIR, nome_arquivo)
        if os.path.exists(caminho):
            print(f"✓ Opening: {titulo}")
            img = Image.open(caminho)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(titulo)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠ File not found: {caminho}")

def main():
    """Main function"""
    if not os.path.exists(RESULTADOS_DIR):
        print(f"❌ Results directory not found: {RESULTADOS_DIR}")
        return

    print(f"📊 Loading results from: {RESULTADOS_DIR}\n")

    # Load and display metrics
    metrics = carregar_metricas()
    if metrics:
        exibir_metricas(metrics)

    # Display images
    exibir_imagens()

if __name__ == "__main__":
    main()
