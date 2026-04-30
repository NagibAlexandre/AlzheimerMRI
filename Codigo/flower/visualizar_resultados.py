"""
Script para visualizar os resultados do Federated Learning
Lê os arquivos salvos durante o treinamento e exibe os gráficos e métricas
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RESULTADOS_DIR = "../resultados"

def carregar_metricas():
    """Carrega as métricas do JSON"""
    metrics_path = os.path.join(RESULTADOS_DIR, "metricas_fedlearning.json")
    
    if not os.path.exists(metrics_path):
        print(f"❌ Arquivo de métricas não encontrado: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def exibir_metricas(metrics):
    """Exibe as métricas em formato de texto"""
    print("\n" + "="*60)
    print("RELATÓRIO FINAL - FEDERATED LEARNING")
    print("="*60)
    print(f"Tempo total de treinamento: {metrics['tempo_total']:.2f} segundos")
    print(f"Tempo médio por rodada: {metrics['tempo_medio_rodada']:.2f} segundos")
    print(f"Acurácia Global Média: {metrics['acuracia_global_media']*100:.2f}%")
    print(f"Acurácia Local Média (clientes): {metrics['acuracia_local_media']*100:.2f}%")
    print(f"Loss Global Médio: {metrics['loss_global_medio']:.4f}")
    print("="*60 + "\n")

def exibir_imagens():
    """Exibe as imagens dos gráficos e matriz de confusão salvas"""
    imagens = [
        ("metricas_graficos.png", "Gráficos de Desempenho do Federated Learning"),
        ("confusion_matrix.png", "Matriz de Confusão - Avaliação Final")
    ]
    
    for nome_arquivo, titulo in imagens:
        caminho = os.path.join(RESULTADOS_DIR, nome_arquivo)
        if os.path.exists(caminho):
            print(f"✓ Abrindo: {titulo}")
            img = Image.open(caminho)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(titulo)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠ Arquivo não encontrado: {caminho}")

def main():
    """Função principal"""
    if not os.path.exists(RESULTADOS_DIR):
        print(f"❌ Diretório de resultados não encontrado: {RESULTADOS_DIR}")
        return
    
    print(f"📊 Carregando resultados de: {RESULTADOS_DIR}\n")
    
    # Carregar e exibir métricas
    metrics = carregar_metricas()
    if metrics:
        exibir_metricas(metrics)
    
    # Exibir imagens
    exibir_imagens()

if __name__ == "__main__":
    main()
