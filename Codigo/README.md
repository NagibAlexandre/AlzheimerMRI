# Código do Sistema de Detecção de Alzheimer

Este diretório contém todo o código-fonte do sistema de classificação de imagens de ressonância magnética para detecção de Alzheimer.

## Estrutura de Diretórios

```
Codigo/
├── dataset/
│   ├── OriginalDataset/              # Dataset original organizado por classe
│   │   ├── NonDemented/
│   │   ├── VeryMildDemented/
│   │   ├── MildDemented/
│   │   └── ModerateDemented/
│   ├── AugmentedAlzheimerDataset/   # Dataset com dados aumentados
│   │   ├── NonDemented/
│   │   ├── VeryMildDemented/
│   │   ├── MildDemented/
│   │   └── ModerateDemented/
│   └── Processed/                   # Dataset processado (gerado por TestTrain.py)
│       ├── train/
│       ├── val/
│       └── test/
├── flower/                          # Implementação de aprendizado federado
│   ├── client.py                    # Cliente Flower para treinamento distribuído
│   ├── server.py                    # Servidor Flower
│   └── model.py                     # Definição do modelo ResNet18
├── inteface/                        # Interface gráfica
│   └── interface.py                 # Interface PyQt5 para classificação de imagens
├── CNN.py                           # Script principal de treinamento do modelo
├── TestTrain.py                     # Script para preparar e dividir o dataset
├── infer.py                         # Script de inferência e otimização de performance
├── model.pth                        # Modelo treinado (gerado após CNN.py)
└── README.md                        # Este arquivo
```

## Scripts Principais

### TestTrain.py
Prepara o dataset dividindo as imagens dos diretórios `OriginalDataset` e `AugmentedAlzheimerDataset` em conjuntos de treino (70%), validação (15%) e teste (15%).

**Uso:**
```bash
python TestTrain.py
```

**Requisitos:**
- Dataset organizado em `dataset/OriginalDataset/` e `dataset/AugmentedAlzheimerDataset/`
- Cada diretório deve conter subpastas com os nomes das classes

### CNN.py
Script principal para treinamento do modelo ResNet18. Implementa transfer learning com duas fases:
1. **Warmup** (5 épocas): Treina apenas a camada final
2. **Fine-tuning** (5 épocas): Descongela e ajusta layer3, layer4 e fc

**Uso:**
```bash
python CNN.py
```

**Saída:**
- `model.pth`: Modelo treinado
- Matriz de confusão exibida no final

**Parâmetros configuráveis:**
- `IMG_SIZE = 224`: Tamanho das imagens
- `BATCH = 32`: Tamanho do batch
- `WARMUP_EPOCHS = 5`: Épocas de warmup
- `FINETUNE_EPOCHS = 5`: Épocas de fine-tuning

### infer.py
Script para inferência e otimização de performance. Realiza grid search testando diferentes configurações de batch size, workers e threads para encontrar a melhor configuração de throughput.

**Uso:**
```bash
python infer.py
```

**Saída (em `results/`):**
- `grid_search_results.csv`: Resultados em formato CSV
- `grid_search_results.json`: Resultados em formato JSON
- `throughput_analysis.png`: Gráficos de análise
- `heatmap_throughput.png`: Heatmaps de throughput
- `best_confusion_matrix.png`: Matriz de confusão da melhor configuração
- `best_config_metrics.txt`: Métricas detalhadas

### interface.py
Interface gráfica desenvolvida em PyQt5 para classificação de imagens. Suporta formatos JPG, PNG e NIFTI.

**Uso:**
```bash
cd inteface
python interface.py
```

**Funcionalidades:**
- Carregamento de imagens (JPG, PNG, NIFTI)
- Visualização com zoom
- Classificação usando modelo treinado
- Exibição de probabilidades por classe
- Interface acessível com ajuste de tamanho de fonte

## Aprendizado Federado (Flower)

### server.py
Servidor Flower para coordenação de treinamento federado.

**Uso:**
```bash
cd flower
python server.py
```

**Configuração:**
- Porta: `8080`
- Mínimo de clientes: `3`
- Rodadas: `1`

### client.py
Cliente Flower que participa do treinamento federado.

**Uso:**
```bash
cd flower
python client.py
```

**Requisitos:**
- Servidor rodando em `localhost:8080`
- Dataset processado disponível

## Modelo

O modelo utiliza ResNet18 pré-treinada no ImageNet com as seguintes características:
- **Arquitetura**: ResNet18
- **Input**: Imagens 224x224x3
- **Output**: 4 classes (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
- **Transfer Learning**: Camadas iniciais congeladas, fine-tuning nas camadas finais

## Dependências

Principais bibliotecas utilizadas:
- `torch`, `torchvision`: Deep learning
- `PyQt5`: Interface gráfica
- `scikit-learn`: Métricas de avaliação
- `matplotlib`, `seaborn`: Visualização
- `nibabel`: Processamento NIFTI
- `opencv-python`: Processamento de imagens
- `flwr`: Aprendizado federado
- `tqdm`: Barras de progresso
- `pandas`: Análise de dados

## Notas

- O modelo treinado (`model.pth`) deve estar presente para usar a interface ou scripts de inferência
- Para melhor performance, recomenda-se uso de GPU (CUDA)
- O dataset deve ser preparado com `TestTrain.py` antes do treinamento
