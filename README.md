# Sistema de Detecção de Alzheimer em Imagens de Ressonância Magnética

Este projeto implementa um sistema de classificação de imagens de ressonância magnética (MRI) para detecção de Alzheimer utilizando redes neurais convolucionais (CNN). O sistema é capaz de classificar imagens em quatro categorias: NonDemented (sem demência), VeryMildDemented (demência muito leve), MildDemented (demência leve) e ModerateDemented (demência moderada).

O projeto utiliza uma arquitetura ResNet18 com transfer learning, implementa uma interface gráfica para classificação de imagens e suporta aprendizado federado através do framework Flower. O sistema processa tanto imagens em formato JPG quanto arquivos NIFTI (formato comum para imagens médicas).

## Alunos integrantes da equipe

* João Vítor de Freitas Scarlatelli
* Nagib Alexandre Verly Borjaili
* Vitor Dias de Britto Militão
* Vitória Símil Araújo
* Yasmin Cassemiro Viegas

## Professores responsáveis

* Henrique Cota de Freitas
* João Paulo Coelho Furtado

## Estrutura do Projeto

```
.
├── Codigo/                    # Código-fonte principal
│   ├── CNN.py                 # Script de treinamento do modelo
│   ├── TestTrain.py           # Script para preparar dataset (train/val/test)
│   ├── infer.py               # Script de inferência e otimização de performance
│   ├── model.pth              # Modelo treinado (gerado após treinamento)
│   ├── dataset/               # Datasets
│   │   ├── OriginalDataset/   # Dataset original organizado por classe
│   │   ├── AugmentedAlzheimerDataset/  # Dataset com dados aumentados
│   │   └── Processed/         # Dataset processado (train/val/test)
│   ├── flower/                # Implementação de aprendizado federado
│   │   ├── client.py          # Cliente Flower
│   │   ├── server.py          # Servidor Flower
│   │   └── model.py           # Definição do modelo
│   └── inteface/              # Interface gráfica
│       └── interface.py       # Interface PyQt5 para classificação
├── Artefatos/                 # Artefatos do projeto
├── Documentacao/              # Documentação adicional
└── Divulgacao/                # Materiais de divulgação
```

## Instruções de utilização

### Pré-requisitos

- Python 3.8 ou superior
- CUDA (opcional, para aceleração GPU)
- Sistema operacional: Windows, Linux ou macOS

### Instalação de dependências

```bash
pip install torch torchvision
pip install PyQt5
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install numpy
pip install nibabel
pip install opencv-python
pip install flwr
pip install pandas
```

### Preparação do Dataset

1. Organize o dataset original na pasta `Codigo/dataset/OriginalDataset/` com a seguinte estrutura:
   ```
   OriginalDataset/
   ├── NonDemented/
   ├── VeryMildDemented/
   ├── MildDemented/
   └── ModerateDemented/
   ```

2. Execute o script de preparação do dataset:
   ```bash
   cd Codigo
   python TestTrain.py
   ```
   Este script criará a estrutura `Processed/train/`, `Processed/val/` e `Processed/test/` com divisão 70/15/15.

### Treinamento do Modelo

Para treinar o modelo ResNet18:

```bash
cd Codigo
python CNN.py
```

O modelo será treinado em duas fases:
1. **Warmup**: Treina apenas a camada final (5 épocas)
2. **Fine-tuning**: Descongela e ajusta as camadas layer3, layer4 e fc (5 épocas)

O modelo treinado será salvo como `model.pth`.

### Executar Interface Gráfica

Para usar a interface gráfica de classificação:

```bash
cd Codigo/inteface
python interface.py
```

A interface permite:
- Carregar imagens (JPG, PNG, NIFTI)
- Visualizar imagens com zoom
- Classificar imagens usando o modelo treinado
- Ver probabilidades para cada classe

### Inferência e Testes de Performance

Para executar testes de inferência e otimização de parâmetros:

```bash
cd Codigo
python infer.py
```

Este script realiza um grid search testando diferentes combinações de:
- Batch sizes: [2, 4, 8, 16, 32, 64, 128, 256]
- Num workers: [2, 4, 8]
- Num threads: [1, 2, 4, 8]

Os resultados são salvos em `Codigo/results/` incluindo:
- CSV e JSON com métricas
- Gráficos de análise de throughput
- Matriz de confusão da melhor configuração

### Aprendizado Federado (Flower)

Para executar treinamento federado:

1. **Iniciar o servidor** (em um terminal):
   ```bash
   cd Codigo/flower
   python server.py
   ```

2. **Iniciar clientes** (em terminais separados, um para cada cliente):
   ```bash
   cd Codigo/flower
   python client.py
   ```

O servidor aguarda 3 clientes por padrão. Cada cliente treina localmente e envia atualizações para o servidor.

## Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning
- **ResNet18**: Arquitetura de rede neural pré-treinada
- **PyQt5**: Interface gráfica
- **Flower**: Framework para aprendizado federado
- **scikit-learn**: Métricas de avaliação
- **nibabel**: Processamento de arquivos NIFTI
- **OpenCV**: Processamento de imagens

## Classes de Classificação

O modelo classifica imagens em 4 categorias:
- **NonDemented**: Sem demência
- **VeryMildDemented**: Demência muito leve
- **MildDemented**: Demência leve
- **ModerateDemented**: Demência moderada
