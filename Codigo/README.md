# Sistema de Classificação de Níveis de Alzheimer

Este projeto implementa um classificador de níveis de Alzheimer utilizando processamento de imagens de ressonância magnética e machine learning com descritores de textura GLCM.

## 📋 Descrição

O sistema classifica imagens em 4 categorias:
- **NonDemented**: Sem demência
- **VeryMildDemented**: Demência muito leve
- **MildDemented**: Demência leve
- **ModerateDemented**: Demência moderada

## 🔧 Tecnologias

- **PIL**: Processamento de imagens
- **Scikit-image**: Descritores GLCM e segmentação
- **Scikit-learn**: Classificação SVM
- **NumPy/Matplotlib**: Manipulação de dados e visualização

## 📁 Estrutura

```
TI6/
├── OriginalDataset/           # Dataset organizado por classe
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
├── alzheimer_classifier.py   # Classificador principal
├── test_alzheimer_classification.py # Script de teste
└── README.md
```

## 🚀 Uso

1. **Instalar dependências:**
```bash
pip install pillow scikit-image scikit-learn numpy matplotlib
```

2. **Executar:**
```bash
python test_alzheimer_classification.py
```

## 🔬 Metodologia

### Pré-processamento
- Conversão para escala de cinza
- Filtros de nitidez, contraste e brilho
- **Segmentação Otsu** (sempre ativa)
- Quantização para 32 tons de cinza

### Extração de Características
- **GLCM** (Gray-Level Co-occurrence Matrix)
- Descritores: Energia, Homogeneidade, Correlação, Dissimilaridade
- Entropia de Shannon
- Múltiplas distâncias e ângulos

### Classificação
- **SVM** com kernel linear
- Separação treino/teste: 75%/25%
- Classes balanceadas automaticamente

## 📊 Saída

- Acurácia e especificidade do modelo
- Matriz de confusão visual (`matriz-confusao.png`)
- Métricas de tempo de execução