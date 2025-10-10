# CNN

Importe a dataset para esta pasta considerando a seguinte estrutura,

```
Codigo/
├── dataset
|     ├─OriginalDataset/           # Dataset organizado por classe
|     |    ├── NonDemented/
|     |    ├── VeryMildDemented/
|     |    ├── MildDemented/
|     |    └── ModerateDemented/
|     └─AugmentedAlzheimerDataset/ 
├── alzheimer_classifier.py   # Classificador principal
├── test_alzheimer_classification.py # Script de teste
└── README.md
```

Rode o TestTrain.py e em seguida o CNN.py