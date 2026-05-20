## A Federated Learning Approach for Distributed Alzheimer’s Disease Diagnosis Using Brain MRI

This project implements a magnetic resonance imaging (MRI) classification system for Alzheimer's detection using convolutional neural networks (CNN). The system is capable of classifying images into four categories: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented.

The project uses a ResNet18 architecture with transfer learning, implements a graphical interface for image classification, and supports federated learning through the Flower framework. The system processes both JPG images and NIFTI files (a common format for medical images).

### Team Members

- João Vítor de Freitas Scarlatelli
- Nagib Alexandre Verly Borjaili
- Vitor Dias de Britto Militão
- Vitória Símil Araújo
- Yasmin Cassemiro Viegas

### Supervising Professors

- Henrique Cota de Freitas
- Felipe Domingos da Cunha

### PITCH

[https://www.youtube.com/watch?v=qCqW1E061YA](https://www.youtube.com/watch?v=qCqW1E061YA)

### Project Structure

```
├── Codigo/                    # Main source code
│   ├── CNN.py                 # Model training script
│   ├── TestTrain.py           # Script to prepare dataset (train/val/test)
│   ├── infer.py               # Inference and performance optimization script
│   ├── model.pth              # Trained model (generated after training)
│   ├── dataset/               # Datasets
│   │   ├── OriginalDataset/   # Original dataset organized by class
│   │   ├── AugmentedAlzheimerDataset/  # Dataset with augmented data
│   │   └── Processed/         # Processed dataset (train/val/test)
│   ├── flower/                # Federated learning implementation
│   │   ├── client.py          # Flower client
│   │   ├── server.py          # Flower server
│   │   └── model.py           # Model definition
│   └── inteface/              # Graphical interface
│       └── interface.py       # PyQt5 interface for classification
├── Artefatos/                 # Project artifacts
├── Documentacao/              # Additional documentation
└── Divulgacao/                # Dissemination materials
```

## Usage Instructions

#### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- Operating system: Windows, Linux, or macOS

#### Installing Dependencies

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

### Dataset Preparation

1. Organize the original dataset in the `Codigo/dataset/OriginalDataset/` folder with the following structure:
   ```
   OriginalDataset/
   ├── NonDemented/
   ├── VeryMildDemented/
   ├── MildDemented/
   └── ModerateDemented/
   ```

2. Run the dataset preparation script:
   ```bash
   cd Codigo
   python TestTrain.py
   ```
   This script will create the `Processed/train/`, `Processed/val/`, and `Processed/test/` structure with a 70/15/15 split.

### Model Training

Para treinar o modelo ResNet18:

```bash
cd Codigo
python CNN.py
```

The model will be trained in two phases:
1. **Warmup**: Trains only the final layer (5 epochs)
2. **Fine-tuning**: Unfreezes and adjusts the layer3, layer4, and fc layers (5 epochs)

The trained model will be saved as `model.pth`.

### Running the Graphical Interface

To use the graphical classification interface:

```bash
cd Codigo/inteface
python interface.py
```

The interface allows you to:
- Load images (JPG, PNG, NIFTI)
- View images with zoom
- Classify images using the trained model
- View probabilities for each class

### Inference and Performance Testing

To run inference tests and parameter optimization:

```bash
cd Codigo
python infer.py
```

This script performs a grid search testing different combinations of:
- Batch sizes: [2, 4, 8, 16, 32, 64, 128, 256]
- Num workers: [2, 4, 8]
- Num threads: [1, 2, 4, 8]


Results are saved in `Codigo/results/`, including:
- CSV and JSON with metrics
- Throughput analysis charts
- Confusion matrix of the best configuration

### Federated Learning (Flower)

To run federated training:

1. **Start the server** (in one terminal):
   ```bash
   cd Codigo/flower
   python server.py
   ```

2. **Start clients** (in separate terminals, one per client):
   ```bash
   cd Codigo/flower
   python client.py
   ```

The server waits for 3 clients by default. Each client trains locally and sends updates to the server.

### Technologies Used

- **PyTorch**: Deep learning framework
- **ResNet18**: Pre-trained neural network architecture
- **PyQt5**: Graphical interface
- **Flower**: Federated learning framework
- **scikit-learn**: Evaluation metrics
- **nibabel**: NIFTI file processing
- **OpenCV**: Image processing

## Classification Classes

The model classifies images into 4 categories:
- **NonDemented**: No dementia
- **VeryMildDemented**: Very mild dementia
- **MildDemented**: Mild dementia
- **ModerateDemented**: Moderate dementia
