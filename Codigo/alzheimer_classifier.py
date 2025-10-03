# =============================================================================
# TI 6 - Classificador de Níveis de Alzheimer - Grupo 7
# =============================================================================

import os
import time
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics
from matplotlib import pyplot as plt


ALZHEIMER_CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

DEFAULT_CONFIG = {
    'kernel': 'linear',
    'C': 1.4,
    'class_weight': 'balanced'
}

DEFAULT_PREPROCESSING = {
    'n_colors': 32,
    'gaussian_radius': 0,
    'sharpness_boost': 1.4,
    'contrast_boost': 2.2,
    'brightness_boost': 1.1,
    'percentage_train': 0.75
}

DEFAULT_SEGMENTATION = {
    'min_object_size': 100,
    'gaussian_sigma': 1.0
}

# configurações padrão GLCM (Gray-Level Co-occurrence Matrix)
DEFAULT_GLCM = {
    'distances': [1, 2, 4, 8, 16],
    'angles': [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8],
    'descriptors': ["energy", "homogeneity", "correlation", "dissimilarity"]
}

class ImageClassifier:
    """
    Para a classificação vamos utilizar um SVM para classificar as imagens de acordo com o nível de Alzheimer.
    Para extrair características das imagens vamos utilizar descritores de textura GLCM (Gray-Level Co-occurrence Matrix).
    
    Atributos:
        _model: Modelo SVM treinado
        _model_kernel: Tipo de kernel do SVM
        _images_train: Conjunto de treino 
        _images_test: Conjunto de teste 
        _answers_train: Labels do conjunto de treino
        _answers_test: Labels do conjunto de teste
        _predictions: Predições do modelo no conjunto de teste
        _images_dir: Diretório contendo as imagens organizadas por classe
        _runtime_metrics: Dicionário com métricas de tempo de execução
    """

    # inicialização do classificador
    def __init__(self):
        """Inicializa o classificador com configurações padrão."""
        self._initialize_model()
        self._initialize_data_arrays()
        self._initialize_preprocessing_config()
        self._initialize_segmentation_config()
        self._initialize_glcm_config()
        self._initialize_metrics()
    
    # inicialização do modelo SVM
    def _initialize_model(self):
        self._model_kernel = DEFAULT_CONFIG['kernel']
        self._model = svm.SVC(
            kernel=self._model_kernel,
            C=DEFAULT_CONFIG['C'],
            class_weight=DEFAULT_CONFIG['class_weight']
        )
    
    # inicialização dos arrays para armazenar os dados de treino e teste
    def _initialize_data_arrays(self):
        self._images_train = np.empty(0, dtype=np.float64)
        self._images_test = np.empty(0, dtype=np.float64)
        self._predictions = np.empty(0, dtype=np.float64)
        self._answers_train = np.empty(0, dtype=np.int32)
        self._answers_test = np.empty(0, dtype=np.int32)
    
    # inicialização das configurações de pré-processamento
    def _initialize_preprocessing_config(self):
        self._n_colors = DEFAULT_PREPROCESSING['n_colors']
        self._gaussian_radius = DEFAULT_PREPROCESSING['gaussian_radius']
        self._sharpness_boost = DEFAULT_PREPROCESSING['sharpness_boost']
        self._contrast_boost = DEFAULT_PREPROCESSING['contrast_boost']
        self._brightness_boost = DEFAULT_PREPROCESSING['brightness_boost']
        self._percentage_train = DEFAULT_PREPROCESSING['percentage_train']
    
    # inicialização das configurações de segmentação
    def _initialize_segmentation_config(self):
        self._min_object_size = DEFAULT_SEGMENTATION['min_object_size']
        self._gaussian_sigma = DEFAULT_SEGMENTATION['gaussian_sigma']
    
    # inicialização das configurações do GLCM
    def _initialize_glcm_config(self):
        self._distances_glcm = np.array(DEFAULT_GLCM['distances'])
        self._angles_glcm = np.array(DEFAULT_GLCM['angles'])
        self._texture_descriptors = np.array(DEFAULT_GLCM['descriptors'])
    
    # inicialização das métricas de tempo de execução
    def _initialize_metrics(self):
        self._runtime_metrics = {}

    # definição do kernel do modelo SVM
    def set_model_kernel(self, kernel: str) -> None:
        self._model_kernel = kernel
        self._model = svm.SVC(
            kernel=self._model_kernel,
            C=DEFAULT_CONFIG['C'],
            class_weight=DEFAULT_CONFIG['class_weight']
        )

    # utilização do kernel do modelo SVM
    def get_model_kernel(self) -> str:
        return self._model_kernel

    # definição do número de cores para quantização
    def set_n_colors(self, n_colors: int) -> None:
        self._n_colors = n_colors
    
    
    def get_n_colors(self) -> int:
        return self._n_colors
    
    # definição do raio do filtro gaussiano
    def set_gaussian_radius(self, radius: int) -> None:
        self._gaussian_radius = radius
    
    # utilização do raio do filtro gaussiano
    def get_gaussian_radius(self) -> int:
        return self._gaussian_radius
    
    # definição da força do realce de nitidez
    def set_sharpness_boost_strength(self, strength: float) -> None:
        self._sharpness_boost = strength
    
    # utilização da força do realce de nitidez
    def get_sharpness_boost_strength(self) -> float:
        return self._sharpness_boost
    
    # definição da força do realce de contraste 
    def set_contrast_boost_strength(self, strength: float) -> None:
        self._contrast_boost = strength
    
    # utilização da força do realce de contraste
    def get_contrast_boost_strength(self) -> float:
        return self._contrast_boost
    
    # definição da força do realce de brilho
    def set_brightness_boost_strength(self, strength: float) -> None:
        self._brightness_boost = strength
    
    # utilização da força do realce de brilho
    def get_brightness_boost_strength(self) -> float:
        return self._brightness_boost

    # definição da porcentagem de dados para treino
    def set_percentage_train(self, percentage: int) -> None:
        self._percentage_train = percentage / 100
    
    # utilização da porcentagem de dados para treino
    def get_percentage_train(self) -> int:
        return int(self._percentage_train * 100)

    # definição do diretório contendo as imagens organizadas por classe    
    def set_images_dir(self, directory: str) -> None:
        self._images_dir = directory
    
    # utilização do diretório contendo as imagens organizadas por classe
    def get_images_dir(self) -> str:
        return self._images_dir
    
    # definição das extensões de imagem suportadas
    def get_supported_img_extensions(self) -> List[str]:
        return SUPPORTED_EXTENSIONS

    # definição das distâncias para cálculo da matriz GLCM
    def set_distances_glcm(self, distances: List[int]) -> None:
        self._distances_glcm = np.array(distances)
    
    # utilização das distâncias para cálculo da matriz GLCM
    def get_distances_glcm(self) -> np.ndarray:
        return self._distances_glcm
    
    # definição dos ângulos para cálculo da matriz GLCM
    def set_angles_glcm(self, angles: List[float]) -> None:
        self._angles_glcm = np.array(angles)
    
    # utilização dos ângulos para cálculo da matriz GLCM
    def get_angles_glcm(self) -> np.ndarray:
        return self._angles_glcm
    
    # definição dos descritores de textura para cálculo da matriz GLCM
    def set_texture_descriptors(self, descriptors: List[str]) -> None:
        self._texture_descriptors = np.array(descriptors)
    
    # utilização dos descritores de textura para cálculo da matriz GLCM
    def get_texture_descriptors(self) -> np.ndarray:
        return self._texture_descriptors
    
    # utilização do conjunto de treino  
    def get_images_train(self) -> np.ndarray:
        return self._images_train
    
    # utilização dos labels do conjunto de treino
    def get_answers_train(self) -> np.ndarray:
        return self._answers_train
    
    # utilização do conjunto de teste
    def get_images_test(self) -> np.ndarray:
        return self._images_test
    
    # utilização dos labels do conjunto de teste
    def get_answers_test(self) -> np.ndarray:
        return self._answers_test
    
    # utilização das predições do modelo no conjunto de teste
    def get_predictions(self) -> np.ndarray:
        return self._predictions
    
    # definição da métrica de tempo de execução
    def set_runtime_metric(self, metric_name: str, measured_time: float) -> None:
        self._runtime_metrics[metric_name] = f"{measured_time:.6f} s"
    
    # utilização das métricas de tempo de execução
    def get_runtime_metrics(self) -> Dict[str, str]:
        return self._runtime_metrics
    
    # definição do tamanho mínimo do objeto
    def set_min_object_size(self, size: int) -> None:
        self._min_object_size = size
    
    # utilização do tamanho mínimo do objeto
    def get_min_object_size(self) -> int:
        return self._min_object_size
    
    # definição do sigma gaussiano para segmentação
    def set_gaussian_sigma(self, sigma: float) -> None:
        self._gaussian_sigma = sigma
    
    # utilização do sigma gaussiano para segmentação
    def get_gaussian_sigma(self) -> float:
        return self._gaussian_sigma
    
    

    # aplica segmentação por threshold de Otsu na imagem para extrair as regiões de interesse
    def segment_image(self, image_array: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        
        # converte para escala de cinza se necessário
        if len(image_array.shape) == 3:
            image_gray = rgb2gray(image_array)
        else:
            image_gray = image_array.astype(np.float64) / 255.0
        
        # aplica suavização gaussiana
        image_smooth = gaussian(image_gray, sigma=self._gaussian_sigma)
        
        # aplica segmentação por threshold de Otsu
        segmented = self._otsu_segmentation(image_smooth)
        
        # aplica máscara da segmentação na imagem original
        # cria máscara binária
        mask = segmented > 0
        # aplica máscara na imagem original
        result = image_gray.copy()
        result[~mask] = 0  # define pixels fora da região de interesse como 0
        segmented = result
        
        end = time.perf_counter()
        self.set_runtime_metric("Segmentação da imagem", end - start)
        
        return segmented
    
    # segmentação por threshold de Otsu
    def _otsu_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Aplica segmentação por threshold de Otsu."""
        # threshold de Otsu
        threshold = threshold_otsu(image)
        binary = image > threshold
        
        # operações morfológicas para limpeza
        binary = opening(binary, disk(3))
        binary = closing(binary, disk(3))
        
        # remove objetos pequenos
        binary = remove_small_objects(binary, min_size=self._min_object_size)
        
        return binary.astype(np.float64)

    """
        Aplica o pre-processamento na imagem, foram utilizados os seguintes filtros:
        - Filtro gaussiano 
        - Realce de nitidez
        - Ajuste de contraste
        - Ajuste de brilho
        - Segmentação por threshold de Otsu (se habilitada)
        - Quantização para N tons de cinza
        """
    def pre_process_img(self, image: Image.Image) -> Image.Image:
        
        start = time.perf_counter()
        
        # converte para escala de cinza se não estiver
        if image.mode != 'L':
            image = image.convert('L')
        
        # aplica o filtro gaussiano
        if self._gaussian_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=self._gaussian_radius))
        
        # realce de nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self._sharpness_boost)
        
        # ajuste de contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self._contrast_boost)
        
        # ajuste de brilho
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self._brightness_boost)
        
        # converte para array numpy para segmentação
        image_array = np.array(image)
        segmented_array = self.segment_image(image_array)
        
        # converte de volta para PIL Image
        segmented_array = (segmented_array * 255).astype(np.uint8)
        image = Image.fromarray(segmented_array, mode='L')
        
        # quantização para N tons de cinza
        image = image.quantize(self._n_colors)
        
        end = time.perf_counter()
        self.set_runtime_metric("Pré-processamento", end - start)
        
        return image
    
    # processa uma imagem e extrai descritores de textura GLCM
    def process_img(self, filepath: str) -> np.ndarray:
        start = time.perf_counter()
        
        # carrega e pré-processa imagem
        image = Image.open(filepath)
        image = self.pre_process_img(image)
        
        # converte para array numpy
        image_array = np.array(image, dtype=np.uint8)
        image_array = image_array.copy()  # garantir que seja writable
        
        # calcula matriz GLCM
        glcm = graycomatrix(
            image_array,
            self._distances_glcm,
            self._angles_glcm,
            levels=self._n_colors
        )
        
        # extrair descritores de textura
        texture_descriptors = []
        
        # adiciona entropia de Shannon
        texture_descriptors.append(shannon_entropy(glcm, base=2))
        
        # adiciona outros descritores
        for descriptor in self._texture_descriptors:
            texture_descriptors.append(graycoprops(glcm, descriptor))
        
        # concatena todos os descritores
        descriptors = np.concatenate(texture_descriptors, axis=None)
        
        end = time.perf_counter()
        self.set_runtime_metric("Processamento de uma imagem", end - start)
        
        return descriptors
    
    # separa os dados em conjuntos de treino e teste
    def split_train_test(self) -> None:
        tmp_train_set = []
        tmp_test_set = []
        
        # reinicializa arrays
        self._answers_test = np.empty(0, dtype=np.int32)
        self._answers_train = np.empty(0, dtype=np.int32)
        
        start = time.perf_counter()
        
        # processa cada classe
        for alzheimer_class in ALZHEIMER_CLASSES:
            class_dir = os.path.join(self._images_dir, alzheimer_class)
            
            # lista imagens da classe
            image_files = [
                f for f in os.listdir(class_dir)
                if f.endswith(tuple(self.get_supported_img_extensions()))
            ]
            
            # separa treino/teste
            train_files, test_files = train_test_split(
                image_files,
                train_size=self._percentage_train,
                shuffle=True
            )
            
            # processa imagens de treino
            for image_file in train_files:
                filepath = os.path.join(class_dir, image_file)
                descriptors = self.process_img(filepath)
                
                tmp_train_set.append(descriptors)
                self._answers_train = np.append(self._answers_train, [alzheimer_class])
            
            # processa imagens de teste
            for image_file in test_files:
                filepath = os.path.join(class_dir, image_file)
                descriptors = self.process_img(filepath)
                
                tmp_test_set.append(descriptors)
                self._answers_test = np.append(self._answers_test, [alzheimer_class])
        
        end = time.perf_counter()
        self.set_runtime_metric("Separação das imagens de treino e teste", end - start)
        
        # converte para arrays numpy
        self._images_train = np.asarray(tmp_train_set)
        self._images_test = np.asarray(tmp_test_set)
    
    # treina o modelo SVM com os dados de treino (chamar apenas após o split_train_test())
    def train_model(self) -> None:
        start = time.perf_counter()
        
        self._model.fit(self._images_train, self._answers_train)
        
        end = time.perf_counter()
        self.set_runtime_metric("Etapa de treinamento do modelo", end - start)
    
    # faz predições no conjunto de teste (chamar apeapós o split_train_test() e train_model())
    def predict_test_images(self) -> None:
        start = time.perf_counter()
        
        self._predictions = self._model.predict(self._images_test)
        
        end = time.perf_counter()
        self.set_runtime_metric("Identificação das imagens de teste", end - start)
    
    def predict_single_image(self, filepath: str) -> np.ndarray:
        start = time.perf_counter()
        
        descriptors = self.process_img(filepath)
        prediction = self._model.predict([descriptors])
        
        end = time.perf_counter()
        self.set_runtime_metric("Identificação de uma única imagem", end - start)
        
        return prediction
    
    # aplica pré-processamento e retorna a imagem para visualização
    def preview_single_image(self, filepath: str) -> Image.Image:
        
        start = time.perf_counter()
        
        image = Image.open(filepath)
        image = self.pre_process_img(image)
        
        end = time.perf_counter()
        self.set_runtime_metric("Aplicação dos efeitos na imagem", end - start)
        
        return image

    # calcula métricas de avaliação e gera matriz de confusão
    def get_prediction_metrics(self) -> Tuple[float, float]:
        start = time.perf_counter()
        
        # calcula métricas
        accuracy = sklearn.metrics.accuracy_score(self._answers_test, self._predictions)
        confusion_matrix = sklearn.metrics.confusion_matrix(self._answers_test, self._predictions)
        specificity = (100 - accuracy) / 300  # Fórmula original mantida
        
        # gera visualização da matriz de confusão
        self._generate_confusion_matrix_plot(confusion_matrix)
        
        end = time.perf_counter()
        self.set_runtime_metric("Geração das métricas de avaliação", end - start)
        
        return accuracy, specificity
    
    def _generate_confusion_matrix_plot(self, confusion_matrix: np.ndarray) -> None:
        # gera e salva o gráfico da matriz de confusão
        plt.figure(figsize=(7, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title("Matriz de Confusão", fontsize=18, pad=20)
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # configura eixos
        tick_marks = np.arange(len(ALZHEIMER_CLASSES))
        plt.xticks(tick_marks, ALZHEIMER_CLASSES, rotation=20, fontsize=12)
        plt.yticks(tick_marks, ALZHEIMER_CLASSES, fontsize=12)
        
        # adiciona valores nas células
        thresh = confusion_matrix.max() / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                color = "black" if confusion_matrix[i, j] < thresh else "white"
                plt.text(
                    j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='none', edgecolor='0.3')
                )
        
        # configura labels dos eixos
        plt.ylabel("Classe correta", labelpad=15, fontsize=14)
        plt.xlabel("Classe estimada", labelpad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig("./matriz-confusao.png", bbox_inches='tight')
        plt.close()