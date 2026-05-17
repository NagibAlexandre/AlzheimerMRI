import sys
import os
import shutil

import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import nibabel as nib
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTextEdit, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

IMG_SIZE = 224
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ MODEL ------------------

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model.pth')


def getResNetModel(model_path=model_path):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES)) 
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(DEVICE)
    
    return model, DEVICE

# ------------------ STYLES ------------------

def getMainStylesheet(font_size=16):
    return f"""
        #sidebar {{
            background: #2768a5;
            border: none;
        }}
        
        QScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollArea > QWidget {{
            background: transparent;
        }}
        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}
        QWidget#qt_scrollarea_viewport {{
            background: transparent;
        }}
        
        #sidebarHeader {{
            background: #1b446f;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        }}
        
        #sidebarTitle {{
            color: #ffffff;
            font-size: {font_size + 2}pt;
            font-weight: bold;
        }}
        
        #sidebarSubtitle {{
            color: rgba(255, 255, 255, 0.85);
            font-size: {font_size - 4}pt;
        }}
        
        #menuSectionTitle {{
            color: rgba(255, 255, 255, 0.85);
            font-size: {font_size - 6}pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        QPushButton#menuItem {{
            background: transparent;
            color: #ffffff;
            border: none;
            border-left: 4px solid transparent;
            text-align: left;
            padding: 10px 0 10px 20px;
            font-size: {font_size - 2}pt;
        }}
        
        QPushButton#menuItem:hover {{
            background: rgba(255, 255, 255, 0.12);
            padding-left: 25px;
        }}
        
        QPushButton#menuItem[active="true"] {{
            background: rgba(255, 255, 255, 0.18);
            border-left: 4px solid #63b3ed;
            padding-left: 16px;
        }}
        
        #sidebarFooter {{
            background: #1b446f;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
        }}
        
        #accessibilityLabel {{
            color: rgba(255, 255, 255, 0.85);
            font-size: {font_size - 6}pt;
            font-weight: bold;
            letter-spacing: 1px;
        }}
        
        QPushButton#accessibilityBtn {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 6px;
            color: #ffffff;
            font-size: {font_size - 2}pt;
            padding: 8px;
        }}
        
        QPushButton#accessibilityBtn:hover {{
            background: rgba(255, 255, 255, 0.20);
        }}
        
        #contentArea {{
            background: #f7fafc;
        }}
        
        #contentHeader {{
            background: white;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        #pageTitle {{
            color: #1a365d;
            font-size: {font_size + 6}pt;
            font-weight: bold;
        }}
        
        #breadcrumb {{
            color: #718096;
            font-size: {font_size - 3}pt;
        }}
        
        #contentWidget {{
            background: #f7fafc;
        }}
        
        #statCard, #imageViewerCard {{
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }}

        #infoLabel {{
            background: white;
            border: none;
        }}
        
        #cardTitle {{
            color: #2d3748;
            font-size: {font_size - 1}pt;
            font-weight: bold;
        }}
        
        #cardValue {{
            color: #1a365d;
            font-size: {font_size + 12}pt;
            font-weight: bold;
        }}
        
        #cardLabel {{
            color: #718096;
            font-size: {font_size - 3}pt;
        }}
        
        #imageViewer {{
            background: #000000;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }}
        
        QPushButton#primaryBtn {{
            background: #1b446f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: {font_size - 2}pt;
            font-weight: 500;
        }}
        
        QPushButton#primaryBtn:hover {{
            background: #1a3f66;
        }}
        
        QPushButton#secondaryBtn {{
            background: #edf2f7;
            color: #2d3748;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: {font_size - 2}pt;
            font-weight: 500;
        }}
        
        QPushButton#secondaryBtn:hover {{
            background: #e2e8f0;
        }}
        
        QPushButton#secondaryBtn[selected="true"] {{
            background: #ebf8ff;
            border: 2px solid #3182ce;
            color: #2c5282;
        }}
        
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 5px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: rgba(0, 0, 0, 0.3);
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        
        #sectionTitle {{
            color: #1a365d;
            font-size: {font_size + 2}pt;
            font-weight: bold;
        }}
    """

def load_image_file(filePath):
    if filePath.lower().endswith(('.nii', '.nii.gz')):
        niiImage = nib.load(filePath)
        niiDate = niiImage.get_fdata()
        
        if len(niiDate.shape) == 3:
            midSlice = niiDate.shape[2] // 2
            imageSlice = niiDate[:, :, midSlice]
        else:
            imageSlice = niiDate
        
        imageSlice = ((imageSlice - imageSlice.min()) / (imageSlice.max() - imageSlice.min()) * 255)
        imageSlice = imageSlice.astype(np.uint8)
        
        imageSlice = np.rot90(imageSlice)
        return imageSlice
    else:
        return cv2.imread(filePath)
    
def numpyToQpixmap(image):
    if len(image.shape) == 2:
        imageRgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width, channel = imageRgb.shape
    bytesPline = 3 * width
    q_image = QImage(imageRgb.data, width, height, bytesPline, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)
    
# ------------------ ZOOM CONTROL ------------------
class ZoomClass:

    def setup_zoom(self):
        self.zoomFactor = 1.0
        self.originalPixmap = None
    
    def applyZoomToLabel(self, label, containerSize=None):
        if self.originalPixmap and not self.originalPixmap.isNull():
            if containerSize:
                containerWidth, containerHeight = containerSize
            else:
                containerWidth = label.width() - 20
                containerHeight = label.height() - 20
            
            newWidth = int(self.originalPixmap.width() * self.zoomFactor)
            newHeight = int(self.originalPixmap.height() * self.zoomFactor)
            
            scaledPixmap = self.originalPixmap.scaled(
                min(newWidth, containerWidth),
                min(newHeight, containerHeight),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            label.setPixmap(scaledPixmap)
            return int(self.zoomFactor * 100)
        return 100
    
    def changeZoom(self, delta, minZoom=0.1, maxZoom=5.0):
        if delta > 0:
            self.zoomFactor = min(self.zoomFactor * 1.25, maxZoom)
        elif delta < 0:
            self.zoomFactor = max(self.zoomFactor / 1.25, minZoom)
        else:
            self.zoomFactor = 1.0
        return self.zoomFactor
    
# ------------------ LOAD IMAGE WIDGET ------------------

class ImageLoaderWidget(QWidget, ZoomClass):
    loadedImg = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentImg = None
        self.imagePath = None
        self.setup_zoom()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        layout.addLayout(self.createControls())

        separator = QFrame()
        separator.setFrameShape(QFrame.NoFrame)
        separator.setFixedHeight(1)
        separator.setStyleSheet("border: 1px solid #e2e8f0;")
        layout.addWidget(separator)

        layout.addWidget(self.create_image_container())
    
    def createControls(self):
        controlsLayout = QHBoxLayout()
        
        fileControls = QHBoxLayout()
        
        self.loadBtn = QPushButton("Carregar Imagem")
        self.loadBtn.setObjectName("primaryBtn")
        self.loadBtn.setCursor(Qt.PointingHandCursor)
        self.loadBtn.clicked.connect(self.loadImage)
        fileControls.addWidget(self.loadBtn)

        self.clearBtn = QPushButton("Limpar")
        self.clearBtn.setObjectName("secondaryBtn")
        self.clearBtn.setCursor(Qt.PointingHandCursor)
        self.clearBtn.clicked.connect(self.clearImage)
        self.clearBtn.setEnabled(False)
        fileControls.addWidget(self.clearBtn)

        controlsLayout.addLayout(fileControls)
        controlsLayout.addStretch()
        
        zoomControls = QHBoxLayout()
        
        self.zoomOutBtn = QPushButton("-")
        self.zoomOutBtn.setObjectName("secondaryBtn")
        self.zoomOutBtn.setCursor(Qt.PointingHandCursor)
        self.zoomOutBtn.clicked.connect(lambda: self.handleZoom(-1))
        self.zoomOutBtn.setEnabled(False)
        zoomControls.addWidget(self.zoomOutBtn)
        
        self.resetZoomBtn = QPushButton("Reset")
        self.resetZoomBtn.setObjectName("secondaryBtn")
        self.resetZoomBtn.setCursor(Qt.PointingHandCursor)
        self.resetZoomBtn.clicked.connect(lambda: self.handleZoom(0))
        self.resetZoomBtn.setEnabled(False)
        zoomControls.addWidget(self.resetZoomBtn)
        
        self.zoomInBtn = QPushButton("+")
        self.zoomInBtn.setObjectName("secondaryBtn")
        self.zoomInBtn.setCursor(Qt.PointingHandCursor)
        self.zoomInBtn.clicked.connect(lambda: self.handleZoom(1))
        self.zoomInBtn.setEnabled(False)
        zoomControls.addWidget(self.zoomInBtn)
        
        self.zoomLabel = QLabel("Zoom: 100%")
        self.zoomLabel.setObjectName("cardLabel")
        self.zoomLabel.setFixedWidth(80)
        zoomControls.addWidget(self.zoomLabel)
        
        controlsLayout.addLayout(zoomControls)
        return controlsLayout
    
    def create_image_container(self):
        self.imageContainer = QFrame()
        self.imageContainer.setObjectName("imageViewer")
        self.imageContainer.setMinimumHeight(400)
        self.imageContainer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        imageLayout = QVBoxLayout(self.imageContainer)
        imageLayout.setAlignment(Qt.AlignCenter)
        
        self.imageLabel = QLabel("Nenhuma imagem carregada")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(500, 400)
        
        imageLayout.addWidget(self.imageLabel)
        return self.imageContainer
    
    def loadImage(self):
        fileFilter = "Imagens (*.png *.jpg *.nii *.nii.gz);;Todos (*.*)"
        filePath, _ = QFileDialog.getOpenFileName(self, "Carregar Imagem", "", fileFilter)
        
        if filePath:
            try:
                image = load_image_file(filePath)
                if image is None:
                    self.showError("Não foi possível carregar a imagem")
                    return
                
                self.currentImg = image
                self.imagePath = filePath
                
                self.displayImage(image)
                self.enableControls(True)
                
                self.loadedImg.emit(filePath, image)
                
            except Exception as e:
                self.showError(f"Erro ao carregar imagem: {str(e)}")
    
    def displayImage(self, image):
        try:
            self.originalPixmap = numpyToQpixmap(image)
            self.handleZoom(0)
            self.imageLabel.setText("")
            self.imageLabel.setStyleSheet("background: transparent; border: none;")
        except Exception as e:
            self.showError(f"Erro ao exibir imagem: {str(e)}")
    
    def handleZoom(self, direction):
        self.changeZoom(direction)
        zoom_percentage = self.applyZoomToLabel(self.imageLabel)
        self.zoomLabel.setText(f"Zoom: {zoom_percentage}%")
    
    def enableControls(self, enabled):
        self.clearBtn.setEnabled(enabled)
        self.zoomInBtn.setEnabled(enabled)
        self.zoomOutBtn.setEnabled(enabled)
        self.resetZoomBtn.setEnabled(enabled)
    
    def clearImage(self):
        self.currentImg = None
        self.imagePath = None
        self.originalPixmap = None
        self.zoomFactor = 1.0
        
        self.imageLabel.clear()
        self.imageLabel.setText("Nenhuma imagem carregada")
        
        self.zoomLabel.setText("Zoom: 100%")
        self.enableControls(False)
    
    def showError(self, message):
        self.imageLabel.setText(f"Erro: {message}")

# ------------------ WIDGET MODELO ------------------
class ModelsWidget(QFrame):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_viewer = ImageLoaderWidget()
        self.originalFilePath = None
        self.model = None
        self.device = None
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.setSpacing(0)

        contentWidget = QFrame()
        contentLayout = QVBoxLayout(contentWidget)
        contentLayout.setContentsMargins(15, 15, 15, 15)
        contentLayout.setSpacing(20)

        contentLayout.addWidget(self.original_viewer)

        self.btnClassifier = QPushButton("Classificar")
        self.btnClassifier.setObjectName("primaryBtn")
        self.btnClassifier.setMaximumWidth(400)
        self.btnClassifier.setCursor(Qt.PointingHandCursor)
        self.btnClassifier.setEnabled(False)
        self.btnClassifier.clicked.connect(self.runModel)
        contentLayout.addWidget(self.btnClassifier)

        contentLayout.addWidget(self.create_results_section())

        mainLayout.addWidget(contentWidget)

        self.original_viewer.loadedImg.connect(self.onImageLoaded)
    
    def load_model(self):
        try:
            self.model, self.device = getResNetModel(model_path=model_path)
            print(f"Modelo carregado com sucesso no dispositivo: {self.device}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            self.model = None
            self.device = None

    def onImageLoaded(self, file_path, image):
        self.originalFilePath = file_path
        self.btnClassifier.setEnabled(True)
        self.clearResults()
    
    def clearResults(self):
        pass
    
    def clearResultsDisplay(self):
        self.results_text.setText("Adicione uma imagem e clique em classificar para ver o resultado")
    
    def create_results_section(self):
        frame = QFrame()
        frame.setMinimumHeight(600)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(15)
        
        headerLayout = QHBoxLayout()

        self.results_title = QLabel("Resultados")
        self.results_title.setAlignment(Qt.AlignLeft)
        self.results_title.setObjectName("sectionTitle")
        self.results_title.setFont(QFont("Roboto", 16, QFont.Bold))
        headerLayout.addWidget(self.results_title)
        
        headerLayout.addStretch()
        
        self.btn_clear_results = QPushButton("Limpar Resultados")
        self.btn_clear_results.setObjectName("secondaryBtn")
        self.btn_clear_results.setCursor(Qt.PointingHandCursor)
        self.btn_clear_results.setMaximumWidth(200)
        self.btn_clear_results.clicked.connect(self.clearResultsDisplay)
        headerLayout.addWidget(self.btn_clear_results)
        
        layout.addLayout(headerLayout)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setObjectName("infoLabel")
        self.results_text.setReadOnly(True)
        self.results_text.setText("Adicione uma imagem e clique em classificar para ver o resultado")
        self.results_text.setStyleSheet("""
            QTextEdit#infoLabel {
                color: #718096;
                font-family: Roboto;
                font-size: 14px;
                background: transparent;
                border: none;
                padding: 10px;
            }
        """)
        layout.addWidget(self.results_text)
        
        return frame
    
    def runModel(self):
        """Run the ResNet classification model"""
        try:
            result = self.runResNetClassifier()
            self.displayResults(result)
        except Exception as e:
            self.showError(f"Erro ao executar modelo: {str(e)}")
    
    def processNiftiImage(self, nifti_path, slice_idx=None):
        try:
            nifti_img = nib.load(nifti_path)
            image_data = nifti_img.get_fdata()
            
            if len(image_data.shape) == 3:
                if slice_idx is None:
                    slice_idx = image_data.shape[2] // 2
                slice_2d = image_data[:, :, slice_idx]
            elif len(image_data.shape) == 2:
                slice_2d = image_data
            elif len(image_data.shape) == 4:
                if slice_idx is None:
                    slice_idx = image_data.shape[2] // 2
                slice_2d = image_data[:, :, slice_idx, 0]
            else:
                raise ValueError(f"Formato não suportado: {image_data.shape}")
            
            slice_normalized = ((slice_2d - slice_2d.min()) / 
                            (slice_2d.max() - slice_2d.min()) * 255)
            slice_normalized = slice_normalized.astype(np.uint8)
            
            slice_rgb = np.stack([slice_normalized] * 3, axis=-1)
            pil_image = Image.fromarray(slice_rgb)
            
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(pil_image).unsqueeze(0)
            return image_tensor
        
        except Exception as e:
            raise Exception(f"Erro ao processar imagem NIFTI: {str(e)}")
    
    def preprocessImage(self, image_path):
        try:
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # Adiciona batch dimension
            
            return image_tensor
        except Exception as e:
            raise Exception(f"Erro ao processar imagem: {str(e)}")
    
    def runResNetClassifier(self):
        try:
            if self.model is None or self.device is None:
                return {
                    'sucesso': False,
                    'mensagem': 'Modelo não carregado. Verifique o arquivo model.pth',
                    'modelo': 'ResNet18'
                }
            
            is_nifti = self.originalFilePath.endswith(('.nii', '.nii.gz'))
            
            if is_nifti:
                image_tensor = self.processNiftiImage(self.originalFilePath)
            else:
                image_tensor = self.preprocessImage(self.originalFilePath)
            
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class_idx = predicted.item()
                confidence_score = confidence.item()
            
            result = {
                'sucesso': True,
                'classe': CLASSES[predicted_class_idx],
                'confianca': confidence_score,
                'modelo': 'ResNet18',
                'probabilidades': {
                    CLASSES[i]: float(probabilities[0][i]) 
                    for i in range(len(CLASSES))
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'sucesso': False,
                'mensagem': f'Erro no ResNet18: {str(e)}',
                'modelo': 'ResNet18'
            }
    
    def displayResults(self, result):
        text = ""
        
        if result.get('sucesso', False):
            classe = result.get('classe', '')
            confianca = result.get('confianca', 0) * 100
            modelo = result.get('modelo', '')
            probabilidades = result.get('probabilidades', {})
            
            class_colors = {
                'NonDemented': '#10b981',
                'VeryMildDemented': '#3b82f6',
                'MildDemented': '#f59e0b',
                'ModerateDemented': '#ef4444'
            }

            cor_classe = class_colors.get(classe, '#667eea')
            
            text += f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; 
                        border-radius: 12px; 
                        margin: 20px 0;
                        text-align: center;'>
                <h2 style='color: #1a365d; margin: 0 0 10px 0; font-size: 10px; font-weight: normal;'>
                    CLASSIFICAÇÃO
                </h2>
                <h1 style='color: #1a365d; margin: 0; font-size: 36px; font-weight: bold;'>
                    {classe}
                </h1>
            </div>
            
            <div style=' 
                        padding: 25px; 
                        border-radius: 12px; 
                        margin: 20px 0;'>
                <div style='display: grid; 
                            grid-template-columns: 1fr 1fr; 
                            gap: 15px;
                            margin-bottom: 20px;'>
                    <div style='
                                padding: 20px; 
                                border-radius: 8px; 
                                border: 2px solid #e2e8f0;
                                text-align: center;'>
                        <p style='color: #718096; 
                                  font-size: 11px; 
                                  text-transform: uppercase; 
                                  margin: 0 0 8px 0;'>
                            CONFIANÇA
                        </p>
                        <p style='color: #1a365d; 
                                  font-size: 24px; 
                                  font-weight: bold; 
                                  margin: 0;'>
                            {confianca:.1f}%
                        </p>
                    </div>
                    
                    <div style='; 
                                padding: 20px; 
                                border-radius: 8px; 
                                border: 2px solid #e2e8f0;
                                text-align: center;'>
                        <p style='color: #718096; 
                                  font-size: 11px; 
                                  text-transform: uppercase; 
                                  margin: 0 0 8px 0;'>
                            MODELO
                        </p>
                        <p style='color: #1a365d; 
                                  font-size: 24px; 
                                  font-weight: bold; 
                                  margin: 0;'>
                            {modelo}
                        </p>
                    </div>
                </div>
                
                <p style='color: #718096; 
                          font-size: 12px; 
                          text-transform: uppercase; 
                          letter-spacing: 1px; 
                          margin: 15px 0 10px 0;'>
                    PROBABILIDADES POR CLASSE
                </p>
                
                <div style='; 
                            padding: 15px; 
                            border-radius: 8px; 
                            border: 2px solid #e2e8f0;'>
            """
            
            for class_name, prob in sorted(probabilidades.items(), key=lambda x: x[1], reverse=True):
                prob_percent = prob * 100
                text += f"""
                    <div style='margin-bottom: 10px;'>
                        <div style='display: flex; 
                                    justify-content: space-between; 
                                    margin-bottom: 5px;'>
                            <span style='color: #2d3748; 
                                        font-weight: 500;'>
                                {class_name}
                            </span>
                            <span style='color: #718096;'>
                                {prob_percent:.2f}%
                            </span>
                        </div>
                        <div style='background: #e2e8f0; 
                                    height: 8px; 
                                    border-radius: 4px; 
                                    overflow: hidden;'>
                            <div style='background: {class_colors.get(class_name, "#667eea")}; 
                                        height: 100%; 
                                        width: {prob_percent:.1f}%;'>
                            </div>
                        </div>
                    </div>
                """
            
            text += """
                </div>
            </div>
            """
        else:
            text += f"""
            <div style='background: #fee2e2; 
                        padding: 20px; 
                        border-radius: 8px; 
                        border-left: 4px solid #ef4444;
                        margin: 20px 0;'>
                <h4 style='color: #ef4444; margin: 0;'>
                    {result.get('mensagem', 'Erro desconhecido')}
                </h4>
            </div>
            """
            
        self.results_text.setHtml(text)
    
    def showError(self, message):
        """Show error message"""
        self.results_text.setHtml(
            f"""
            <div style='background: #fee2e2; 
                        padding: 20px; 
                        border-radius: 8px; 
                        border-left: 4px solid #ef4444;
                        margin: 20px 0;'>
                <h4 style='color: #ef4444; margin: 0;'>
                    ERRO: {message}
                </h4>
            </div>
            """
        )

class SideBar(QFrame):
    pageChanged = pyqtSignal(str)  
    fontSizeChanged = pyqtSignal(int)  
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.activeButton = None
        self.init_ui()
        
    def init_ui(self):
        self.setObjectName("sidebar")
        self.setFixedWidth(260)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        layout.addWidget(self.createHeader())
        layout.addWidget(self.createMenu())
        layout.addWidget(self.createFooter())
        
    def createHeader(self):
        header = QFrame()
        header.setObjectName("sidebarHeader")
        header.setFixedHeight(100)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 25, 20, 25)
        
        title = QLabel("MRI Analyzer")
        title.setObjectName("sidebarTitle")
        title.setFont(QFont("Roboto", 16, QFont.Bold))
        layout.addWidget(title)
        
        subtitle = QLabel("Alzheimer's Detection")
        subtitle.setObjectName("sidebarSubtitle")
        subtitle.setFont(QFont("Roboto", 10))
        layout.addWidget(subtitle)
        
        return header
    
    def createMenu(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        widget = QWidget()
        self.menuLayout = QVBoxLayout(widget)
        self.menuLayout.setContentsMargins(0, 20, 0, 20)
        self.menuLayout.setSpacing(0)
        
        menuStructure = {
            "CLASSIFICAÇÃO": ["CNN"]
        }
        
        self.menuButtons = []
        
        for sectionTitle, items in menuStructure.items():
            self.addSectionTitle(sectionTitle)
            for item in items:
                self.addMenuItem(item)
        
        self.menuLayout.addStretch()
        scroll.setWidget(widget)
        return scroll
    
    def addSectionTitle(self, title):
        label = QLabel(title)
        label.setObjectName("menuSectionTitle")
        label.setContentsMargins(20, 15, 20, 10)
        label.setFont(QFont("Roboto", 9, QFont.Bold))
        self.menuLayout.addWidget(label)
    
    def addMenuItem(self, text):
        btn = QPushButton(text)
        btn.setObjectName("menuItem")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(44)
        btn.clicked.connect(lambda: self.onMenuClick(btn, text))
        
        if text == "CNN":
            btn.setProperty("active", True)
            self.activeButton = btn
        
        self.menuButtons.append(btn)
        self.menuLayout.addWidget(btn)
    
    def onMenuClick(self, button, page_name):
        if self.activeButton:
            self.activeButton.setProperty("active", False)
            self.activeButton.style().unpolish(self.activeButton)
            self.activeButton.style().polish(self.activeButton)
        
        button.setProperty("active", True)
        button.style().unpolish(button)
        button.style().polish(button)
        self.activeButton = button
        
        self.pageChanged.emit(page_name)
    
    def createFooter(self):
        footer = QFrame()
        footer.setObjectName("sidebarFooter")
        footer.setFixedHeight(100)
        
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(20, 20, 20, 20)
        
        label = QLabel("ACESSIBILIDADE")
        label.setObjectName("accessibilityLabel")
        label.setFont(QFont("Roboto", 9, QFont.Bold))
        layout.addWidget(label)
        
        buttonsLayout = QHBoxLayout()
        buttonsLayout.setSpacing(10)
        
        btnDecrease = QPushButton("A-")
        btnDecrease.setObjectName("accessibilityBtn")
        btnDecrease.setCursor(Qt.PointingHandCursor)
        btnDecrease.clicked.connect(lambda: self.fontSizeChanged.emit(-2))
        
        btnIncrease = QPushButton("A+")
        btnIncrease.setObjectName("accessibilityBtn")
        btnIncrease.setCursor(Qt.PointingHandCursor)
        btnIncrease.clicked.connect(lambda: self.fontSizeChanged.emit(2))
        
        buttonsLayout.addWidget(btnDecrease)
        buttonsLayout.addWidget(btnIncrease)
        
        layout.addLayout(buttonsLayout)
        
        return footer

class Content(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.loadPage("CNN") 
        
    def init_ui(self):
        self.setObjectName("contentArea")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        layout.addWidget(self.createHeader())
        
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        
        self.contentWidget = QWidget()
        self.contentWidget.setObjectName("contentWidget")
        self.contentLayout = QVBoxLayout(self.contentWidget)
        self.contentLayout.setContentsMargins(30, 30, 30, 30)
        self.contentLayout.setSpacing(20)
        
        self.scrollArea.setWidget(self.contentWidget)
        layout.addWidget(self.scrollArea)
        
    def createHeader(self):
        header = QFrame()
        header.setObjectName("contentHeader")
        header.setFixedHeight(100)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(30, 20, 30, 20)
        
        self.pageTitle = QLabel("Classificação")
        self.pageTitle.setObjectName("pageTitle")
        self.pageTitle.setFont(QFont("Roboto", 20, QFont.Bold))
        layout.addWidget(self.pageTitle)
        
        self.breadcrumb = QLabel("Início / Classificação")
        self.breadcrumb.setObjectName("breadcrumb")
        self.breadcrumb.setFont(QFont("Roboto", 11))
        layout.addWidget(self.breadcrumb)
        
        return header
    
    def loadPage(self, pageName):
        self.pageTitle.setText(pageName)
        self.breadcrumb.setText(f"Início / {pageName}")
        
        while self.contentLayout.count():
            child = self.contentLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if pageName == "CNN":
            models = ModelsWidget()
            self.contentLayout.addWidget(models)
            self.contentLayout.addStretch()

        else:
            placeholder = QLabel(f"Página: {pageName}\n\n(Em desenvolvimento)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setFont(QFont("Roboto", 16))
            placeholder.setStyleSheet("color: #718096; padding: 100px;")

            self.contentLayout.addWidget(placeholder)
            self.contentLayout.addStretch()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_font_size = 16
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MRI Analyzer - Alzheimer's Disease Detection")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        mainLayout = QHBoxLayout(centralWidget)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.setSpacing(0)

        self.sidebar = SideBar(self)
        mainLayout.addWidget(self.sidebar)

        self.contentArea = Content(self)
        mainLayout.addWidget(self.contentArea)

        self.applyStylesheet()

        self.sidebar.pageChanged.connect(self.navigate_to)
        self.sidebar.fontSizeChanged.connect(self.updateFontSize)

    def navigate_to(self, page_name):
        self.contentArea.loadPage(page_name)
        
    def updateFontSize(self, delta):
        new_size = self.current_font_size + delta
        if 12 <= new_size <= 24:  
            self.current_font_size = new_size
            self.applyStylesheet()
            QTimer.singleShot(50, lambda: (self.update(), self.repaint()))

    def applyStylesheet(self):
        self.setStyleSheet(getMainStylesheet(self.current_font_size))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())