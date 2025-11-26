"""
Aplicação GUI para Classificação de Alzheimer em Imagens de Ressonância Magnética
Arquitetura: MVC (Model-View-Controller) com separação de responsabilidades
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import torch
import torch.nn as nn
from torchvision import transforms, models

# ============================================================================
# CAMADA DE MODELO (Model Layer)
# ============================================================================

@dataclass
class ClassificationResult:
    """Resultado da classificação"""
    has_alzheimer: bool
    confidence: float
    category: str
    

class ImageProcessor:
    """Responsável pelo processamento de imagens"""
    
    @staticmethod
    def load_and_preprocess(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Carrega e preprocessa a imagem"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Usa as mesmas transformações que em infer.py
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            tensor = transform(img)
            return tensor
        except Exception as e:
            raise ValueError(f"Erro ao processar imagem: {str(e)}")


class IAlzheimerClassifier(ABC):
    """Interface para classificadores de Alzheimer"""
    
    @abstractmethod
    def predict(self, image_array: np.ndarray) -> ClassificationResult:
        """Realiza a predição"""
        pass


class MockAlzheimerClassifier(IAlzheimerClassifier):
    """
    Classificador mock para demonstração
    Em produção, substitua por um modelo real (TensorFlow, PyTorch, etc.)
    """
    
    def predict(self, image_array: np.ndarray) -> ClassificationResult:
        """
        Simula uma predição
        Em produção, carregue um modelo treinado e faça a inferência real
        """
        # Simulação: analisa a intensidade média da imagem
        mean_intensity = np.mean(image_array)
        
        # Simula probabilidade baseada na intensidade
        confidence = abs(mean_intensity - 0.5) * 2
        has_alzheimer = mean_intensity < 0.45
        
        if has_alzheimer:
            if confidence > 0.7:
                category = "Alzheimer Moderado a Severo"
            else:
                category = "Alzheimer Leve"
        else:
            category = "Sem Alzheimer"
            
        return ClassificationResult(
            has_alzheimer=has_alzheimer,
            confidence=min(confidence, 0.95),
            category=category
        )


class RealAlzheimerClassifier(IAlzheimerClassifier):
    """
    Classificador real usando ResNet18 (mesmo modelo usado em infer.py)
    """
    
    CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, model_path: str = "model.pth"):
        """
        Inicializa o classificador com modelo ResNet18 treinado
        """
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        """Carrega o modelo ResNet18 treinado"""
        from pathlib import Path
        
        # Procura o modelo.pth no diretório do script ou no diretório Codigo
        possible_paths = [
            self.model_path,
            Path(__file__).parent / self.model_path,
            Path(__file__).parent / "model.pth"
        ]
        
        model_found = None
        for path in possible_paths:
            if Path(path).exists():
                model_found = path
                break
        
        if not model_found:
            raise FileNotFoundError(
                f"Modelo não encontrado. Procurou em:\n" + 
                "\n".join(str(p) for p in possible_paths)
            )
        
        print(f"Carregando modelo de: {model_found}")
        
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.CLASSES))
        
        try:
            model.load_state_dict(torch.load(model_found, map_location=self.DEVICE, weights_only=True))
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar estado do modelo: {str(e)}")
        
        model = model.to(self.DEVICE)
        model.eval()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        return model
    
    def predict(self, image_tensor: torch.Tensor) -> ClassificationResult:
        """
        Realiza predição com modelo ResNet18
        """
        try:
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor).float()
            
            # Adiciona batch dimension se necessário
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Garante que o tensor está em float
            image_tensor = image_tensor.float().to(self.DEVICE)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence = float(confidence[0])
            predicted_class = self.CLASSES[int(predicted_idx[0])]
            
            # Mapeia classe para categoria
            category_map = {
                'NonDemented': 'Sem Alzheimer',
                'VeryMildDemented': 'Alzheimer Muito Leve',
                'MildDemented': 'Alzheimer Leve',
                'ModerateDemented': 'Alzheimer Moderado a Severo'
            }
            
            category = category_map.get(predicted_class, predicted_class)
            has_alzheimer = predicted_class != 'NonDemented'
            
            return ClassificationResult(
                has_alzheimer=has_alzheimer,
                confidence=confidence,
                category=category
            )
        except Exception as e:
            raise RuntimeError(f"Erro durante predição: {str(e)}")


# ============================================================================
# CAMADA DE CONTROLE (Controller Layer)
# ============================================================================

class AlzheimerClassifierController:
    """Controlador principal da aplicação"""
    
    def __init__(self, use_real_model: bool = True):
        if use_real_model:
            try:
                self.classifier = RealAlzheimerClassifier()
            except FileNotFoundError:
                print("Aviso: Modelo não encontrado. Usando classificador mock.")
                self.classifier = MockAlzheimerClassifier()
        else:
            self.classifier = MockAlzheimerClassifier()
        
        self.image_processor = ImageProcessor()
        self.current_image_path: Optional[str] = None
        
    def load_image(self, file_path: str) -> Image.Image:
        """Carrega imagem para visualização"""
        self.current_image_path = file_path
        return Image.open(file_path)
    
    def classify_image(self, file_path: str) -> ClassificationResult:
        """Classifica a imagem"""
        if not Path(file_path).exists():
            raise FileNotFoundError("Arquivo não encontrado")
        
        # Processa a imagem
        img_array = self.image_processor.load_and_preprocess(file_path)
        
        # Realiza a classificação
        result = self.classifier.predict(img_array)
        
        return result


# ============================================================================
# CAMADA DE VISÃO (View Layer)
# ============================================================================

class AlzheimerClassifierView(ctk.CTk):
    """Interface gráfica principal"""
    
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.title("Classificador de Alzheimer - MRI")
        self.geometry("900x700")
        
        # Controlador
        self.controller = AlzheimerClassifierController()
        
        # Variáveis
        self.current_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        
        # Configura tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura a interface do usuário"""
        
        # Frame principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Cabeçalho
        self._create_header()
        
        # Área de conteúdo
        self._create_content_area()
        
        # Rodapé
        self._create_footer()
    
    def _create_header(self):
        """Cria o cabeçalho"""
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title = ctk.CTkLabel(
            header_frame,
            text="🧠 Classificador de Alzheimer",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=10)
        
        subtitle = ctk.CTkLabel(
            header_frame,
            text="Análise de Imagens de Ressonância Magnética",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.pack()
    
    def _create_content_area(self):
        """Cria a área de conteúdo principal"""
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Botão de upload
        upload_btn = ctk.CTkButton(
            content_frame,
            text="📁 Selecionar Imagem MRI",
            command=self._on_upload_click,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        upload_btn.grid(row=0, column=0, pady=20, padx=20, sticky="ew")
        
        # Frame para imagem e resultados
        display_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        display_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        display_frame.grid_columnconfigure((0, 1), weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        
        # Área de visualização da imagem
        image_frame = ctk.CTkFrame(display_frame)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        image_label = ctk.CTkLabel(
            image_frame,
            text="Nenhuma imagem carregada",
            font=ctk.CTkFont(size=14)
        )
        image_label.pack(expand=True)
        self.image_label = image_label
        
        # Área de resultados
        results_frame = ctk.CTkFrame(display_frame)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        results_title = ctk.CTkLabel(
            results_frame,
            text="Resultados da Análise",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_title.pack(pady=20)
        
        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=ctk.CTkFont(size=14),
            wrap="word"
        )
        self.results_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.results_text.insert("1.0", "Aguardando análise...\n\nSelecione uma imagem de ressonância magnética para começar.")
        self.results_text.configure(state="disabled")
        
        # Botão de classificar
        classify_btn = ctk.CTkButton(
            content_frame,
            text="🔍 Classificar Imagem",
            command=self._on_classify_click,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        classify_btn.grid(row=2, column=0, pady=20, padx=20, sticky="ew")
        self.classify_btn = classify_btn
    
    def _create_footer(self):
        """Cria o rodapé"""
        footer_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent", height=40)
        footer_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        
        footer_label = ctk.CTkLabel(
            footer_frame,
            text="⚠️ Este sistema é apenas para fins demonstrativos. Consulte um profissional médico.",
            font=ctk.CTkFont(size=11),
            text_color="orange"
        )
        footer_label.pack()
    
    def _on_upload_click(self):
        """Manipula o clique no botão de upload"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem MRI",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.dcm"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                self._load_and_display_image(file_path)
                self.classify_btn.configure(state="normal")
                self._update_results("Imagem carregada com sucesso!\n\nClique em 'Classificar Imagem' para análise.")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")
    
    def _load_and_display_image(self, file_path: str):
        """Carrega e exibe a imagem"""
        self.current_image = self.controller.load_image(file_path)
        
        # Redimensiona para exibição
        display_size = (350, 350)
        img_copy = self.current_image.copy()
        img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Usa CTkImage em vez de PhotoImage para melhor suporte a HighDPI
        self.photo_image = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=display_size)
        self.image_label.configure(image=self.photo_image, text="")
    
    def _on_classify_click(self):
        """Manipula o clique no botão de classificar"""
        if not self.controller.current_image_path:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro.")
            return
        
        # Desabilita botão durante processamento
        self.classify_btn.configure(state="disabled", text="⏳ Analisando...")
        self._update_results("Processando imagem...\n\nPor favor, aguarde.")
        
        # Executa classificação em thread separada
        thread = threading.Thread(target=self._classify_image_thread)
        thread.daemon = True
        thread.start()
    
    def _classify_image_thread(self):
        """Executa a classificação em thread separada"""
        try:
            if not self.controller.current_image_path:
                raise ValueError("Nenhuma imagem carregada")
            result = self.controller.classify_image(self.controller.current_image_path)
            self.after(0, lambda: self._display_results(result))
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Erro na classificação: {error_msg}")
            self.after(0, lambda msg=error_msg: self._show_error(msg))
    
    def _display_results(self, result: ClassificationResult):
        """Exibe os resultados da classificação"""
        self.classify_btn.configure(state="normal", text="🔍 Classificar Imagem")
        
        # Formata resultados
        status = "POSITIVO" if result.has_alzheimer else "NEGATIVO"
        color = "🔴" if result.has_alzheimer else "🟢"
        confidence_pct = result.confidence * 100
        
        results_text = f"""
{color} DIAGNÓSTICO: {status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Categoria: {result.category}

Confiança: {confidence_pct:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERPRETAÇÃO:

"""
        if result.has_alzheimer:
            results_text += """A análise indica possíveis sinais
compatíveis com Alzheimer nas
imagens de ressonância magnética.

⚠️ IMPORTANTE: Este resultado
requer validação por profissional
médico qualificado."""
        else:
            results_text += """A análise não identificou sinais
significativos de Alzheimer nas
imagens analisadas.

✓ Recomenda-se acompanhamento
médico regular preventivo."""
        
        self._update_results(results_text)
    
    def _update_results(self, text: str):
        """Atualiza o texto dos resultados"""
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")
    
    def _show_error(self, error_msg: str):
        """Exibe mensagem de erro"""
        self.classify_btn.configure(state="normal", text="🔍 Classificar Imagem")
        messagebox.showerror("Erro", f"Erro ao classificar imagem:\n{error_msg}")


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main():
    """Função principal"""
    app = AlzheimerClassifierView()
    app.mainloop()


if __name__ == "__main__":
    main()