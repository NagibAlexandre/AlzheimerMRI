import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- Config ---
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODEL_PATH = "Codigo/model.pth"
INFER_FOLDER = "Codigo/infer"

# --- Transform para inferência ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Carregar modelo ---
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --- Função de inferência ---
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_probs = list(zip(CLASSES, probs))
    # Ordena do maior para o menor
    class_probs.sort(key=lambda x: x[1], reverse=True)
    return class_probs

# --- Rodar inferência na pasta ---
if __name__ == "__main__":
    for filename in os.listdir(INFER_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(INFER_FOLDER, filename)
            class_probs = predict(path)
            pred_class, pred_prob = class_probs[0]
            print(f"\nArquivo observado: {filename}")
            print("+----------------------+-------------------------------+")
            print(f"| {'Predição':<20} | {pred_class:<29} |")
            print(f"| {'Probabilidade':<20} | {pred_prob:<29} |")
            print("+----------------------+-------------------------------+")
            print("| Recusado:                                            |")
            for cls, prob in class_probs[1:]:
                print(f"| {cls:<20} | {prob:<29} |")
            print("+------------------------------------------------------+")

