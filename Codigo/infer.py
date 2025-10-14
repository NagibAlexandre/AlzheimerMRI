import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import sys

# --- Config ---
CLASSES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Codigo/model.pth"

# --- Transform (mesmo usado no teste) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Carregar modelo ---
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

# --- Predição ---
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        _, pred = output.max(1)

    return CLASSES[pred.item()]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python infer.py caminho/da/imagem.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    result = predict_image(model, image_path)
    print(f"Predição: {result}")
