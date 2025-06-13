import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pt")
LABELS = ["cat", "dog"]

def load_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_img(img_path):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    xb = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(xb).argmax(1).item()
    return LABELS[pred]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="ścieżka do obrazka")
    args = parser.parse_args()
    label = predict_img(args.img)
    print(f"Predykcja: {label}")
