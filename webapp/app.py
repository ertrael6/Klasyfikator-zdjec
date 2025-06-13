
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pt")
LABELS = ["cat", "dog"]

@st.cache_resource
def load_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_img(img):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    xb = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(xb).argmax(1).item()
    return LABELS[pred]

st.title("Cat vs Dog Classifier 🐱🐶")
st.write("Wgraj zdjęcie kota lub psa, a AI powie co widzi.")

img_file = st.file_uploader("Wybierz obrazek JPG/PNG", type=["jpg","jpeg","png"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Wgrany obrazek", width=300)
    label = predict_img(img)
    st.success(f"Predykcja: **{label.upper()}**")
