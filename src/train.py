import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cats-vs-dogs-classification", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pt")
BATCH_SIZE = 16
EPOCHS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Trening...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for xb, yb in tqdm(train_loader):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader.dataset):.4f} acc={acc:.3f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model zapisany do: {MODEL_PATH}")
