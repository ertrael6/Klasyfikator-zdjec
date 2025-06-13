
# Cat vs Dog Classifier

**PyTorch | Web UI (Streamlit) | CLI | Prosty CNN lub ResNet18 | Otwarty mini-dataset**

---

## Demo

![Web demo screenshot](notebook/demo_result.png)

---

## Jak uruchomić?

1. **Pobierz dane:**
   ```
   python data/download_data.py
   ```

2. **Instaluj zależności:**
   ```
   pip install -r requirements.txt
   ```

3. **Trenuj model:**
   ```
   python src/train.py
   ```
   (albo użyj już wytrenowanego modelu z model/model.pt)

4. **CLI predykcja:**
   ```
   python src/predict.py --img example.jpg
   ```

5. **Web UI:**
   ```
   streamlit run webapp/app.py
   ```

---

## Architektura

- PyTorch, ResNet18 (ImageNet-pretrained, fine-tuned)
- Mini-dataset psów i kotów (ok. 1000 zdjęć)
- CLI i web UI do predykcji
- Notebook do eksploracji/wizualizacji

## Inspiracje

- [OpenML cats_dogs](https://www.openml.org/search?type=data&id=40927)
- [HuggingFace cats_vs_dogs_small](https://huggingface.co/datasets/andfoy/cats-vs-dogs-classification)

