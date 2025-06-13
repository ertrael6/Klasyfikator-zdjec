"""
Pobiera mini-dataset (koty/psy) z HuggingFace (~900 zdjęć, rozpakowuje do data/raw)
"""
import os
from huggingface_hub import snapshot_download

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "raw")
    print("Pobieram mini-dataset kotów/psów (ok. 900 zdjęć)...")
    snapshot_download(repo_id="andfoy/cats-vs-dogs-classification", repo_type="dataset", local_dir=out_dir)
    print(f"Gotowe: {out_dir} (do treningu użyj src/train.py)")

if __name__ == "__main__":
    main()
