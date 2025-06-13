import os
from src.predict import predict_img

def test_cat_image():
    # Użyj prawdziwego testowego zdjęcia lub podmień ścieżkę
    test_img = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cats-vs-dogs-classification", "data", "cat", "cat_1.jpg")
    if os.path.exists(test_img):
        result = predict_img(test_img)
        assert result == "cat" or result == "dog"
    else:
        print("Brak testowego zdjęcia do testu.")

if __name__ == "__main__":
    test_cat_image()
