import unittest
import requests
from PIL import Image
import io


class TestClassifyFunction(unittest.TestCase):
    def test_home_page(self):
        """Test if the home page's contents load correctly."""
        response = requests.get("http://127.0.0.1:5000/")
        content = response.text
        self.assertEqual(content, "Welcome to the Home Page.")

    def test_classifier(self):
        """Test the classifier API and its accuracy."""
        with (
            Image.open("data/external/pembroke.jpg") as image,
            io.BytesIO() as buffer
        ):
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            response = requests.post(
                "http://127.0.0.1:5000/classify/imagenet", buffer)
            predicted_labels = response.json()
            self.assertIn("Pembroke, Pembroke Welsh corgi", predicted_labels)

    def test_binary_classifier(self):
        """Test the classifier API and its accuracy."""
        with (
            Image.open("data/external/pembroke.jpg") as image,
            io.BytesIO() as buffer
        ):
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            response = requests.post(
                "http://127.0.0.1:5000/classify/pet-images", buffer)
            predicted_probability = float(response.text)
            self.assertGreaterEqual(0.5, predicted_probability)
            

if __name__ == "__main__":
    unittest.main()
