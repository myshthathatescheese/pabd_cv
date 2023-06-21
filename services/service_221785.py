from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)
imagenet_classifier = tf.keras.applications.resnet.ResNet101()
with open("data/external/imagenet-labels.txt", "r") as f:
    imagenet_classifier_labels = [
        i.rstrip().strip("\"") for i in f.readlines()
        ]
pet_images_classifier = tf.keras.models.load_model("models/binary-classifier")


@app.route("/")
def home_page():
    return "Welcome to the Home Page."


@app.post("/classify/imagenet")
def classify():
    """Receive an image and return its predicted label."""
    data = request.data
    image = tf.io.decode_jpeg(data)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, (224, 224))
    output = imagenet_classifier(image)
    indices = tf.argsort(output, direction="DESCENDING")[0, :3]
    output = [imagenet_classifier_labels[int(i)] for i in indices]
    return output


@app.post("/classify/pet-images")
def binary_classify():
    """Receive an image and return its predicted label."""
    data = request.data
    image = tf.io.decode_jpeg(data)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, (180, 180))
    output = str(float(pet_images_classifier.predict(image)[0]))
    return output
