from flask import Flask, render_template, request
from bson.binary import Binary
import numpy as np
import os

from utills.database import collection
from utills.preprocess import preprocess_image
from config import IMAGE_SIZE

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("register.html")

# Register Person
@app.route("/register", methods=["POST"])
def register():

    name = request.form["name"]
    age = request.form["age"]
    job = request.form["job"]
    phone = request.form["phone"]
    address = request.form["address"]

    files = request.files.getlist("photos")

    # Auto-generate label
    label = collection.count_documents({})

    # Convert images to binary
    images_binary = [Binary(file.read()) for file in files]

    person = {
        "name": name,
        "age": age,
        "job": job,
        "phone": phone,
        "address": address,
        "label": label,
        "images": images_binary
    }

    collection.insert_one(person)

    return "Person Registered Successfully!"

# Recognition Page
@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html")

# Predict Person
@app.route("/predict", methods=["POST"])
def predict():

    model_path = "model/face_model.h5"

    if not os.path.exists(model_path):
        return "Model not trained yet! Please run train_model.py first."

    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    file = request.files["photo"]
    img = preprocess_image(file.read())

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    person = collection.find_one({"label": int(predicted_label)})

    if person:
        return render_template("result.html", person=person)
    else:
        return "Unknown Person"

if __name__ == "__main__":
    app.run(debug=True)

