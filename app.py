from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('sheet_count_model.keras')


# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/smart_counting', methods=['GET', 'POST'])
def smart_counting():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file:
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            prediction = np.maximum(prediction, 0)
            rounded_prediction = int(np.round(prediction[0][0]))

            return jsonify({'count': rounded_prediction})

    return render_template('smart_counting.html')


if __name__ == '__main__':
    app.run(debug=True)
