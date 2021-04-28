from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow.keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# PIL
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Modelo/keras_model.h5'

# Load the model and labels
model = tensorflow.keras.models.load_model(MODEL_PATH)
text_file = open("Modelo/labels.txt", "r") 
labels = text_file.readlines()

print('Model loaded. Check http://127.0.0.1:5000/')

# !Cambio para hacer preciccion directamente
def model_predict(img_path, model):
    image = Image.open(img_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    preds = model.predict(data)
    return preds[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #! Make prediction
        pred = model_predict(file_path, model)
        # print(pred)
        max_value = np.amax(pred)
        max_index = np.where(pred == np.amax(pred))
        # print('Tuple of arrays returned : ', max_index)
        indice = max_index[0][0]
        predText = str(labels[int(indice)]+" = "+str(max_value))
        os.remove(file_path)

        return predText

    return None


if __name__ == '__main__':
    app.run(debug=True)
