import os
import numpy as np
import cv2
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model

# Load model once
model = load_model('brain_tumor_model.h5')

def index(request):
    context = {}

    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = fs.path(filename)

        # Image Processing
        img = cv2.imread(file_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.reshape(1, 128, 128, 1)

        # Prediction
        prediction = model.predict(img)

        if prediction > 0.5:
            result = "Tumor Detected "
        else:
            result = "No Tumor "

        context['result'] = result
        context['image_url'] = fs.url(filename)

    return render(request, 'index.html', context)