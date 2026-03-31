import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load data
image_folder = 'dataset/images'
label_file = 'dataset/labels/labels.txt'

images = []
labels = []

# Read labels
with open(label_file, 'r') as f:
    for line in f:
        img_name, label = line.strip().split()
        img_path = os.path.join(image_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))

        images.append(img)
        labels.append(int(label))

# Convert to numpy
X = np.array(images) / 255.0
y = np.array(labels)
# Limit dataset to 20,000 samples
X = X[:20000]
y = y[:20000]

# Add channel dimension
X = X.reshape(-1, 128, 128, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# Save model
model.save("brain_tumor_model.h5")                                                                                                                 CNN