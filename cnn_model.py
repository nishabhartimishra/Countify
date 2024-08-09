import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Define image size
image_height = 128
image_width = 128

# Function to load your dataset with specific labels
def load_your_dataset(folder_path):
    images = []
    labels = []
    label_dict = {
        '1.jpeg': 50,
        '2.jpeg': 29,
        '3.jpeg': 25,
        '4.jpeg': 31,
        '5.jpeg': 37,
        '6.jpeg': 81,
        '7.jpeg': 22,
        '8.jpeg': 51,
        '9.jpeg': 40
    }

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.resize((image_width, image_height))  # Resize images to the desired size
            images.append(np.array(image))
            labels.append(label_dict.get(filename, 0))  # Use 0 if filename not found in label_dict

    return np.array(images), np.array(labels)

# Path to the folder containing the images
folder_path = "sample_image"

# Load and preprocess your dataset
X, y = load_your_dataset(folder_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the image data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # Add Dropout layer
    Dense(1)  # Output layer with one neuron for regression
])



# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Train the model using the data generator
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=50,  # Increased number of epochs
                    validation_data=(X_test, y_test))

# Save the trained model
model.save('sheet_count_model.keras')

# Load the model and recompile it
model = tf.keras.models.load_model('sheet_count_model.keras')
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Assume new_images is a numpy array of new images you want to predict
new_images = X_test  # For demonstration purposes
predictions = model.predict(new_images)

# Ensure predictions are non-negative
predictions = np.maximum(predictions, 0)

# Round predictions to the nearest integer
rounded_predictions = np.round(predictions).astype(int)

# Print rounded predictions
print(rounded_predictions)
