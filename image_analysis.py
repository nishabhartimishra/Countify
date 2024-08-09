import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path):
    try:
        # Load the image
        image = Image.open(image_path)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Use the Canny edge detection method to detect edges
        edges = cv2.Canny(blurred_image, 50, 150)

        # Apply morphological operations to enhance the edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        return image_cv, dilated_edges

    except Exception as e:
        print(f"An error occurred with {image_path}: {e}")
        return None, None


def find_and_filter_contours(dilated_edges, image_cv):
    # Find contours in the preprocessed image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape to isolate the sheet boundaries
    min_contour_length = 100  # Minimum length of contours to be considered as sheet boundaries
    filtered_contours = []

    for contour in contours:
        if cv2.arcLength(contour, True) > min_contour_length:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the contour is roughly horizontal
            if w > h * 2:
                filtered_contours.append(contour)

    # Draw the filtered contours on the original image
    contours_image = image_cv.copy()
    cv2.drawContours(contours_image, filtered_contours, -1, (0, 255, 0), 2)

    return contours_image, len(filtered_contours)


# Path to the folder containing the images
folder_path = "sample_image"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)

        # Preprocess the image
        image_cv, dilated_edges = preprocess_image(image_path)

