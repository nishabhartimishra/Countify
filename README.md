# Countify - Sheet Counting Application

This repository contains the code for Countify, an application designed to automate the counting of sheet stacks in a manufacturing plant using computer vision techniques.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Countify automates the process of counting sheets in a stack from images, helping to eliminate errors and reduce time spent on manual counting. The application uses a Convolutional Neural Network (CNN) model to analyze the images and predict the number of sheets.

## Features

- Image preprocessing for enhanced accuracy.
- Deep learning-based model for accurate sheet counting.
- User-friendly web interface for image upload and result display.
- Optimized for fast processing.

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Flask
- OpenCV
- NumPy
- Pillow

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nishabhartimishra/countify.git
   cd countify
   
2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the application:**
   ```bash
   python app.py

Access the application in your browser at http://127.0.0.1:5000.

## Usage

1. Open the application in your browser.
2. Upload an image of the sheet stack.
3. The application will process the image and display the count of sheets.

## File Structure

countify/
│
├── app.py                   # Main application file
├── cnn_model.py             # Script to train the CNN model
├── static/                  # Images used for UI/UX
├── templates/               # HTML templates
├── sample_image/            # Directory for storing images
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.

## License
This project is licensed under the MIT License.
