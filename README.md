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

