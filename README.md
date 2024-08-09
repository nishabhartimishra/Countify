Countify - Sheet Counting Application
This repository contains the code for Countify, an application designed to automate the counting of sheet stacks in a manufacturing plant using computer vision techniques.

Table of Contents
Introduction
Features
Requirements
Installation
Usage
File Structure
Contributing
License
Introduction
Countify automates the process of counting sheets in a stack from images, helping to eliminate errors and reduce time spent on manual counting. The application uses a Convolutional Neural Network (CNN) model to analyze the images and predict the number of sheets.

Features
Image preprocessing for enhanced accuracy.
Deep learning-based model for accurate sheet counting.
User-friendly web interface for image upload and result display.
Optimized for fast processing.
Requirements
Python 3.8+
TensorFlow/Keras
Flask
OpenCV
NumPy
Pillow
MySQL (for database support)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/countify.git
cd countify
Set up a virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up MySQL database:

Create a database and update the config.py file with your database credentials.
Run the script db_setup.py to create the necessary tables.
Train the CNN Model (optional):

Place your dataset in the appropriate folder.
Run the training script:
bash
Copy code
python train_model.py
Run the application:

bash
Copy code
python app.py
Access the application in your browser at http://127.0.0.1:5000.
Usage
Open the application in your browser.
Upload an image of the sheet stack.
The application will process the image and display the count of sheets.
File Structure
bash
Copy code
countify/
│
├── app.py                   # Main application file
├── train_model.py           # Script to train the CNN model
├── db_setup.py              # Database setup script
├── static/                  # Static files (CSS, JS)
├── templates/               # HTML templates
├── sample_image/            # Directory for storing images
├── config.py                # Configuration file for database
├── model/                   # Directory for trained models
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.

License
This project is licensed under the MIT License.

