# Facial Emotion Recognition (FER) using Pretrained Mini-Xception Model

Real-time facial emotion recognition using a pretrained Mini-Xception model, built with TensorFlow and OpenCV for webcam-based emotion detection.

## Overview

This project implements real-time Facial Emotion Recognition (FER) using a pretrained Mini-Xception model (`fer2013_mini_XCEPTION.102-0.66.hdf5`) trained on the FER-2013 dataset. It leverages OpenCV for face detection with Haar Cascade and TensorFlow/Keras for emotion prediction from webcam footage. Detected faces are outlined with rectangles, and their predicted emotions are displayed above them in real-time.

This project was developed as part of a Data Science and AI course internship, focusing on Computer Vision (CV) and Deep Learning (DL).

## Features

- Real-time face detection and emotion recognition using a webcam.
- Predicts seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Visualizes results with blue rectangles around faces and emotion labels.

## Technologies Used

- **Python**: 3.12 (required due to TensorFlow compatibility)
- **Libraries**:
    - `opencv-python`: For webcam capture and Haar Cascade face detection.
    - `numpy`: For array manipulation and image preprocessing.
    - `tensorflow`: For loading and running the pretrained Mini-Xception model.
- **IDE**: Visual Studio Code
- **Model**: `fer2013_mini_XCEPTION.102-0.66.hdf5`

## Prerequisites

- Python 3.12 (TensorFlow does not yet support Python 3.13 as of March 2025).
- A working webcam.
- The pretrained model file: `fer2013_mini_XCEPTION.102-0.66.hdf5` (place it in the project directory).
- Virtual environment (recommended).

## Installation

1. **Clone the Repository**:
    ```
    git clone https://github.com/[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
2. **Create a Virtual Environment (Python 3.12)**:
    ```
    py -3.12 -m venv venv
    ```
3. **Activate the Virtual Environment**:

    - On Windows:
        ```
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```
        source venv/bin/activate
        ```
4. **Install Dependencies**:

    - Using `requirements.txt` (recommended):
        ```
        pip install -r requirements.txt
        ```
    - Or manually:
        ```
        pip install opencv-python numpy tensorflow
        ```
## Usage

1. Place the `fer2013_mini_XCEPTION.102-0.66.hdf5` file in the project directory.
2. Run the script:
    ```
    python fer_pretrained.py
    ```
3. A window will open displaying the webcam feed with detected faces and their predicted emotions.
4. Press `q` to exit the application.

## Code Explanation

- **Model Loading**: Loads the pretrained Mini-Xception model and compiles it with the Adam optimizer (`learning_rate=0.0001`).
- **Face Detection**: Uses OpenCV’s Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect faces.
- **Preprocessing**: Converts detected faces to grayscale, resizes to 64x64, and normalizes pixel values for model input.
- **Prediction**: Predicts emotions using the model and displays the highest-probability label above each face.

## Sample Output

- Faces are outlined with blue rectangles.
- Predicted emotions (e.g., `Happy`) are displayed in blue text above each face.
- Console logs the input shape: `(1, 64, 64, 1)`.

## Troubleshooting

- **Python Version**: Use Python 3.12. TensorFlow may fail to install on Python 3.13 (`ERROR: Could not find a version...`).
- **Model File Missing**: Ensure `fer2013_mini_XCEPTION.102-0.66.hdf5` is in the project directory or update the path in the script.
- **Webcam Issues**: Verify your webcam is connected and not in use by another application.
- **VS Code Warning**: If Pylance shows `reportMissingImports` for `tensorflow.keras.models`, select the `venv` interpreter in VS Code (`Ctrl+Shift+P` > `Python: Select Interpreter` > choose `venv\Scripts\python.exe`).

## Acknowledgments

- Built as part of a Data Science and AI internship course.
- Thanks to the FER-2013 dataset contributors, OpenCV, and TensorFlow communities.

## License

This project is licensed under the MIT License - see the LICENSE file for details