import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


# Load the pre-trained emotion detection model (adjust the path to where you have it stored)
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)


model.compile(
    optimizer=Adam(learning_rate=0.0001), # Use the correct argument learning_rate
    loss='categorical_crossentropy',
    metrics=['accuracy']

    )

# Define emotion labels (as per FER-2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam capture
cap = cv2.VideoCapture(0)

# Initialize the face detector (Haarcascade or Dlib can also be used)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y +h), (255, 0,0),2)

        # Crop the face from the frame
        face = frame[y:y +h, x:x + w]

        # Convert face to grayscale and normalize
        grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(grayscale_face, (64, 64)) # Resize to 64x64
        face_input = np.expand_dims(resized_face, axis =- 1) # Add channel dimension
        face_input = np.expand_dims(face_input, axis=0) # Add batch dimension
        face_input = face_input.astype('float32') / 255.0 # Normalize pixel values (optional)
        emotion_prediction = model.predict(face_input)
        print("Face input shape:", face_input. shape)

        # Should print: (1, 64, 64, 1)

        max_index = np.argmax(emotion_prediction[0])
        predicted_emotion = emotion_labels[max_index]

        # Display the predicted emotion on the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()