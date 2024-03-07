import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# Load the pre-trained model
classifier = load_model('Emotion_recognition.h5')

# Define the list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the face detection classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

# Main loop to detect emotions in real-time
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Convert the ROI to a format compatible with the model
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Make a prediction using the model
        prediction = classifier.predict(roi)[0]

        # Get the predicted emotion label
        label = emotion_labels[np.argmax(prediction)]

        # Draw the label on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with the labels
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
