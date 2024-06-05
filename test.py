import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the trained model
custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
model = load_model('face_mask_detector_model.h5', custom_objects=custom_objects)

# Initialize MTCNN for face detection
detector = MTCNN()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame using MTCNN!
    faces = detector.detect_faces(frame)

    for face in faces:
        # Get the bounding box coordinates
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height

        # Extract the face region
        face_region = frame[y:y2, x:x2]

        # Preprocess the face region
        face_resized = cv2.resize(face_region, (128, 128))  # Resize to match the input shape of the model
        face_normalized = face_resized / 255.0              # Normalize the pixel values
        face_reshaped = np.reshape(face_normalized, (1, 128, 128, 3))  # Reshape for the model input

        # Predict mask or no mask
        prediction = model.predict(face_reshaped)
        (mask, without_mask) = prediction[0]

        # Determine the class label and color
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()