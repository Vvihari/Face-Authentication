import cv2
import os

# Path to the haarcascade_frontalface_default.xml file
cascade_path = 'haarcascade_frontalface_default.xml'  # Change this to the full path if necessary

# Ensure the cascade file exists
if not os.path.exists(cascade_path):
    raise IOError(f'Cannot find the cascade classifier xml file at {cascade_path}')

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f'Cannot load the cascade classifier xml file from {cascade_path}')

# To capture video stream from webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open webcam.')

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

try:
    while True:
        # Read the frame
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect the faces in the color frame
        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display
        cv2.imshow('Face Detection', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
except KeyboardInterrupt:
    print("Interrupted by user")

# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
