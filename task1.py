import cv2
import os
import time
from picamera2 import Picamera2

# Directory setup
train_folder = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-12/data/train"
test_folder = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-12/data/test"
cascade_path = "haarcascade_frontalface_default.xml"  # Ensure this file is in your working directory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Initialize the Pi camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

def capture_images(person_id, image_count=60, train_split=50):
    # Define folders for training and testing images
    train_path = os.path.join(train_folder, str(person_id))
    test_path = os.path.join(test_folder, str(person_id))
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    captured_count = 0

    while captured_count < image_count:
        # Capture frame from the camera
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]  # Crop to face
            face_resized = cv2.resize(face, (64, 64))  # Resize to 64x64 pixels

            # Save the image to either train or test folder
            if captured_count < train_split:
                filename = os.path.join(train_path, f"{captured_count}.jpg")
            else:
                filename = os.path.join(test_path, f"{captured_count - train_split}.jpg")

            cv2.imwrite(filename, face_resized)
            print(f"Image saved: {filename}")
            captured_count += 1

            if captured_count >= image_count:
                break

        time.sleep(0.1)  # Optional: Add a delay to manage capture speed

for person_id in [0, 1]:  # Replace with actual IDs if needed
    capture_images(person_id)

# Release resources
picam2.stop()
print("Image capture complete.")
