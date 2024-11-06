import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from picamera2 import Picamera2
from torchvision.models import mobilenet_v3_small

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = mobilenet_v3_small(num_classes=2)
model.load_state_dict(torch.load('lab8/final_model.pth', map_location=device))
model.to(device)
model.eval()

# Define the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Initialize the Raspberry Pi camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

def process_frame(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    for (x, y, w, h) in faces:
        # Crop the detected face
        face = frame[y:y+h, x:x+w]

        # Convert the cropped face to a PIL image and apply transformations
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Add batch dimension

        # Classify the face
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.softmax(output, dim=1)  # Get probabilities
            confidence, predicted = torch.max(probabilities, 1)

            # Apply a confidence threshold
            if confidence.item() > 0.5:  # Adjust threshold if needed
                label = "Teammate 0" if predicted.item() == 0 else "Teammate 1"
            else:
                label = "Uncertain"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

try:
    print("Press 'q' to quit.")
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Process the frame for face detection and classification
        processed_frame = process_frame(frame)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Live Face Recognition', processed_frame)

        # Wait for key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
