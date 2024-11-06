import cv2

def check_camera():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Capture a single frame
    ret, frame = cap.read()
    
    if ret:
        print("Camera is working. Press any key in the window to exit.")
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
    else:
        print("Error: Frame not captured correctly.")
    
    # Release the camera
    cap.release()

check_camera()
