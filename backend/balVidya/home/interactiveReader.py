import cv2
import cvzone
import numpy as np
import time
from cvzone.FaceMeshModule import FaceMeshDetector

def interactiveReader():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    # Alphabet list
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = 0  # Start with 'A'
    start_time = time.time()  # Track time

    while True:
        success, img = cap.read()
        imgText = np.zeros_like(img)  # Blank image for text overlay
        img, faces = detector.findFaceMesh(img, draw=False)

        # Change letter every 3 seconds
        if time.time() - start_time > 3:
            index = (index + 1) % len(alphabet)  # Cycle through letters
            start_time = time.time()  # Reset timer

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3  # Approximate eye distance in cm

            # Finding the distance or depth
            f = 500  # Focal length (arbitrary but consistent)
            d = (W * f) / w
            print(d)

            # Display depth info
            cvzone.putTextRect(img, f'Depth : {int(d)} cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

            # Calculate text position and scaling based on depth
            singleHeight = 20 + int(d / 5)
            scale = 0.4 + (int(d*5 / 10) * 10) / 80

            # Display the current letter
            cv2.putText(imgText, alphabet[index], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4)

        # Stack images for display
        imgStacked = cvzone.stackImages([img, imgText], 2, 1)
        cv2.imshow("Image", imgStacked)

        # Exit on 'q' press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"status": "success"}