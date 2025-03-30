import cv2
import mediapipe as mp
import random

def sense_organ_labeling():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Key landmark indices for sense organs
    SENSE_ORGANS = {
        "Left Eye": (159, (-100, -70)),
        "Right Eye": (386, (100, -70)),
        "Nose": (1, (0, -120)),
        "Mouth": (13, (0, 100)),
        "Lips": (0, (-90, 110)),
        "Left Ear": (234, (-140, 0)),
        "Right Ear": (454, (140, 0)),
        "Chin": (152, (0, 140)),
        "Left Cheek": (50, (-120, 70)),
        "Right Cheek": (280, (120, 70)),
        "Left Eyebrow": (70, (-100, -100)),
        "Right Eyebrow": (300, (100, -100))
    }

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                for label, (idx, (offset_x, offset_y)) in SENSE_ORGANS.items():
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                    bg_color = (0, 0, 0)
                    
                    # Define label position with unique offset
                    label_x, label_y = x + offset_x, y + offset_y
                    
                    # Draw smaller arrow from face landmark to label
                    cv2.arrowedLine(frame, (label_x, label_y), (x, y), color, 1, tipLength=0.2)
                    
                    # Draw filled rectangle as background for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                    cv2.rectangle(frame, (label_x - 5, label_y - text_size[1] - 5), 
                                (label_x + text_size[0] + 5, label_y + 5), bg_color, -1)
                    
                    # Draw text label
                    cv2.putText(frame, label, (label_x, label_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
                    cv2.circle(frame, (x, y), 5, color, -1)  # Smaller dots
        
        cv2.imshow("Sense Organ Labeling App", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"status": "success"}