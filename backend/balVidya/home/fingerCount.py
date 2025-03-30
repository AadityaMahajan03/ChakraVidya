import cv2
import mediapipe as mp
import numpy as np
import random


def finger_counter():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    # Define finger tips based on landmark indices
    FINGER_TIPS = [4, 8, 12, 16, 20]

    # Open camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip image and convert color space
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        total_count = 0
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                
                # Get landmarks in a list
                landmarks = hand_landmarks.landmark
                finger_up = []
                
                for tip in FINGER_TIPS:
                    if tip == 4:  # Thumb special case
                        if hand_label == "Right":  # Right hand thumb
                            if landmarks[tip].x < landmarks[tip - 1].x:
                                finger_up.append(1)
                            else:
                                finger_up.append(0)
                        else:  # Left hand thumb (opposite logic)
                            if landmarks[tip].x > landmarks[tip - 1].x:
                                finger_up.append(1)
                            else:
                                finger_up.append(0)
                    else:
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            finger_up.append(1)
                        else:
                            finger_up.append(0)
                
                total_count += sum(finger_up)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                    mp_draw.DrawingSpec(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=5, circle_radius=5),
                                    mp_draw.DrawingSpec(color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=3))
        
        # Display the count with cool effects
        text_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.putText(frame, f'Fingers: {total_count}', (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, text_color, 5)
        
        # Add animated circles for a fun effect
        for _ in range(10):
            x, y = random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0])
            cv2.circle(frame, (x, y), random.randint(5, 15), text_color, -1)
        
        cv2.imshow("Finger Counter - Gamified", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # return success status in json format
    return {"status": "success"}