import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define function to detect gestures
def detect_gesture(landmarks):
    if landmarks is None:
        return "No Hand Detected"

    # Extract key points (thumb, index, middle, ring, pinky fingertips)
    thumb_tip = landmarks[4]  
    index_tip = landmarks[8]  
    middle_tip = landmarks[12]  
    ring_tip = landmarks[16]  
    pinky_tip = landmarks[20]  
    wrist = landmarks[0]  

    # Gesture: "Thumbs Up" (Thumb above wrist, other fingers down)
    if thumb_tip[1] < wrist[1] and all(thumb_tip[1] < tip[1] for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "👍 Thumbs Up!: Good Job"

    # Gesture: "Open Palm" (All fingers stretched)
    if all(landmarks[i][1] < landmarks[i - 2][1] for i in range(8, 21, 4)):  
        return "🖐 Open Palm!: Stop"

    # Gesture: "Pointing Up" (Index finger up, others down)
    if index_tip[1] < wrist[1] and all(index_tip[1] < tip[1] for tip in [middle_tip, ring_tip, pinky_tip]):
        return "☝️ Pointing Up!: One Moment"

    return "🤷 Unknown Gesture"

def manners():
    # Open camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (Mirror effect)
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        # Draw landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw HEROIC Hand (Thicker, glowing effect)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=6),
                                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=3))

                # Convert landmarks into a list of (x, y) coordinates
                landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark]

                # Detect gesture
                gesture_text = detect_gesture(landmarks)

                # Display gesture text with shadow effect
                text_position = (50, 50)
                cv2.putText(frame, gesture_text, (text_position[0] + 2, text_position[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)  # Shadow
                cv2.putText(frame, gesture_text, text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Main Text

        # Show the frame
        cv2.imshow("Heroic Hand Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"status": "success"}