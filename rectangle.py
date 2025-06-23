import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
camera = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        success, frame = camera.read()
        if not success:
            continue

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect all landmark points
                h, w, _ = frame.shape
                landmark_array = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # Unzip into x and y coordinate lists
                x_list, y_list = zip(*landmark_array)

                # Calculate bounding box with padding
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                padding = 10  # pixels
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()