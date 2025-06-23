import cv2
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                if coords:
                    x_vals, y_vals = zip(*coords)
                    x_min, x_max = min(x_vals), max(x_vals)
                    y_min, y_max = min(y_vals), max(y_vals)

                    # Print box values
                    print(f"🟩 Box: ({x_min},{y_min}) to ({x_max},{y_max})")

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw debug circles
                    cv2.circle(frame, (x_min, y_min), 6, (255, 0, 0), -1)   # blue = top-left
                    cv2.circle(frame, (x_max, y_max), 6, (0, 255, 255), -1) # yellow = bottom-right

        else:
            print("No hand detected.")

        # Show video
        cv2.imshow("Hand Tracking + Bounding Box", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()