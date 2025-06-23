import cv2
import mediapipe as mp

# Init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Safety check
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

# MediaPipe Hands
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert to pixel coords
                pixel_coords = []
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    pixel_coords.append((x_px, y_px))

                # Unpack coords
                x_vals = [pt[0] for pt in pixel_coords]
                y_vals = [pt[1] for pt in pixel_coords]

                # Get bounding box corners
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)

                # Print for debug
                print(f"Box: Top-left ({x_min},{y_min}) → Bottom-right ({x_max},{y_max})")

                # DEBUG: draw corner points
                cv2.circle(frame, (x_min, y_min), 8, (0, 0, 255), -1)  # Red dot
                cv2.circle(frame, (x_max, y_max), 8, (255, 0, 0), -1)  # Blue dot

                # Draw rectangle
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        # Show frame
        cv2.imshow("Hand with Box", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()