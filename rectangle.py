import cv2
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
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

                # Gather all (x, y) points
                coords = []
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    coords.append((x_px, y_px))
                    cv2.circle(frame, (x_px, y_px), 2, (0, 0, 255), -1)

                # DEBUG PRINT
                print("Landmark coordinates:")
                for pt in coords:
                    print(pt)

                if coords:
                    xs = [x for x, _ in coords]
                    ys = [y for _, y in coords]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    print(f"Bounding box: ({x_min},{y_min}) to ({x_max},{y_max})")

                    # Draw bounding box and corner dots
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(frame, (x_min, y_min), 6, (255, 0, 0), -1)   # blue dot
                    cv2.circle(frame, (x_max, y_max), 6, (0, 255, 255), -1) # yellow dot
        else:
            print("No hand detected.")

        cv2.imshow("Hand Bounding Box", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()