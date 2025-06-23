import cv2
import mediapipe as mp

# Initialize
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not working.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Flip and convert
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand skeleton
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get landmark pixel positions
            x_vals = []
            y_vals = []

            for lm in handLms.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                x_vals.append(x)
                y_vals.append(y)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Visualize all points

            # Bounding box
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            print(f"Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")

            # Draw green box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()