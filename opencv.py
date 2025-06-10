# Import necessary libraries
import cv2  # For camera access
import mediapipe as mp  # For hand tracking

# Initialize webcam
camera = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module and drawing utility
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create Hands object with parameters
hands = mp_hands.Hands(
    static_image_mode = False,      # Detect hands continuously
    max_num_hands = 1,               # Track only one hand for simplicity (for now, could be changed to 2 hands later on for more hand signs)
    min_detection_confidence = 0.7  # Confidence threshold for detection
)

# Loop to continuously get camera frames
while camera.isOpened():
    # Read camera frame
    #frame = get_current_frame_from_camera()
    ret, frame = camera.read()
    if not ret:
        break  # Exit the loop if the frame couldn't be read

    # Flip the frame horizontally for mirrored view
    flipped_frame = cv2.flip(frame, 1)

    # Draw a static green rectangle to test drawing works
    cv2.rectangle(flipped_frame, (50, 50), (400, 400), (0, 0, 255), 8)
    cv2.putText(flipped_frame, "Test Box", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(rgb_frame)

    print("Hands detected:", bool(result.multi_hand_landmarks))

    # If any hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks (useful visual check)
            mp_drawing.draw_landmarks(
                flipped_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            h, w, _ = flipped_frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Clamp bounding box coordinates to image frame
            #x_min = max(x_min - 20, 0)
            #y_min = max(y_min - 20, 0)
            #x_max = min(x_max + 20, w)
            #y_max = min(y_max + 20, h)

            #print(f"Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")  # Debug print

            x_min = max(x_min - 20, 0)
            y_min = max(y_min - 20, 0)
            x_max = min(x_max + 20, w - 1)
            y_max = min(y_max + 20, h - 1)

            print(f"Final box coords: ({x_min}, {y_min}) to ({x_max}, {y_max})")

            # Draw bounding box rectangle
            #cv2.rectangle(flipped_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(flipped_frame, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 3)

            # Center coordinates for label
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            coord_text = f"({cx}, {cy})"

            cv2.putText(flipped_frame, coord_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # SHow result frame in window
    cv2.imshow("Hand Tracking", flipped_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close any windows
camera.release()
cv2.destroyAllWindows()
