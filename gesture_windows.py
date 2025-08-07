import cv2
import mediapipe as mp
import keyboard
import time
import numpy as np

# Media control functions
def play_pause():
    keyboard.send("play/pause")

def next_track():
    keyboard.send("next track")

def previous_track():
    keyboard.send("previous track")

def mute_volume():
    keyboard.send("volume mute")

def volume_up():
    keyboard.send("volume up")

def volume_down():
    keyboard.send("volume down")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam init
cap = cv2.VideoCapture(0)
last_gesture = None
prev_y = None
last_volume_adjust_time = time.time()
volume_level = 50  # initial volume (0-100 range for display)

# Finger detection helper
def fingers_up(hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if hand.landmark[tip_ids[i]].y < hand.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Gesture classifier
def classify_gesture(finger_state):
    if finger_state == [0, 0, 0, 0, 0]:
        return "Fist"
    elif finger_state == [1, 0, 0, 0, 0]:
        return "Thumb Up"
    elif finger_state == [0, 1, 1, 0, 0]:
        return "Peace"
    elif finger_state == [0, 1, 0, 0, 0]:
        return "Point"
    elif finger_state == [1, 1, 1, 1, 1]:
        return "Palm"
    elif finger_state[1:] == [1, 1, 1, 1]:
        return "Volume"
    else:
        return "Unknown"

# Draw volume bar
def draw_volume_bar(frame, volume_level):
    x, y, w, h = 50, 100, 30, 300
    vol_height = int(h * (volume_level / 100))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)
    cv2.rectangle(frame, (x, y + h - vol_height), (x + w, y + h), (0, 255, 0), -1)
    cv2.putText(frame, f'{volume_level}%', (x - 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Main loop
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "No Hand"
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(handLms)
            gesture = classify_gesture(fingers)
            gesture_text = gesture

            # Run action if new gesture
            if gesture != last_gesture:
                last_gesture = gesture

                if gesture == "Thumb Up":
                    play_pause()
                elif gesture == "Point":
                    next_track()
                elif gesture == "Fist":
                    previous_track()
                elif gesture == "Palm":
                    mute_volume()

            # Volume Control
            if gesture == "Volume":
                wrist_y = handLms.landmark[0].y
                now = time.time()

                if prev_y is not None and now - last_volume_adjust_time > 0.3:
                    delta = wrist_y - prev_y
                    if delta > 0.03:
                        volume_down()
                        volume_level = max(0, volume_level - 5)
                        last_volume_adjust_time = now
                    elif delta < -0.03:
                        volume_up()
                        volume_level = min(100, volume_level + 5)
                        last_volume_adjust_time = now

                prev_y = wrist_y
    else:
        last_gesture = None
        prev_y = None

    # Draw gesture text & volume bar
    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    draw_volume_bar(frame, volume_level)

    cv2.imshow("Gesture Media Control (Windows)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()