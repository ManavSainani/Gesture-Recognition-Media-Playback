import cv2
import mediapipe as mp
import time
import os

# Media control functions (macOS only)
def play_pause():
    os.system("osascript -e 'tell application \"System Events\" to key code 16 using {command down}'")  # F8

def next_track():
    os.system("osascript -e 'tell application \"System Events\" to key code 17 using {command down}'")  # F9

def previous_track():
    os.system("osascript -e 'tell application \"System Events\" to key code 18 using {command down}'")  # F7

def mute_volume():
    os.system("osascript -e 'set volume output muted not (output muted of (get volume settings))'")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam init
cap = cv2.VideoCapture(0)
last_gesture = None

# Finger detection helper
def fingers_up(hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand.landmark[4].x < hand.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for i in range(1, 5):
        if hand.landmark[tip_ids[i]].y < hand.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Gesture classifier
def classify_gesture(finger_state):
    if finger_state == [0, 0, 0, 0, 0]:
        return "fist"
    elif finger_state == [1, 0, 0, 0, 0]:
        return "thumb_up"
    elif finger_state == [0, 1, 1, 0, 0]:
        return "peace"
    elif finger_state == [0, 1, 0, 0, 0]:
        return "point"
    elif finger_state == [1, 1, 1, 1, 1]:
        return "palm"
    else:
        return "unknown"

# Main loop
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_up(handLms)
            gesture = classify_gesture(fingers)

            if gesture != last_gesture:
                last_gesture = gesture

                if gesture == "thumb_up":
                    play_pause()
                    print("Play/Pause")

                elif gesture == "point":
                    next_track()
                    print("Next Track")

                elif gesture == "fist":
                    previous_track()
                    print("Previous Track")

                elif gesture == "palm":
                    mute_volume()
                    print("Mute Toggle")

    else:
        last_gesture = None  # Reset if no hand detected

    cv2.imshow("Gesture Media Control (macOS)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()