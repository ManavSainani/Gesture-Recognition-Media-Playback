import cv2
import mediapipe as mp
import pygame
import time
import os

# Load multiple mp3 files
music_folder = "/Users/manavsainani/Desktop"
playlist = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith(".mp3")]
current_track = 0

# Pygame init
pygame.mixer.init()
pygame.mixer.music.load(playlist[current_track])

# MediaPipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam init
cap = cv2.VideoCapture(0)
is_playing = False
last_gesture = None

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

def play_track(index):
    pygame.mixer.music.load(playlist[index])
    pygame.mixer.music.play()
    print(f"▶️ Playing: {os.path.basename(playlist[index])}")

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

                if gesture == "thumb_up" and not is_playing:
                    pygame.mixer.music.play()
                    is_playing = True
                    print("Play")
                elif gesture == "fist" and is_playing:
                    pygame.mixer.music.pause()
                    is_playing = False
                    print("Pause")
                elif gesture == "peace":
                    pygame.mixer.music.unpause()
                    is_playing = True
                    print("Resume")
                elif gesture == "palm":
                    pygame.mixer.music.stop()
                    is_playing = False
                    print("Stop")
                elif gesture == "point":
                    current_track = (current_track + 1) % len(playlist)
                    play_track(current_track)
                    is_playing = True

    else:
        last_gesture = None  # Reset if no hand detected

    cv2.imshow("Gesture Audio Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()