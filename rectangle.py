import cv2
import mediapipe as mp
import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(r"/Users/manavsainani/Desktop/Tate McRae - Just Keep Watching (From F1 The Movie) (Official Video).mp3")
pygame.mixer.music.set_volume(1.0)  # Max volume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
is_playing = False


def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)  # Thumb up
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            finger_state = fingers_up(handLms)

            if finger_state == [1, 0, 0, 0, 0]:  # Thumb up
                if not is_playing:
                    pygame.mixer.music.play()
                    is_playing = True
                    print("Playing")

            elif finger_state == [0, 0, 0, 0, 0]:  # Fist
                if is_playing:
                    pygame.mixer.music.pause()
                    is_playing = False
                    print("Paused")

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()