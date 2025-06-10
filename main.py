import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            fingers = []
            # Thumb
            fingers.append(1 if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1] else 0)
            # Other fingers
            for id in range(1, 5):
                fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)

            total_fingers = fingers.count(1)

            if total_fingers == 1:
                pyautogui.press('playpause')
            elif total_fingers == 2:
                pyautogui.press('nexttrack')
            elif total_fingers == 3:
                pyautogui.press('prevtrack')
            elif total_fingers == 4:
                pyautogui.press('volumeup')
            elif total_fingers == 5:
                pyautogui.press('volumedown')

    cv2.imshow("Media Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if results.multi_hand_landmarks:
        print("Hand detected!")
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        print("No hand detected.")

cap.release()
cv2.destroyAllWindows()