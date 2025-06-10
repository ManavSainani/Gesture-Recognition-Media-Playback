import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a green rectangle
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)
    cv2.putText(frame, "Test Box", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Test Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()