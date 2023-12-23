import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

thumb_tip = 4
index_tip = 8
middle_tip = 12

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            if lm_list[thumb_tip].x < lm_list[thumb_tip - 1].x and \
                    lm_list[index_tip].y < lm_list[index_tip - 2].y and \
                    lm_list[middle_tip].y < lm_list[middle_tip - 2].y:
                cv2.putText(img, "Thumbs down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Thumbs up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

            key = cv2.waitKey(1)
            if key == 32:
                break

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)
