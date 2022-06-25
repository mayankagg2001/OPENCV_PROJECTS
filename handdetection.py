import cv2
import mediapipe as mp

import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
ptime = 0
ctime = 0


while True:
    ret,img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    # print(results.multi_hand_landmarks)
    h,w,c = img.shape


    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            for id,lm in enumerate(hand.landmark):
                # print(id,lm)
                if id%4 == 0:
                    cv2.circle(img, (int(lm.x*w),int(lm.y*h)), 15, (255,0,0),-1)

            mpDraw.draw_landmarks(img, hand,mpHands.HAND_CONNECTIONS)
        
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255),3)
    cv2.imshow("webcam",img)
    if(cv2.waitKey(1)==ord('q')):
        break



