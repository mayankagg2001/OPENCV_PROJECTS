import cv2
import mediapipe
import handdetectionmodule as hdm
import os
import time
import numpy as np

#############
color = (255,0,255)
xp,yp = 0,0
brushthickness = 15
eraserthickness = 30
#############

videowidth = 1280
videoheight = 720


cap = cv2.VideoCapture(0)
cap.set(3,videowidth)
cap.set(4,videoheight)
ptime = 0


folderPath = "Header Files"

files = os.listdir(folderPath)
overlaylist = []
canvas = np.zeros((720,1280,3),np.uint8)
for filename in files:
    path = folderPath+"/"+filename
    img = cv2.imread(path)
    overlaylist.append(img)

header = overlaylist[0]


detector = hdm.handDetector(detectionCon = 0.8,trackCon = 0.8)


while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)

    img = detector.findHands(img)


    lmlist = detector.pointindexes(img)


    if len(lmlist)!=0:


        x1 , y1 = lmlist[0][8]
        x2 , y2 = lmlist[0][12]
        fingers = detector.checkfingers()
        # print(fingers)

        if y1<125:
            if x1>250 and x1<450:
                header = overlaylist[0]
                color = (255,0,255)
            elif x1>550 and x1<750:
                header = overlaylist[1]
                color = (235,52,58)
            elif x1>800 and x1<950:
                header = overlaylist[2]
                color = (0,255,0)
            elif x1>1050 and x1<1200:
                header = overlaylist[3]
                color = (0,0,0)
        
        
        
        if fingers[1] and fingers[2]:
            # print("selection mode")
            xp,yp = 0,0
            cv2.rectangle(img,(x1,y1-20),(x2,y2+20),color,-1)

        if fingers[2]==False and fingers[1]:
            # print("drawing mode")
            cv2.circle(img,(x1,y1),20,color,-1)
            if xp==0 and yp==0:
                xp,yp = x1,y1
            if color == (0,0,0):
                cv2.line(canvas,(xp,yp),(x1,y1),color,eraserthickness)
            else:
                cv2.line(canvas,(xp,yp),(x1,y1),color,brushthickness)
            xp,yp = x1,y1    


    imgGray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,canvas)
    img[0:125,0:1280] = header


    cv2.imshow("webcam",img)
    cv2.imshow("canvas",imgInv)
    if cv2.waitKey(1)==ord('q'):
        break
