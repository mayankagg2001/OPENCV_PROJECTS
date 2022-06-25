import cv2
import mediapipe
import os
import handdetectionmodule as hm
import time
cap = cv2.VideoCapture(0)

ptime = 0

width , height = 640,480

cap.set(3,width)
cap.set(4,height)

folderPath = "fingers"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    path = folderPath+"/"+imPath
    image = cv2.imread(path)
    overlayList.append(image)

# print(overlayList)

detector = hm.handDetector(detectionCon=0.75)

tipindexes = [8,12,16,20]

while True:
    ret,img = cap.read()
    img = detector.findHands(img)
    points = detector.pointindexes(img)
    # print(points)
    fingers = []
    count = 0
    if len(points)!=0:

        if(points[0][4][0]>points[0][3][0]):
            fingers.append(1)
        else:
            fingers.append(0)

        for x in tipindexes:
            if(points[0][x][1]<points[0][x-2][1]):
                fingers.append(1)
            else:
                fingers.append(0)
        for i in fingers:
            if i==1:
                count+=1
    img[0:200,0:200] = overlayList[count]
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (430,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,78,255),3)
    cv2.putText(img,str(count),(430,470),cv2.FONT_HERSHEY_COMPLEX,2,(255,78,255),3)
    cv2.imshow("img",img)

    if cv2.waitKey(1)==ord('q'):
        break
