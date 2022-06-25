import cv2
import time
import handdetectionmodule as hdm
import mediapipe
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv2.VideoCapture(0)
width , height = 640,480
cap.set(3,width)
cap.set(4,height)




devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volrange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minvol = volrange[0]
maxvol = volrange[1]




ptime  = 0
vol = 0
volBar = 0
detector = hdm.handDetector()

while True:
    ret,img = cap.read()
    ctime = time.time()

    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(430,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,87,56),3)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)

    img = detector.findHands(img)
    points = detector.pointindexes(img)

    dist = 0

    if len(points)!=0:
        x1 = points[0][4][0]
        x2 = points[0][8][0]
        y1 = points[0][4][1]
        y2 = points[0][8][1]
        cv2.circle(img,(x1,y1),12,(255,0,0),-1)
        cv2.circle(img,(x2,y2),12,(255,0,0),-1)
        cv2.circle(img,((x1+x2)//2,(y1+y2)//2),12,(255,0,0),-1)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        dist = math.hypot(x1-x2,y1-y2)


        vol = np.interp(dist,[50,200],[minvol,maxvol])
        volBar = np.interp(dist,[50,200],[400,150])
        volume.SetMasterVolumeLevel(vol,None)

        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),-1)
        # print(dist)

    cv2.imshow("img",img)
    if cv2.waitKey(2) == ord('q'):
        break 