import cv2
import mediapipe as mp

import time


class handDetector():
    def __init__(self,mode=False,maxHands = 1,detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.points = []
        self.tipindexes = [8,12,16,20]


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw = True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)
        h,w,c = img.shape


        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:

                # for id,lm in enumerate(hand.landmark):
                #     # print(id,lm)
                #     if id%4 == 0:
                #         cv2.circle(img, (int(lm.x*w),int(lm.y*h)), 15, (255,0,0),-1)
                if draw:
                    self.mpDraw.draw_landmarks(img, hand,self.mpHands.HAND_CONNECTIONS)

        return img   


    def pointindexes(self,img):
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # results = self.hands.process(imgrgb)

        h,w,c = img.shape

        self.points = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                pointsinglehand = []
                for lm in hand.landmark:
                    pointsinglehand.append([int(lm.x*w),int(lm.y*h)])
                self.points.append(pointsinglehand)

        return self.points


    def checkfingers(self):

        self.fingers = []
        if len(self.points)!=0:

            if(self.points[0][4][0]<=self.points[0][3][0]):
                self.fingers.append(1)
            else:
                self.fingers.append(0)

            for x in self.tipindexes:
                if(self.points[0][x][1]<self.points[0][x-2][1]):
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)
        return self.fingers



def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret,img = cap.read()
        img = detector.findHands(img)
        # print(results.multi_hand_landmarks)
        h,w,c = img.shape
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255),3)
        cv2.imshow("webcam",img)
        if(cv2.waitKey(1)==ord('q')):
            break


if __name__=="__main__":
    main()