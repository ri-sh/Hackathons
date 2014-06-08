#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""-----------------||| hand gesture recognition ||||----------------"""


from pymouse import PyMouse
import numpy as np
import pickle, time, os, threading
import cv2
from numpy import sqrt, arccos, rad2deg

if not cv2.__version__ >= "2.4":
    print "OpenCV version to old!!."
    print "need version >= 2.4."
    exit()
m= PyMouse()#for controlliing the mouse

class Tracking:


    def __init__(self):

        self.debugMode = True
        self.camera = cv2.VideoCapture(1)


        self.camera.set(3,640)#setting the resolution of  cam
        self.camera.set(4,480)#and size and width

        self.posPre = 0

        self.Data = {"angles less 90" : 0,
                     "cursor" : (0, 0),
                     "hulls" : 0,
                     "defects" : 0,
                     "fingers": 0,
                     "fingers history": [0],
                     "area": 0,
                     }
        #
        self.lastData = self.Data

        #
        #
        try:  self.Vars = pickle.load(open(".configuration", "r"))
        except:
            print "Config file («.configuration») not found."
            exit()


        #
        self.addText = lambda image, text, point:cv2.putText(image,text, point, cv2.FONT_HERSHEY_PLAIN, 1.0,(255,255,255))

        #
        while True:
            self.run()
            self.interprete()
            self.updateMousePos()  #
            if self.debugMode:
                if cv2.waitKey(1) == 27: break


    #----------------------------------------------------------------------
    def run(self):
        ret, im = self.camera.read()
        im = cv2.flip(im, 1)
        self.imOrig = im.copy()
        self.imNoFilters = im.copy()#initial image video raw

        #
        im = cv2.blur(im, (self.Vars["smooth"], self.Vars["smooth"]))# before background detection we  are smothening the captured video

        #
        filter_ = self.filterSkin(im)


        filter_ = cv2.erode(filter_,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["erode"], self.Vars["erode"])))


        filter_ = cv2.dilate(filter_,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.Vars["dilate"], self.Vars["dilate"])))



        if self.debugMode: cv2.imshow("Filter Skin", filter_)


        contours, hierarchy = cv2.findContours(filter_,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)




        allIdex = []
        for index in range(len(contours)):
            area = cv2.contourArea(contours[index])
            if area < 5e3: allIdex.append(index)
        allIdex.sort(reverse=True)
        for index in allIdex: contours.pop(index)


        if len(contours) == 0: return

        allIdex = []
        index_ = 0

        for cnt in contours:
            self.Data["area"] = cv2.contourArea(cnt)

            tempIm = im.copy()
            tempIm = cv2.subtract(tempIm, im)


            hull = cv2.convexHull(cnt)
            self.last = None
            self.Data["hulls"] = 0
            for hu in hull:
                if self.last == None: cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                else:
                    distance = self.distance(self.last, tuple(hu[0]))
                    if distance > 40:
                        self.Data["hulls"] += 1

                        cv2.circle(tempIm, tuple(hu[0]), 10, (0,0,255), 5)
                self.last = tuple(hu[0])

            M = cv2.moments(cnt)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            cv2.circle(tempIm, (centroid_x, centroid_y), 20, (0,255,255), 10)
            self.Data["cursor"] = (centroid_x, centroid_y)

            #)
            hull = cv2.convexHull(cnt,returnPoints = False)
            angles = []
            defects = cv2.convexityDefects(cnt,hull)
            if defects == None: return

            self.Data["defects"] = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if d > 1000 :
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    self.Data["defects"] += 1
                    cv2.circle(tempIm,far,5,[0,255,255],-1)  #
                    #
                    cv2.line(tempIm, start, far, [255, 0, 0], 5)
                    cv2.line(tempIm, far, end, [255, 0, 0], 5)
                    #s
                    angles.append(self.angle(far, start, end))

            #.
            b = filter(lambda a:a<90, angles)

            #
            self.Data["angles less 90"] = len(b)
            self.Data["fingers"] = len(b) + 1

            #.
            self.Data["fingers history"].append(len(b) + 1)

            if len(self.Data["fingers history"]) > 10: self.Data["fingers history"].pop(0)
            self.imOrig = cv2.add(self.imOrig, tempIm)

            index_ += 1

        #
        cv2.drawContours(self.imOrig,contours,-1,(64,255,85),-1)


        self.debug()
        if self.debugMode: cv2.imshow("\"Hulk\" Mode", self.imOrig)


    #----------------------------------------------------------------------
    def distance(self, cent1, cent2):
        """Returns  points displacement"""
        x = abs(cent1[0] - cent2[0])
        y = abs(cent1[1] - cent2[1])
        d = sqrt(x**2+y**2)
        return d

    #----------------------------------------------------------------------
    def angle(self, cent, rect1, rect2):

        v1 = (rect1[0] - cent[0], rect1[1] - cent[1])
        v2 = (rect2[0] - cent[0], rect2[1] - cent[1])
        dist = lambda a:sqrt(a[0] ** 2 + a[1] ** 2)
        angle = arccos((sum(map(lambda a, b:a*b, v1, v2))) / (dist(v1) * dist(v2)))
        angle = abs(rad2deg(angle))
        return angle

    #----------------------------------------------------------------------
    def filterSkin(self, im):

        UPPER = np.array([self.Vars["upper"], self.Vars["filterUpS"], self.Vars["filterUpV"]], np.uint8)
        LOWER = np.array([self.Vars["lower"], self.Vars["filterDownS"], self.Vars["filterDownV"]], np.uint8)
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        filter_im = cv2.inRange(hsv_im, LOWER, UPPER)
        return filter_im

    #----------------------------------------------------------------------
    def debug(self):

        yPos = 10
        if self.debugMode: self.addText(self.imOrig, "Debug", (yPos, 20))
        pos = 50
        for key in self.Data.keys():
            if self.debugMode: self.addText(self.imOrig, (key+": "+str(self.Data[key])), (yPos, pos))
            pos += 20

    #----------------------------------------------------------------------
    def updateMousePos(self):
        pos = self.Data["cursor"]
        posPre = self.posPre
        npos = np.subtract(pos, posPre)
        self.posPre = pos

        if self.Data["fingers"] in [1]:
            try: self.t.__stop.set()
            except: pass

            self.t = threading.Thread(target=self.moveMouse, args=(npos))
            self.t.start()

    #----------------------------------------------------------------------
    def interprete(self):


        cont = 3

        if self.Data["fingers history"][:cont] == [5] * cont:
            x,y=m.position()

            m.click(x,y,1)
            self.Data["fingers history"] = [0]

        elif self.Data["fingers history"][:cont] == [3] * cont:
            x,y=m.position()
            m.click(x,y,3)

            self.Data["fingers history"] = [0]

    def moveMouse(self, x, y):

        mini = 10
        mul = 2
        x *= mul
        y *= mul

        posy = lambda n:(y/x) * n
        stepp = 8

        if x > 0:
            pos=m.position()
            for i in range(0, x, stepp): m.move(i, posy(i))
        if x < 0:
            pos=m.position()

            for i in range(x, 0, stepp): m.move(i, posy(i))

        time.sleep(0.2)


if __name__=='__main__':
    Tracking()