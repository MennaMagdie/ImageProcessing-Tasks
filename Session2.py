import numpy as np
import cv2 as cv


# Task 1: Generating a gray image from a colored one
def Task1():
    normalPic = cv.imread("yellow.jpg")
    cv.imshow('Before', normalPic)
    
    cv.imshow('After', cv.cvtColor(normalPic, cv.COLOR_BGR2GRAY))
    cv.waitKey(0)


# Task 2: Creating minimum and maximum bounds trackbars that control which color to be extracted on a running video capture
# not exactly mine tho
def Task2():
    camera = cv.VideoCapture(0)

    def nothing(x):
        pass

    cv.namedWindow('marking')

    cv.createTrackbar('H Lower','marking',0,179,nothing)
    cv.createTrackbar('H Higher','marking',179,179,nothing)
    cv.createTrackbar('S Lower','marking',0,255,nothing)
    cv.createTrackbar('S Higher','marking',255,255,nothing)
    cv.createTrackbar('V Lower','marking',0,255,nothing)
    cv.createTrackbar('V Higher','marking',255,255,nothing)


    while(1):
        _,img = camera.read()
        img = cv.flip(img,1)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        hL = cv.getTrackbarPos('H Lower','marking')
        hH = cv.getTrackbarPos('H Higher','marking')
        sL = cv.getTrackbarPos('S Lower','marking')
        sH = cv.getTrackbarPos('S Higher','marking')
        vL = cv.getTrackbarPos('V Lower','marking')
        vH = cv.getTrackbarPos('V Higher','marking')

        LowerRegion = np.array([hL,sL,vL],np.uint8)
        upperRegion = np.array([hH,sH,vH],np.uint8)

        redObject = cv.inRange(hsv,LowerRegion,upperRegion)

        kernal = np.ones((1,1),"uint8")

        red = cv.morphologyEx(redObject,cv.MORPH_OPEN,kernal)
        red = cv.dilate(red,kernal,iterations=1)

        res1=cv.bitwise_and(img, img, mask = red)


        cv.imshow("Masking ",res1)

        if cv.waitKey(10) & 0xFF == ord('q'):
            camera.release()
            cv.destroyAllWindows()
            break

Task1()
Task2()
