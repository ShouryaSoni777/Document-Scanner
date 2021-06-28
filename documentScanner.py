import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def preProcess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel,iterations=2)
    imgThresh = cv2.erode(imgDial,kernel,iterations=1)
    return imgThresh

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        biggest = np.array([])
        maxarea = 0
        area = cv2.contourArea(contour)
        if area>500:
            # cv2.drawContours(img, contour,-1,(255,255,0),2)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1*peri, True)
            if area>maxarea and len(approx) == 4:
                maxarea = area
                biggest = approx

    cv2.drawContours(imgContour, contour,-1,(255,255,0),2)
    print(biggest)
    return biggest

def reorder(mypoints):
    mypoints.reshape(4,2)
    mypointsNew = np.zeros((4,1,2),np.int32)
    add = mypoints.sum(1)
    
    mypointsNew[0] = mypoints[np.argmin(add)]
    mypointsNew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    mypointsNew[1] = mypoints[np.argmin(diff)]
    mypointsNew[2] = mypoints[np.argmax(diff)]
    return mypointsNew

    

def warp_img(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
    transformed = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img, transformed,(640,480))
    return imgOutput
    
while True:
    ret, frames = cap.read()

    imgContour = frames.copy()

    imgThresh = preProcess(frames)

    biggest = getContours(imgThresh)

    if list(biggest) != [] :
        ovr = warp_img(frames, biggest)
        cv2.imshow("warp",ovr)

    cv2.resize(frames, (640,480))

    cv2.imshow("frames", imgContour)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()