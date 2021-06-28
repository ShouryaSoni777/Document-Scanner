import cv2
import numpy as np

img = cv2.imread("anotherpaper.jpg")

# img = cv2.resize(img,(640,480))

def preprocess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgGray,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel,iterations=2)
    imgThresh = cv2.erode(imgDial,kernel,iterations=1)
    return imgThresh

imgThresh = preprocess(img)




def contour(img):
    biggest_cnt = np.array([])
    max_area = 0  
    contours,hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.2*perimeter, True)
            print(len(approx))
            if area>max_area & len(approx) == 4:
                biggest_cnt = approx
                print(biggest_cnt)
                max_area = area
    cv2.drawContours(imgBig_cnt, biggest_cnt, -1, (255,255,0),10)
    return biggest_cnt

imgBig_cnt = img.copy()

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

def warp(img,biggest_cnt):
    if biggest_cnt.size != 0:
        biggest_cnt = reorder(biggest_cnt)
        pts1 = np.float32(biggest_cnt)
        pts2 = np.float32([[0,0], [341,0], [0,148], [341,148]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output = cv2.warpPerspective(img, matrix, (341,148))
    return output


biggest_cnt = contour(imgThresh)
# output = warp(img, biggest_cnt)

cv2.imshow("MWIN",imgBig_cnt)
cv2.waitKey(0)