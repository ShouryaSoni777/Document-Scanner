import cv2
import numpy as np

img = cv2.imread("white-empty-paper-sheet-with-curl_1284-43065.jpg")

height, width, _ = img.shape

print(width,height)


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
            approx = cv2.approxPolyDP(contour, 0.1*perimeter, True)
            if area>max_area and len(approx) == 4:
                biggest_cnt = approx
                max_area = area
    cv2.drawContours(imgBig_cnt, biggest_cnt, -1, (255,255,0),10)
    return biggest_cnt

imgBig_cnt = img.copy()

def reorder(mypoints):


    mypoints1 = mypoints
    mypoints2 = mypoints
    mypoints_ovr = np.zeros((4,2))

    mypoints_ovr[0] = mypoints[0]
    mypoints_ovr[1] = mypoints[1]
    mypoints_ovr[2] = mypoints1[3]
    mypoints_ovr[3] = mypoints2[2]

    return mypoints_ovr



def warp(img,biggest_cnt):
    if biggest_cnt.size != 0:
        biggest_cnt = reorder(biggest_cnt)
        pts1 = np.float32(biggest_cnt)
        pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output = cv2.warpPerspective(img, matrix, (width,height))
    return output

biggest_cnt = contour(imgThresh)
output = warp(img, biggest_cnt)

cv2.imshow("MWIN",output)
cv2.waitKey(0)