import cv2
import numpy as np
import test
import datetime
import keyboard
import os

cap = cv2.VideoCapture(0)


time = datetime.datetime.now().strftime("%I %M")

def takeimg(img):
    cv2.imwrite("imgs\main.jpg",img)
    img = cv2.imread("imgs\main.jpg")
    print("image taken")
    return img
    

while True:
    ret, frames = cap.read()


    none_img = frames.copy()

    biggest = test.contour(frames)

    cv2.drawContours(frames, biggest, -1, (255,255,0),20)
    # cv2.line(frames, biggest[0], biggest[1], (255,255),2)
    # cv2.line(frames, biggest[1], biggest[2], (255,255),2)
    # cv2.line(frames, biggest[2], biggest[3], (255,255),2)
    # cv2.line(frames, biggest[3], biggest[0], (255,255),2)


    cv2.imshow("frames", frames)
    # cv2.imshow("warped", warped)
    
    if keyboard.is_pressed('s'):
        img = takeimg(none_img)
        warped = test.warp(biggest,img)
        os.remove("imgs\main.jpg")

        cv2.imwrite("imgs\Document-Scanner %s.jpg"%time, warped)
        
    
        cv2.imshow("out",warped)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()