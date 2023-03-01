import cv2
import os
import numpy as np
import serial
import time

webcam = cv2.VideoCapture(0)
# arduino = serial.Serial(port='COM9', baudrate=115200, timeout=0.1)
global value
value = 0


'''
def write_data(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.06)
    data = arduino.readline()
    return data

'''
'''
while (1):
    _, imageFrame = webcam.read()
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([133, 90, 110], np.uint8)
    red_upper = np.array([175, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(value == "12"):
            #write_data("0")
            value = "0"
        if (area > 300 and area < 1500):
            #value = write_data("11")
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))


# Program Termination
    cv2.imshow("Window", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        imageFrame.release()
        cv2.destroyAllWindows()
        break

'''

pathImg = r'C:\Languages\python\PBL5\img\meo2.jpg'

img = cv2.imread(pathImg)
cascade = cv2.CascadeClassifier('cat.xml')

meo_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
meos = cascade.detectMultiScale(meo_img)

for(x,y,w,h) in meos:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)
    cv2.putText(img, "cat", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

cv2.imshow('image', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()