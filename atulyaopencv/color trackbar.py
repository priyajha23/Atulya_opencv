
import cv2
import numpy as np

img = cv2.imread('media/CVtask.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img=cv2.resize(img,(512,512))

cv2.namedWindow("trackbar")
cv2.resizeWindow("trackbar",640,240)
def any(x):
    pass
cv2.createTrackbar("hue min","trackbar",0,255,any)
cv2.createTrackbar("hue max","trackbar",255,255,any)
cv2.createTrackbar("sat min","trackbar",0,255,any)
cv2.createTrackbar("sat max","trackbar",255,255,any)
cv2.createTrackbar("val min","trackbar",0,255,any)
cv2.createTrackbar("val max","trackbar",255,255,any)

while True:
    h_min=cv2.getTrackbarPos("hue min","trackbar")
    h_max=cv2.getTrackbarPos("hue max","trackbar")
    sat_min=cv2.getTrackbarPos("sat min","trackbar")
    sat_max=cv2.getTrackbarPos("sat max","trackbar")
    val_min=cv2.getTrackbarPos("val min","trackbar")
    val_max=cv2.getTrackbarPos("val max","trackbar")
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([h_min,sat_min,val_min])
    upper = np.array([h_max,sat_max,val_max])
    mask = cv2.inRange(img_hsv,lower,upper)
    bitand = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("original",img)
    cv2.imshow("hsv",img_hsv)
    cv2.imshow("mask",mask)
    cv2.imshow("segmented",bitand)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()