import cv2
import numpy as np
import cv2.aruco as aruco
import math

img = cv2.imread('media/CVtask.jpg')
cv2.imshow("img",img)
imgcopy=img.copy()
Ha=cv2.imread("media/Ha.jpg")
HaHa=cv2.imread("media/HaHa.jpg")
LMAO=cv2.imread("media/LMAO.jpg")
XD=cv2.imread("media/XD.jpg")

"""using trackbar we found Green has hue=(32,170),saturation=(73,173),value=(191,225)
                           Peach-pink has hue=(14,25),saturation=(18,67),value=(30,255)
                           orange has hue=(5,255),saturation=(181,255),value=(0,255)
                           Black has hue=(0,255),saturation=(0,255),value=(0,34)"""

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_green = np.array([35,73,191])
lower_pink = np.array([14,18,30])
lower_orange= np.array([5,181,0])
lower_black = np.array([0,0,0])
upper_green = np.array([170,173,225])
upper_pink = np.array([25,67,255])
upper_orange = np.array([255,255,255])
upper_black = np.array([255,255,34])

g = cv2.inRange(hsv,lower_green,upper_green)
p = cv2.inRange(hsv,lower_pink,upper_pink)
o = cv2.inRange(hsv,lower_orange,upper_orange)
b = cv2.inRange(hsv,lower_black,upper_black)
green = cv2.bitwise_and(img,img,mask=g)
pink = cv2.bitwise_and(img,img,mask=p)
orange = cv2.bitwise_and(img,img,mask=o)
black = cv2.bitwise_and(img,img,mask=b)

#removing any noise that's remaining in the image
kernel = np.ones((5,5),np.uint8)
g=cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel,iterations=2)
p=cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel,iterations=3)
o=cv2.morphologyEx(o, cv2.MORPH_OPEN, kernel)
b=cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)



#finding contours of each color from their mask
contourgreen, _ = cv2.findContours(g, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

contourpink, _ = cv2.findContours(p, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

contourorange, _ = cv2.findContours(o, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

contourblack, _ = cv2.findContours(b, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)






#function to find whether a contour is a square and return coordinates of an upright box that bounds them and the angle by which it's rotated
def squares(contours):
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            (x1 ,y1), (w, h),r = cv2.minAreaRect(approx)
            aspectRatio = float(w) / h
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                xd = int(x1 - (w*1.4142135 / 2 * math.cos(math.pi * (-90 + r + 45) / 180)))
                yd = int(y1 - (h *1.4142135/ 2 * math.sin(math.pi * (90 - r + 45) / 180)))
                xu = int(x1 + (w*1.4142135 / 2 * math.cos(math.pi * (-90 + r + 45) / 180)))
                yu = int(y1 + (h*1.4142135/ 2 * math.sin(math.pi * (90 - r + 45) / 180)))
                return [yd, yu, xd, xu,r]



blackbound=squares(contourblack)
greenbound=squares(contourgreen)
orangebound=squares(contourorange)
pinkbound=squares(contourpink)

#finding the width and height of the upright square that bounds our inclined squares
blackw=blackbound[3]-blackbound[2]
blackh=blackbound[1]-blackbound[0]
orangew=orangebound[3]-orangebound[2]
orangeh=orangebound[1]-orangebound[0]
greenw=greenbound[3]-greenbound[2]
greenh=greenbound[1]-greenbound[0]
pinkw=pinkbound[3]-pinkbound[2]
pinkh=pinkbound[1]-pinkbound[0]


#function to detect arucoid and find their angle
def detectArucoid(img,markerSize=5,totalMarkers=250,draw=True):
    imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict=aruco.Dictionary_get(aruco.DICT_5X5_250)
    arucop=aruco.DetectorParameters_create()
    corners,id,r= aruco.detectMarkers(imgGray,arucoDict,parameters=arucop)
    imgID=id[0][0]
    aruangle=-math.degrees(math.atan((corners[0][0][3][1]-corners[0][0][2][1])/(corners[0][0][2][0]-corners[0][0][3][0])))
    return imgID,aruangle


HaId=detectArucoid(Ha)
HaHaId=detectArucoid(HaHa)
LMAOId=detectArucoid(LMAO)
XDId=detectArucoid(XD)
print("HaId="+str(HaId[0]),"HaHaId="+str(HaHaId[0]),"LMAOId="+str(LMAOId[0]),"XDId="+str(XDId[0]))

#warping the aruco markers to match the squares
id3=cv2.resize(Ha,(blackw,blackh))
angle = -blackbound[4]+HaId[1]
rot_point = blackw//2, blackh//2
rot_mat = cv2.getRotationMatrix2D(rot_point,angle,1)
id3 = cv2.warpAffine(id3,rot_mat,(blackw,blackh),borderValue=(255,255,255))

id4=cv2.resize(HaHa,(pinkw,pinkh))
angle4=-HaHaId[1]
rotpoint4=pinkw//2,pinkh//2
rotmat4=cv2.getRotationMatrix2D(rotpoint4,angle4,1)
id4=cv2.warpAffine(id4,rotmat4,(pinkw,pinkh),borderValue=(255,255,255))

id1=cv2.resize(LMAO,(greenw,greenh))
angle1=-greenbound[4]+LMAOId[1]
rotpoint1=greenw//2,greenh//2
rotmat1=cv2.getRotationMatrix2D(rotpoint1,angle1,1)
id1=cv2.warpAffine(id1,rotmat1,(greenw,greenh),borderValue=(255,255,255))

id2=cv2.resize(XD,(orangew,orangeh))
angle2=orangebound[4]+XDId[1]
rotpoint2=orangew//2,orangeh//2
rotmat2=cv2.getRotationMatrix2D(rotpoint2,angle2,1)
id2=cv2.warpAffine(id2,rotmat2,(orangew,orangeh),borderValue=(255,255,255))



#pasting the aruco markers on the squares
img[blackbound[0]:blackbound[1],blackbound[2]:blackbound[3]]=id3
img[pinkbound[0]:pinkbound[1],pinkbound[2]:pinkbound[3]]=id4
img[greenbound[0]:greenbound[1],greenbound[2]:greenbound[3]]=id1
img[orangebound[0]:orangebound[1],orangebound[2]:orangebound[3]]=id2

"""this next bit of code is to bring back the other shapes to their original shape,as they had been ruined by the white borders of the aruco markers.

(the coordinates have been found using ms paint as I realized this method to correct the deformations almost at the last minute. 
 Had I realised it sooner, I would have done the same using the methods available in opencv)
 
 """
img[504:713,976:1248]=imgcopy[504:713,976:1248]
img[486:874,83:273]=imgcopy[486:874,83:273]
img[1030:1158,123:455]=imgcopy[1030:1158,123:455]
img[852:1053,182:401]=imgcopy[852:1053,182:401]
img[797:913,1185:1394]=imgcopy[797:913,1185:1394]




cv2.imshow("final",img)



cv2.waitKey(0)


