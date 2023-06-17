import tkinter as tk
import cv2 as cv
from cv2 import cvtColor
from cv2 import GaussianBlur
from cv2 import BORDER_DEFAULT 
import cvzone


def onclick1():
 img=cv.imread('D:\projects\IMG20220318112523.jpg')
 resized=cv.resize(img,(300,600))
 gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
 cv.imshow('grayscale',gray)

def onclick2():
 img=cv.imread('D:\projects\IMG20220318112523.jpg')
 resized=cv.resize(img,(300,600))
 blur=cv.GaussianBlur(resized,(3,3),cv.BORDER_ISOLATED)
 cv.imshow('blur',blur)

def onclick3():
 img=cv.imread('D:\projects\IMG20220318112523.jpg')
 resized=cv.resize(img,(300,600))
 blur=cv.GaussianBlur(resized,(3,3),cv.BORDER_ISOLATED)
 canny=cv.Canny(blur,125,175)
 cv.imshow('canny',canny)

def onclick4():
 img=cv.imread('D:\projects\IMG20220318112523.jpg')
 resized=cv.resize(img,(300,600))
 blur=cv.GaussianBlur(resized,(3,3),cv.BORDER_ISOLATED)
 canny=cv.Canny(blur,125,175)
 dilated=cv.dilate(canny,(3,3),iterations=3)
 cv.imshow('dilation',dilated)

def onclick5():
 img=cv.imread('D:\projects\IMG20220318112523.jpg')
 resized=cv.resize(img,(300,600))
 blur=cv.GaussianBlur(resized,(3,3),cv.BORDER_ISOLATED)
 canny=cv.Canny(blur,125,175)
 dilated=cv.dilate(canny,(3,3),iterations=3)
 eroded=cv.erode(dilated,(3,3),iterations=3)
 cv.imshow('erode',eroded)

def onclick6():
 import cv2
 import numpy as np
 img1 = cv2.imread('D:\projects\IMG20220318112523.jpg')
 img=cv2.resize(img1,(300,600))
 def cartoonize(img, k):
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
     data = np.float32(img).reshape((-1, 3))
     print("shape of input data: ", img.shape)
     print('shape of resized data', data.shape)
     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
     _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
     center = np.uint8(center)
     result = center[label.flatten()]
     result = result.reshape(img.shape)
     blurred = cv2.medianBlur(result, 3)
     cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
     return cartoon
 cartoonized = cartoonize(img, 8)
 cv2.imshow('input', img)
 cv2.imshow('output',cartoonize)
 
 



 
global num 
def onclick7():
 import cv2 as cv
 import numpy as np
 cam = cv.VideoCapture(0)
 cv.namedWindow("CAMERAAAAA")
 img_counter = 0
 while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("CAMERAAAAA", frame)
    nun=cv.waitKey(1)
    if nun%256 == 27:
        print("Escape hit, closing...")
        break
    elif nun%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
 cv.imshow('ypur img',frame)
 img=cv.imread('D:\projects\opencv_frame_0.png')
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 smoothGrayScale = cv.medianBlur(gray, 5)
 edges = cv.adaptiveThreshold(smoothGrayScale, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 7)
 ReSized4 = cv.resize(edges, (960, 540))
 colorImage = cv.bilateralFilter(frame, 9, 300, 300)
 ReSized5 = cv.resize(colorImage, (960, 540))
 criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
 cartoon = cv.bitwise_and(ReSized5, ReSized5, mask=ReSized4)
 cv.imshow('final',cartoon)
 cv.waitkey(0)
 cam.release()
#cv.imread('D:\projects\opencv_frame_0.png')
#cv.imshow('ypur img',img)
#cv.destroyAllWindows()
 

def onclick8():
 import cv2 as cv
 cam = cv.VideoCapture(0)
 cv.namedWindow("CAMERAAAAA")
 img_counter = 0
 while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("CAMERAAAAA", frame)
    nun=cv.waitKey(1)
    if nun%256 == 27:
        print("Escape hit, closing...")
        break
    elif nun%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
 cv.imshow('ypur img',frame)
 cv.waitkey(0)
 cam.release()  


root=tk.Tk()
root.title("dip")
btn1=tk.Button(root, text="GRAYSCALE",command=onclick1)
btn2=tk.Button(root, text="BLUR",command=onclick2)
btn3=tk.Button(root, text="CANNY",command=onclick3)
btn4=tk.Button(root, text="DILATING",command=onclick4)
btn5=tk.Button(root, text="EROSION",command=onclick5)
btn6=tk.Button(root, text="CARTOONIFY",command=onclick6)
btn7=tk.Button(root, text="CAMERA",command=onclick7)
btn8=tk.Button(root, text="CYI",command=onclick8)


btn1.pack()
btn2.pack()
btn3.pack()
btn4.pack()
btn5.pack()
btn6.pack()
btn7.pack()
btn8.pack()
root.mainloop()
