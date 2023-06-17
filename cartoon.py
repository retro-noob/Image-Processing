
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
img=cv.imread('D:\projects\opencv_frame_0.png')
resized=cv.resize(img,(300,600))
gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow('grayscale',gray)
cv.waitkey(0)
cam.release()
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
 cv2.imshow('output',cartoonized)
 cv2.waitKey(0)
 cv2.destroyAllWindows() 