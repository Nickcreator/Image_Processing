import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import PIL.Image, PIL.ImageTk

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""


def get_licence_plate(image):
    im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for i in range(0, len(im_hsv)):
        for j in range(0, len(im_hsv[i])):
            #print(im_hsv[i,j,0])
            if (im_hsv[i, j, 0] > 35 or im_hsv[i,j,0] < 4 or im_hsv[i,j,1] < 45 or im_hsv[i,j,2] < 70):
                im_hsv[i, j, 2] = 0
                #print("new ", im_hsv[i, j, 2])
    im2 = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
    return im2

def plate_detection(im):
    cv2.imshow('BGR image', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    bgr_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2 = get_licence_plate(bgr_im)
    im_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    threshold = 100
    binary_im = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return im_gray

im = cv2.imread('Nummerbord.jpg', cv2.IMREAD_COLOR)
binary_im = plate_detection(im)
cv2.imshow('Binary image', binary_im)
cv2.waitKey()
cv2.destroyAllWindows()
