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
#def empty():
#    pass

#cv2.namedWindow("Parameters")
#cv2.resizeWindow("Parameters", 640, 240)
#cv2.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
#cv2.createTrackbar("Threshold2", "Parameters", 100, 255, empty)

def stack_images(scale, images):
    rows = len(images)
    cols = len(images[0])
    rowsAvailable = isinstance(images[0], list)
    width = images[0][0].shape[1]
    height = images[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if images[x][y].shape[:2] == images[0][0].shape[:2]:
                    images[x][y] = cv2.resize(images[x][y], (0,0), None, scale, scale)
                else:
                    images[x][y] = cv2.resize(images[x][y], (images[0][0].shape[1], images[0][0].shape[0]), None, scale, scale)
                if len(images[x][y].shape) == 2:
                    images[x][y] = cv2.cvtColor(images[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width))
        hor = [imageBlank]* rows
        hor_con = [imageBlank]* rows
        for x in range(0, rows):
            hor[x] = np.hstack(images[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if images[x].shape[:2] == images[0].shape[:2]:
                images[x] = cv2.resize(images[x], (0,0), None, scale, scale)
            else:
                images[x] = cv2.resize(images[x], (images[0].shape[1], images[0].shape[0]), None, scale, scale)
            if len(images[x].shape) == 2:
                images[x] = cv2.cvtColor(images[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(images)
        ver= hor
    return ver

def gray_stretch(img):
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            a = (255/50)*(img[i,j]-50)
            b = min(a,255)
            img[i,j] = max(b,0)
    return img

def get_licence_plate(image):
    im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    max_color = 0
    for i in range(0, len(im_hsv)):
        for j in range(0, len(im_hsv[i])):
            #print(im_hsv[i,j,0])
            if (im_hsv[i, j, 0] > 37 or im_hsv[i,j,0] < 1 or im_hsv[i,j,1] < 27 or im_hsv[i,j,2] < 20):
                im_hsv[i, j, 2] = 0
            if im_hsv[i,j,2] > max_color:
                max_color = im_hsv[i,j,2]
                #print("new ", im_hsv[i, j, 2])
    im2 = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return im2, max_color

def transform_points(im, approx, targets):
    curdist1 = 9999999
    curdist2 = 9999999
    curdist3 = 9999999
    curdist4 = 9999999
    for a in approx:
        dist1 = calc_distance(targets[0], a)
        if dist1 < curdist1:
            curdist1 = dist1
            point1 = a
        dist2 = calc_distance(targets[1], a)
        if dist2 < curdist2:
            curdist2 = dist2
            point2 = a
        dist3 = calc_distance(targets[2], a)
        if dist3 < curdist3:
            curdist3 = dist3
            point3 = a
        dist4 = calc_distance(targets[3], a)
        if dist4 < curdist4:
            curdist4 = dist4
            point4 = a
    pts1 = np.float32([point1, point2, point3, point4])
    pts2 = np.float32([[0, 0], [512, 0], [0, 110], [532, 110]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (512, 110))
    return result

def calc_distance(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0][0])**2 + (pt1[1] - pt2[0][1])**2)
    return dist

def line_detection(im_gray, im):
    new_im = np.copy(im)
    # Apply edge detection method on the image
    contours1, _ = cv2.findContours(im_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1
    print("len contours: ", len(contours))
    licence_pic = np.zeros(0)
    im_arr = []

    for cnt in contours:
        if (cv2.contourArea(cnt) > 800):
            approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
            print("shape: ", len(approx))
            cv2.drawContours(new_im, [approx], 0, (0, 255, 0), 2)
            #cv2.drawContours(im_gray, [approx], 0, (0, 255, 0), 2)
            licence_pic = np.array([[0, 0], [1, 0]])
            if (len(approx) == 4):
                print("approx", approx)
                x, y, w, h = cv2.boundingRect(approx)
                #cv2.rectangle(new_im, (x, y), (w + x, h + y), (225, 0, 0), 2)
                print("rectangle: ", x, y, w, h)
                licence_pic = new_im[np.maximum(y-15, 0):y+h+15, np.maximum(x-15, 0):x+w+15]
                print(licence_pic)
                cv2.imshow('Licence pic', licence_pic)
                #gray_image = cv2.COLOR_RGB2GRAY(licence_pic)
                targets = []
                targets.append([x,y])
                targets.append([x+w, y])
                targets.append([x, y+h])
                targets.append([x+w, y+h])
                print("targets: ", targets)
                #gray_licence_pic = cv2.COLOR_RGB2GRAY(licence_pic)
                result = transform_points(new_im, approx, targets)
                im_arr.append(result)
                im_arr.append(licence_pic)
                cv2.imshow('Transform', result)
            if (len(approx) == 5):
                approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                licence_pic = new_im[y - 15:y + h + 15, x - 15:x + w + 15]
                im_arr.append(licence_pic)
    # This returns an array of r and theta values
    return im_arr

def interest_detection(im_gray, im):
    # Apply edge detection method on the image
#    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    print("len contours: ", len(contours))

    corners = cv2.goodFeaturesToTrack(im_gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(im, (x,y), 10, (0, 0, 255), 10)
    return im

def plate_detection(im):
    bgr_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im2, max_color = get_licence_plate(bgr_im)
    print(max_color)
    im_blur = cv2.GaussianBlur(im2, (9,9), 1)
    im_gray = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)
    gray_stretch(im_gray)
    #im_gray = cv2.erode(im_gray, np.ones((5, 5), np.uint8), iterations=1)
    #im_gray = cv2.dilate(im_gray, np.ones((5, 5), np.uint8), iterations=1)

    car_binary = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3))
    car_binary_rect = cv2.erode(car_binary[1], kernel, iterations=2)
    im_canny = cv2.Canny(car_binary_rect, 180, 255)
    im_dil = cv2.dilate(im_canny, kernel, iterations=3)
    im_er = cv2.erode(im_dil, kernel, iterations = 3)
    im_dil1 = cv2.dilate(im_er, kernel, iterations=3)
    im_er1 = cv2.erode(im_dil1, kernel, iterations = 3)
    im_dil2 = cv2.dilate(im_er1, kernel, iterations=1)

    rect_detect = line_detection(im_dil2, im2)
    bin_plates = []
    print('Number of licence plates:', len(rect_detect))
    for i, image in enumerate(rect_detect):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bin_plate_low = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY_INV)
        bin_plate_high = cv2.threshold(image_gray, 120, 255, cv2.THRESH_BINARY_INV)
        bin_plates.append(bin_plate_low[1])
        bin_plates.append(bin_plate_high[1])
        name1 = "bin_plate_high" + str(i)
        name2 = "bin_plate_low" + str(i)
        cv2.imshow(name1, bin_plate_low[1])
        cv2.imshow(name2, bin_plate_high[1])
    im_stack = stack_images(0.5, ([im, im_dil, im_dil2],
                                    [im_gray, im2, car_binary_rect]))
    cv2.imshow('Result', im_stack)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return bin_plates


#im = cv2.imread('Nummerbord.jpg', cv2.IMREAD_COLOR)
#binary_im = plate_detection(im)