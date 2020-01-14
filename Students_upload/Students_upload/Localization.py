import cv2
import numpy as np
import math
import time
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
# def empty():
#    pass
#
# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 100, 255, empty)

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

# def gray_stretch(img):
#     for i in range(0, len(img)):
#         for j in range(0, len(img[i])):
#             a = (255/50)*(img[i,j]-50)
#             b = min(a,255)
#             img[i,j] = max(b,0)
#     return img

def get_licence_plate(image):
    im_hsv_plate = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i in range(0, len(im_hsv_plate)):
        for j in range(0, len(im_hsv_plate[i])):
#             #print(im_hsv[i,j,0])
            if (im_hsv_plate[i, j, 0] > 27 or im_hsv_plate[i,j,0] < 6 or im_hsv_plate[i,j,1] < 65 or im_hsv_plate[i,j,2] < 85):
                im_hsv_plate[i, j, 2] = 0
#                 #print("new ", im_hsv[i, j, 2])
    im_bgr_plate = cv2.cvtColor(im_hsv_plate, cv2.COLOR_HSV2BGR)
    return im_bgr_plate

def get_plate_thres(image):
    im_hsv_letters = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i in range(0, len(im_hsv_letters)):
        for j in range(0, len(im_hsv_letters[i])):
#             #print(im_hsv[i,j,0])
            if (im_hsv_letters[i, j, 0] > 34 or im_hsv_letters[i, j, 0] < 0 or im_hsv_letters[i, j, 1] < 60 or im_hsv_letters[i, j, 2] < 40):
                im_hsv_letters[i, j, 2] = 0
#                 #print("new ", im_hsv[i, j, 2])
    im_bgr_letters = cv2.cvtColor(im_hsv_letters, cv2.COLOR_HSV2BGR)
    return im_bgr_letters

def order_points(approx, x, w):
    max_left = 0
    max_right = 0
    p1 = [[0, 0]]
    p2 = [[0, 0]]
    p3 = [[0, 0]]
    p4 = [[0, 0]]
    for a in approx:
        if a[0][0] < x + w * 0.5:
            if a[0][1] > max_left:
                max_left = a[0][1]
                p1 = p3
                p3 = a
            else:
                p1 = a
        else:
            if a[0][1] > max_right:
                max_right = a[0][1]
                p2 = p4
                p4 = a
            else:
                p2 = a
    pts1 = np.float32([p1, p2, p3, p4])
#     # print('pts1', pts1)
    return pts1

def calc_distance(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0][0])*2 + (pt1[1] - pt2[0][1])*2)
    return dist


def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def line_detection(im_canny, im_plate, original):
    w = 0
    green_im = cv2.cvtColor(im_plate, cv2.COLOR_GRAY2BGR)
    # Apply edge detection method on the image
    contours, _ = cv2.findContours(im_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("len contours: ", len(contours))
    im_arr_letters = []
    im_arr_plate = []
    arrxy = []
    for cnt in contours:
#         # print(cv2.contourArea(cnt))
        if (cv2.contourArea(cnt) > 230):
            approx = cv2.approxPolyDP(cnt, 0.030 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(green_im, [approx], 0, (0, 255, 0), 2)
            #licence_pic = np.array([[0, 0], [1, 0]])
            # print('shape:', approx)
            if (len(approx) == 4):
#                 # print("approx", approx)
                x, y, w, h = cv2.boundingRect(approx)
                rec = True
                for t in arrxy:
                    # print(t[0] - x)
                    if abs(t[0] - x) < 5:
                        rec = False
                if rec == True:
                    cv2.rectangle(green_im, (x, y), (w + x, h + y), (225, 0, 0), 2)
#                     #cv2.imshow('Detect green', green_im)
                # cv2.rectangle(green_im, (x, y), (w + x, h + y), (0, 255, 255), 2)
#                 # print("rectangle: x, y, w, h: ", x, y, w, h)
                    if w > h:
                        arrxy.append([x,y])
                        original_letters = original[np.maximum((y-3)*4, 0):(y+h+3)*4, np.maximum((x-3)*4, 0):(x+w+3)*4]
                        #licence_pic_letters = im_letters[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
                        licence_pic_plate = im_plate[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
#                         # cv2.imshow('Licence pic', licence_pic_letters)
                        # gray_image = cv2.COLOR_RGB2GRAY(licence_pic)
                        # targets = []
                        # targets.append([x,y])
                        # targets.append([x+w, y])
                        # targets.append([x, y+h])
                        # targets.append([x+w, y+h])
#                         #print("targets: ", targets)
                        #gray_licence_pic = cv2.COLOR_RGB2GRAY(licence_pic)
                        pts1 = order_points(approx, x, w)
                        # pts3 = order_points2(approx, x, w, targets)
#                         # print('pts3', pts3)
                        # pts2 = np.float32([[0, 0], [512, 0], [0, 110], [512, 110]])
                        # matrix3D = cv2.getPerspectiveTransform(pts1, pts2)
                        # result3D_plate = cv2.warpPerspective(licence_pic_letters, matrix3D, (512, 110))
#                         # cv2.imshow('result3d', result3D_plate)
                        # result3D_letters = cv2.warpPerspective(new_im_letters, matrix3D, (512, 110))
                        myradians = math.atan2(pts1[3][0][1] - pts1[2][0][1], pts1[3][0][0] - pts1[2][0][0])
                        mydegrees = math.degrees(myradians)
                        # print('degrees:', mydegrees)
                        if abs(mydegrees) > 4:
                            #licence_pic_letters = rotate(licence_pic_letters, mydegrees)
                            licence_pic_plate = rotate(licence_pic_plate, mydegrees)
                            original_letters = rotate(original_letters, mydegrees)
#                         # cv2.imshow('Rotated im letters', rotated_im_letters)
#                         # cv2.imshow('Rotated im plate', rotated_im_plate)
                        im_arr_letters.append(original_letters)
                        im_arr_plate.append(licence_pic_plate)
                        #im_arr_plate.append(result3D_letters)
            elif (len(approx) == 5):
                x, y, w, h = cv2.boundingRect(approx)
                rec = True
                for t in arrxy:
                    # print(t[0] - x)
                    if abs(t[0] - x) < 5:
                        rec = False
                if rec == True:
                #cv2.rectangle(green_im, (x, y), (w + x, h + y), (225, 0, 0), 2)
                # cv2.rectangle(green_im, (x, y), (w + x, h + y), (0, 255, 255), 2)
#                 # print("rectangle: x, y, w, h: ", x, y, w, h)
                    if w > h:
                        arrxy.append([x,y])
                        original_letters = original[np.maximum((y-3)*4, 0):(y+h+3)*4, np.maximum((x-3)*4, 0):(x+w+3)*4]
                        #licence_pic_letters = im_letters[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
                        licence_pic_plate = im_plate[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
                        im_arr_letters.append(original_letters)
                        im_arr_plate.append(licence_pic_plate)
        # This returns an array of r and theta values
#     # print('arrxy', arrxy)
    return im_arr_letters, im_arr_plate, w

def scaleimage(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def plate_detection(im):
    start = time.time()
    color = [0,0,0]
    top, bottom, left, right = [3] * 4
    img_with_border = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    bgr_im = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2RGB)
    resized = scaleimage(bgr_im, 25)
#     # cv2.imshow('resized image', resized)
    im_plate = get_licence_plate(resized)
    biggerst = scaleimage(resized, 400)
#     #cv2.imshow('big', biggerst)
    im_gray_plate = cv2.cvtColor(im_plate, cv2.COLOR_BGR2GRAY)
#     #cv2.imshow('gray', im_gray_plate)
    blur = cv2.blur(im_gray_plate, (2, 2))
    car_binary = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((6, 6))
    car_binary_rect = cv2.erode(car_binary[1], kernel, iterations=1)
    im_canny = cv2.Canny(car_binary_rect, 180, 255, 2)
#     #cv2.imshow('canny', im_canny)
    # im_dil2 = cv2.dilate(im_canny, kernel, iterations=1)

    letters_detect, plate_detect, size = line_detection(im_canny, im_gray_plate, bgr_im)
    bin_plates = []
    # res = cv2.bitwise_and(maskim, bgr_im, mask=maskim)
#     # cv2.imshow('res', res)
    if len(letters_detect) > 0:
        for i, image in enumerate(letters_detect):
            ravel_img = plate_detect[i].ravel()
            zeros = np.count_nonzero(plate_detect[i] == 0)
#             # print("length: ", zeros/len(ravel_img))
            if 100 * zeros/len(ravel_img) < 85:
                name0 = 'gray_plate_' + str(i)
                # cv2.imshow(name0 + 'plate', plate_detect[i])
                # cv2.imshow(name0 + 'letters', letters_detect[i])
                ravel_img = ravel_img[ravel_img != 0]
                #plt.hist(image_gray.ravel(),256,[1,256]); plt.show()
                correction = min(size*0.40, 31)
                thres = np.median(ravel_img) - correction
                # print(correction)
                # thresname = 'Threshold_' + str(i)
#                 # print(thresname, thres)
                # det1 = "letters_detect_" + str(i)
#                 # cv2.imshow(det1, letters_detect[i])
                return_image = get_plate_thres(letters_detect[i])
                gray_letters = cv2.cvtColor(return_image, cv2.COLOR_BGR2GRAY)
                bin_plate_low = cv2.threshold(gray_letters, thres, 255, cv2.THRESH_BINARY_INV)
                bin_plates.append(bin_plate_low[1])
                name1 = "bin_plate_thres_" + str(i)
                # cv2.imshow(name1, bin_plate_low[1])
    # im_stack = stack_images(0.5, ([img_with_border, im_canny, im_canny],
    #                                  [im_plate, im_letters, car_binary[1]]))
#     # cv2.imshow('Result', im_stack)
    end = time.time()
    # print('time', end - start)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return bin_plates


#im = cv2.imread('Nummerbord.jpg', cv2.IMREAD_COLOR)
#binary_im = plate_detection(im)