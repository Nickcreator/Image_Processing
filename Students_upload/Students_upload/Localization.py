import cv2
import numpy as np
import math
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
    im_hsv_plate = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    im_hsv_letters = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    max_color = 0
    for i in range(0, len(im_hsv_plate)):
        for j in range(0, len(im_hsv_plate[i])):
            #print(im_hsv[i,j,0])
            if (im_hsv_plate[i, j, 0] > 34 or im_hsv_plate[i,j,0] < 1 or im_hsv_plate[i,j,1] < 60 or im_hsv_plate[i,j,2] < 50):
                im_hsv_plate[i, j, 2] = 0
            if (im_hsv_letters[i, j, 0] > 34 or im_hsv_letters[i, j, 0] < 0 or im_hsv_letters[i, j, 1] < 60 or im_hsv_letters[i, j, 2] < 40):
                im_hsv_letters[i, j, 2] = 0
            if im_hsv_plate[i,j,2] > max_color:
                max_color = im_hsv_plate[i,j,2]
                #print("new ", im_hsv[i, j, 2])
    im_bgr_plate = cv2.cvtColor(im_hsv_plate, cv2.COLOR_HSV2BGR)
    im_bgr_letters = cv2.cvtColor(im_hsv_letters, cv2.COLOR_HSV2BGR)
    return im_bgr_plate, im_bgr_letters, max_color

def order_points(approx, targets):
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
    return pts1

def calc_distance(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0][0])**2 + (pt1[1] - pt2[0][1])**2)
    return dist


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def line_detection(im_gray, im_letters, im_plate):
    new_im_letters = np.copy(im_letters)
    new_im_plate =np.copy(im_plate)
    green_im = np.copy(im_plate)
    biggestw = 0
    # Apply edge detection method on the image
    contours1, _ = cv2.findContours(im_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1
    print("len contours: ", len(contours))
    licence_pic = np.zeros(0)
    im_arr_letters = []
    im_arr_plate = []
    small = False
    for cnt in contours:
        if (cv2.contourArea(cnt) > 1000):
            approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)
            print("shape: ", len(approx))
            cv2.drawContours(green_im, [approx], 0, (0, 255, 0), 2)
            #cv2.drawContours(im_gray, [approx], 0, (0, 255, 0), 2)
            #licence_pic = np.array([[0, 0], [1, 0]])
            if (len(approx) == 4):
                print("approx", approx)
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                print("box", box)
                x, y, w, h = cv2.boundingRect(approx)
                xa, ya, wa, ha = cv2.boundingRect(box)
                cv2.rectangle(green_im, (x, y), (w + x, h + y), (225, 0, 0), 2)
                #cv2.rectangle(green_im, (x, y), (w + x, h + y), (0, 255, 255), 2)
                print("rectangle: x, y, w, h: ", x, y, w, h)
                if w > h:
                    if w > biggestw:
                        w = biggestw
                        if w < 125:
                            small = True
                        else:
                            small = False
                    licence_pic_letters = new_im_letters[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
                    licence_pic_plate = new_im_plate[np.maximum(y-3, 0):y+h+3, np.maximum(x-3, 0):x+w+3]
                    cv2.imshow('Licence pic', licence_pic_letters)
                    cv2.imshow('Detect green', green_im)
                    #gray_image = cv2.COLOR_RGB2GRAY(licence_pic)
                    targets = []
                    targets.append([x,y])
                    targets.append([x+w, y])
                    targets.append([x, y+h])
                    targets.append([x+w, y+h])
                    print("targets: ", targets)
                    #gray_licence_pic = cv2.COLOR_RGB2GRAY(licence_pic)
                    pts1 = order_points(approx, targets)
                    pts2 = np.float32([[0, 0], [512, 0], [0, 110], [512, 110]])
                    matrix3D = cv2.getPerspectiveTransform(pts1, pts2)
                    result3D_plate = cv2.warpPerspective(new_im_plate, matrix3D, (512, 110))
                    result3D_letters = cv2.warpPerspective(new_im_letters, matrix3D, (512, 110))
                    print('pts1', pts1)
                    print('pts1[0]', pts1[0])
                    print('pts1[0][0][0]', pts1[0][0][0])
                    myradians = math.atan2(pts1[3][0][1] - pts1[2][0][1], pts1[3][0][0] - pts1[2][0][0])
                    print(myradians)
                    mydegrees = math.degrees(myradians)
                    rotated_im_letters = rotate_bound(licence_pic_letters, mydegrees)
                    rotated_im_plate = rotate_bound(licence_pic_plate, mydegrees)
                    #M = cv2.getRotationMatrix2D(licence_pic, myradians, 1.0)
                    #cv2.warpAffine(licence_pic, M, )
                    cv2.imshow('Rotated im letters', rotated_im_letters)
                    cv2.imshow('Rotated im plate', rotated_im_plate)
                    #if loops > 0:
                    #    result_bw = transform_points(trans_gray, approx, targets)
                    #    results = line_detection(result_bw, result, loops - 1)
                    #    im_arr.append(results)
                    #else:

                    #im_arr_letters.append(result3D_plate)
                    im_arr_letters.append(rotated_im_letters)
                    im_arr_plate.append(rotated_im_plate)
                    #im_arr_plate.append(result3D_letters)
    # This returns an array of r and theta values
    return im_arr_letters, im_arr_plate, small

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
    im_plate, im_letters, max_color = get_licence_plate(bgr_im)
    print(max_color)
    im_gray = cv2.cvtColor(im_plate, cv2.COLOR_BGR2GRAY)
    gray_stretch(im_gray)
    #im_gray = cv2.erode(im_gray, np.ones((5, 5), np.uint8), iterations=1)
    #im_gray = cv2.dilate(im_gray, np.ones((5, 5), np.uint8), iterations=1)
    cv2.imshow("gray image", im_gray)
    car_binary = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5))
    car_binary_rect = cv2.erode(car_binary[1], kernel, iterations=2)
    im_canny = cv2.Canny(car_binary_rect, 180, 255)
    #im_dil = cv2.dilate(im_canny, kernel, iterations=3)
    #im_er = cv2.erode(im_dil, kernel, iterations = 3)
    #im_dil1 = cv2.dilate(im_er, kernel, iterations=3)
    #im_er1 = cv2.erode(im_dil1, kernel, iterations =3)
    im_dil2 = cv2.dilate(im_canny, kernel, iterations=1)

    rect_detect, plate_detect, small = line_detection(im_dil2, im_letters, im_plate)
    bin_plates = []
    print('Number of licence plates:', len(rect_detect))
    if len(rect_detect) > 0:
        for i, image in enumerate(rect_detect):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray_plate = cv2.cvtColor(plate_detect[i], cv2.COLOR_BGR2GRAY)
            ravel_img = image_gray_plate.ravel()
            zeros = np.count_nonzero(image_gray_plate == 0)
            print("length: ", zeros/len(ravel_img))
            if 100 * zeros/len(ravel_img) < 75:
                name0 = "gray_plate_" + str(i)
                cv2.imshow(name0, image_gray_plate)
                cv2.imshow(name0 + 'letters', image_gray)
                ravel_img = ravel_img[ravel_img != 0]
                #plt.hist(image_gray.ravel(),256,[1,256]); plt.show()
                if small == False:
                    thres = np.median(ravel_img) - 20
                else:
                    thres = np.median(ravel_img)
                thresname = 'Threshold_' + str(i)
                print(thresname, thres)
                bin_plate_low = cv2.threshold(image_gray, thres, 255, cv2.THRESH_BINARY_INV)
                bin_plates.append(bin_plate_low[1])
                name1 = "bin_plate_thres_" + str(i)
                cv2.imshow(name1, bin_plate_low[1])
    im_stack = stack_images(0.5, ([im, im_canny, im_dil2],
                                    [im_plate, im_letters, car_binary_rect]))
    cv2.imshow('Result', im_stack)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return bin_plates


#im = cv2.imread('Nummerbord.jpg', cv2.IMREAD_COLOR)
#binary_im = plate_detection(im)