import cv2
import numpy as np
import time

import Students_upload.Students_upload.Localization as local
import Students_upload.Students_upload.CaptureFrame_Process as cap

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""


def recognize(img):
	start = time.time()

	letters = segment(img)
	string = read(letters)

	end = time.time()
	print('time: ',  end - start)
	return string


# Scales the image to the black areas
def scaleToData(img, emptyColor):
	startCol = 0
	endCol = 0
	startRow = 0
	endRow = 0

	(height, width) = np.shape(img)
	emptyCol = np.array([emptyColor]*height)
	transpose = img.T
	for i in np.arange(width - 1):
		if not np.array_equal(transpose[i], emptyCol):
			startCol = i
			break
	for i in np.arange(width):
		if not (np.array_equal(transpose[width - i - 1], emptyCol)):
			endCol = width - i - 1
			break

	transposeNew = np.split(transpose, [startCol, endCol])[1]
	img = transposeNew.T

	emptyRow = np.array([emptyColor] * (endCol - startCol))
	for i in np.arange(height - 1):
		if not np.array_equal(img[i], emptyRow):
			startRow = i
			break
	for i in np.arange(height):
		if not (np.array_equal(img[height - i - 1], emptyRow)):
			endRow = height - i - 1
			break
	imgNew = np.split(transposeNew.T, [startRow, endRow])[1]
	return imgNew

def segment(img):
	#cv2.imshow("after dilating", img)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	#img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
	#img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)
	h, w = img.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	#cv2.floodFill(img, mask, (0,0), 0)
	#cv2.floodFill(img, mask, (0,h-1), 0)
	#cv2.floodFill(img, mask, (w-1,0), 0)
	#cv2.floodFill(img, mask, (w-1,h-1), 0)
	letterList = []
	# dilation
	#cv2.imshow("after dilating + filling", img)
	#cv2.waitKey()
	#cv2.destroyAllWindows()

	N, regions, stats, centroids = cv2.connectedComponentsWithStats(img)
	img = np.array(img)
	letters = []
	centers = []
	areas = []
	for i in np.arange(N):
		(xStart, yStart, width, height, area) = stats[i]
		letter = img[yStart:(yStart + height), xStart:(xStart + width)]
		if 1.5 > width/height > 0.3 :
			letter = img[yStart:(yStart + height), xStart:(xStart + width)]
			letterList.append((letter, area, centroids[i][0] - centroids[i - 1][0], centroids[i][0]))
			#letterList.append((letter, area, centroids[i][0] - centroids[i-1][0], centroids[i][0]))
			#print(centroids[i][0] - centroids[i-1][0])
		elif np.average(letter.ravel()) > 200:
			letterList.append((None, area, None, centroids[i][0]))

	letterList.sort(key=lambda letter: -letter[1])
	letterList = letterList[0:8]
	letterList.sort(key=lambda letter: letter[3])

	#meanList = []
	#for i in np.arange(len(letterList)):
	#	meanList.append(letterList[i][2])
	#	np.sort(meanList)
	#meanListSorted = np.sort(meanList)
	#thresh = meanListSorted[5]*0.9
	#letters = []
	for i in np.arange(len(letterList)):
		if letterList[i][0] is None:
			letters.append(None)
		else:
			letters.append(letterList[i][0])
	return letters

# Segments into seperate letter
def segment1(img):

	cv2.imshow('img', img)
	cv2.waitKey()

	# Remove noise
	img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
	img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)

	cv2.imshow('after dilating & eroding', img)
	cv2.waitKey()
	# Fill frame
	h, w = img.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	#cv2.floodFill(img, mask, (0,0), 0)
	#cv2.floodFill(img, mask, (0,h-1), 0)
	#cv2.floodFill(img, mask, (w-1,0), 0)
	#cv2.floodFill(img, mask, (w-1,h-1), 0)

	cv2.imshow('after filling frame', img)
	cv2.waitKey()

	# Split letters
	letterList = []
	(height, width) = np.shape(img)
	rowStart = 0
	for col in np.arange(width):
		rowEmpty = True
		for row in np.arange(height):
			if img[row][col] == 255:
				rowEmpty = False
				break
		if rowEmpty == True: #never true
			letter = imageUntilColumn(img, height, rowStart, col)
			if not np.array_equal(letter,np.array([-1])):
				#letter = scaleToData(letter, 0)
				(widthL, heightL) = letter.shape
				if widthL > 0 and heightL > 0:
					letterList.append(letter)
					cv2.imshow('letter', letter)
					cv2.waitKey()
			rowStart = col
	return letterList

def imageUntilColumn(img, height, rowStart, rowEnd):
	if rowEnd-rowStart > 1 :
		letter = np.split(img.T, [rowStart, rowEnd])[1]
		return letter.T
	return np.array([-1])

# Read the letters
def read(letters):


	result = []
	characters = np.array(['-', 'B', 'D', 'F', 'D', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z'])
	for img in letters:
		if img is None:
			result.append('-')
			continue

		'''cv2.imshow('Hello', img)
		cv2.waitKey()
		contours, hierarchy = cv2.findContours(img, 1, 2)
		(x,y),(width,height),theta = cv2.minAreaRect(contours[0])

		rows, cols = img.shape
		rect = cv2.minAreaRect(contours[0])
		center = rect[0]
		angle = rect[2]
		rot = cv2.getRotationMatrix2D(center, angle - 90, 1)
		print(rot)
		img = cv2.warpAffine(img, rot, (rows, cols))
		cv2.imshow('after rotating', img)
		cv2.waitKey()'''


		minDifferenceL = (0, None)
		# Check letters
		for i in range(1, 18):
			#template = scaleToData(cv2.imread(f"SameSizeLetters/{i}.bmp", 0), 0)
			template = cv2.imread(f"SameSizeLetters/{i}.bmp", 0)
			difference = getDifference(img, template)
			if difference > minDifferenceL[0]:
				minDifferenceL = (difference, i)

		minDifferenceN = (0, None)
		# Check numbers
		for i in range(0, 10):
			template = None
			#template = scaleToData(cv2.imread(f"SameSizeNumbers/{i}.bmp", 0), 0)
			template = cv2.imread(f"SameSizeNumbers/{i}.bmp", 0)
			difference = getDifference(img, template)
			cv2.destroyAllWindows()
			if difference > minDifferenceN[0]:
				minDifferenceN = (difference, i)

		if minDifferenceL[0] > minDifferenceN[0]:
			if minDifferenceL[1] == 1: ######## it's a D
				(height, width) = img.shape
				imgSlice = img[:,[int(width/2)]]
				N, regions, stats, centroids  = cv2.connectedComponentsWithStats(imgSlice)
				if N > 3:
					character = 'B'
				else:
					character = 'D'
			else:
				character = characters[minDifferenceL[1] + 1]
			result.append(character)
		else:
			result.append(minDifferenceN[1])
	string = ''.join(" ".join(str(x) for x in result))
	print('result: ', string)
	return string

# Measures difference between 2 images
def getDifference(img, template):

	(imgHeight, imgWidth) = img.shape
	(temHeight, temWidth) = template.shape

	factor = temHeight/imgHeight
	imgHeight = int(imgHeight * factor)
	imgWidth = int(imgWidth * factor)
	img = cv2.resize(img, (imgWidth, imgHeight))

	#cv2.imshow('template', template)
	#cv2.imshow('letter', img)
	#  cv2.waitKey()

	'''difference = 0
	for i in np.arange(temHeight):
		for j in np.arange(min(imgWidth, temWidth)):
			#print(j)

			if(template[i][j] != img[i][j]):
				difference = difference + 1
	return difference'''

	similarity = 0
	for i in np.arange(min(temHeight, imgHeight)):
		for j in np.arange(min(imgWidth, temWidth)):
			# print(j)

			if (template[i][j] == img[i][j]):
				similarity += 1
	return similarity


img = cv2.imread('nl_gl-395-x_template.jpg', 1)
(a, b, img) = cap.CaptureFrame_Process("trainingsvideo.avi", 1, '')[12]
#cv2.imshow('Hello', img)
#cv2.waitKey()
plates = local.plate_detection(img)

for plate in plates:
	print('----------------------', recognize(plate))
	cv2.imshow('Plate', plate)
	cv2.waitKey()





