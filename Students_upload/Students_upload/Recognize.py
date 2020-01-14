import cv2
import numpy as np
import time
import Students_upload.Students_upload.Localization as local

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


def recognize(img, templatesN, templatesL):
	start = time.time()

	letters = segment(img)
	string = read(letters, templatesN, templatesL)

	end = time.time()
	print('time: ',  end - start)
	return string


# Scales the image to the black areas

def segment(img):
	img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)
	img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
	# h, w = img.shape[:2]
	# mask = np.zeros((h + 2, w + 2), np.uint8)
	# cv2.floodFill(img, mask, (0,0), 0)
	# cv2.floodFill(img, mask, (0,h-1), 0)
	# cv2.floodFill(img, mask, (w-1,0), 0)
	# cv2.floodFill(img, mask, (w-1,h-1), 0)
	letterList = []
	# dilation
	# cv2.imshow("after dilating + filling", img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

	N, regions, stats, centroids = cv2.connectedComponentsWithStats(img)
	if 3 < N < 6:
		return -1
	if N < 3:
		return
	img = np.array(img)
	letters = []
	for i in np.arange(N):
		(xStart, yStart, width, height, area) = stats[i]
		# letter = img[yStart:(yStart + height), xStart:(xStart + width)]
		if 1.5 > width / height > 0.3 and area > 50:
			letter = img[yStart:(yStart + height), xStart:(xStart + width)]
			letterList.append([letter, area, (xStart, xStart + width), centroids[i][0], 0])

		# print(centroids[i][0] - centroids[i-1][0])
	# elif np.average(letter.ravel()) > 200:
	#	letterList.append((None, area, None, centroids[i][0]))

	if len(letterList) < 6:
		return
	letterList.sort(key=lambda letter: -letter[1])
	letterList = letterList[0:6]                  ## takes 6 biggest segments
	letterList.sort(key=lambda letter: letter[3])

	for i in range(1, len(letterList)):
		(startLast, endLast) = letterList[i - 1][2]
		(start, end) = letterList[i][2]
		letterList[i][4] = start - endLast       ## distance between letters

	letters = []
	for i in range(6):
		letters.append(letterList[i][0])

	meanList = []
	for i in range(1, 6):
		meanList.append((i, letterList[i][4]))
	meanList.sort(key=lambda key: key[1])
	letters.insert(max(meanList[-1][0], meanList[-2][0]), None)   ## insert None ('-') where distances are biggest
	letters.insert(min(meanList[-1][0], meanList[-2][0]), None)

	return letters


# Segments into seperate letter
# def segment1(img):
# 	cv2.imshow('img', img)
# 	cv2.waitKey()
#
# 	# Remove noise
# 	img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
# 	img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)
#
# 	cv2.imshow('after dilating & eroding', img)
# 	cv2.waitKey()
# 	# Fill frame
# 	# h, w = img.shape[:2]
# 	# mask = np.zeros((h + 2, w + 2), np.uint8)
# 	# cv2.floodFill(img, mask, (0,0), 0)
# 	# cv2.floodFill(img, mask, (0,h-1), 0)
# 	# cv2.floodFill(img, mask, (w-1,0), 0)
# 	# cv2.floodFill(img, mask, (w-1,h-1), 0)
#
# 	cv2.imshow('after filling frame', img)
# 	cv2.waitKey()
#
# 	# Split letters
# 	letterList = []
# 	(height, width) = np.shape(img)
# 	rowStart = 0
# 	for col in np.arange(width):
# 		rowEmpty = True
# 		for row in np.arange(height):
# 			if img[row][col] == 255:
# 				rowEmpty = False
# 				break
# 		if rowEmpty == True:  # never true
# 			letter = imageUntilColumn(img, height, rowStart, col)
# 			if not np.array_equal(letter, np.array([-1])):
# 				# letter = scaleToData(letter, 0)
# 				(widthL, heightL) = letter.shape
# 				if widthL > 0 and heightL > 0:
# 					letterList.append(letter)
# 					cv2.imshow('letter', letter)
# 					cv2.waitKey()
# 			rowStart = col
# 	return letterList


#def imageUntilColumn(img, height, rowStart, rowEnd):
#	if rowEnd - rowStart > 1:
#		letter = np.split(img.T, [rowStart, rowEnd])[1]
#		return letter.T
#	return np.array([-1])


# Read the letters
def read(letters, templatesN, templatesL):
	if letters is None:
		return
	if letters is -1:
		return -1
	resultList = []

	num_letters = 0
	num_numbers = 0

	strength = 0
	temp_list = []
	for (k, img) in enumerate(letters):
		if img is None:
			isLetter = temp_list[0][2]

			##check validity of slice
			for i in temp_list:
				if i[2] != isLetter:
					num = evaluateSlice(temp_list)
					if num != -1:
						temp_list[num] = readLetter(letters[k - len(temp_list) + num], templatesN, templatesL, not temp_list[num][2])

			if len(temp_list) > 3:
				return
			for i in temp_list:
				strength += i[1]
				if i[2]:
					num_letters += 1
				else:
					num_numbers += 1
				resultList.append(i)
			resultList.append(('-', None, None))
			temp_list = []
			continue
		temp = readLetter(img, templatesN, templatesL)

		temp_list.append(temp)

	isLetter = temp_list[0][2]
	for i in temp_list:
		if i[2] != isLetter:
			num = evaluateSlice(temp_list)
			if num != -1:
				temp_list[num] = readLetter(letters[8 - len(temp_list) + num],templatesN,templatesL, not temp_list[num][2])
	for i in temp_list:
		strength += i[1]
		if i[2]:
			num_letters += 1
		else:
			num_numbers += 1
		resultList.append(i)

	if num_numbers > 4 or num_letters > 4:
		return
	string = ''.join("".join(str(x[0]) for x in resultList))
	print('result: ', string)
	return string, strength


def evaluateSlice(slice):
	#print('different values')
	length = len(slice)
	if length == 2:
		minDif = 0
		if slice[1][3] < slice[0][3]:
			minDif = 1
		return minDif
	if length == 3:
		counter = 0
		for i in np.arange(3):
			if slice[i][2]:
				counter += 1
			else:
				counter -= 1
		if counter > 0:
			for i in np.arange(3):
				if not slice[i][2]:
					return i
		else:
			for i in np.arange(3):
				if slice[i][2]:
					return i
	return -1




def readLetter(img, templatesN, templatesL, finalLetter=None):
	characters = np.array(['-', 'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z'])
	(imgHeight, imgWidth) = img.shape
	temHeight = 85

	factor = temHeight / imgHeight
	imgHeight = int(imgHeight * factor)
	imgWidth = int(imgWidth * factor)
	img = cv2.resize(img, (imgWidth, imgHeight))

	minDifferenceL = (0, None)
	minDifferenceN= (0, None)
	tim = 0
	# Check letters
	if finalLetter != False:
		for i in range(0, 17):
			# template = scaleToData(cv2.imread(f"SameSizeLetters/{i}.bmp", 0), 0)
			template = templatesL[i] #cv2.imread(f"SameSizeLetters/{i}.bmp", 0)
			difference = getDifference(img, template)
			if difference > minDifferenceL[0]:
				minDifferenceL = (difference, i+1)

	# Check numbers
	if finalLetter != True:
		for i in range(0, 10):
			# template = scaleToData(cv2.imread(f"SameSizeNumbers/{i}.bmp", 0), 0)
			template = templatesN[i] #cv2.imread(f"SameSizeNumbers/{i}.bmp", 0)
			difference = getDifference(img, template)
			if difference > minDifferenceN[0]:
				minDifferenceN = (difference, i)

	if minDifferenceL[0] > minDifferenceN[0]:
		if minDifferenceL[1] == 1:  ######## it's a D
			(height, width) = img.shape
			imgSlice = img[:, [int(width / 2)]]
			N, regions, stats, centroids = cv2.connectedComponentsWithStats(imgSlice)
			if N > 3:
				character = 'B'
			else:
				character = 'D'
			return (character, minDifferenceL[0], True, minDifferenceL[0] - minDifferenceN[0])  ## True for letter
		else:
			character = characters[minDifferenceL[1]]
			return (character, minDifferenceL[0], True, minDifferenceL[0] - minDifferenceN[0])
	else:
		return (minDifferenceN[1], minDifferenceN[0], False, minDifferenceN[0] - minDifferenceL[0])


# Measures difference between 2 images
def getDifference(img, template):
	(imgHeight, imgWidth) = img.shape
	(temHeight, temWidth) = template.shape

	# factor = temHeight/imgHeight
	# imgHeight = int(imgHeight * factor)
	# imgWidth = int(imgWidth * factor)
	# img = cv2.resize(img, (imgWidth, imgHeight))

	# cv2.imshow('template', template)
	# cv2.imshow('letter', img)
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

# img = cv2.imread('nl_gl-395-x_template.jpg', 1)
# (a, b, img) = cap.CaptureFrame_Process("trainingsvideo.avi", 1, '')[7]
# cv2.imshow('Hello', img)
# cv2.waitKey()
# plates = local.plate_detection(img)

# for plate in plates:
#	print('----------------------', recognize(plate))
#	cv2.imshow('Plate', plate)
#	cv2.waitKey()
