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


def recognize(img):
	start = time.time()

	letters = segment(img)
	read(letters)

	end = time.time()
	print(end - start)


# Scales the image to the black areas
def scaleToData(img, emptyColor):
	startCol = 0
	startRow = 0
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

# Segments into seperate letter
def segment(img):

	# Remove noise
	img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
	img = cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)

	# Fill frame
	h, w = img.shape[:2]
	mask = np.zeros((h + 2, w + 2), np.uint8)
	cv2.floodFill(img, mask, (0,0), 0)

	# Split letters
	letterList = []
	(height, width) = np.shape(img)
	rowStart = 0
	for col in np.arange(width):
		rowEmpty = True
		for row in np.arange(height):
			if img[row][col] == 255:
				rowEmpty = False
		if rowEmpty == True: #never true
			letter = imageUntilColumn(img, height, rowStart, col)
			if not np.array_equal(letter,np.array([-1])):
				letter = scaleToData(letter, 0)
				letterList.append(letter)
			rowStart = col
	return letterList

def imageUntilColumn(img, height, rowStart, rowEnd):
	if rowEnd-rowStart > 1 :
		letter = np.split(img.T, [rowStart, rowEnd])[1]
		return letter.T
	return np.array([-1])

# Read the letters
def read(imgs):
	result = []
	characters = np.array(['-', 'B', 'D', 'F', 'D', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z'])
	for img in imgs:
		(width, height) = img.shape


		average = img.mean(axis=0).mean(axis=0)
		if average > 200: #################### adjust variable
			result.append('-')
			continue

		minDifferenceL = (9999, None)
		# Check letters
		for i in range(1, 18):
			template = scaleToData(cv2.imread(f"SameSizeLetters/{i}.bmp", 0), 0)
			difference = getDifference(img, template)
			if difference < minDifferenceL[0]:
				minDifferenceL = (difference, i)

		minDifferenceN = (9999, None)
		# Check numbers
		for i in range(0, 10):
			template = scaleToData(cv2.imread(f"SameSizeNumbers/{i}.bmp", 0), 0)
			difference = getDifference(img, template)
			if difference < minDifferenceN[0]:
				minDifferenceN = (difference, i)

		if minDifferenceL[0] < minDifferenceN[0]:
			character = characters[minDifferenceL[1] + 1]
			result.append(character)
		else:
			result.append(minDifferenceN[1])
	string = ''.join(" ".join(str(x) for x in result))
	print(string)
	return string

# Measures difference between 2 images
def getDifference(img, template):
	(imgHeight, imgWidth) = img.shape
	(temHeight, temWidth) = template.shape

	imgHeight = int(temHeight)
	imgWidth = int(imgWidth*temHeight/imgHeight)
	img = np.resize(img, (imgHeight, imgWidth))

	difference = 0
	for i in np.arange(temHeight):
		for j in np.arange(min(imgWidth, temWidth)):
			#print(j)

			if(template[i][j] != img[i][j]):
				difference = difference + 1
	return difference


# TO RUN CODE
img = cv2.imread('Nummerbord.jpg', 1)
#plates = local.plate_detection(img)
#for plate in plates:
	#recognize(plate)




