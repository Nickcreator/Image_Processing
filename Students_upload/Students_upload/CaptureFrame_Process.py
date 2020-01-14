import cv2
import numpy as np
from Students_upload.Students_upload import Recognize as rec
from Students_upload.Students_upload import Localization as local
from difflib import SequenceMatcher
import time
#from difflib import SequenceMatcher
# from Students_upload.Students_upload import Recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    start = time.time()
    images = capture(file_path, sample_frequency)
    recAndSave(images, save_path)


def capture(file_path, sample_frequency):
    vidcap = cv2.VideoCapture(file_path)

    # extracting meta information about the video
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # calculating new array length
    lenNew = 0
    if (fps != 0.0):
        lenNew = int(length * sample_frequency)

    # setting up the array
    emptyFrame = (int, float, np.array((height, width, 3)))
    images = np.array([emptyFrame] * int(lenNew + 1))

    # inserting images
    counter = 0
    allImages = []
    for i in np.arange(length):
        ret, frame = vidcap.read()
        allImages.append(frame)

    relevantImages = allImages[::int(1/sample_frequency)]
    for i, frame in enumerate(relevantImages):
        images[i] = (i*(1/sample_frequency), (i*(1/sample_frequency)) / fps, frame)

    return images


def recAndSave(images, save_path):
    templatesN = []
    for i in np.arange(10):
        templatesN.append(cv2.imread(f"SameSizeNumbers/{i}.bmp", 0))
    templatesL = []
    for i in range(1, 18):
        templatesL.append(cv2.imread(f"SameSizeLetters/{i}.bmp", 0))
    data = ["License plate,Frame no.,Timestamp(seconds)".split(',')]
    result = ''
    strength = 0
    lastResult = ''
    lastStrength = 0
    for image in images:
        (frameNr, timeStamp, frame) = image
        (a,localised,b) = local.plate_detection(frame)

        for locals in localised:

            recognized = rec.recognize(locals, templatesN, templatesL)
            if recognized is None:
                continue
            if recognized == -1:
                continue  ## Nick do your thing
            result, strength = recognized
            ratio = SequenceMatcher(None, result, lastResult).ratio()
            if ratio < 0.6:
                data.append([str(result), str(frameNr) + ' ', timeStamp])
            elif (ratio > 0.5 and strength > lastStrength):
                data.pop()
                data.append([str(result), str(frameNr) + ' ', timeStamp])
            lastResult, lastStrength = recognized


    import csv
    myFile = open(save_path, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

def reco(image):
    (frameNr, timeStamp, frame) = image
    (gray, localised, thresh_list) = local.plate_detection(frame)
    for i, locals in enumerate(localised):
        recognized = rec.recognize(locals)
        if recognized == -1:
            locals = local.rethreshhold(gray[i], thresh_list[i])
            recognized = rec.recognize(locals)
        if recognized is None:
            continue
        result, strength = recognized
        return [str(result), str(frameNr) + ' ', timeStamp]

def show_images(images):
    for i in images:
        (frameNr, timeStamp, frame) = i
        cv2.imshow('Frame number: %s, TimeStamp: %s' % (frameNr, timeStamp), frame / 256)
        #print(frame)
        cv2.destroyAllWindows()
        cv2.waitKey()
        images = local.plate_detection(frame)
        for i, image in enumerate(images):
            name = "image_" + str(i)
            cv2.imshow(name, image)
            # Recognize.recognize(image)


## example for how to use the functions

