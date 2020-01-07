import cv2
import numpy as np
import os
import pandas as pd
from Students_upload.Students_upload import Localization
from Students_upload.Students_upload import Recognize
import matplotlib as plt

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
    vidcap = cv2.VideoCapture(file_path)

    # extracting meta information about the video
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # calculating new array length
    lenNew = 0
    if (fps != 0.0):
        lenNew = int(length * (sample_frequency / fps))

    # setting up the array
    emptyFrame = (int, float, np.array((height, width, 3)))
    images = np.array([emptyFrame] * lenNew)

    # inserting images
    counter = 0
    for i in np.arange(int(lenNew)):
        for j in np.arange(int(fps / sample_frequency) - 1):  #skip the next (fps/sampling_frequency) - 1 frames
            counter = counter + 1
            ret, frame = vidcap.read()
        images[i] = (counter, counter / fps, frame)

    return images


def show_images(images):
    for i in images:
        (frameNr, timeStamp, frame) = i
        cv2.imshow('Frame number: %s, TimeStamp: %s' % (frameNr, timeStamp ), frame / 256)
        print(frame)
        cv2.destroyAllWindows()
        cv2.waitKey()
        images = Localization.plate_detection(frame)
        for i, image in enumerate(images):
            name = "image_" + str(i)
            cv2.imshow(name, image)
            #Recognize.recognize(image)


## example for how to use the functions
arr = CaptureFrame_Process('trainingsvideo.avi', 1.0, 'abc')
show_images(arr)