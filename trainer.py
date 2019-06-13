import os
os.system('cls' if os.name == 'nt' else 'clear')


# DEBUG STUFF
from inspect import currentframe, getframeinfo

LOAD_LOG = 0
DATASET_LOG = 1

LIMIT_DATA = None

DEBUG_LOAD = True
DISABLE_LOAD_BAR = False

DEBUG_DATASET = True
DISABLE_DATASET_BAR = False


def log(tag = None, message = None):

    if(tag == None):
        print("Here")
    elif(tag == LOAD_LOG and DEBUG_LOAD):
        print(message)
    elif(tag == DATASET_LOG and DEBUG_DATASET):
        print(message)
    





#Current Line String
# str(getframeinfo(currentframe()).lineno)
#-------------------------------------


import time
import csv
from tqdm import tqdm
import random

import numpy as np
import cv2

from CarSample import CarSample



log(LOAD_LOG, "Loading Data...")

DATA_PATH = '../data/'

lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)

carSamples = []

for line in tqdm(lines[1:LIMIT_DATA], disable=DISABLE_LOAD_BAR):
    sourceCenter = line[0].lstrip()
    sourceLeft = line[1].lstrip()
    sourceRight = line[2].lstrip()
    steering = float(line[3])
    throttle = float(line[4])
    brake = float(line[5])
    speed = float(line[6])

    centerImg = cv2.imread(DATA_PATH + sourceCenter)
    leftImg = cv2.imread(DATA_PATH + sourceLeft)
    rightImg = cv2.imread(DATA_PATH + sourceRight)

    name = sourceCenter.split('/')[-1]
    name = name.split('.')[-2]
    name = name[7:]

    sample = CarSample(centerImg, leftImg, rightImg, steering, throttle, brake, speed, name)
    
    carSamples.append(sample)


random.shuffle(carSamples)










os.system('cls' if os.name == 'nt' else 'clear')

log(DATASET_LOG, "Datasets Setup...")

completeDatasetSize = len(carSamples)

trainSize = int(completeDatasetSize * 0.8)
validSize = completeDatasetSize - trainSize

imageShape = carSamples[0].centerImg.shape

xTrainShape = (trainSize, imageShape[0], imageShape[1], imageShape[2])
xValidShape = (validSize, imageShape[0], imageShape[1], imageShape[2])

xTrainCenter = []
xTrainLeft = []
xTrainRight = []
xTrainSpeed = []

xValidCenter = []
xValidLeft = []
xValidRight = []
xValidSpeed = []

yTrain = []
yValid = []


with tqdm(total=12, disable=DISABLE_DATASET_BAR) as pbar:

    for sample in carSamples[0:trainSize]:
        xTrainCenter.append(sample.centerImg)
        xTrainLeft.append(sample.leftImg)
        xTrainRight.append(sample.rightImg)
        xTrainSpeed.append(sample.speed)
        yTrain.append(sample.steering)

    pbar.update(1)


    for sample in carSamples[trainSize:]:
        xValidCenter.append(sample.centerImg)
        xValidLeft.append(sample.leftImg)
        xValidRight.append(sample.rightImg)
        xValidSpeed.append(sample.speed)
        yValid.append(sample.steering)

    pbar.update(1)

    xTrainCenter = np.array(xTrainCenter)
    pbar.update(1)
    xTrainLeft = np.array(xTrainLeft)
    pbar.update(1)
    xTrainRight = np.array(xTrainRight)
    pbar.update(1)
    xTrainSpeed = np.array(xTrainSpeed)
    pbar.update(1)
    yTrain = np.array(yTrain)
    pbar.update(1)

    xValidCenter = np.array(xValidCenter)
    pbar.update(1)
    xValidLeft = np.array(xValidLeft)
    pbar.update(1)
    xValidRight = np.array(xValidRight)
    pbar.update(1)
    xValidSpeed = np.array(xValidSpeed)
    pbar.update(1)
    yValid = np.array(yValid)
    pbar.update(1)

del carSamples






