import os
os.system('cls' if os.name == 'nt' else 'clear')

import time
import csv
from tqdm import tqdm

import numpy as np
import cv2

from CarSample import CarSample


print("Loading Data...")


DATA_PATH = '../data/'

lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)


carSamples = []

for line in tqdm(lines[1:]):
    sourceCenter = line[0]
    sourceLeft = line[1]
    sourceRight = line[2]
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

print("Data Loaded")




print("\n\n\n")