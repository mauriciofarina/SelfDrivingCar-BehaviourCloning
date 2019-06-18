import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.system('cls' if os.name == 'nt' else 'clear')

# DEBUG STUFF
from inspect import currentframe, getframeinfo

LIMIT_DATA = 10


#Current Line String
# str(getframeinfo(currentframe()).lineno)
#-------------------------------------

import time
import csv
from tqdm import tqdm
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

from CarSample import CarSample

from keras.models import Sequential, Input, Model
from keras.layers import concatenate, Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.utils import plot_model


os.system('cls' if os.name == 'nt' else 'clear')





STEPS = 3



print("---Behaviour Clonning Trainer Application---")

#Load Data

DATA_PATH = '../data/'

lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)

carSamples = []

for line in tqdm(lines[1:LIMIT_DATA], desc="  (1/" +str(STEPS) + ") Loading Data"):
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



#Data Augmentation

carSamplesAugmented = []

for sample in tqdm(carSamples[:], desc="  (2/" +str(STEPS) + ") Augmenting Data"):

        flippedCenter = np.fliplr(sample.centerImg)
        flippedRight = np.fliplr(sample.leftImg)
        flippedLeft = np.fliplr(sample.rightImg)
        flippedSteering = -sample.steering

        flippedSample = CarSample(
                                    flippedCenter,
                                    flippedLeft,
                                    flippedRight,
                                    flippedSteering,
                                    sample.throttle,
                                    sample.brake,
                                    sample.speed,
                                    sample.name + "FLIPPED"
                                )


        carSamplesAugmented.append(sample)
        carSamplesAugmented.append(flippedSample)

        

del carSamples






# Convert to Dataset

datasetSize = len(carSamplesAugmented)

imageShape = carSamplesAugmented[0].centerImg.shape


xTrainCenter = []
xTrainLeft = []
xTrainRight = []
xTrainSpeed = []
yTrain = []



with tqdm(total=6, desc="  (3/" +str(STEPS) + ") Converting to Dataset") as pbar:

    for sample in carSamplesAugmented[:]:
        xTrainCenter.append(sample.centerImg)
        xTrainLeft.append(sample.leftImg)
        xTrainRight.append(sample.rightImg)
        xTrainSpeed.append(sample.speed)
        yTrain.append(sample.steering)

    pbar.update()


    xTrainCenter = np.array(xTrainCenter)
    pbar.update()
    xTrainLeft = np.array(xTrainLeft)
    pbar.update()
    xTrainRight = np.array(xTrainRight)
    pbar.update()
    xTrainSpeed = np.array(xTrainSpeed)
    pbar.update()
    yTrain = np.array(yTrain)
    pbar.update()


del carSamplesAugmented








# Preprocess Dataset
















print('\n\n\n')

#Neural Network Model

camCenter = Input(shape=imageShape, name="Center")
camLeft = Input(shape=imageShape, name="Left")
camRight = Input(shape=imageShape, name="Right")

cropC = Cropping2D(cropping=((50,20), (0,0)))(camCenter)
cropL = Cropping2D(cropping=((50,20), (0,0)))(camLeft)
cropR = Cropping2D(cropping=((50,20), (0,0)))(camRight)

#input_shape=(160,320,3)
lamC = Lambda(lambda x: ((x -128.0) / 128.0))(cropC)
lamL = Lambda(lambda x: ((x -128.0) / 128.0))(cropL)
lamR = Lambda(lambda x: ((x -128.0) / 128.0))(cropR)


conv1 = Conv2D(32, (3, 3), name="Conv1")(lamC)
act1 = Activation('relu', name="act1")(conv1)

conv2 = Conv2D(32, (3, 3), name="Conv2")(lamL)
act2 = Activation('relu', name="act2")(conv2)

conv3 = Conv2D(32, (3, 3), name="Conv3")(lamR)
act3 = Activation('relu', name="act3")(conv3)


merge = concatenate([act1, act2, act3], name="merge")


flat = Flatten( name="flat")(merge)

dens = Dense(1, name="dense")(flat)

model = Model(inputs=[camCenter, camLeft, camRight], outputs=dens)

#model.summary()

plot_model(model, to_file='./model.png', rankdir='TR', show_shapes=True)

model.compile(optimizer='adam', loss='mse')


history_object = model.fit(
    x=[xTrainCenter, xTrainLeft, xTrainRight],
    y=yTrain,
    validation_split=0.2,
    shuffle=True,
    epochs=5
    )


try:
  os.remove("model.h5")
except:
    pass


model.save('model.h5')



plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



print("All Done!")
time.sleep(3)
