import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.system('cls' if os.name == 'nt' else 'clear')

# DEBUG STUFF
from inspect import currentframe, getframeinfo

LIMIT_DATA = None


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
from keras.layers import concatenate, Lambda, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint

from keras.utils import plot_model


os.system('cls' if os.name == 'nt' else 'clear')





STEPS = 3



print("---Behaviour Clonning Trainer Application---")

#Load Data

DATA_PATH = '/opt/carnd_p3/data/'
#DATA_PATH = '../data/'

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


random.shuffle(carSamplesAugmented)



# Convert to Dataset

datasetSize = len(carSamplesAugmented)

imageShape = carSamplesAugmented[0].centerImg.shape




xTrain = []
yTrain = []



with tqdm(total=3, desc="  (3/" +str(STEPS) + ") Converting to Dataset") as pbar:

    for sample in carSamplesAugmented[:]:
        xTrain.append(sample.centerImg)
        yTrain.append(sample.steering)

    pbar.update()

    xTrain = np.array(xTrain)
    pbar.update()

    yTrain = np.array(yTrain)
    pbar.update()


del carSamplesAugmented








# Preprocess Dataset
















print('\n\n\n')

#Neural Network Model

model = Sequential()


model.add( Cropping2D(cropping=((70,25),(0,0)), name="Cropped" ) )

model.add( Lambda(lambda x: ((x / 255.0) - 0.5) , name="Normalized") )

model.add( Conv2D(24, (5,5),  name="Convolution_1" ) )
model.add( MaxPooling2D((2,2), name="MaxPool_1" ) )
model.add( Dropout(0.2, name="Dropout_1" ) )
model.add( Activation('relu', name="Activation_1" ) )

model.add( Conv2D(36, (5,5),  name="Convolution_2" ) )
model.add( MaxPooling2D((2,2), name="MaxPool_2" ) )
model.add( Dropout(0.3, name="Dropout_2" ) )
model.add( Activation('relu', name="Activation_2" ) )



model.add( Conv2D(48, (5,5),  name="Convolution_3" ) )
#model.add( MaxPooling2D((2,2), name="MaxPool_3" ) )
model.add( Dropout(0.2, name="Dropout_3" ) )
model.add( Activation('relu', name="Activation_3" ) )



model.add( Conv2D(64, (3,3),  name="Convolution_4" ) )
#model.add( MaxPooling2D((2,2), name="MaxPool_4" ) )
#model.add( Dropout(0.2, name="Dropout_4" ) )
model.add( Activation('relu', name="Activation_4" ) )


model.add( Conv2D(64, (3,3),  name="Convolution_5" ) )
#model.add( MaxPooling2D((2,2), name="MaxPool_5" ) )
#model.add( Dropout(0.1, name="Dropout_5" ) )
model.add( Activation('relu', name="Activation_5" ) )


model.add( Flatten( name="flat") )


model.add( Dense(1024, name="FL1"))
model.add( Dropout(0.3, name="Dropout_FL1" ) )
model.add( Activation('relu', name="Activation_FL1" ) )

model.add( Dense(512, name="FL2") )
model.add( Dropout(0.2, name="Dropout_FL2" ) )
model.add( Activation('relu', name="Activation_FL2" ) )

model.add( Dense(256, name="FL3") )
model.add( Dropout(0.1, name="Dropout_FL3" ) )
model.add( Activation('relu', name="Activation_FL3" ) )

model.add( Dense(128, name="FL4") )
#model.add( Dropout(0.1, name="Dropout_FL4" ) )
model.add( Activation('relu', name="Activation_FL4" ) )

model.add( Dense(64, name="FL5") )
#model.add( Dropout(0.1, name="Dropout_FL5" ) )
model.add( Activation('relu', name="Activation_FL5" ) )

model.add( Dense(1, name="FL6") )




#

model.compile(optimizer='adam', loss='mse')

checkpoint = ModelCheckpoint(filepath='bestModel.h5', monitor='val_loss', save_best_only=True)

history_object = model.fit(
    x=xTrain,
    y=yTrain,
    validation_split=0.2,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint]
    )

#model.summary()
#plot_model(model, to_file='./model.png', rankdir='TR', show_shapes=True)
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

plt.savefig('results.png')

#pip install pydot

print("All Done!")
