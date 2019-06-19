import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   #Ignore Tensorflow Warnings

import csv
from tqdm import tqdm
import random

import numpy as np
import cv2

from keras.models import Sequential, Input, Model
from keras.layers import concatenate, Lambda, Flatten
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model



os.system('cls' if os.name == 'nt' else 'clear')
print("---Behaviour Clonning Trainer Application---")



###################################################
# Load Training Data
###################################################

# Object to store loaded training data
class CarSample:
    
    def __init__(self, centerImg, leftImg, rightImg, steering, throttle, brake, speed, name):
        self.centerImg = centerImg
        self.leftImg = leftImg
        self.rightImg = rightImg
        self.steering = float(steering)
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.speed = float(speed)
        self.name = name


# Path to Training Data
DATA_PATH = '/opt/carnd_p3/data/'


# Load CSV File
lines = []
with open(DATA_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)


# Load Images and Create CarSample Objects
carSamples = []

for line in tqdm(lines[1:10], desc="  (1/3) Loading Data"):

    # Get Image File Paths (Clean string white spaces)
    sourceCenter = line[0].lstrip()
    sourceLeft = line[1].lstrip()
    sourceRight = line[2].lstrip()
    
    # Get Measures
    steering = float(line[3])
    throttle = float(line[4])
    brake = float(line[5])
    speed = float(line[6])

    # Load Images
    centerImg = cv2.imread(DATA_PATH + sourceCenter)
    leftImg = cv2.imread(DATA_PATH + sourceLeft)
    rightImg = cv2.imread(DATA_PATH + sourceRight)

    # Get CarSample Name
    name = sourceCenter.split('/')[-1]
    name = name.split('.')[-2]
    name = name[7:]

    # Create Object
    sample = CarSample(centerImg, leftImg, rightImg, steering, throttle, brake, speed, name)
    
    # Add to List
    carSamples.append(sample)





###################################################
# Data Augmentation
###################################################


carSamplesAugmented = []


# Add Flipped Images and Measures to Training Data
for sample in tqdm(carSamples[:], desc="  (2/3) Augmenting Data"):

        # Get Flipped Image
        flippedCenter = np.fliplr(sample.centerImg)
        flippedRight = np.fliplr(sample.leftImg)
        flippedLeft = np.fliplr(sample.rightImg)

        # Get Flipped Steering
        flippedSteering = -sample.steering

        # Get Object
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

        # Add Original
        carSamplesAugmented.append(sample)
        
        # Add Flipped
        carSamplesAugmented.append(flippedSample)

        
# Delete carSamples List to Free Memory Space
del carSamples

# Shuffle Data
random.shuffle(carSamplesAugmented)




###################################################
# Convert to Dataset
###################################################

# Get Dataset Size
datasetSize = len(carSamplesAugmented)

# Get Image Shape
imageShape = carSamplesAugmented[0].centerImg.shape


# Separate Data into Two Datasets (Inputs and Labels)
xTrain = [] # Center Image
yTrain = [] # Steering Measure

with tqdm(total=3, desc="  (3/3) Converting to Dataset") as pbar:

    # Separate Data
    for sample in carSamplesAugmented[:]:
        xTrain.append(sample.centerImg)
        yTrain.append(sample.steering)

    pbar.update()
    pbar.refresh()

    # Convert To Numpy Array
    xTrain = np.array(xTrain)
    pbar.update()
    pbar.refresh()

    yTrain = np.array(yTrain)
    pbar.update()
    pbar.refresh()


# Delete carSamplesAugmented List to Free Memory Space
del carSamplesAugmented





###################################################
# Define Neural Network Model
###################################################

print('\n\n\n')

# Define Sequential Model
model = Sequential()

# Crop Input Image
model.add( Cropping2D(cropping=((70,25),(0,0)), name="Cropped" ) )

# Normalize Data
model.add( Lambda(lambda x: ((x / 255.0) - 0.5) , name="Normalized") )


# Convolution Model

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


# Fully Connected Model

# Flatten Convolution Output
model.add( Flatten( name="flat") )


model.add( Dense(1024, name="Fully_Connected_1"))
model.add( Dropout(0.3, name="Dropout_Fully_Connected_1" ) )
model.add( Activation('relu', name="Activation_Fully_Connected_1" ) )

model.add( Dense(512, name="Fully_Connected_2") )
model.add( Dropout(0.2, name="Dropout_Fully_Connected_2" ) )
model.add( Activation('relu', name="Activation_Fully_Connected_2" ) )

model.add( Dense(256, name="Fully_Connected_3") )
model.add( Dropout(0.1, name="Dropout_Fully_Connected_3" ) )
model.add( Activation('relu', name="Activation_Fully_Connected_3" ) )

model.add( Dense(128, name="Fully_Connected_4") )
#model.add( Dropout(0.1, name="Dropout_Fully_Connected_4" ) )
model.add( Activation('relu', name="Activation_Fully_Connected_4" ) )

model.add( Dense(64, name="Fully_Connected_5") )
#model.add( Dropout(0.1, name="Dropout_Fully_Connected_5" ) )
model.add( Activation('relu', name="Activation_Fully_Connected_5" ) )

model.add( Dense(1, name="Fully_Connected_6") )





###################################################
# Train Neural Network Model
###################################################

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Define Checkpoint Callback
# This Callback saves the best model found
checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)

# Train Model
history_object = model.fit(
    x=xTrain,
    y=yTrain,
    validation_split=0.2,
    shuffle=True,
    epochs=10,
    callbacks=[checkpoint]
    )

# Get Model Info
#model.summary()
#plot_model(model, to_file='./model.png', rankdir='TR', show_shapes=True)



print("All Done!")
