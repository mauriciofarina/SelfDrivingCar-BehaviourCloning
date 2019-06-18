from keras.models import Sequential, Input, Model
from keras.layers import concatenate, Lambda, Flatten
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D




def convolutionInput(shape, name=None):

    center = Input(shape=shape, name=(name + "_Center") )
    left   = Input(shape=shape, name=(name + "_Left")   )
    right  = Input(shape=shape, name=(name + "_Right")  )

    return (center, left, right)


def crop(modelInput, cropValues , name=None):

    center = Cropping2D(cropping=cropValues, name=(name + "_Center") )(modelInput[0])
    left   = Cropping2D(cropping=cropValues, name=(name + "_Left")   )(modelInput[1])
    right  = Cropping2D(cropping=cropValues, name=(name + "_Right")  )(modelInput[2])

    return (center, left, right)


def normalize(modelInput , name=None):
    
    center = Lambda(lambda x: ((x -128.0) / 128.0) , name=(name + "_Center") )(modelInput[0])
    left   = Lambda(lambda x: ((x -128.0) / 128.0) , name=(name + "_Left")   )(modelInput[1])
    right  = Lambda(lambda x: ((x -128.0) / 128.0) , name=(name + "_Right")  )(modelInput[2])

    return (center, left, right)





def convolution(modelInput ,convFilter, convKernel, activationFunction=None, name=None):

    center = Conv2D(convFilter, convKernel, activation=activationFunction, name=(name + "_Center") )(modelInput[0])
    left   = Conv2D(convFilter, convKernel, activation=activationFunction, name=(name + "_Left")   )(modelInput[1])
    right  = Conv2D(convFilter, convKernel, activation=activationFunction, name=(name + "_Right")  )(modelInput[2])

    return (center, left, right)




def activation(modelInput ,activationFunction, name=None):

    center = Activation(activationFunction, name=(name + "_Center") )(modelInput[0])
    left   = Activation(activationFunction, name=(name + "_Left")   )(modelInput[1])
    right  = Activation(activationFunction, name=(name + "_Right")  )(modelInput[2])

    return (center, left, right)




def maxPooling(modelInput ,size, name=None):

    center = MaxPooling2D(size, name=(name + "_Center") )(modelInput[0])
    left   = MaxPooling2D(size, name=(name + "_Left")   )(modelInput[1])
    right  = MaxPooling2D(size, name=(name + "_Right")  )(modelInput[2])

    return (center, left, right)



