import numpy as np
import cv2
import matplotlib.pyplot as plt

class CarSample:
    
    def __init__(self, centerImg, leftImg, rightImg, steering, throttle, brake, speed, name):
        self.centerImg = np.array(centerImg)
        self.leftImg = np.array(leftImg)
        self.rightImg = np.array(rightImg)
        self.steering = float(steering)
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.speed = float(speed)
        self.name = name


