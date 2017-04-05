import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines[1:]:
    steering_center = float(line[3])
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    center_image = cv2.imread('./data/IMG/' + line[0].split('/')[-1])
    images.append(center_image)
    measurements.append(steering_center)
    left_image = cv2.imread('./data/IMG/' + line[1].split('/')[-1])
    images.append(left_image)
    measurements.append(steering_left)
    right_image = cv2.imread('./data/IMG/' + line[2].split('/')[-1])
    images.append(right_image)
    measurements.append(steering_right)

augmented_images = []
augmented_measurements = []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

## Use a gain of 1.5 to increase agressitivity 
y_train = y_train * 1.5

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

model.save('model.h5')
