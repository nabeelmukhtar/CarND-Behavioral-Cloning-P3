import csv
import cv2
import sklearn
import numpy as np
from random import shuffle

lines = []
skipHeader = True
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    if skipHeader:
        skipHeader = False
    else:
        for line in reader:
            lines.append(line)
        
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

samples_per_line = 6

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
    
                center_image = cv2.imread('./data/IMG/' + batch_sample[0].split('/')[-1])
                images.append(center_image)
                angles.append(steering_center)
                
                images.append(cv2.flip(center_image, 1))
                angles.append(steering_center * -1.0)
                
                left_image = cv2.imread('./data/IMG/' + batch_sample[1].split('/')[-1])
                images.append(left_image)
                angles.append(steering_left)

                images.append(cv2.flip(left_image, 1))
                angles.append(steering_left * -1.0)

                right_image = cv2.imread('./data/IMG/' + batch_sample[2].split('/')[-1])
                images.append(right_image)
                angles.append(steering_right)

                images.append(cv2.flip(right_image, 1))
                angles.append(steering_right * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

# Preprocess incoming data, centered around zero with small standard deviation 

## Use a gain of 1.5 to increase agressitivity 
# y_train = y_train * 1.5

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

# model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

model.fit_generator(train_generator, samples_per_epoch= samples_per_line * len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=samples_per_line * len(validation_samples), nb_epoch=3)

model.save('model.h5')
