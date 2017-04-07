import csv
import cv2
import sklearn
import numpy as np
from random import shuffle

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
    
def translate_image(image,steer,trans_range):
    rows,cols,_ = image.shape
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang
    
def flip_image(image,steer):
    image_flip = cv2.flip(image, 1)
    steer_flip = -1.0 * steer
    
    return image_flip,steer_flip
        
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    skipHeader = True
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split


samples_per_line = 6
steering_correction = 0.25
steering_aggressivity = 1.5
steering_min = 0.1
steering_keep_prob = 0.5

filtered_samples = [x for x in lines[1:] if float(x[3]) > steering_min or np.random.uniform() < steering_keep_prob]

train_samples, validation_samples = train_test_split(filtered_samples, test_size=0.2)

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
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
    
                center_image = cv2.imread('./data/IMG/' + batch_sample[0].split('/')[-1])
                images.append(center_image)
                angles.append(steering_center)
                
                flipped_image, flipped_steering = flip_image(center_image, steering_center)
                images.append(flipped_image)
                angles.append(flipped_steering)
                
                left_image = cv2.imread('./data/IMG/' + batch_sample[1].split('/')[-1])
                images.append(left_image)
                angles.append(steering_left)

                flipped_image, flipped_steering = flip_image(left_image, steering_left)
                images.append(flipped_image)
                angles.append(flipped_steering)

                right_image = cv2.imread('./data/IMG/' + batch_sample[2].split('/')[-1])
                images.append(right_image)
                angles.append(steering_right)

                flipped_image, flipped_steering = flip_image(right_image, steering_right)
                images.append(flipped_image)
                angles.append(flipped_steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles) * steering_aggressivity
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

model.fit_generator(train_generator, samples_per_epoch=samples_per_line * len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=samples_per_line * len(validation_samples), nb_epoch=5)

model.save('model.h5')
