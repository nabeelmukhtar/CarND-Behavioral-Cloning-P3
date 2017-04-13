import csv
import cv2
import sklearn
import numpy as np
import random

'''
This method is inspired from the blog:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
[quote]Changing brightness to simulate day and night conditions. 
We will generate images with different brightness by first converting images to HSV,
scaling up or down the V channel and converting back to the RGB channel.[quote]
'''
def augment_brightness(image, steer):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1, steer

'''
This method is inspired from the blog:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
[quote]We will shift the camera images horizontally to simulate the effect of 
car being at different positions on the road, and add an offset corresponding 
to the shift to the steering angle.[quote]
'''
def translate_image(image, steer, trans_range = 100):
    rows,cols,_ = image.shape
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr, steer_ang
    
'''
Flip image and steering.
'''
def flip_image(image, steer):
    image_flip = cv2.flip(image, 1)
    steer_flip = -1.0 * steer
    
    return image_flip, steer_flip

'''
A copy image function.
'''
def copy_image(image, steer):
    return image, steer

'''
Increase contrast of an image
'''
def increase_contrast(image, steer):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)    
    return img_output, steer
        
lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split

# An array of augmentation functions. Function is chosen as random from this array.
augmentations = [copy_image, flip_image, translate_image, augment_brightness]

# How many samples should be augmented from the original data.
samples_per_line = 5

# Steering correction for left and right images.
steering_correction = 0.3

# Steering aggressivity to apply, 1.0 means no change.
steering_aggressivity = 1.0

# Minimum steering for random filtering
steering_min = 0.1

# Steering  probability for random filtering
steering_keep_prob = 0.8

# dropout value for model.
dropout_probability = 0.5

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        # remove images where steering is close to 0.
        samples = [x for x in samples if abs(float(x[3])) > steering_min or np.random.uniform() < steering_keep_prob]
        random.shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering = float(batch_sample[3])
                image_index = random.randint(0, 2)
                image = cv2.imread('./data/IMG/' + batch_sample[image_index].split('/')[-1])
                if image_index == 1:
                    steering = steering + steering_correction
                elif image_index == 2:
                    steering = steering - steering_correction
                    
                augment = random.choice(augmentations)
                augmented_image, augmented_steer = augment(image, steering)
                
                images.append(augmented_image)
                angles.append(augmented_steer)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles) * steering_aggressivity
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(3, 1, 1, activation='elu'))

model.add(Convolution2D(32, 3, 3, activation='elu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_probability))

model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_probability))

model.add(Convolution2D(128, 3, 3, activation='elu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_probability))

model.add(Flatten())

model.add(Dense(128, activation='elu'))

model.add(Dense(64, activation='elu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch=samples_per_line * len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=samples_per_line * len(validation_samples), nb_epoch=5)

model.save('model.h5')
