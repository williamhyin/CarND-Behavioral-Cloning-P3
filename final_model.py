import csv
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, Conv2D
from keras.optimizers import Adam
from keras.regularizers import l2

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

CORRECTION_FACTOR = 0.25
EPOCHS = 5
SPLIT_SIZE = 0.15
BTACH_SIZE = 32
PATH = "./all_data/"

def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Generate new random brightness
    random_bright = random.uniform(0.3, 1.0)
#     random_bright = .25 + np.random.uniform()
    hsv[:, :, 2] = random_bright * hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


def flip(image, angle):
    # flip the image vertical
    new_image = cv2.flip(image, 1)
    # negative angle
    new_angle = angle * (-1)
    return new_image, new_angle


def crop_resize(image):
    # resize the image to throw unnecessary information
    # original 320x160x3
    cropped = cv2.resize(image[60:140, :], (64, 64))
    return cropped


def load_image(path):
    # CV2 reads an image in BGR, which need to be converted to RGB
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def panda_csv(path):
    # add titles to each column and extract data of each column with panda package
    titles = ["Center_img", "Left_img", "Right_img", "Steering_angle", "Throttle", "Break", "Speed"]
    drive_log = pd.read_csv(path + 'driving_log.csv', skiprows=[0], names=titles)

    center = drive_log.Center_img.tolist()
    left = drive_log.Left_img.tolist()
    right = drive_log.Right_img.tolist()
    steering = drive_log.Steering_angle.tolist()

    return center, left, right, steering


def reader_csv(path):
    # extract data of each row with csv package
    samples = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) #this is necessary to skip the first record as it contains the headings(if title exists)
        for line in reader:
            samples.append(line)
    return samples


def split_data(samples, test_size):
    # split data to two parts: train-sample and validation-sample
    train_samples, validation_samples = train_test_split(samples, test_size=test_size)
    return train_samples, validation_samples


def process_data(img, angle, aug):
    # flip the image to augment data, for train-sample(aug=True) we can use random-brightness method to augment data again
    img, angle = flip(img, angle)
    if aug == True:
        if random.randint(0,2)==1:
            img = random_brightness(img)

    return img, angle

####################

# help fuction

def visualization(PATH):
    samples = reader_csv(PATH)
    aug = True

    images = []
    angles = []
    for num in range(0, 8):
        sample = samples[random.randint(0, len(samples))]
        for i in range(0, 3):
            path = PATH + "IMG/" + sample[i].split("\\")[-1]
            # print(path)
            center_image = load_image(path)  #
            center_angle = float(sample[3])
            if (i == 0):
                images.append(center_image)
                angles.append(center_angle)
                aug_img, aug_angle = process_data(center_image, center_angle, aug)
                images.append(crop_resize(aug_img))
                angles.append(aug_angle)
            elif (i == 1):
                images.append(center_image)
                angles.append(center_angle + CORRECTION_FACTOR)
                aug_img, aug_angle = process_data(center_image, center_angle + CORRECTION_FACTOR, aug)
                images.append(crop_resize(aug_img))
                angles.append(aug_angle)

            elif (i == 2):
                images.append(center_image)
                angles.append(center_angle - CORRECTION_FACTOR)
                aug_img, aug_angle = process_data(center_image, center_angle - CORRECTION_FACTOR, aug)
                images.append(crop_resize(aug_img))
                angles.append(aug_angle)

    X = np.array(images)
    y = np.array(angles)

    for i in range(0, 8):
        figs, axes = plt.subplots(1, 6, figsize=(20, 5))
        figs.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for ii in range(0, 6):
            image = X[i * 6 + ii]
            angle = y[i * 6 + ii]
            if ii < 2:
                title = "Center steering angle " + str(float('%.3f' % angle))
            elif ii < 4:
                title = "Left steering angle " + str(float('%.3f' % angle))
            else:
                title = "Right steering angle " + str(float('%.3f' % angle))
            axes[ii].set(title=title)
            axes[ii].imshow(image)
            axes[ii].axis('off')
        plt.show()


def data_statistic(path):
    data_frame = pd.read_csv(path + 'driving_log.csv', usecols=[0, 1, 2, 3])
    #     print(data_frame)
    data_frame.describe(include='all')
    data_frame.hist(column='steering')


# visualization(PATH)
# data_statistic(PATH)

####################

def generator_data(samples, batch_size, aug):
    num_samples = len(samples)

    while 1:
        shuffle(samples)  # shuffling the total images
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0, 3):  # for each row, first one is center, second is left and third is right,fourth is steering angle

                    path = PATH+"IMG/"+batch_sample[i].split("\\")[-1]
                    # print(path)
                    center_image = load_image(path)  #
                    center_angle = float(batch_sample[3])  # get the steering angle

                    # introducing correction for left and right images
                    # if image is in left we increase the steering angle by 0.25
                    # if image is in right we decrease the steering angle by 0.25

                    if (i == 0):
                        images.append(center_image)
                        angles.append(center_angle)
                        aug_img, aug_angle = process_data(center_image, center_angle, aug)
                        images.append(aug_img)
                        angles.append(aug_angle)
                    elif (i == 1):
                        images.append(center_image)
                        angles.append(center_angle + CORRECTION_FACTOR)
                        aug_img, aug_angle = process_data(center_image, center_angle + CORRECTION_FACTOR, aug)
                        images.append(aug_img)
                        angles.append(aug_angle)

                    elif (i == 2):
                        images.append(center_image)
                        angles.append(center_angle - CORRECTION_FACTOR)
                        aug_img, aug_angle = process_data(center_image, center_angle - CORRECTION_FACTOR, aug)
                        images.append(aug_img)
                        angles.append(aug_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)  # here we do not hold the values of X_train and y_train instead we yield the values which means we hold until the generator is running


def cnn_1():
    samples = reader_csv(PATH)
    train_samples, validation_samples = split_data(samples, test_size=SPLIT_SIZE)
    # get train-sample batch and validation sample batch
    train_generator = generator_data(train_samples, batch_size=BTACH_SIZE, aug=True)
    validation_generator = generator_data(validation_samples, batch_size=BTACH_SIZE, aug=False)

    input_shape = (160, 320, 3)  # original image size
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # resize image with Cropping2D fuction, which is more faster
    model.add(Conv2D(24, kernel_size=(5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(36, kernel_size=(5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, kernel_size=(5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))

    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(80, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(40, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(Dense(1, kernel_regularizer=l2(0.001)))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    history_object = model.fit_generator(train_generator, validation_data=validation_generator,
                                         steps_per_epoch=len(train_samples), epochs=EPOCHS,
                                         validation_steps=len(validation_samples), verbose=1)

    print('Done Training')
    # Save model
    model.save("model_cnn1.h5.h5")
    print("Saved model to disk")
    print(history_object.history.keys())

    ## plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def cnn_2():
    samples = reader_csv(PATH)
    train_samples, validation_samples = split_data(samples, test_size=SPLIT_SIZE)
    #     print(len(train_samples))

    train_generator = generator_data(train_samples, batch_size=BTACH_SIZE, aug=True)
    validation_generator = generator_data(validation_samples, batch_size=BTACH_SIZE, aug=False)

    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,  input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(Activation('elu'))

    # layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('elu'))

    # layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('elu'))

    # layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))

    # layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))

    # flatten image from 2D to side by side
    model.add(Flatten())

    # layer 6- fully connected layer 1
    model.add(Dense(100))
    model.add(Activation('elu'))

    # Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
    model.add(Dropout(0.25))

    # layer 7- fully connected layer 1
    model.add(Dense(50))
    model.add(Activation('elu'))

    # layer 8- fully connected layer 1
    model.add(Dense(10))
    model.add(Activation('elu'))

    # layer 9- fully connected layer 1
    model.add(
        Dense(1))  # here the final layer will contain one value as this is a regression problem and not classification

    # the output is the steering angle
    # using mean squared error loss function is the right choice for this regression problem
    # adam optimizer is used here
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

    history_object = model.fit_generator(train_generator, validation_data=validation_generator,
                                         steps_per_epoch=len(train_samples), epochs=EPOCHS,
                                         validation_steps=len(validation_samples), verbose=1)

    print('Done Training')
    # Save model
    model.save("model_cnn1.h5")
    print("Saved model to disk")
    print(history_object.history.keys())

    ## plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    cnn_1()
    # cnn_2()
