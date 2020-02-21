# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* final_ model.py, final_ model.html and final_ model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_cnn1.h5 containing a trained convolution neural network 
* my_writeup.md/html summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The final_ model.ipynb/py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Overview


I decided to use NVIDA model, which is proved to be more powerful. The input of my model is (160x30x3). After that the data is normalized in the model using a Keras lambda layer.

	model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
But there are many unnecessary information in the images. So we need to crop and resize the image to augmente data.

	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
![](https://i.imgur.com/YhKz7yG.png)

[//]: ![](/md_images/NVIDIA.PNG)

#### 2. Loading and spliting Data 

- I uesd Udacity simulator to record data. I use the mouse to control the movement direction of the vehicle, so that the value of the steering angle changes smoothly. There are 4 anticlockwise loop, 3 clockwise loop.

- Because CV2 reads an image in BGR, so we  need to be converted to RGB, which can be processed in drive.py. 
	
code：

	def load_image(path):
	    # CV2 reads an image in BGR, which need to be converted to RGB 
	    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
	    return img

- Since we have 3 camera image, but the steering angle is associated with center camera, so I need to introduce a correction factor for left(0.25) and right(-0.25) images. Then I combined images and steering angle from 3 camera to augment my dataset. Before that we need split the original center dataset to train-samples and validation samples. I decided to keep 15% of the data in Validation Set and remaining in Training Set.

code:

	def split_data(samples, test_size):
	    # split data to two parts: train-sample and validation-sample
	    train_samples, validation_samples = train_test_split(samples, test_size=test_size)
	    return train_samples, validation_samples

#### 3. Preprocessing

- filp: I decided to flip the image horizontally and adjust steering angle accordingly to balance the distribution of left steering and right steering, and increse the numbers of my dataset samples.

code: 

	def flip(image, angle):
	    # flip the image vertical 
	    new_image = cv2.flip(image, 1)
	    # negative angle
	    new_angle = angle * (-1)
	    return new_image, new_angle

Distribution of raw center steering is shown below:

![](https://i.imgur.com/Impj9de.png)
[//]:![](/md_images/9.png)

- brightness: I decided to use random brightness changes to make my model can adapt different bright condition.

code:

	def random_brightness(image):
	    # Convert 2 HSV colorspace from RGB colorspace
	    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	    # Generate new random brightness
	    random_bright = random.uniform(0.3, 1.0) 
	    hsv[:, :, 2] = random_bright * hsv[:, :, 2]
	    # Convert back to RGB colorspace
	    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	    return new_img
	
- crop and resize images: There are many unnecessary information in the images. So we need to crop and resize the image to augmente data. (but I use keras.layers.Cropping2D to crop and resize images instead of cv.resize function.

code:

	def crop_resize(image):
	    # resize the image to throw unnecessary information
	    # original 320x160x3
	    cropped = cv2.resize(image[60:140, :], (64, 64))
	    return cropped
	
The example-results of  data loading and preprocessing are shown below:

The 1st, 3rd and 5th columns are raw center, left and right images.
The 2nd, 4th and 6th columns are augmented (flip, crop-resize, random brightness) images.

![](https://i.imgur.com/lpmTK6L.png)
![](https://i.imgur.com/ML6hsef.png)
![](https://i.imgur.com/gwuasG4.png)
![](https://i.imgur.com/Z6c7NVR.png)
![](https://i.imgur.com/u1apHeM.png)
![](https://i.imgur.com/aqWnvoE.png)
![](https://i.imgur.com/MrgKznQ.png)
![](https://i.imgur.com/pEbMoVw.png)
[//]:![](/md_images/1.png)
[//]:![](/md_images/2.png)
[//]:![](/md_images/3.png)
[//]:![](/md_images/4.png)
[//]:![](/md_images/5.png)
[//]:![](/md_images/6.png)
[//]:![](/md_images/7.png)
[//]:![](/md_images/8.png)

	
#### 4. Final model

The model includes 5 convolution layers and 5 dense Layers. RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

The model contains dropout layers (0.5)in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

![](https://i.imgur.com/Gaf0MAo.png)
[//]:![](/md_images/model.png)

Code:

	def cnn_1(_):
	    
	    samples = reader_csv(PATH)
	    train_samples, validation_samples = split_data(samples, test_size=SPLIT_SIZE)
	    # get train-sample batch and validation sample batch
	    train_generator = generator_data(train_samples, batch_size=BTACH_SIZE, aug=True)
	    validation_generator = generator_data(validation_samples, batch_size=BTACH_SIZE, aug=False)
	
	    input_shape = (160, 320, 3) # original image size 
	    model = Sequential()
	    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
	    model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # resize image with Cropping2D fuction, which is more faster
	    model.add(Conv2D(24,kernel_size= (5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
	    model.add(Activation('relu'))
	
	    model.add(Conv2D(36, kernel_size=(5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
	    model.add(Activation('relu'))
	
	    model.add(Conv2D(48, kernel_size=(5, 5), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
	
	    model.add(Activation('relu'))
	
	    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2), kernel_regularizer=l2(0.001)))
	    model.add(Activation('relu'))
	
	    model.add(Conv2D(64,kernel_size= (3, 3), padding='valid', strides=(2, 2), kernel_regularizer=l2(0.001)))
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
	    model.compile(optimizer=adam, loss='mse',metrics=['accuracy']) 
	    model.summary()
	    history_object =model.fit_generator(train_generator, validation_data=validation_generator,
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

The training results are shown below:

![](https://i.imgur.com/7LlvM5N.png)
[//]:![](/md_images/results.png)

AS you see, the accuracy didnot decrease and the loss is small . I think I can't get better results unless I find more better paramaters or add more data samples.

#### 5. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. And I use l2-regulation.

CORRECTION_FACTOR = 0.25

EPOCHS = 5

SPLIT_SIZE = 0.15

BTACH_SIZE = 32

PATH = "./all_data/"

l2-kernel_regularizer=0.001

Adam-learning rate=0.0001

#### 5. Output Video


<iframe width="560" height="315" src="https://www.youtube.com/embed/iE2Q_QQVEr0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


# CarND-Behavioral-Cloning-P3
