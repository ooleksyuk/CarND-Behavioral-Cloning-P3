# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/image_processing.png "Processing Images Before Training"
[image6]: ./examples/run-track1.gif "Track 1"
[image7]: ./examples/run-track2.gif "Track 2"
[video1]: http://youtu.be/vt5fpE0bzSY "Track 1"
[video2]: https://youtu.be/Vt1JVnPHcjA "Track 2" 

---
### Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py contains help methods to process images and create baches
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.ipynd notebook with same code from model.py with visualization of model.summary and training results per epoch
* readme.md summarizing the results
* video1 [video summary of car driving on a track 1](http://youtu.be/vt5fpE0bzSY) (full video)
* video2 [video summary of car driving on a track 2](https://youtu.be/Vt1JVnPHcjA) (full video)

![Track 1][image6] ![Track 2][image7]

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a model that trains fast and car stays on the road.

My first step was to use a convolution neural network model similar to the NVidia Newural Network. I thought this model might be appropriate because it has all layers and filters to train on drinving data.

I stumbled upon the article about 1x1 kenel Convolution network. Descided to give it a try as a data I was using for training was already augmented by 3D modeling of the simulator. This model proved to be working well. It was training much faster than NVidia one and was producing similar results.

I was watching mean square error on training set to be higher than on validation set. It means that my model was not overfitting.

I split my data into training and validation set in ratio 90:10.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I added image processing and image resizing.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py method `create_model()` lines 124-137) consisted of a convolution neural network with the following layers and layer sizes: one 2D convolution, max pooling, dropout, flatten, dense.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
----------------------------------------------
| Layer (type) | Output Shape | Param # | Connected to |                     
----------------------------------------------
| Normalization (Lambda) | (None, 25, 65, 1) | 0 | lambda_input_1[0][0] |             
----------------------------------------------
| convolution2d_1 | (Convolution2D) | (None, 23, 63, 2) | 20 | Normalization[0][0] |              
----------------------------------------------
| maxpooling2d_1 | (MaxPooling2D) | (None, 5, 15, 2) | 0 | convolution2d_1[0][0] |            
----------------------------------------------
| dropout_1 | (Dropout) | (None, 5, 15, 2) | 0 | maxpooling2d_1[0][0] |             
----------------------------------------------
| flatten_1 | (Flatten) | (None, 150) | 0 | dropout_1[0][0] |                  
----------------------------------------------
| dense_1 | (Dense) | (None, 1) | 151 | flatten_1[0][0] |               
----------------------------------------------
| Total params: 171 |
| Trainable params: 171 |
| Non-trainable params: 0 |
----------------------------------------------

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving and processed vertion of it. To augment data I used image resizing and image cropping. I cropped off a sky, trees and a very bottom leyer of the picture. 

![alt text][image3]

I adjusted left and right camera images by adding/substraction delta from steernig data.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

Train on 43394 samples, validate on 4822 samples.

