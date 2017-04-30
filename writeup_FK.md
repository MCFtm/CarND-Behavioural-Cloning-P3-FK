# Behavioral Cloning

This is the writeup to project 3 "Behavioural Cloning" of the Udacity Self-Driving Car Engineer NanoDegree Program. 
The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/center.jpg "Center Lane Driving"
[image6]: ./images/Picture1.png "Normal Image"
[image7]: ./images/Picture1_flipped.png "Flipped Image"


## 1. Files Submitted & Code Quality

### 1.1 Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_FK.md summarizing the results
* run_FK.mp4 a video-file that was recorded while driving in autonomous mode based on the trained network

### 1.2 Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 1.3 Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## 2. Model Architecture and Training Strategy

### 2.1 An appropriate model architecture has been employed

My model is based on the CNN-architecture that was created by the Nvidia autonomous driving team. In detail it consists of the layers as follows:

* 1. Layer: Input-pictures with a shape of 160x320x3 are normalized and zero-centered
* 2. Layer: Pictures are cropped at the upper and lower edge to eliminate areas of the pictures that are not containing information relevant for driving-parameters to train the network 
* 3. Layer: Convolution with 24 filters, kernel-size of 5x5, stride of 2x2 and rectified linear unit for activation
* 4. Layer: Convolution with 36 filters, kernel-size of 5x5, stride of 2x2 and rectified linear unit for activation
* 5. Layer: Convolution with 48 filters, kernel-size of 5x5, stride of 2x2 and rectified linear unit for activation
* 6. Layer: Convolution with 64 filters, kernel-size of 3x3, stride of 1x1 and rectified linear unit for activation
* 7. Layer: Convolution with 64 filters, kernel-size of 3x3, stride of 1x1 and rectified linear unit for activation
* 8. Layer: Flattening of the input
* 9. Layer: Fully connected layer reducing output-size to 100
* 10. Layer: Fully connected layer reducing output-size to 50
* 11. Layer: Fully connected layer reducing output-size to 10
* 12. Layer: Output-Layer = Fully connected layer reducing output-size to 1

The model includes RELU layers to introduce nonlinearity (code lines 59 to 63), and the data is normalized in the model using a Keras lambda layer (code line 57). 

### 2.2 Attempts to reduce overfitting in the model

The model doesn't contain  specific dropout layers in order to prevent overfitting. Instead the strategy to prevent overfitting was to increase/augment the amount of input data, which was done by flipping the images and considering both the data as it was provided by the initial project-repository and images as they were recorded while driving the car in the simulator.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 73-76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 2.3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

### 2.4 Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I generated training data by the center lane driving, recovering from the left and right sides of the road. The car should stay in the center of the road as much as possible. If the car veered off to the side, I recovered it back to center.
I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data set, I also flipped images and angles thinking that this would augment the data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 19074 number of data points. I then preprocessed this data by normalizing and mean-centering.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by the validation loss.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

In order to increase the amount of data I augmented by flipping the images and using considering both recorded images and data as it was provided by the project-repository.

There was still some potentials left to improve/increase the input-data, which are:
* driving counter-clockwise can help the model generalize
* collecting data from the second track can also help generalize the model
* more emphasis on alternating while driving (from outer bounderary to center of the lane)

## 3. Training Strategy

### 3.1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple architecture (just flattening layer) in order to test whether the pipeline was working.

Then I step by step added convolutional and fully-connected layers. By that I the network made progress and validation/testing results improved.

For further improvement of the network-architecture my strategy was then take the full advantage of the experience and lessons learned that the autonomous driving team at Nvidia made. Thus I improved my network architecture accordingly.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, which I eliminated by actions as described. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 3.2 Final Model Architecture

The final model architecture was described already in chapter 2.1

### 3.3 Training Process

The training process in general followed the guidance and instructions given by the lectures. However I detected that at a certain point the training/validation showed symtoms of overfitting, which could be seen by the fluctuating validation results after a few epochs. 
So increased the amount of input-data as already described above. By this and developping the network-architecture my network iteratively achieved the final goal. 

If this wasn't the case I could have employed further improvement steps as follows:

* driving counter-clockwise can help the model generalize
* driving on the second track
* collecting data from the second track can also help generalize the model
* adjusting training epochs
* establishing/adjusting learning rate
* use of dropout layers
* augmenting left and right images 
* applying grayscaling

