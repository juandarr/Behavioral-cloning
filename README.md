# **Behavioral Cloning** 

The steps carried out during this project are the following:

1. Use the simulator to collect data of good driving behavior
2. * a. Build a convolution neural network in Keras that predicts steering angles from images
   * b. Train and validate the model with a training and validation set
3. Test that the model successfully drives around track one without leaving the road


My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 showing the driving skills of the new algorithm

[//]: # (Image References)

[image1]: ./images/architecture.png "Model architecture"
[image2]: ./images/training_result.png "Training result"
[image3]: ./images/normal.jpg "Normal Image"
[image4]: ./images/recovery1.jpg "Recovery Image"
[image5]: ./images/recovery2.jpg "Recovery Image"
[image6]: ./images/recovery3.jpg "Recovery Image"
[image7]: ./images/flipped.png "Flipped Image"

## 1. Use the simulator to collect data of good driving behavior

The simulator was downloaded to my personal computer and several training datasets were collected during this process. To achieve the best performance several aspects were considered during data collection. In general, these were the rules considered when collecting training data in the simulator:

* Vehicle should move guided by the center of the lane most of the time. The following image serves as an example of this scenario. 
![alt text][image3]

* For the network to learn how to go back to the center of the lane when it is getting too close to the lane limits I recorded the vehicle going back from the right/left lane lines to the center in straight and curved portions of the road. The following set of three images shows the progress of recovery from the left line lane to the center:
![alt text][image4]
![alt text][image5]
![alt text][image6]

* To avoid bias due to overstimulation to just one type of turning direction, I recorded several loops of the vehicle moving clockwise and counter-clockwise.

After the data was collected in the simulator, image augmentation was also implemented. In this case, we mirrored each image horizontally and change the sign of the label variable to match the reflexion.

Before going to the network, the input data is preprocessed using a lambda layer to normalize and move the mean of the pixel values close to zero. A cropping layer also crops each image removing a top and bottom regions of height 70 (top) and 25 (bottom) pixels. This addition removes undesired objects from the top of the image, such as trees, clouds and from the bottom such as the front of the vehicle.

## 2a-2b. Build a convolution neural network in Keras that predicts steering angles from images

The overall strategy for deriving a model architecture was to follow an iterative method. Initially started with a simple fully connected layer. The results were not attractive since the vehicle was turning left and right in a constant basis and the moving was far from smooth. After this, I made a transition to a LeNet like network with three convolutional layers and just one fully connected layer at the end in addition to the output. During model training, the training loss was decreasing  while validation loss changed randomly around a fixed value. In simulation testing, the car was able to drive in the straight road but could not complete the curves in several scenarios. The driving style was also far from smooth.  

The best result was achieved with an architecture similar to the one recommended in the video lectures. My final model consists of a deep convolutional neural network with a set of five convolutions at the beginning and three fully connected layers in addition to the output. The first 3 convolutional layers use a filter of size 5x5, stride of 2x2 and have depths of 24, 32 and 48 respectively. The other 2 convolutional layers use a filter of 3x3, stride of 1x1 and have a depth of 64 (both). To avoid overfitting, validation sets were defined as the 20% of the training set in each epoch. Dropout layers were also used in the convolutional layers. The batch size was set to 32 and Adam optimizer was used to train the model, so the learning rate was not tuned manually (model.py line 94). The activation function RELU was used in the whole network. 

The following image shows the architecture overall architecture of the deep neural network:

![alt text][image1]

A detailed description of dimensions, depth, filter size and more is described in the following table.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   							| 
| Lambda layer     	| Normalization and moves the input to mean zero 	|
| Cropping layer      	| inputs 320x160x3, outputs 320x65x3 	|
| Convolution 5x5     	| 2x2 stride, 'VALID' padding, outputs 158x31x24 	|
| RELU					|												|
| Dropout               | Keep prob = 0.5                               |
| Convolution 5x5     	| 2x2 stride, 'VALID' padding, outputs 77x14x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, 'VALID' padding, outputs 37x5x48 	|
| RELU					|												|
| Dropout               | Keep prob = 0.5                               |
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 35x3x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 33x1x64 	|
| RELU					|												|
| Dropout               | Keep prob = 0.5                             |
| Flatten layer		| Reshapes data from 33x1x64 to 2112       				|
| Fully connected layer (FC1) 		| inputs 2112, outputs 100       				|
| RELU				|       									|
| Fully connected layer (FC2)	| inputs 100, outputs 50       				|
| RELU				|       									|
| Fully connected layer (FC3)		| inputs 50, outputs 10       				|
| RELU				|       									|
| Logits				| inputs 10, outputs 1						|

After training the model with the best data set collected and the final model for 20 epochs we get the following plot of the training and validation loss:

![alt text][image2]

Adding a validation set and dropout layers were useful to avoid overfitting as we see from the graph were validation and training losses have a downward trend. 

## 3. Test that the model successfully drives around track one without leaving the road

The final implementation was successfully tested. The video recorded from the point of view of the central camera installed in the car can be found in the project directory.
