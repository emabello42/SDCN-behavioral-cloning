## Project: Behavioral Cloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project, a convolutional neural network is trained using images taken from three cameras (mounted in front of a car) to output the steering angle to an autonomous vehicle, so that the driving behavior is cloned. The model is trained, validated and tested using Keras.

The [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) is used to steer a car around two tracks for data collection and testing.

The Project
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_2018_11_08_22_29_23_545.jpg "Example 1 - Track 1"
[image3]: ./examples/center_2018_11_11_00_19_32_755.jpg "Example 1 - Track 2"
[image4]: ./examples/center_2018_11_08_21_38_39_562.jpg "Example 2 - Track 1"
[image5]: ./examples/center_2018_11_11_00_26_33_914.jpg "Example 2 - Track 2"
[image6]: ./examples/center_2018_11_08_21_39_02_139.jpg "Example 3 - Track 1"
[image7]: ./examples/center_2018_11_11_00_16_30_102.jpg "Example 3 - Track 2"
[image8]: ./examples/center_2018_11_11_00_20_21_737.jpg "Example center"
[image9]: ./examples/left_2018_11_11_00_20_21_737.jpg "Example left"
[image10]: ./examples/right_2018_11_11_00_20_21_737.jpg "Example right"
[image11]: ./examples/center_2018_11_11_00_19_32_755_cropped.jpg "Example cropped"
[image12]: ./examples/loss.png "Loss"
[image13]: ./examples/loss_previous_model.png "Loss - previous model"
### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* *data_loader.py* containing the functions to load and augment the dataset
* *model.py* containing the script to create and train the model
* *drive.py* for driving the car in autonomous mode
* *model.h5* containing a trained convolutional neural network 
* *README.md* summarizing the results
* *visualize_loss.py* for loss visualization

#### 2. Submission includes functional code
Using the Udacity provided simulator and my *drive.py* file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolutional neural network. Additionally, *data_loader.py* file contains the code for loading the dataset.
Both files show the pipeline I used for training and validating the model, and they contain comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with three 5x5 filters (with depths: 24, 36 and 48) , followed by two 3x3 filters (both with depth: 64) and 3 fully connected layers  (with 100, 50 and 10 neurons). See lines 35-51 of *model.py* file. 

The model includes RELU layers after each convolutional layer to introduce nonlinearities, and the data is normalized in the model using a Keras lambda layer (code line 34). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (*model.py* lines 36, 38, 40, 42, 44, 47 and 49). For every dropout layer is used the same probability of dropout: 0.2: 
On the other hand, the model was trained and validated using not only the images from the camera in the center, but also from the left and right cameras, using a correction factor of 0.3
to assign the corresponding steering angle for them (see functions
*load_samples_X* in *data_loader.py* file). The images were collected from both
Track 1 and 2.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (*model.py* line 52).

#### 4. Appropriate training data


Training data was chosen to keep the vehicle driving on the road. Trained data
was collected from both tracks. For every track car was driven through multiple
laps trying to keep the car centered on the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


The strategy followed to derive a model architecture to drive autonomously the
car in the simulator was to start with the neural network architecture
proposed in [End to End Learning for Self-Driving
Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
and experimenting with different image data sets taken from both tracks,
adjusting the hyperparameters (such as dropout and correction factors for the
left and right camera images), and observing the training and validation losses
to see possible signs of underfitting (high training loss) or
overfitting (validation loss smaller than training loss). However, the final
model was derived paying more attention to the ability of the model to keep the car
in the road for both tracks than to the validation loss.

For the first version of the model, the neural network was trained with the
car images of the Track 1 taken from the center as well as from the left and right cameras
applying a correction factor of 0.2. Additionally, the dataset was
augmented flipping all the images and calculating the steering angle as the
negative of the original one. The resulting model was able to drive the car
without problems in the first track but had problems to keep the car on the
road during the second track.

For the second version of the model, I decided to collect images from the second
track too. The correction factor for the left and right cameras was 0.2 too. These additional images were also augmented flipping all the images
like in the first version. This time, the neural network was trained with dropout layers as to reduce the overfitting. The resulting model was
capable of passing most of the curves in the Track 2, but it was not able to
finish a lap without crashing.

To obtain the final model, the correction factor was increased to 0.3, because
in the second version was observed that the car crashed due to the steering
angle was not enough pass some curves. Furthermore, it was observed that augmenting the
training set as described above was not necessary at all, because the bias of
the images taken from Track 1 (that always turns to the left) is
compensated with the images taken from Track 2, so that the training time was
considerably reduced because the training set was small enough to be placed in
the RAM memory, removing the need of python generators (which read images from
disk in batches of 32). With this final model the car is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (*model.py* lines 33-52) consisted of a convolutional neural network with the following layers:

| Layer (type)		|	Description					| 	Output Shape	|Param #|   
|:---------------------:|:-----------------------------------------------------:|:---------------------:|:-----:|
|lambda_1 (Lambda)      |Normalization						|(None, 65, 320, 3)	|0	|
|conv2d_1 (Conv2D)      |24 filters, 5x5 kernel size, 2x2 stride, valid padding	|(None, 31, 158, 24)    |1824   |
|RELU			|							|			|	|
|dropout_1 (Dropout)    |dropout probability: 0.2     				|(None, 31, 158, 24)    |0      |   
|conv2d_2 (Conv2D)      |36 filters, 5x5 kernel size, 2x2 stride, valid padding	|(None, 14, 77, 36)     |21636  |   
|RELU			|							|			|	|
|dropout_2 (Dropout)    |dropout probability: 0.2				|(None, 14, 77, 36)     |0      |   
|conv2d_3 (Conv2D)      |48 filters, 5x5 kernel size, 2x2 stride, valid padding	|(None, 5, 37, 48)      |43248  |   
|RELU			|							|			|	|
|dropout_3 (Dropout)    |dropout probability: 0.2				|(None, 5, 37, 48)      |0      |   
|conv2d_4 (Conv2D)      |64 filters, 3x3 kernel size, 1x1 stride, valid padding	|(None, 3, 35, 64)      |27712  |   
|RELU			|							|			|	|
|dropout_4 (Dropout)    |dropout probability: 0.2				|(None, 3, 35, 64)      |0      |   
|conv2d_5 (Conv2D)      |64 filters, 3x3 kernel size, 1x1 stride, valid padding	|(None, 1, 33, 64)      |36928  |   
|RELU			|							|			|	|
|dropout_5 (Dropout)    |dropout probability: 0.2				|(None, 1, 33, 64)      |0      |   
|flatten_1 (Flatten)    |      							|(None, 2112)           |0      |   
|dense_1 (Dense)        |Fully connected layer    				|(None, 100)            |211300 |   
|dropout_6 (Dropout)    |dropout probability: 0.2				|(None, 100)            |0      |   
|dense_2 (Dense)        |Fully connected layer    				|(None, 50)             |5050   |   
|dropout_7 (Dropout)    |dropout probability: 0.2				|(None, 50)             |0      |   
|dense_3 (Dense)        |Fully connected layer    				|(None, 10)             |510    |   
|dense_4 (Dense)        |      							|(None, 1)              |11     |

The total number of parameters to train is: 348,219

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process
To capture good driving behavior, I recorded my driving behavior through Track 1 and 2 during multiple laps, following the next strategy:

* Three laps trying to stay the car in the middle of the road. Here are
  examples from both tracks:

![alt text][image2]
![alt text][image3]

* One lap keeping the car on the left side of the road. Examples:

![alt text][image4]
![alt text][image5]

* One lap keeping the car on the right side of the road. Examples:

![alt text][image6]
![alt text][image7]

* One lap traying to drive the car as smoothly as possible in every curve.

As described in previous sections, the images taken from the left and right
cameras were also added to the training and validation sets with their corresponding
steering angles calculated as:

steering_angle for the left camera image = *measured steering angle* + 0.3

steering_angle for the right camera image = *measured steering angle* - 0.3

Here is an example image taken from the center camera:

![alt text][image8]

And the corresponding images taken from the left and right cameras,
respectively:

![alt text][image9]
![alt text][image10]


After the collection process, I had 64179 number of data points. I then preprocessed this data by cropping them so that the neural network is feed with images focused on the road,
removing the top portion of the image from pixel 0 to 70 and the bottom portion from pixel 135 to 160. Example:

![alt text][image11]

I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

The ideal number of epochs was 5. The evidence was taken from the loss obtained
during the training of a previous model, where 20 epochs were used for training:

![alt text][image13]

As can be seen, from epoch 5 no considerable improvement was achieved.

Here is the visualization of the loss obtained during the training of the final model:

![alt text][image13]

Although with the final model the validation and training loss are bigger than
in the previous model, this final model was able to drive the car through both
tracks during a full lap.
