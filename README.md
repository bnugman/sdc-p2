**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sample]: ./sample.png "Sample Signs"
[distro]: ./distro.png "Sign counts"
[image1]: ./from_web/1.jpg "Traffic Sign"
[image2]: ./from_web/2.jpg "Traffic Sign"
[image3]: ./from_web/3.jpg "Traffic Sign"
[image4]: ./from_web/4.jpg "Traffic Sign"
[image5]: ./from_web/5.jpg "Traffic Sign"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bnugmanov/sdc-p2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here's a sample of 5 images:

![sample][sample]

The counts of images for each traffic sign suggest that the dataset has many more instances of some classes, compared to others:

![Histogram][distro]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I've decided against converting the images to grayscale because, unlike digit recognition, the color in
a traffic sign carries information, and I didn't want to discard it.

As the first pre-processing step, I normalized the image data because neural networks work best with
normalized inputs.

As the second step, I've added random noise to that normalized data, to improve generalization.

I decided not to generate additional data, so as not to slow down the training.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have opted to adapt the LeNet architecture, with an eye on preventing overfitting.
I have added two dropout layers, in the fully-connected layers of the LeNet.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16				    |
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | 5x5x16 -> 400                                 |
| Fully connected       | 400 inputs, 120 outputs                       |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | 120 inputs, 84 outputs                        |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | 84 inputs, 43 outputs                         |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After some experimenting, I've ended up using AdamOptimizer with loss defined as a cross entropy
defined on the softmax of the logits output of LeNet.

I've used the learning rate of 0.004, with 10 epochs and a batch size of 128.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 95.0%
* test set accuracy of 93.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I started with LeNet architecture.

* What were some problems with the initial architecture?

The problem seemed to be with network's ability to generalize well.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I've added dropout layers in the most dense parts of the network, to improve its generalization.

* Which parameters were tuned? How were they adjusted and why?

I've adjusted the learning rate and ended up decreasing the learning rate and increasing the number of epochs


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolutions layers were important to make the network insensitive to shifts of the image.

The deep architecture was important to give the network enough expressive power to learn complex image features with large variances (different lighting, backgrounds, etc)


If a well known architecture was chosen:
* What architecture was chosen?

LeNet architecture was chosen as the course material suggested it.

* Why did you believe it would be relevant to the traffic sign application?

That architecture proved relevant to image recognition problems

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The 95% validation and 93% testing accuracy attest to decent performance. The tiny sample of 5 images found on the internet all got recognized correctly, which also inspires confidence.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![1][image1] ![2][image2] ![3][image3] ![4][image4] ![5][image5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No vehicles      		| No vehicles  									|
| Beware of ice/snow    | Beware of ice/snow            				|
| Roundabout mandatory	| Roundabout mandatory							|
| Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| No passing			| No passing          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.1%, although with a set of just 5 images the accuracy figure is far from being statistically significant.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Except for the 2nd and 4th image, the model is extremely sure about the sign it was presented:

9.99999404e-01,  8.17978442e-01, 1.00000000e+00, 4.44036514e-01, 1.00000000e+00 ]

Let's consider the 2nd and 4th image, however.

For the second image:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .82                   | Beware of ice/snow                            |
| .14                   | Right-of-way at the next intersection         |
| .03                   | Slippery road                                 |
| .008                  | Children crossing                             |
| .004                  | Pedestrians                                   |



For the fourth image:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .44                   | Speed limit (30km/h)                          |
| .37                   | Speed limit (50km/h)                          |
| .18                   | Speed limit (80km/h)                          |
| .0005                 | Speed limit (60km/h)                          |
| 0.00001               | End of speed limit (80km/h)                   |


It can be seen that these secondary and tertiary guesses do make sense.



