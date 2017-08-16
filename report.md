#**Project II: Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Selection_001.png "Deer Dark"
[image2]: ./Selection_002.png "Deer Bright"
[image3]: ./Selection_003.png "features"
[web1]: ./web_images/web1.png "web1"
[web2]: ./web_images/web2.png "web2"
[web3]: ./web_images/web3.png "web3"
[web4]: ./web_images/web4.png "web4"
[web5]: ./web_images/web5.png "web5"
[web6]: ./web_images/web6.png "web6"
[web7]: ./web_images/web7.png "web7"
[web8]: ./web_images/web8.png "web8"
[web9]: ./web_images/web9.png "web9"

[signs_matrix]: ./traffic_signs.png "Traffic Signs Ensemble"
[hist]: ./hist.png "The histogram of training dataset by labels"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yingweiy/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The figure below is an exploratory visualization of the data set. It shows some random samples that are drawn from the training dataset.  

![alt text][signs_matrix]

The second exploratory figure is a bar chart showing how the record count of each class. The x-axis is the class labels ranged from 0 to 42, while the 
y-axis is the count of the corresponding class in the training dataset.

![alt text][hist]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to equalize the histogram of the image because I noticed that the color and brightness/contrast
are not well balanced for many images. This is done by the following operation:

* convert the image from RGB to YUV encoding
* apply the equalizeHist function from OpenCV to the first channel (the energy)
* back convert the image to RGB channels 

Here is an example of a traffic sign image before and after color histogram equalization.

![alt text][image1]
![alt text][image2]

Please note that I did not convert the image to gray scale, because that I observe the color convey 
important information for classification tasks, such as red, blue, etc.

As a last step, I normalized the image data because the data range is not within [-1,1] range.
This is done by the equation below

image = (image-128.0)/128.0

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3   	| 1x1 stride, same padding 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                     				|
| Convolution 5x5x6    	| 1x1 stride, same padding 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3x16 	| 1x1 stride, same padding 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride 				|
| Fully connected		| 1024 cells        									|
| RELU					|												|
| Fully connected		| 1024 cells        									|
| RELU					|												|
| Fully connected		| 512 cells        									|
| RELU					|												|
| Fully connected		| 128 cells        									|
| Softmax				|         									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer as the optimizer, cross entropy as loss function. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100.000%
* validation set accuracy of 96.0%
* test set accuracy of 93.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Answer: the initial try is with the default LeNet architect 

* What were some problems with the initial architecture?

Answer: It seems the number of cells or the depth of the networks is not enough to memorize all the data.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Answer: the architecture is adjusted by adding more layers and cells. 

* Which parameters were tuned? How were they adjusted and why?

Answer: the parameters are use the default LeNet parameters from previous quiz. It works fine.


* What are some of the important design choices and why were they chosen? 

Answer: 

1. RELU in the fully connected layer is an important choice. It is simple, and introduces
the non-linearality into the networks.

2. The additional number of convolutional layers and fully connected layers are useful to increase the training accuracy.

Note: in my test, the dropout does not help much, so I did not use them.

###Test a Model on New Images

####1. Choose nine German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

(sorry, these images are adapted with different image sizes.)

![alt text][web1] ![alt text][web2] ![alt text][web3]
 
![alt text][web4] ![alt text][web5]![alt text][web6]

![alt text][web7] ![alt text][web8]![alt text][web9] 


The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first convolution layer outputs from six filters are extracted and illustrated as shown below:

![alt text][image3]

* FeatureMap0: It seems the text "STOP" is enhanced.
* FeatureMap1: This filter appears to extract the derivatives of the image,
* FeatureMap2: This filter has a preference of 40-45 degree angle features.
* FeatureMap3: This appears to be the reverse of FeatureMap1.
* FeatureMap4: This filter highlights the stright lines of horizontal or vertical oriented patterns.
* FeatureMap5: This filter extract high frequency vertical lines.
 