# Machine Learning: Machine Learning Crash Course

## Why Now?

- Data is being collected at an unprecedented scale

- Computers are getting faster and faster

- We have more data than ever before

To the models get a better performance, we need to use more data and more complex models.

## What is Machine Learning?

Machine Learning is the science of programming computers so they can learn from data.

### Learning From Data

#### Supervised Learning

The best way to understand supervised learning is beyond an child example. Imagine that you have a child and you want to teach him/her the difference between a dog and a cat. You show him/her a lot of pictures of dogs and cats and tell him/her which one is a dog and which one is a cat. After a while, he/she will be able to distinguish between a dog and a cat. This is the same way that supervised learning works. As long you give the model more data, it will be able to learn and make predictions.

The learning will scale with the amount of data that you give to the model. The more data you give, the better the model will be.

#### Unsupervised Learning

Unsupervised learning is the opposite of supervised learning. In this case, you don't have a label to tell the model what is what. You just give the model a lot of data and it will try to find patterns in the data. For example, you give a lot of pictures of dogs and cats to the model and it will try to find patterns in the data. It will try to find the difference between a dog and a cat. After that, you can give a new picture to the model and it will try to predict if it is a dog or a cat.

#### Reinforcement Learning

In reinforcement learning, you have an agent that will interact with the environment. The agent will try to maximize the reward that it gets from the environment. For example, you have a robot that will try to learn how to walk. The robot will try to maximize the reward that it gets from the environment. The reward can be the distance that the robot walks. The robot will try to maximize the distance that it walks.

## Features

Basically they are the driving force to any machine learning algorithm. They are the variables that you will use to make predictions. For example, if you want to predict the price of a house, you will use features like the number of rooms, the size of the house, the location of the house, etc.

The data that is used to train a model is called training data. The training data is composed of features and labels. The features are the variables that you will use to make predictions and the labels are the values that you want to predict.

In python we define a variable X to store the features and a variable y to store the labels.

### Practical Example

The pratical example can be found [Here](/Machine%20Learning%20Crash/practice/features_practice.ipynb)

## Regression

Regression is a supervised learning algorithm. It is used to predict a continuous value. For example, if you want to predict the price of a house, you will use regression. The price of a house is a continuous value. It can be any value. It can be 1000, 2000, 3000, etc.

So for supervised problems which your target variable is not categorical, but a continuous value that is called regression. The goal of regression is to predict a continuous value.

The most common regression algorithms are:

- Linear Regression

- Polynomial Regression

- Support Vector Regression

- Decision Tree Regression

- Random Forest Regression

### Practical Example

The pratical example can be found [Here](/Machine%20Learning%20Crash/practice/regression_practice.ipynb)

## Classification

Classification is a supervised learning algorithm. It is used to predict a discrete value. For example, if you want to predict if a person has cancer or not, you will use classification. The person can have cancer or not. It is a discrete value. It can be 0 or 1.

So for supervised problems which your target variable is categorical, that is called classification. The goal of classification is to predict a discrete value.

The most common classification algorithms are:

- Logistic Regression

- K-Nearest Neighbors

- Support Vector Machine

### Practical Example

The pratical example can be found [Here](/Machine%20Learning%20Crash/practice/classification_practice.ipynb)

## Clustering

Clustering is a unsupervised learning algorithm. It is used to find patterns in data. For example, if you have a lot of data and you want to find patterns in the data, you will use clustering. The algorithm will try to find patterns in the data.

So for unsupervised problems which your target variable is not defined, that is called clustering. The goal of clustering is to find patterns in data.

The most common clustering algorithms are:

- K-Means

- Hierarchical Clustering

- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

### Practical Example

The pratical example can be found [Here](/Machine%20Learning%20Crash/practice/clustering_practice.ipynb)

## Different format of data

In regression we use a continuous value to make predictions. In classification we use a discrete value to make predictions. But in some cases we don't have a continuous or discrete value to make predictions. In some cases we have a image, a text or a sound. In this cases we need to use different algorithms to make predictions.

### Image

Images are basically a matrix (2d-array) of numbers, each number the gray scale of a pixel. For example, if you have a image with 100x100 pixels, you will have a matrix with 100 rows and 100 columns. Each number in the matrix will be the gray scale of a pixel. The gray scale can be a number between 0 and 255. 0 is black and 255 is white.

Each pixel is a feature.

Before the CNN (Convolutional Neural Network) we used the SVM (Support Vector Machine) to make predictions with images. The SVM is a classification algorithm. It is used to classify images. The SVM is a supervised learning algorithm. It needs labels to make predictions. So we need to label the images before we can use the SVM to make predictions.

### Video

Videos are basically a sequence of images. We need to convert the vídeo to an array of features.

For video is recommended to use the RNN (Recurrent Neural Network) to make predictions. This is because the RNN can learn the temporal patterns in the data. For example, if you have a video of a person walking, the RNN will learn the temporal patterns in the data. It will learn the sequence of images that represents a person walking.

But is necessary a lot of data to train a RNN. If you don't have a lot of data, you can use a CNN to extract the features from the video and then use a RNN to make predictions. But if you don't have data to CNN and RNN, you can use classical machine learning algorithms to make predictions.

### Audio

Normally audio is a one dimensional array. Each number in the array is the amplitude of the sound wave at a particular time. For example, if you have a audio with 1000 samples, you will have a array with 1000 numbers. Each number in the array is the amplitude of the sound wave at a particular time.

Each sample is a feature.


### Text

Text can appear in many different forms. It can be use to represent a value, for example **"The number of disciplines are 5"** or it can be use to represent a category, for example **"The color of the car is red"**. In the first case we can use regression to make predictions and in the second case we can use classification to make predictions.

