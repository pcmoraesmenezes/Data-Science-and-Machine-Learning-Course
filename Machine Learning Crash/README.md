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

## Model

Model is a function in the feature space. It is a function that maps the features to the labels. For example, if you have a house with 3 rooms and you want to predict the price of the house, you will use a model to make the prediction. The model will map the features (3 rooms) to the label (price of the house).

The model is the core of machine learning. It is the function that maps the features to the labels. The model is the function that you will use to make predictions.

### Dimensions

Basically is the number of efatrues of the data set.

### Parameters

The parameters are the variables that the model will use to make predictions. For example, if you have a linear regression model, the parameters are the slope and the intercept. The model will use the slope and the intercept to make predictions.

The process of estimating the parameters is called training the model. The model will learn the parameters from the data. The model will learn the parameters from the training data.

$c = ax_{1} + bx_{2} + d$

In example above, the parameters are a, b and d. The model will learn the parameters from the data. The goal of $a$, $b$ and $d$ is to minimize the error of the model.

If we have a classification problem, the parameters are the coefficients of the decision boundary. For example 

```python

if c < 0:
    y  - pred = -1
else:
    y - pred = 1

```

### Hyperparameters 

The hyperparameters are the variables that you will use to control the model. For example, if you have a linear regression model, the hyperparameters are the learning rate and the number of iterations.

### Error/Cost and Optimization

**Result of Training**: is the value of the parameters of our model

$y^{'}_{i}$ = Prediction (In python we call it y_pred)

$y_{i}$ = Real Value

What we want is to $y^{'}_{i}$ to be as close as possible to $y_{i}$

**Error/Cost**: is the difference between the prediction and the real value

$E$ = ($y^{'}_{1}$ - $y_{1}$) + ($y^{'}_{2}$ - $y_{2}$) + ... + ($y^{'}_{n}$ - $y_{n}$)

The signs of the erros can be positive or negative. This can be a problem because the errors can cancel each other out. To solve this problem we have a lot of options. One of them is to square the errors.

$E$ = ($y^{'}_{1}$ - $y_{1}$)$^{2}$ + ($y^{'}_{2}$ - $y_{2}$)$^{2}$ + ... + ($y^{'}_{n}$ - $y_{n}$)$^{2}$

This is called the **Mean Squared Error**. The goal of the model is to minimize the error.

There are several others errors. For example, the **Mean Absolute Error**.

$E$ = |($y^{'}_{1}$ - $y_{1}$)| + |($y^{'}_{2}$ - $y_{2}$)| + ... + |($y^{'}_{n}$ - $y_{n}$)|

To define the best error for your model, you need to understand the problem that you are trying to solve.
Defining an best error can be an hyperparameter.

### Linear Regression

$y_{i} = ax + b$

$y^{'}_{i} = ax + b$

We can use matrix notation to define $y_{i}$

Let's start with $x_{i}$

First of all, $b$ is a constant $1$ and $a$ is the value of the feature $x_{i}$

$ax_{i}+b$ = $\begin{bmatrix} x_{i} & 1 \end{bmatrix}$ $X$ $\begin{bmatrix} a \\ b \end{bmatrix}$ 

We can extend the matrix notation to all the features, because $x_{i}$ is a vector

$x_{i}$ = $\begin{bmatrix} x_{i1} & x_{i2} & \cdots & x_{in}\end{bmatrix}$

$b$ = $\begin{bmatrix} 1 & 1 & \cdots & 1\end{bmatrix}$

Let's now expand the matrix of $y_{i}$

$y_{i}$ = $\begin{bmatrix} y_{1} \\ y_{2} \\ \vdots \\ y_{n} \end{bmatrix}$

Now we can define $y_{i}$ in matrix notation

$\begin{bmatrix} x_{i} & 1 \\ x_{i} & 1 \\ \vdots & \vdots \\ x_{i} & 1 \end{bmatrix}$ $X$ $\begin{bmatrix} a \\ b \end{bmatrix}$ =  $\begin{bmatrix} y_{1} \\ y_{2} \\ \vdots \\ y_{n} \end{bmatrix}$

We have a $X_{n x 2}$ matrix, a $a_{2 x 1}$ matrix and a $y_{n x 1}$ matrix

$Xa$ = $y$

To solve this equation, we use the training data to build the matrix $X$ and the matrix $y$. Then we use the linear algebra to solve the equation (from numpy) 

Linear Regression uses the **Mean Squared Error** to minimize the error.

### Practical Example

The pratical example can be found [Here](/Machine%20Learning%20Crash/practice/model_/linear_regression.ipynb)

