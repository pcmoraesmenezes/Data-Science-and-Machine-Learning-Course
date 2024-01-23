# Machine Learning: Feature Engineering and Dimensionality Reduction

## Table of Contents

- [Activity 1: Explore dimensionality of different data](#activity-1-explore-dimensionality-of-different-data)
    - [Description](#description)
    - [References](#references)

## Activity 1: Explore dimensionality of different data

The dataset can be found [here](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/dataset/Autism-Adult-Data.arff)

The code can be found [here](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/codes/feature_practice.ipynb)

### Description

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/1.png)

This show us the first 5 rows of the dataset. Here we also can se some columns names.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/2.png)

Here we use the method `info` to view the data types of each column. We can see that the column `gender` is an object, so we need to convert it to a numeric value.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/3.png)

Here we use `shape` to see the number of rows and columns of the dataset. We can see that we got 704 rows and 21 columns.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/4.png)

Here we call `.columns` to see the columns names.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/5.png)

Here we use `dtypes` to see the data types of each column.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/6.png)

Here we use `describe` to see the statistical summary of the dataset. We got only two numerical columns, so we can see the statistical summary of only two columns.

![Image](/Machine%20Learning:%20Feature%20Engineering%20and%20Dimensionality%20Reduction/images/7.png)

Here we see the target column `Class/ASD` and we can see that we got 704 rows and 2 unique values.

The features of this dataset are 20 columns, and the target is the column `Class/ASD`.

### References

Thabtah,Fadi. (2017). Autism Screening Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5F019.

## Why Dimensionality Reduction?

Have more and more data actually gives a better aproach to the function. However, there are some problems with this:

- **Curse of Dimensionality**: The more dimensions we have, the more data we need to generalize well. This is because the volume of the space increases so fast that the available data become sparse.

- **More features means more time**: The more features we have, the more time it takes to train our model.

- **More features means more space**: The more features we have, the more space it takes to store the data.

- **More features means more chance of overfitting**: The more features we have, the higher the chance of overfitting.

However having more features is not always bad, but we need to be careful. The more data, more chances to find patterns.

### What's overfitting has relation with dimensionality reduction?

Overfitting is when the model is too complex for the amount of data we have. This means that the model will memorize the data instead of learning from it. This is a problem because the model will not generalize well.

So, if we have a lot of features, we have a lot of parameters to learn. This means that the model will be more complex. So, we need to be careful with the number of features we have.

### Feature Selection vs Feature Extraction

**Feature Selection**: Selecting a subset of the original features.

**Feature Extraction**: Creating new features from the original ones.

### Feature Selection

Feature selection is the process of selecting a subset of the original features. This is done by removing the features that are not useful for the model. This is done by using some criteria.

By dealing with dimensionality reduction ,the expect is to reduce the number of features and improve the generalization of the model.

#### Feature Selection Methods

- **Filter Methods**: Filter methods are used as a preprocessing step. They are used to remove features that are not useful for the model. This is done by using some criteria. The criteria can be correlation, chi-square, information gain, etc.

- **Wrapper Methods**: Wrapper methods are used to select a set of features by training several models. The criteria can be accuracy, precision, recall, etc.

- **Embedded Methods**: Embedded methods are used to select a set of features by training a single model. The criteria can be the weights of the model.