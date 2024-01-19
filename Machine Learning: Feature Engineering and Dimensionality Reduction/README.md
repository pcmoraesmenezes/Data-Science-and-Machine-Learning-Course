# Machine Learning: Feature Engineering and Dimensionality Reduction

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