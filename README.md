# Deep Learning - Function Approximation

[中文筆記](https://hackmd.io/@GoldBaby/B1mTsx9Xn)  
[Kaggle](https://www.kaggle.com/competitions/function-approximation/)

## Problem Description
The dataset contains the data which are the outputs from an unknown function **$y=f(x_1,x_2)$**
Note that the given training dataset has been disturbed by some noise. You are now required to design a network to remove the introduced noise. After finishing your network training, you need to produce the outputs for a given testing dataset. Kaggle will compute the **mean square error** (MSE) between your outputs and the true outputs for the performance evaluation of your network.

## Problem Evaluation
The criterion used to evaluate your submission is the MSE loss $L=\frac{1}{n}\sum_{i=1}^n(\hat{y}_i-y)^2$

## Dataset Description
Two datasets are given:
* train.csv: the training dataset which contains three columns of data. Column 1 and Column 2 are inputs x1 and x2, while Column 3 is the target value y.
* test.csv: the testing dataset for evaluating your model. Only two columns of data, i.e., x1 and x2, are given.