# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the standard Libraries.
2.  Set variables for assigning dataset values.
3.  Import linear regression from sklearn.
4.  Assign the points for representing in the graph.
5.  Predict the regression for the marks by using the representation of the graph.
6.  Hence we obtained the linear regression for the given dataset.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Shaik Shoaib Nawaz 
RegisterNumber: 212222240094 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1_os.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:

### 1.) Dataset:

![2 1](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/80bf0bfa-ec54-4ae3-8bc8-48d00379e5ed)


### 2.) Graph of plotted data:
![2 2](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/d48e2839-e2f2-4636-be2e-60d979370e4b)



### 3.) Performing Linear Regression:

![2 3](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/a43d41f8-ceb5-46af-9140-7623c3b5b654)


### 4.) Trained data:
![2 4](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/9a73c35e-84e8-4ec8-8277-cb4a2ee5c630)


### 5.) Predicting the line of Regression:

![2 5](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/39a42430-1770-4782-b055-4e811242f25e)

### 6.) Coefficient and Intercept values:


![2 6](https://github.com/shaikSameerbasha5404/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707756/ba0716a7-fb13-41bc-be9d-5ba6d43ce395)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
