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
/*
Program to implement the simple linear regression model for predicting the marks scored.
#### Developed by:Shaik Sameer Basha
#### RegisterNumber: 212222240093 
*/
```python
Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARSHITHA MS
RegisterNumber: 212223240015
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
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
