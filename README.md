# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.  

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VANA BHARATH D
RegisterNumber:212223040231 
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
1.PLACEMENT DATA
![image](https://github.com/user-attachments/assets/13fc0d7e-3dad-48ac-885c-382b66e40d32)
2.SALARY DATA

![image](https://github.com/user-attachments/assets/314edf7c-6b7c-4229-96c2-d30b0756ddb8)
3.

![image](https://github.com/user-attachments/assets/9497c399-8aa4-4167-817f-891075fef09c)
4.

![image](https://github.com/user-attachments/assets/d3bf1118-6019-4df4-8cec-ca732a0edb5f)
5.

![image](https://github.com/user-attachments/assets/96701ae6-13c0-4f43-8916-274ab92ef375)
6.

![image](https://github.com/user-attachments/assets/4c2ab516-6092-4e85-8636-e2606c2f536c)

![image](https://github.com/user-attachments/assets/29365602-0630-4ab1-b544-85fdce0118ee)
7.

![image](https://github.com/user-attachments/assets/82bf9c36-c7ca-43e7-9af4-d5f29ba44204)
8.

![image](https://github.com/user-attachments/assets/18548119-ba16-4377-9547-36b01f3f3031)
9.

![image](https://github.com/user-attachments/assets/c3fdfa86-a0c2-4642-ad1e-51241aa1219c)
10.

![image](https://github.com/user-attachments/assets/9c2e095b-e4ed-4da3-a869-d9e4098f8e0f)













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
