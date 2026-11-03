# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required Python libraries such as Pandas, NumPy and Scikit-learn.
2. Load the Placement_Data.csv dataset and select the required academic performance columns as input features.
3. Convert the placement status into numerical values and split the dataset into training and testing sets.
4. Train the Logistic Regression model, predict the placement status of students and display the accuracy.. 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Jaden Samuel Abraham
RegisterNumber: 212225040138

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data=pd.read_csv("Placement_Data.csv")
X=data[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]
y=data['status'].map({'Placed':1,'Not Placed':0})
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Actual Placement Status:",y_test.values)
print("Predicted Placement Status:",y_pred)
print("Accuracy:",accuracy_score(y_test,y_pred)*100,"%")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))
student=[[75,80,70,85,78]]
prediction=model.predict(student)
if prediction[0]==1:
    print("Student is Placed")
else:
    print("Student is Not Placed") 

```

## Output:
<img width="1377" height="537" alt="image" src="https://github.com/user-attachments/assets/edc6208a-7b3b-4e05-879f-75ae575c7fb7" />
<img width="1162" height="407" alt="image" src="https://github.com/user-attachments/assets/57df1a0e-9c94-4547-b5ca-97bb19f9a1cc" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
