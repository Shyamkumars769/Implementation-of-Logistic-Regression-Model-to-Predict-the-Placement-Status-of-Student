# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Shyam Kumar.S

RegisterNumber: 212224040315

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![image](https://github.com/user-attachments/assets/41e47659-161c-453a-b687-a1d0871e213e)

### Checking the null() function
![image](https://github.com/user-attachments/assets/05152417-eb8b-4569-be6b-b78536945332)

### Print Data:
![image](https://github.com/user-attachments/assets/7b102e20-da1f-4fe9-82ef-3f0f0eaedd2a)

### Y_prediction array
![image](https://github.com/user-attachments/assets/936e6639-15ca-4022-9673-07fc026d99bb)

### Accuracy value
![image](https://github.com/user-attachments/assets/05203798-370a-4a1f-9915-417cfd002256)

### Confusion array
![image](https://github.com/user-attachments/assets/8960bb46-236c-483b-90cd-3f90163a776f)


### Classification Report
![image](https://github.com/user-attachments/assets/f3b36c2e-3134-4b6b-aeb7-4366c64e4ccf)

### Prediction of LR
![image](https://github.com/user-attachments/assets/168cf381-1492-440e-af50-1e8a4321281b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
