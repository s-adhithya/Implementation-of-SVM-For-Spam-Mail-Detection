# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the necessary python packages using import statements.

2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3. Split the dataset using train_test_split.

4. Calculate Y_Pred and accuracy.

5. Print all the outputs.

6. End the Program.
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Adhithya.S
RegisterNumber: 212222240003
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

import chardet 
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### Result output
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/94c3a24e-9677-486e-9c70-dc70961926b0)


### data.head()
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/58fc1564-4953-4653-b85f-eb0a7dfbd4d0)


### data.info()
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/accb0bc3-5363-46f1-864a-64f1b0171409)


### data.isnull().sum()
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/075fc65c-d827-436c-9b2f-1f4c9be3898d)


### Y_prediction value
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/363cf3f7-fbc4-4155-8967-0553d8be311b)


### Accuracy value
![image](https://github.com/s-adhithya/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497423/1ae9e592-581e-43cd-bdc0-b157b09cb8fa)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
