import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#preprocessing
dataset=pd.read_csv('data.csv')
x=dataset.iloc[:, :-1].values
y = dataset.iloc[:, 21].values
#splitting dataest
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier= SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#accuracy
count=0
for i in range(len(y_pred)):
    if(y_pred[i]==y_test[i]):
        count=count+1
print(count)