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
#fitting model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#predicting results
y_pred=regressor.predict(x_test)
#converting float to int
y_final=np.round(y_pred)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_final)
#accuracy
count=0
for i in range(len(y_final)):
    if(y_final[i]==y_test[i]):
        count=count+1
print(count)            
      