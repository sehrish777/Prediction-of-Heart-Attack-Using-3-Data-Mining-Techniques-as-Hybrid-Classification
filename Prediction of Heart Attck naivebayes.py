# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 02:17:16 2019

@author: Shahid Abbas
"""

# Import packages
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score

# Import data
#training = pd.read_csv('.csv')
#test = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')
data=pd.read_csv('DatasetIdDrop.csv')
print(data)
X=data.drop('active',axis=1)
Y=data['active']
# Create the X, Y, Training and Test
#xtrain = training.drop('Species', axis=1)
#ytrain = training.loc[:, 'Species']
#xtest = test.drop('Species', axis=1)
#ytest = test.loc[:, 'Species']

#from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20) 
# Init the Gaussian Classifier
model = GaussianNB()

# Train the model 
model.fit(X_train, Y_train)

# Predict Output 
pred = model.predict(X_test)
print("prediction is")
print(pred)
naiveBayes_result=accuracy_score(Y_test,pred)
naiveBayes_result
print(naiveBayes_result)

# Plot Confusion Matrix
mat = confusion_matrix(pred, Y_test)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
print("naive bayes testing start")

print("another data for naive bayes is")
data2=pd.read_csv('two.csv')
print(data2)
predict = model.predict(data2)
print(predict)