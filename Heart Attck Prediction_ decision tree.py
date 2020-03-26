# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:42:39 2019

@author: M AQEEL M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score
dataset=pd.read_csv('DatasetIdDrop.csv')
dataset.shape
dataset.head()
X=dataset.drop('active',axis=1)
Y=dataset['active']
print(X)
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred)) 
decisiontree_result=accuracy_score(Y_test,Y_pred)
decisiontree_result
print(decisiontree_result)
mat = confusion_matrix(Y_pred, Y_test)
names = np.unique(Y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
z=pd.read_csv('two.csv')
print(z)
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, Y_train)
ZdecisionTree = classifier.predict(z)
print(ZdecisionTree)  
