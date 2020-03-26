# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:15 2019

@author: M AQEEL M
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''%matplotlib notebook'''
data=pd.read_csv('DatasetIdDrop.csv')
#column=['age','sex','cp','trsetbp','chol','fbs','rstecg','thalach','exang','oldpeak','slope','ca','thal','target']
#data.columns_column
#print(data.head()) # to display top 5 rows
#print(data.describe()) # to display meean, stndard deviation, row count and percentiles
print(data)
X=data.drop('active',axis=1)
Y=data['active']
#coding of ann started
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
ann_clf=MLPClassifier()
parameters={'solver':['lbfgs'],
            'alpha':[1e-4],
            'hidden_layer_sizes':(11,18,18,2),
            'random_state':[1]}
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  

acc_scorer=make_scorer(accuracy_score)
grid_obj=GridSearchCV(ann_clf,parameters,scoring=acc_scorer)
grid_obj=grid_obj.fit(X_train,Y_train)
ann_clf=grid_obj.best_estimator_
ann_clf.fit(X_train,Y_train)
Y_pred_ann=ann_clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_ann=confusion_matrix(Y_test,Y_pred_ann)
cm_ann
ann_result=accuracy_score(Y_test,Y_pred_ann)
ann_result
print(ann_result)
mat = confusion_matrix(Y_test,Y_pred_ann)
names = np.unique(Y_pred_ann)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
print("this part is for runtime")
#786 my first test
z=pd.read_csv('two.csv')
print(z)
ann_clf.fit(X_train,Y_train)
z_pred_ann=ann_clf.predict(z)
print("ann result is")
print(z_pred_ann)
