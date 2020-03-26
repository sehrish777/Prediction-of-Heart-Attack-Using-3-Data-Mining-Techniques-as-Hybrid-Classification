# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:17:09 2019

@author: Shahid Abbas
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.naive_bayes import GaussianNB
dataset=pd.read_csv('DatasetIdDrop.csv')
print("the complete data set is as follows")
print(dataset)
X=dataset.drop('active',axis=1)
Y=dataset['active']
print(X)
print("the label is:")
print(Y)
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, Y_train)
print("training data of decision tree")
print(X_train, Y_train)
print("testing data of decision tree")
print(X_test,Y_test)
Y_pred = classifier.predict(X_test)
print(Y_pred) 
 
 
decisionTree_result=accuracy_score(Y_test,Y_pred )
a=confusion_matrix(Y_test, Y_pred)  
b=classification_report(Y_test, Y_pred) 
print("confusion matrix of decision tree is")
print(a)
print("confusion matrix REPORT of decision tree is")
print(b)
print("graph of decision tree is")
#the below code is for run time testing of model by passing the value of one patient and checking the prediction of heart Attack,
# this new single row is totally different from testing and training data
print("decision tree single row is")
z=pd.read_csv('three.csv')
print(z)
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, Y_train)
ZdecisionTree = classifier.predict(z)
print(ZdecisionTree) 

names = np.unique(Y_pred)

sns.heatmap(a, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')


print("decisionTree result")
print(decisionTree_result)
#end of decision tree as it has trained and test data now move to ANN
print("ann coding started")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
ann_clf=MLPClassifier()


parameters={'solver':['lbfgs'],
            'alpha':[1e-4],
            'hidden_layer_sizes':(11,18,18,2),
            'random_state':[1]}
#replace X with x_anntrain and Y with y_anntrain
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
print("training data of ann")
print(X_train, Y_train)
print("testing data of ann")
print(X_test,Y_test)
acc_scorer=make_scorer(accuracy_score)
grid_obj=GridSearchCV(ann_clf,parameters,scoring=acc_scorer)
grid_obj=grid_obj.fit(X_train,Y_train)
ann_clf=grid_obj.best_estimator_
ann_clf.fit(X_train,Y_train)
y_anntrain_pred_ann=ann_clf.predict(X_test)
#taken from decision tree
from sklearn.metrics import classification_report, confusion_matrix  
print("confusion matrix of ANN combined with decision tree is")
c=confusion_matrix(Y_test, y_anntrain_pred_ann) 
print(c)
print("classification report of ANN combined with decision tree is")
d=classification_report(Y_test, y_anntrain_pred_ann)
print(d)
ann_result=accuracy_score(Y_test, y_anntrain_pred_ann)
print("the combined result of decision tree and ann is")
combine=ann_result*100
print(combine) 
print("graph of ANN is ")

'''
names = np.unique( y_anntrain_pred_ann)
sns.heatmap(c, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
'''


#ann completed with 1% improved accuracy
print("this part is for runtime")
#786 my first test
z=pd.read_csv('three.csv')
print(z)
ann_clf.fit(X_train,Y_train)
z_pred_ann=ann_clf.predict(z)
print("ann result is")
print(z_pred_ann)
if(ZdecisionTree == z_pred_ann):
    if(ZdecisionTree ==1) and (z_pred_ann==1):
        combination=1
        print(1)
        #return 1 strong chances of having heart attack
        print("yes strong chances taht you  have heart attack in future so follow precautions to avoid it")
    else:
        combination=0
        print(0)
        print("congrats! you are save strong chances that you will not have heart attack")
        #return 3  strong chances of not having heart attack
        #end of inner if
        
else:
    if (ZdecisionTree ==1) and (z_pred_ann==0):#another if within else statement
              #return 2 means weak chances of having heart attack
              print(" no chances")
              combination=0
    else:#another else within outer else
              #return 3 mean weak chances of not having heart attack
              combination=1
              print(" heart attack weak probability")


#now turn to naive bayes
              

print("naive bayes code started")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data=pd.read_csv('DatasetIdDrop.csv')
print(data)
X=dataset.drop('active',axis=1)
Y=dataset['active']
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20) 
model = GaussianNB()

# Train the model 
model.fit(X_train, Y_train)

# Predict Output 

pred = model.predict(X_test)
print("prediction is")
print(pred)
naiveBayes_result=accuracy_score(Y_test,pred)
print("naive bayes accuracy is")
print(naiveBayes_result)
e=confusion_matrix(Y_test,pred)
print("naive bayes confusion matrix")
print(e)
f=classification_report(Y_test,pred)
print("classification report of confusion matrix of naive bayes is")
print(f)

# Plot Confusion Matrix
mat = confusion_matrix(Y_test,pred)
'''
names2 = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names2, yticklabels=names2)
plt.xlabel('Truth')
plt.ylabel('Predicted')
'''

print("naive bayes testing start")

print("another data for naive bayes is")
data2=pd.read_csv('three.csv')
print(data2)
predict = model.predict(data2)
print(predict)


if(predict == combination):
    if(predict ==1) and (combination==1):
        #return 1 strong chances of having heart attack
        final=1
        print(final)
        print("yes strong chances taht you  have heart attack in future so follow precautions to avoid it")
    else:
        print("congrats! you are save strong chances that you will not have heart attack")
        #return 3  strong chances of not having heart attack
        final=0
        print(final)
        #end of inner if
        
else:
    if(predict ==0) and (combination==1):#another if within else statement
              #return 2 means weak chances of having heart attack
              print("YES you may have heart attack weak chances BE CAREFUL WEAK CHANCES")
              final=3
              print(final)
    else:#another else within outer else
              #return 3 mean weak chances of not having heart attack
              final=2
              print(final)
              print(" weak probability of NO")
      
'''
print(predict)
if predict == y_anntrain_pred_ann:
         print("YES results are correct")
else:
         print("incorrect results")
         
'''         
    
             
