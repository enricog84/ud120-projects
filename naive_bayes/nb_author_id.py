#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
#sys.path.append("/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### create classifier
clf = GaussianNB()

tFit = time()
### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
print "training/fit time:", round(time()-tFit, 3), "s"

tPred = time()
### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)
print "preidct time:", round(time()-tPred, 3), "s"

tAcc1 = time()
#This also works
accuracy1 = clf.score(features_test, labels_test)
print "accuracy1 time:", round(time()-tAcc1, 3), "s"

### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
from sklearn.metrics import accuracy_score
tAcc2 = time()
#accuracy = accuracy_score(features_test, labels_test, normalize=False)
accuracy2 = accuracy_score(labels_test, pred)
print "accurracy2 time:", round(time()-tAcc2, 3), "s"

print """EG: prediction: """
print(pred)
print """EG: accurracy1: """
print(accuracy1)
print """EG:  accurracy2: """
print(accuracy2)


#########################################################


