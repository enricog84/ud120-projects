#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
#clf = svm.SVC()
#clf = svm.SVC(kernel="rbf", gamma=1.0, C=3.0)
#clf = svm.SVC(kernel="linear", gamma=1.0, C=3.0)
#clf = svm.SVC(kernel="linear")
clf = svm.SVC(kernel="rbf", C=10000.0)

###################################
#EG: FROM NAIVE BAYES
#import numpy as np
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score
### create classifier
#clf = GaussianNB()
###################################

#Reduce training data, reduces accuracy
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
###################

tFit = time()
clf.fit(features_train, labels_train)
print "training/fit time:", round(time()-tFit, 3), "s"

tPred = time()
pred = clf.predict(features_test)
print "preidct time:", round(time()-tPred, 3), "s"

print "prediction for index 10"
print pred[10]
print "prediction for index 26"
print pred[26]
print "prediction for index 50"
print pred[50]

group1 = 0
group2 = 0
for i in range(len(pred)):
	if pred[i] == 0:
		group1 += 1
	if pred[i] == 1:
		group2 += 1
print "prediction number of label 0:"
print group1
print "prediction number of label 1:"
print group2


tAcc1 = time()
#This also works
accuracy1 = clf.score(features_test, labels_test)
print "accuracy1 time:", round(time()-tAcc1, 3), "s"
print "accuracy1 is:"
print(accuracy1)

### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
from sklearn.metrics import accuracy_score
tAcc2 = time()
#accuracy = accuracy_score(features_test, labels_test, normalize=False)
accuracy2 = accuracy_score(labels_test, pred)
print "accuracy2 time:", round(time()-tAcc2, 3), "s"
print "accuracy2 is:"
print(accuracy2)

### draw the decision boundary with the text points overlaid
#from class_vis_eg import prettyPicture, output_image
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

#########################################################


