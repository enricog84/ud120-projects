#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print "accuracy is:", accuracy

poiCnt = 0
#print labels_test
for value in labels_test:
    if value == 1.0:
        poiCnt += 1

print "poiCnt in test is: ", poiCnt
print "total test data: ", len(labels_test)

#If your identifier predicted 0.0 (not POI) for everyone in the test set, what would your accuracy be?
print 1.0 - 4.0/29

#Get numer of true positives
truePosCnt = 0
for key in xrange(len(pred)):
    if pred[key] == 1.0 and pred[key] == labels_test[key]:
        truePosCnt += 1
print "truePosCnt is: ", truePosCnt
#print pred
#print labels_test

#Calc precision
from sklearn.metrics import precision_score
print "precision score: ", precision_score(labels_test, pred)
#Calc recall
from sklearn.metrics import recall_score
print "recall score: ", recall_score(labels_test, pred)

#True positive on made up data
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
truePosCnt = 0
for key in xrange(len(true_labels)):
    if predictions[key] == 1 and predictions[key] == true_labels[key]:
        truePosCnt += 1
print "made up data truePosCnt is: ", truePosCnt
#True negatives
trueNegCnt = 0
for key in xrange(len(true_labels)):
    if predictions[key] == 0 and predictions[key] == true_labels[key]:
        trueNegCnt += 1
print "true negatives on made up data", trueNegCnt
#False positives
falsePosCnt = 0
for key in xrange(len(true_labels)):
    if predictions[key] == 1 and predictions[key] != true_labels[key]:
        falsePosCnt += 1
print "false positives on made up data", falsePosCnt
#False negatives
falseNegCnt = 0
for key in xrange(len(true_labels)):
    if predictions[key] == 0 and predictions[key] != true_labels[key]:
        falseNegCnt += 1
print "false negatives on made up data", falseNegCnt
#precision
print "precision score on made up data: ", precision_score(true_labels, predictions)
print "recall score on made up data: ", recall_score(true_labels, predictions)

