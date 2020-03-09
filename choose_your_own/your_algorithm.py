#!/usr/bin/python

import matplotlib
#EG: Choose somehting that does NOT require a display: https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
#EG: Save to file instead of displaying it.
plt.savefig('./initial_points.png')
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.metrics import accuracy_score

#Nearest neighbors#######################
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
#clf = KNeighborsClassifier(n_neighbors=3, weights='distance', leaf_size=5)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)
print "accuracy is:"
print(accuracy)
#############################

#Random forest##############
"""
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()
clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 100)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)
print "accuracy is:"
print(accuracy)
"""
#####################

"""
#AdaBoost###########
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)
print "accuracy is:"
print(accuracy)
"""
#############################

#Naive Bayes##################
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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
"""
#############################

#SVM#################
"""
from sklearn import svm
#clf = svm.SVC()
#clf = svm.SVC(kernel="rbf", gamma=1.0, C=3.0)
#clf = svm.SVC(kernel="linear", gamma=1.0, C=3.0)
clf = svm.SVC(kernel="rbf", C=10000.0)

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
"""
##############################

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
