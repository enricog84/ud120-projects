#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )

#EG: Remove outlier by custom stopword things
#EG: Nah, the removal is done in vectorize_text.py and stored in the files above
#for key, word in enumerate(word_data):
#    stopwords = ["sara", "shackleton", "chris", "germani", "sshacklensf"]
#    for toRemove in stopwords:
#        word_data[key] = word.replace(toRemove, "")

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
#EG: Has been deprecated, see https://stackoverflow.com/questions/46572475/module-sklearn-has-no-attribute-cross-validation#46572558
#from sklearn import cross_validation
#features_train, features_test, labels_train, labels_test = cross_validation.cross_validation(word_data, authors, test_size=0.1, random_state=42)
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.metrics import accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "Number of features_train:", len(features_train)
accuracy = accuracy_score(labels_test, pred)
print "accuracy is:", accuracy
featureList = vectorizer.get_feature_names()
for key, feature_importance in enumerate(clf.feature_importances_):
    if feature_importance > 0.2:
        print "feature key:", key
        print "feature importances: ", feature_importance
        print "feature word is: ", featureList[key]
print "MAX feature importances: ", max(clf.feature_importances_)