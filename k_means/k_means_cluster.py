#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""



#EG: Import future division to get a float instead of an integer from division of 2 integers...
from __future__ import division
import pickle
import numpy
#EG: Choose somehting that does NOT require a display: https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    #plt.show()
    #EG: Save to file instead of displaying it.
    plt.savefig('./theFile1.png')



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
#EG: Updated on later task
feature_3 = "total_payments"
#features_list = [poi, feature_1, feature_2, feature_3]
#####

data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

#EG: For later course class
from sklearn import preprocessing
#feature_1_list =  featureFormat(data_dict, [feature_1] )
#feature_2_list =  featureFormat(data_dict, [feature_2] )
#feature_1_list = numpy.append(feature_1_list, [200000.])
#feature_2_list = numpy.append(feature_2_list, [1000000.])
feature_1_list = []
feature_2_list = []
for key in data_dict:
    value = data_dict[key]
    if value[feature_1] != 'NaN':
        feature_1_list.append([value[feature_1]])
    if value[feature_2] != 'NaN':
        feature_2_list.append([value[feature_2]])
feature_1_list.append([200000.])
feature_2_list.append([1000000.])
#print feature_1_list
numpyFeatureList = numpy.array([feature_1_list, feature_2_list])
#print data
#print numpyFeatureList
#print finance_features[1]
scaler = preprocessing.MinMaxScaler()
scaledSalary = scaler.fit_transform(numpy.array(feature_1_list))
#print scaledSalary
scaledExercisedStockOptions = scaler.fit_transform(numpy.array(feature_2_list))
#print scaledExercisedStockOptions
data = numpy.array([data[0], scaledSalary, scaledExercisedStockOptions])
#print finance_features
#poi, finance_features = targetFeatureSplit( data )
#print finance_features
########


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
#for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
#plt.show()
#EG: Save to file instead of displaying it.
plt.savefig('./theFile2.png')

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(finance_features)
pred = kmeans.predict(finance_features)

#Get some maximum and minimum values
minExercisedStockOptions = 999999999999999999
maxExercisedStockOptions = 0
minSalary = 999999999999999999
maxSalary = 0
enron_data = data_dict
for key in enron_data:
    if enron_data[key]['exercised_stock_options'] != 'NaN' and enron_data[key]['exercised_stock_options'] < minExercisedStockOptions:
        minExercisedStockOptions = enron_data[key]['exercised_stock_options']
    if enron_data[key]['exercised_stock_options'] != 'NaN' and enron_data[key]['exercised_stock_options'] > maxExercisedStockOptions:
        maxExercisedStockOptions = enron_data[key]['exercised_stock_options']
    if enron_data[key]['salary'] != 'NaN' and enron_data[key]['salary'] < minSalary:
        minSalary = enron_data[key]['salary']
    if enron_data[key]['salary'] != 'NaN' and enron_data[key]['salary'] > maxSalary:
        maxSalary = enron_data[key]['salary']
print minExercisedStockOptions
print maxExercisedStockOptions
print minSalary
print maxSalary

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
