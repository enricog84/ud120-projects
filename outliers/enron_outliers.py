#!/usr/bin/python

import pickle
import sys
#EG: Choose somehting that does NOT require a display: https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
#EG: done stuff later
data_dict.pop('TOTAL',0) # Added after finding the outlier
#################
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()
#EG: Save to file instead of displaying it.
matplotlib.pyplot.savefig('./theFile3.png')

# To Find outlier
for key, value in data_dict.items():
    if value['bonus'] == data.max():
        print "Max bonus: "
        print key
        print value['bonus']

print "#######################"
biggest = 0
for key, value in data_dict.items():
    if value['bonus'] > biggest:
        biggest = value['bonus']
        print key
        print value['bonus']
print "#######################"
for key, value in data_dict.items():
    if value['bonus'] > 5000000 and value['salary'] > 1000000 and value['salary'] != 'NaN' and value['bonus'] != 'NaN':
        biggest = value['bonus']
        print key
        print value['bonus']
        print value['salary']

