#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
#print len(enron_data)
#print enron_data.keys()
#print len(enron_data['METTS MARK'])
poiCnt = 0
largestPayment = 0
guyWithLargestPayment = ''
quantitiedSalaryCount = 0
knownEmailCount = 0
totalPaymentNanCount = 0
totalPaymentPoiNanCount = 0
numberOfPois = 0
minExercisedStockOptions = 999999999999999999
maxExercisedStockOptions = 0
for key in enron_data:
    if (key in ['SKILLING JEFFREY K', 'LAY KENNETH L', 'FASTOW ANDREW S']) and enron_data[key]['total_payments'] > largestPayment:
        largestPayment = enron_data[key]['total_payments']
        guyWithLargestPayment = key
#    if enron_data[key]['director_fees'] != 'NaN':
#       print key
#       print enron_data[key]['director_fees']
    if enron_data[key]['poi'] == True:
        poiCnt += 1
    if enron_data[key]['salary'] != 'NaN':
        quantitiedSalaryCount += 1
    if enron_data[key]['email_address'] != 'NaN':
        knownEmailCount += 1
    if enron_data[key]['poi'] == True:
        numberOfPois += 1
    if enron_data[key]['total_payments'] == 'NaN':
        totalPaymentNanCount += 1
    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == True:
        totalPaymentPoiNanCount += 1
    if enron_data[key]['exercised_stock_options'] != 'NaN' and enron_data[key]['exercised_stock_options'] < minExercisedStockOptions:
        minExercisedStockOptions = enron_data[key]['exercised_stock_options']
    if enron_data[key]['exercised_stock_options'] != 'NaN' and enron_data[key]['exercised_stock_options'] > maxExercisedStockOptions:
        maxExercisedStockOptions = enron_data[key]['exercised_stock_options']

#print largestPayment
#print guyWithLargestPayment
print poiCnt
#print enron_data['PRENTICE JAMES']['total_stock_value']
#print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
#print enron_data['SKILLING JEFFREY K']['exercised_stock_options']
#print enron_data['SKILLING JEFFREY K']
#print quantitiedSalaryCount
#print knownEmailCount
print totalPaymentNanCount
print len(enron_data.keys())
#print numberOfPois
#print totalPaymentPoiNanCount
print minExercisedStockOptions
print maxExercisedStockOptions

import sys
sys.path.append("../tools/")
from feature_format import featureFormat
#print featureFormat(enron_data, ['total_payments'])