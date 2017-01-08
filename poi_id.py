# -*- coding: utf-8 -*-
#!/usr/bin/python

# importing necessary modules
import sys
import pickle
import os
import matplotlib

# appending local paths
sys.path.append("C:/Users/Lukasz/ud120-projects/tools")
sys.path.append("C:/Users/Lukasz/ud120-projects/final_project")

# importing necessary modules (2nd batch)
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# local directory change to load the dataset
os.chdir('C:/Users/Lukasz/ud120-projects/final_project')

# load the file
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# how many records
print len(data_dict.keys())

# what are the keys
print data_dict.keys()

# what are the keys for each record
print data_dict['LAY KENNETH L']

# how many of features we can get
print len(data_dict['LAY KENNETH L'].keys())

# how many POIs are there
count = 0
for key in data_dict.keys():
    if data_dict[key]['poi'] == 1:
        count += 1
print count 

# Function for adding a new feature.    
# Takes two features where one is a subset of the other.
# Returns data with a new feature (percentage value).

def new_perc_feature(data, name, key1, key2):
    for key in data:
        try:
            x = data[key][key1]
            y = data[key][key2]
            data[key][name] = round(100.0 * int(y) / int(x), 2)
        except (ValueError, ZeroDivisionError):
            data[key][name] = 'NaN'
    return data

# New features added
data_dict2 = new_perc_feature(data_dict, '%frompoi', 'from_messages', 'from_this_person_to_poi') 
data_dict3 = new_perc_feature(data_dict2, '%topoi', 'to_messages', 'from_poi_to_this_person')

# Newly composed dataset
my_dataset = data_dict3

# All features with the exception of an e-mail address, which is a text
# variable.

all_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi', '%frompoi', '%topoi']

# The function that counts the number of NaN values and returns a sorted list
# of tuples, showing NaN values for each feature and label

def count_nans(dictionary):
    result = {}
    for key in dictionary:
        for key2 in dictionary[key]:
            if key2 not in result.keys():
                result[key2] = 0
                if dictionary[key][key2] == 'NaN':
                    result[key2] += 1
            else:
                if dictionary[key][key2] == 'NaN':
                    result[key2] += 1
    sortedd = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return sortedd
    
print count_nans(data_dict3)   



# the following function will help us to first identify graphically if there are
# certain outliers that need to be taken care of

data = featureFormat(my_dataset, all_features_list, sort_keys = True)

# the function that plots a scatterplot of any two variables, useful for
# finding outliers

def plot_outliers(data, i, j):
    for point in data:
        x = point[i]
        y = point[j]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(all_features_list[i])
    matplotlib.pyplot.ylabel(all_features_list[j])
    return matplotlib.pyplot.show()

plot_outliers(data, 1, 3)

# the function that sorts all values of a specific feature, can be used to
# find the record where an outlier shows up.

def identify_outliers(data, trait):
    result = {}
    for key in data:
        if not data[key][trait] == 'NaN':
            result[key] = data[key][trait]
    return sorted(result.items(), key=lambda x: x[1], reverse=True)

identify_outliers(my_dataset, 'total_payments')[:5]

# removing outliers and data quirks

my_dataset.pop('TOTAL', None)
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', None)
my_dataset.pop('LOCKHART EUGENE E', None)

print len(my_dataset.keys())

# The list of features as selected manually before using SelectKBest
# and DT feature_importances_

analyzed_features_list = ['poi', 'salary', 'total_payments', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'restricted_stock', 'shared_receipt_with_poi', '%frompoi', '%topoi']


data = featureFormat(my_dataset, analyzed_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# a function that selects and then returns k best features
# the 'lista' argument refers to the list with feature names
def select_features(how_many, lista):
    int_dict = {'salary' : 0, 'total_payments' : 0, 
                           'total_stock_value' : 0, 'expenses' : 0, 
                           'exercised_stock_options' : 0, 'other' : 0,
                           'restricted_stock' : 0, 'shared_receipt_with_poi' : 0, 
                           '%frompoi' : 0, '%topoi' : 0}
    # I run the selection on different data split as the initial dataset
    # is too small to show meaningful results after trying once
    for i in range(20,120):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        b = SelectKBest(f_classif, k=how_many)
        b.fit_transform(features_train, labels_train)
        indexes = []
        for idx, i in enumerate(b.get_support()):
            if i == True:
                indexes.append(idx)
        for elem in indexes:
            int_dict[lista[elem + 1]] += 1
    return int_dict

# four best features chosen
select_features(4, analyzed_features_list)
fin_feat_kbest = ['poi', 'exercised_stock_options', 'salary', 'total_stock_value', '%frompoi']

# a selection that shows importances of features when used against a DT classifier
# the highest average importances are used to pick the most important features
 
def select_tree_features():
    int_list = [[], [], [], [], [], [], [], [], [], []]
    for i in range(20,120):
        clf = tree.DecisionTreeClassifier()
        features_train, features_test, labels_train, labels_test = \
                    train_test_split(features, labels, test_size=0.3, random_state=i)
        clf.fit(features_train, labels_train)
        for idx, elem in enumerate(clf.feature_importances_):
            int_list[idx].append(elem)
    averages = []
    for elem in int_list:
        averages.append(float(sum(elem) / len(elem)))
    return averages
        
get = select_tree_features()

tree_features = {}

# show which features correspond to average importances
for idx, elem in enumerate(analyzed_features_list[1:]):
    tree_features[elem] = get[idx]
print tree_features
    

fin_feat_tree = ['poi', '%frompoi', 'shared_receipt_with_poi', 'exercised_stock_options', 'expenses']

#Gaussian NB comparison
clf = GaussianNB()
test_classifier(clf, my_dataset, fin_feat_kbest, folds = 1000)

clf = GaussianNB()
test_classifier(clf, my_dataset, fin_feat_tree, folds = 1000)

# preparing ground for DT classifier
data = featureFormat(my_dataset, fin_feat_kbest, sort_keys = True)
data2 = featureFormat(my_dataset, fin_feat_tree, sort_keys = True)
labels, features = targetFeatureSplit(data)
labels2, features2 = targetFeatureSplit(data2)

tuned_parameters_tree = [{'max_depth': [2,3,4,5,6], 'min_samples_leaf': [1,2,3,4]}]

# the function that allows to run the GridSearchCV any number of times on a set 
# of consecutive random states. It shows how many times a specific parameter value
# pops up.

def various_states_val(j,k,feat,lab):
    tuned_dict = {'max_depth' : {2:0, 3:0, 4:0, 5:0, 6:0}, 
                  'min_samples_leaf': {1:0, 2:0, 3:0, 4:0}}
    for i in range (j,k):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(feat, lab, test_size=0.3, random_state=i)
        tree2 = tree.DecisionTreeClassifier()
        clf = GridSearchCV(tree2, tuned_parameters_tree, scoring = 'f1')
        clf.fit(features_train, labels_train)
        tuned_dict['max_depth'][clf.best_params_['max_depth']] += 1
        tuned_dict['min_samples_leaf'][clf.best_params_['min_samples_leaf']] += 1 
    return tuned_dict

print various_states_val(20,520,features,labels)
print various_states_val(20,520,features2,labels2)   

# classifiers tested with specific parameters
clf = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 1)
test_classifier(clf, my_dataset, fin_feat_kbest, folds = 1000)

clf = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 1)
test_classifier(clf, my_dataset, fin_feat_tree, folds = 1000)

dump_classifier_and_data(clf, my_dataset, fin_feat_tree)