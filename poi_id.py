#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import poi_functions as function 
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score,precision_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments', 
                 'loan_advances', 
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income', 
                 'total_stock_value', 
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive', 
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person', 
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 ]
 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = ["THE TRAVEL AGENCY IN THE PARK","TOTAL"]
for outlier in outliers:
    data_dict.pop(outlier,0)
### Task 3: Create new feature(s)
data_dict = function.total_net_worth(data_dict)
data_dict = function.poi_fraction(data_dict)
features_list += ['total_net_worth','poi_total_fraction']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# storing the data of features in a list to plot in matplotlyb
poi = list(data[:,0])
salary = list(data[:,1])
bonus = list(data[:,5])
expenses = list(data[:,9])
exercised_stock_options =list(data[:,10])
long_term_incentive = list(data[:,12])
shared_receipt_with_poi = list(data[:,19])
total_net_worth = list(data[:,20])
poi_total_fraction = list(data[:,21])

print "A few plots to get the insigths of dataset"
plot_name = "salary vs poi_total_fraction"
function.scatter_plot(poi_total_fraction,salary,poi,plot_name)

plot_name = "bonus vs salary"
function.scatter_plot(salary,bonus,poi,plot_name)

plot_name = "total_net_worth vs poi_total_fraction"
function.scatter_plot(poi_total_fraction,total_net_worth,poi,plot_name)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# selecting 10 most informative features out of 21 using SelectKBest
# feature_scores is a list of features and their K_scores
features,features_list,feature_scores = \
function.select_9_features(features,labels,features_list)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#test_classifier(clf,my_dataset,features_list,folds = 1000)
# remove the hash of the above line to see the test results of the clf
print "tried naive bayes"

from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
dt = tree.DecisionTreeClassifier(min_samples_leaf=1)
#test_classifier(dt,my_dataset,features_list,folds = 1000)
# remove the hash of the above line to see the test results of the clf
print "tried decision trees"

clf = AdaBoostClassifier()
#test_classifier(clf,my_dataset,features_list,folds = 1000)
# remove the hash of the above line to see the test results of the clf
print "tried AdaBoost"

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

estimators = [('pca',PCA(n_components = 7)),('Naive bayes',GaussianNB())]
clf = Pipeline(estimators)
#test_classifier(clf,my_dataset,features_list,folds = 1000)
# remove the hash of the above line to see the test results of the clf
print "tried naive bayes with pca"

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100,criterion="gini",min_samples_leaf=1)
#test_classifier(clf,my_dataset,features_list,folds = 1000)
# remove the hash of the above line to see the test results of the clf
print "tried Random Forest"

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
param_grid = {
         'pca__n_components':[5,6,7,8],
         'classifier__C':[0.1,1,10,1000],
         'classifier__gamma':[0.1,0.01,0.001]
          }
estimators = [('pca',PCA()),('classifier',SVC())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1')
gs.fit(features,labels)
clf = gs.best_estimator_
#test_classifier(clf, my_dataset, features_list, folds = 1000)
print "tried pca with svm in grid search"

 ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
estimators = [('pca',PCA()),('Naive bayes',GaussianNB())]
pipe = Pipeline(estimators)
param_grid = {'pca__n_components':[4,5,6,7,8]}
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1',cv=3)
gs.fit(features_train,labels_train)
clf = gs.best_estimator_
pred = clf.predict(features_test)
print "final model test set (30 percent) stats"
print "accuracy score:", accuracy_score(pred, labels_test)
print "recall:", recall_score(labels_test,pred)
print "precision:", precision_score(labels_test,pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "final model result using StratifiedShuffleSplit"
test_classifier(clf,my_dataset,features_list,folds = 1000)
dump_classifier_and_data(clf, my_dataset, features_list)