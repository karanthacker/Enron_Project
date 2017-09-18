# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:02:06 2017

@author: KARAN
"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
feature_scores =[]

# function to add the total payment and total stocks value to get
# the total networth
def total_net_worth (data_dict) :
    features = ['total_payments','total_stock_value']
    
    for key in data_dict :
        name = data_dict[key]
        
        is_null = False 
        
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True
        
        if not is_null:
            name['total_net_worth'] = name[features[0]] + name[features[1]]
        
        else:
            name['total_net_worth'] = 'NaN'
            
    return data_dict                
            
# function to calculate ratio total mails sent/received from poi to total
#mail sent/received 
def poi_fraction(data_dict) :
    for name in data_dict:

        data_point = data_dict[name]

    
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_poi = computeFraction( from_this_person_to_poi, from_messages, from_poi_to_this_person, to_messages )
    
        data_point["poi_total_fraction"] = fraction_poi
    return data_dict
    
def computeFraction(from_this_person_to_poi, from_messages, from_poi_to_this_person, to_messages) :
    if from_this_person_to_poi != 'NaN' and from_poi_to_this_person != 'NaN':
        
        fraction = (from_poi_to_this_person + from_this_person_to_poi)/(float(from_messages + to_messages))
        return fraction
    
    elif from_this_person_to_poi != 'NaN':
        
        fraction = (from_this_person_to_poi)/(float(from_messages + to_messages))
        return fraction
    
    elif from_poi_to_this_person != 'NaN':
        
        fraction = (from_poi_to_this_person)/(float(from_messages + to_messages))
        return fraction
    
    else :
        fraction = 0
        return fraction
    
# function to make a scatter plot from the list of features    
def scatter_plot(x,y,poi,name):   
    for i in range(len(x)):
        if poi[i]:
            plt.scatter(x[i],y[i],color = 'r')
        else:
            plt.scatter(x[i],y[i],color = 'b')
            
    plt.title(name)
    plt.show()

# function to return a list of 10 most informative features out of a list of 21
# the univariate feature selection is based on funct SelectKBest on the criteria
# ANOVA F-value'''    
def select_10_features(features,labels,features_list,k=10) :
    clf = SelectKBest(f_classif,k)
    selected_features = clf.fit_transform(features,labels)
    features_selected=[features_list[i+1] for i in clf.get_support(indices=True)]
    feature_scores = zip(features_list[1:11],clf.scores_[:10])
    feature_scores = sorted(feature_scores,key=lambda x: x[1],reverse=True)
    print 'Final 10 Features selected by SelectKBest:'
    print features_selected
    return selected_features, ['poi'] + features_selected, feature_scores