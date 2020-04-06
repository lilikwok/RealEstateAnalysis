import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn import tree
# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
# import export_graphviz 
from sklearn.tree import export_graphviz  
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = y_true.values
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,2)
#########################################################
########## load data                             ########
#########################################################
m1_c1 = pd.read_csv("m1_c1.csv", sep=',', header=0)
m1_c2 = pd.read_csv("m1_c2.csv", sep=',', header=0)
m1_c3 = pd.read_csv("m1_c3.csv", sep=',', header=0)
m1_full = pd.read_csv("m1_full.csv", sep=',', header=0)
m2_c1 = pd.read_csv("m2_c1.csv", sep=',', header=0)
m2_c2 = pd.read_csv("m2_c2.csv", sep=',', header=0)
m2_c3 = pd.read_csv("m2_c3.csv", sep=',', header=0)
m2_full = pd.read_csv("m2_full.csv", sep=',', header=0)

#print(len(m1_c1))
#print(len(m1_c2))
#print(len(m1_c3))

#print(len(m2_c1))
#print(len(m2_c2))
#print(len(m2_c3))

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

##################################################################################
########## function for full data set                                #############
##################################################################################

def fullModel(m1_c1):

    y = m1_c1['sale_price']

    m1_c1['View_Quality'] = m1_c1['View_Quality'] .astype('category')
    m1_c1['Waterfront_Type'] = m1_c1['Waterfront_Type'] .astype('category')
    m1_c1['withInSewerImprovement'] = m1_c1['withInSewerImprovement'] .astype('category')
    m1_c1['near_firestation'] = m1_c1['near_firestation'] .astype('category')
    m1_c1['near_hospital'] = m1_c1['near_hospital'] .astype('category')
    m1_c1['near_libraries'] = m1_c1['near_libraries'] .astype('category')
    m1_c1['near_policestation'] = m1_c1['near_policestation'] .astype('category')
    m1_c1['near_waterplants'] = m1_c1['near_waterplants'] .astype('category')
    m1_c1['condition'] = m1_c1['condition'] .astype('category')
    m1_c1['quality'] = m1_c1['quality'] .astype('category')
    m1_c1['attic_finished_square_feet'] = m1_c1['attic_finished_square_feet'] .astype('category')
    m1_c1['basement_square_feet'] = m1_c1['basement_square_feet'] .astype('category')
    m1_c1['basement_finished_square_feet'] = m1_c1['basement_finished_square_feet'] .astype('category')
    m1_c1['porch_square_feet'] = m1_c1['porch_square_feet'] .astype('category')
    m1_c1['attached_garage_square_feet'] = m1_c1['attached_garage_square_feet'] .astype('category')
    m1_c1['detached_garage_square_feet'] = m1_c1['detached_garage_square_feet'] .astype('category')
    m1_c1['fireplaces'] = m1_c1['fireplaces'] .astype('category')
    m1_c1['near_private_school'] = m1_c1['near_private_school'] .astype('category')
    m1_c1['near_elementary_school'] = m1_c1['near_elementary_school'] .astype('category')
    m1_c1['near_high_school'] = m1_c1['near_high_school'] .astype('category')
    m1_c1['near_college'] = m1_c1['near_college'] .astype('category')
    m1_c1['Crime_Num'] = m1_c1['Crime_Num'].fillna(0)


    X = m1_c1[['Land_Net_Acres','View_Quality', 
           'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
           'near_firestation', 'near_hospital', 'near_libraries',
           'near_policestation', 'near_waterplants', 'square_feet',
           'condition', 'quality', 'attic_finished_square_feet',
           'basement_square_feet', 'basement_finished_square_feet',
           'porch_square_feet', 'attached_garage_square_feet',
           'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
           'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]

    ###########  decison tree regression ##############

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

    # create a regressor object 
    regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
    random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

    # fit the regressor with X and Y data 
    reg = regressor.fit(X_train, y_train) 
    y_pred = reg.predict(X_test)  
    MSE = mean_squared_error(y_test, y_pred)
    print("MSE of "+ get_df_name(m1_c1) + '\t' ,MSE)
    print('MAPE of '+ get_df_name(m1_c1) + '\t',mean_absolute_percentage_error(y_test, y_pred),'%')
    # export the decision tree to a tree.dot file 
    # for visualizing the plot easily anywhere 
    
    export_graphviz(regressor, out_file =get_df_name(m1_c1) + '_3_cluster_tree.dot',feature_names=X.columns)  

fullModel(m1_c1)
fullModel(m1_c2)
fullModel(m1_c3)
fullModel(m1_full)


##################################################################################
########## function for original data set                            #############
##################################################################################
def orignalModel(m2_c1):
    y = m2_c1['sale_price']

    m2_c1['View_Quality'] = m2_c1['View_Quality'] .astype('category')
    m2_c1['Waterfront_Type'] = m2_c1['Waterfront_Type'] .astype('category')
    m2_c1['condition'] = m2_c1['condition'] .astype('category')
    m2_c1['quality'] = m2_c1['quality'] .astype('category')
    m2_c1['attic_finished_square_feet'] = m2_c1['attic_finished_square_feet'] .astype('category')
    m2_c1['basement_square_feet'] = m2_c1['basement_square_feet'] .astype('category')
    m2_c1['basement_finished_square_feet'] = m2_c1['basement_finished_square_feet'] .astype('category')
    m2_c1['porch_square_feet'] = m2_c1['porch_square_feet'] .astype('category')
    m2_c1['attached_garage_square_feet'] = m2_c1['attached_garage_square_feet'] .astype('category')
    m2_c1['detached_garage_square_feet'] = m2_c1['detached_garage_square_feet'] .astype('category')
    m2_c1['fireplaces'] = m2_c1['fireplaces'] .astype('category')

    X = m2_c1[['Land_Net_Acres','View_Quality', 
        'Waterfront_Type', 'square_feet',
        'condition', 'quality', 'attic_finished_square_feet',
        'basement_square_feet', 'basement_finished_square_feet',
        'porch_square_feet', 'attached_garage_square_feet',
        'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
        'bathrooms', 'year_built']] 

    ###########  decison tree regression ##############

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109)  

    # create a regressor object 
    regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=7, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
    random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

    # fit the regressor with X and Y data 
    reg = regressor.fit(X_train, y_train) 
    y_pred = reg.predict(X_test)  
    MSE = mean_squared_error(y_test, y_pred)
    print("MSE of "+ get_df_name(m2_c1) + '\t',MSE)
    print('MAPE of '+ get_df_name(m2_c1) + '\t',mean_absolute_percentage_error(y_test, y_pred),'%')
    # export the decision tree to a tree_5cluster.dot file 
    # for visualizing the plot easily anywhere 
    export_graphviz(regressor, out_file =get_df_name(m2_c1) + '_3_cluster_tree.dot',feature_names=X.columns)  

orignalModel(m2_c1)
orignalModel(m2_c2)
orignalModel(m2_c3)
orignalModel(m2_full)
    