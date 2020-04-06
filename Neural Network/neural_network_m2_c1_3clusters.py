import pandas as pd
import patsy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools as sm_tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
import graphviz
import os
import matplotlib.pyplot as plt
from sklearn.utils import check_array

df_pierce_county = pd.read_csv("m2_c1_kmean_3clusters.csv", sep=',', header=0)

print(df_pierce_county.describe())
print(df_pierce_county.info())

def mean_absolute_percentage_error(y_test, y_pred):
    y_test = y_test.values
    y_test = y_test.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_test = check_array(y_test)
    y_pred = check_array(y_pred)
    return np.mean(np.abs((y_test - y_pred)/y_test)) * 100

y = df_pierce_county['sale_price']

print(df_pierce_county.dtypes)

df_pierce_county['View_Quality'] = df_pierce_county['View_Quality'] .astype('category')
df_pierce_county['Waterfront_Type'] = df_pierce_county['Waterfront_Type'] .astype('category')
df_pierce_county['condition'] = df_pierce_county['condition'] .astype('category')
df_pierce_county['quality'] = df_pierce_county['quality'] .astype('category')
df_pierce_county['attic_finished_square_feet'] = df_pierce_county['attic_finished_square_feet'] .astype('category')
df_pierce_county['basement_square_feet'] = df_pierce_county['basement_square_feet'] .astype('category')
df_pierce_county['basement_finished_square_feet'] = df_pierce_county['basement_finished_square_feet'] .astype('category')
df_pierce_county['porch_square_feet'] = df_pierce_county['porch_square_feet'] .astype('category')
df_pierce_county['attached_garage_square_feet'] = df_pierce_county['attached_garage_square_feet'] .astype('category')
df_pierce_county['detached_garage_square_feet'] = df_pierce_county['detached_garage_square_feet'] .astype('category')
df_pierce_county['fireplaces'] = df_pierce_county['fireplaces'] .astype('category')
df_pierce_county['Crime_Num'] = df_pierce_county['Crime_Num'].fillna(0)

X = df_pierce_county[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built']] 


# ------------------------------------------------------------- #
# --------------------- Neural Network ------------------------ #
# ------------------------------------------------------------- #


# -------------------- Pierce County Model ------------------------##
# Convert all categorical variables to a matrix of zeros and ones

df_pierce_county = pd.get_dummies(df_pierce_county)
print(df_pierce_county.head())


## Standardizing data improves computations and makes sure all features are weighted equally for NNs
#scaler = StandardScaler()
#df_pierce_county = scaler.fit_transform(df_pierce_county)

## Split the dataset into training and testing data
X_train_PC, X_test_PC, y_train_PC, y_test_PC = train_test_split(df_pierce_county,y,test_size =0.2,random_state=109)

nn2_m2_c1 = MLPClassifier(hidden_layer_sizes = (3), activation='logistic', random_state=109)
nn2_m2_c1.fit(X_train_PC, y_train_PC)
y_pred_nn2_m2_c1 = nn2_m2_c1.predict(X_test_PC)
print("Neural Network Classifier Model logistic: PierceCounty")

MSE = mean_squared_error(y_test_PC, y_pred_nn2_m2_c1)
print("MSE for NN with 1 hidden layers and logistic activation :",MSE)

MAPE = mean_absolute_percentage_error(y_test_PC, y_pred_nn2_m2_c1)
print("MAPE: ", MAPE)


nn2_m2_c1_node2 = MLPClassifier(hidden_layer_sizes = (3,3), activation='logistic', random_state=109)
nn2_m2_c1_node2.fit(X_train_PC, y_train_PC)
y_pred_nn2_m2_c1_node2 = nn2_m2_c1_node2.predict(X_test_PC)
print("Neural Network Classifier Model Logistic: PierceCounty")

MSE = mean_squared_error(y_test_PC, y_pred_nn2_m2_c1_node2)
print("MSE for NN with 2 hidden layers and logistic activation :",MSE)

MAPE = mean_absolute_percentage_error(y_test_PC, y_pred_nn2_m2_c1_node2)
print("MAPE: ", MAPE)
