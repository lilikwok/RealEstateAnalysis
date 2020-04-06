import pandas as pd
import patsy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools as sm_tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
import graphviz
import os
import matplotlib.pyplot as plt
from sklearn.utils import check_array

df_pierce_county = pd.read_csv("m1_c1_kmean_3clusters.csv", sep=',', header=0)

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
df_pierce_county['withInSewerImprovement'] = df_pierce_county['withInSewerImprovement'] .astype('category')
df_pierce_county['near_firestation'] = df_pierce_county['near_firestation'] .astype('category')
df_pierce_county['near_hospital'] = df_pierce_county['near_hospital'] .astype('category')
df_pierce_county['near_libraries'] = df_pierce_county['near_libraries'] .astype('category')
df_pierce_county['near_policestation'] = df_pierce_county['near_policestation'] .astype('category')
df_pierce_county['near_waterplants'] = df_pierce_county['near_waterplants'] .astype('category')
df_pierce_county['condition'] = df_pierce_county['condition'] .astype('category')
df_pierce_county['quality'] = df_pierce_county['quality'] .astype('category')
df_pierce_county['attic_finished_square_feet'] = df_pierce_county['attic_finished_square_feet'] .astype('category')
df_pierce_county['basement_square_feet'] = df_pierce_county['basement_square_feet'] .astype('category')
df_pierce_county['basement_finished_square_feet'] = df_pierce_county['basement_finished_square_feet'] .astype('category')
df_pierce_county['porch_square_feet'] = df_pierce_county['porch_square_feet'] .astype('category')
df_pierce_county['attached_garage_square_feet'] = df_pierce_county['attached_garage_square_feet'] .astype('category')
df_pierce_county['detached_garage_square_feet'] = df_pierce_county['detached_garage_square_feet'] .astype('category')
df_pierce_county['fireplaces'] = df_pierce_county['fireplaces'] .astype('category')
df_pierce_county['near_private_school'] = df_pierce_county['near_private_school'] .astype('category')
df_pierce_county['near_elementary_school'] = df_pierce_county['near_elementary_school'] .astype('category')
df_pierce_county['near_high_school'] = df_pierce_county['near_high_school'] .astype('category')
df_pierce_county['near_college'] = df_pierce_county['near_college'] .astype('category')
df_pierce_county['Crime_Num'] = df_pierce_county['Crime_Num'].fillna(0)

print(df_pierce_county.columns)

X = df_pierce_county[['Land_Net_Acres','View_Quality', 
       'Waterfront_Type', 'Crime_Num', 'withInSewerImprovement',
       'near_firestation', 'near_hospital', 'near_libraries',
       'near_policestation', 'near_waterplants', 'square_feet',
       'condition', 'quality', 'attic_finished_square_feet',
       'basement_square_feet', 'basement_finished_square_feet',
       'porch_square_feet', 'attached_garage_square_feet',
       'detached_garage_square_feet', 'fireplaces', 'stories', 'bedrooms',
       'bathrooms', 'year_built', 'near_private_school', 'near_elementary_school', 'near_college', 'near_high_school']]


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

nn1_m1_c1_3clust = MLPClassifier(hidden_layer_sizes = (3), activation='logistic', random_state=109)
nn1_m1_c1_3clust.fit(X_train_PC, y_train_PC)
y_pred_nn1_m1_c1_3clust = nn1_m1_c1_3clust.predict(X_test_PC)
print("Neural Network Classifier Model Logistic: PierceCounty")

MSE = mean_squared_error(y_test_PC, y_pred_nn1_m1_c1_3clust)
print("MSE for NN with 1 hidden layer and logistic activation :",MSE)

MAPE = mean_absolute_percentage_error(y_test_PC, y_pred_nn1_m1_c1_3clust)
print("MAPE: ", MAPE)

nn1_m1_c1_3clust_node2 = MLPClassifier(hidden_layer_sizes = (3,3), activation='logistic', random_state=109)
nn1_m1_c1_3clust_node2.fit(X_train_PC, y_train_PC)
y_pred_nn1_m1_c1_3clust_node2 = nn1_m1_c1_3clust_node2.predict(X_test_PC)
print("Neural Network Classifier Model Logistic: PierceCounty")

MSE = mean_squared_error(y_test_PC, y_pred_nn1_m1_c1_3clust_node2)
print("MSE for NN with 2 hidden layers and logistic activation :",MSE)

MAPE = mean_absolute_percentage_error(y_test_PC, y_pred_nn1_m1_c1_3clust_node2)
print("MAPE: ", MAPE)

"""
This code is used to visualize
http://deeplearning.net/tutorial/mlp.html#tips-and-tricks-for-training-mlps
Usage: put the following on 389th line
    title = "whatever you want"
    plot_pca(classifier, x, train_set_x, train_set_y, index=epoch, title=title)
"""

def plot_pca(classifier, x_symbol, X, y, index=0,
             title=None, sampling=True):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    axes = axes.flatten()

    apply_hidden = plt.function(inputs=[x_symbol], outputs=classifier.hiddenLayer.output)
    z_data = apply_hidden(X.get_value())
    labels = y.eval()

    numbers = range(10)
    colors = {0: '#263B1C', 1: '#263374', 2: '#3568B5', 3: '#8A5DDF', 4: '#DBB8EE',
              5: '#46B1C9', 6: '#84C0C6', 7: '#9FB7B9', 8: '#BCC1BA', 9: '#F2E2D2'}

    for ax, prod in zip(axes, zip(numbers[:-1], numbers[1:])):
        # print(ax, prod)
        pca = PCA(n_components=2)
        indexer = np.arange(len(labels))[np.in1d(labels, prod)]
        label = labels[indexer]
        z = z_data[indexer]
        pca.fit(z)
        z_pca = pca.transform(z)

        if sampling:
           indexer = np.arange(len(label))
           np.random.shuffle(indexer)
           indexer = indexer[:300]
           z_pca = z_pca[indexer]
           label = label[indexer]

        _c = [colors[l] for l in label]
        ax.scatter(z_pca[:, 0], z_pca[:, 1], color=_c, alpha=0.3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('{0}, {1}'.format(prod[0], prod[1]), size='small')
        plt.show()
    if title is not None:
        fig.suptitle(title)
    plt.savefig('pca_{0:02d}.png'.format(index))