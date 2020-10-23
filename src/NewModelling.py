from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from numpy import absolute
import sklearn
from statistics import mean
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet, SGDRegressor
from copy import deepcopy
from src.extra_functions import plot_model, plot_learning_curves
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_dir = Path("data/")
img_dir = Path("../img")

all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
target_column = "SalePrice"

# find categorical variables
categorical = [var for var in all_data.columns if all_data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
# find numerical variables
numerical = [var for var in all_data.columns if all_data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))


#look at how much data is missing
for var in all_data.columns:
    if all_data[var].isnull().sum()>0:
        print(var, all_data[var].isnull().mean())

#the features for whicha lot of data is missing
for var in all_data.columns:
    if all_data[var].isnull().mean()>0.70:
        print(f"more than 0.70 missing values: {var, all_data[var].unique()}")

#['Alley', 'Pool QC', 'Fence', 'Misc Feature'], it is logical that most houses do not have a pool or alley.


all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['Pool QC'] = all_data['Pool QC'].fillna('None')
all_data['Mics Feature'] = all_data['Misc Feature'].fillna('None')

#numerical values are discrete or continuous
discrete = []
for var in numerical:
    if len(all_data[var].unique())<20:
        print(var, ' values: ', all_data[var].unique())
        discrete.append(var)

continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]

#numerical values that are actually categorical values.
num_to_cat = ['MS SubClass', 'Overall Qual',
 'Overall Cond', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Garage Cars','Yr Sold']
#these are examples of ordinal values
all_data[num_to_cat] = all_data[num_to_cat].apply(lambda x: x.astype("str"))


#encoding categorical variables
#ordinal encoding using Label encoding

for columns in all_data.columns:
    all_data[columns] = all_data[columns].fillna(0)

#looking at boxplots
sns.boxplot(x = all_data["Overall Qual"], y = all_data['SalePrice'])

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

print(X_train2)

X_train2["totalSF"] = X_train2['Gr Liv Area'] + X_train2['Total Bsmt SF']
X_train2["AllfloorSF"] = X_train2["1st Flr SF"] + X_train2["2nd Flr SF"]
X_train2["Qual+Cond"] = X_train2["Overall Qual"] + X_train2["Overall Cond"]

X_test2["totalSF"] = X_test2['Gr Liv Area'] + X_test2['Total Bsmt SF']
X_test2["AllfloorSF"] = X_test2["1st Flr SF"] + X_test2["2nd Flr SF"]
X_test2["Qual+Cond"] = X_test2["Overall Qual"] + X_test2["Overall Cond"]

more_columns = [
    "Lot Area",      #original columns used
    "Overall Qual", #categorical
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr", #categorical
    "AllfloorSF", #new features
    "totalSF",    #new
   # "Qual+Cond",   #new , categorical
'Year Built',
'Fence',
'Pool QC',
'Utilities',   #ordinal values
'House Style',
'Garage Qual',
'Garage Cond']


X_train2 = X_train2[more_columns]
X_test2 =X_test2[more_columns]

X_train2 = X_train2.fillna(0)
X_test2 = X_test2.fillna(0)

print(X_train2.isnull().values.any())

ordinal_col_dicts = {
'Utilities': {'AllPub': 1, 'NoSeWa': 3, 'NoSewr':2, 'ELO':4},
'House Style': {'1Story': 1,  '2Story': 4, '1.5Fin': 2, 'SLvl': 8, 'SFoyer': 7, '1.5Unf': 3, '2.5Unf': 6, '2.5Fin': 5},
'Garage Qual': {'TA': 3, 'Fa': 4, 'Gd': 2, 'Ex': 1, 'Po': 5},
'Garage Cond': {'TA': 3, 'Fa': 4, 'Gd': 2, 'Po': 5, 'Ex': 1},
'Pool QC' : {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'None': 5},
'Fence' : {'GdPrv' : 1, 'MnPrv':	 2, 'GdWo': 3, 'MnWw': 4, 'None': 5},
'Bedroom AbvGr':  {'3' : 3, '2': 2, '5': 5, '1': 1, '4': 4, '0': 0, '6':6 },
'Overall Qual' : {'8':8, '10':10, '7': 7,'6':6, '5': 5,'2':2, '9': 9,'4': 4,'3':3 ,'1': 1}}


def ordinal_encode(data, ordinal_col_dicts):
    """
    Ordinal encode the ordinal columns according to the values in
    ordinal_col_dicts.
    """
    for ord_col in ordinal_col_dicts:
        ord_dict = ordinal_col_dicts[ord_col]
        data[ord_col] = data[ord_col].map(ord_dict)
    return data

X_train2 = ordinal_encode(X_train2, ordinal_col_dicts) #numerical values assigned to feature class
X_test2 = ordinal_encode(X_test2, ordinal_col_dicts)

X_train2 = X_train2.fillna(0)
X_test2 = X_test2.fillna(0)

OverallQualitycheck = X_train2["Overall Qual"]

# find categorical variables in train data
train_categorical = [var for var in X_train2.columns if X_train2[var].dtype=='O']
print('There are {} categorical variables in training data'.format(len(train_categorical)))
# find numerical variables in train data
train_numerical = [var for var in X_train2.columns if X_train2[var].dtype!='O']
print('There are {} numerical variables in training data'.format(len(train_numerical)))

print(X_train2.dtypes)


#turning objects into integer
print(X_train2['Bedroom AbvGr'].unique())
print(X_train2['Overall Qual'].unique())
#print(X_train2['Qual+Cond'].unique())


# Plot histogram grid
plt.figure()
X_train2.hist(figsize=(20,20), xrot=-45)
plt.savefig("graphs/data2histogram")

#looking at new columns
plt.figure()
sns.distplot(X_train2["totalSF"])
sns.distplot(X_train2["AllfloorSF"])

plt.figure()
plt.scatter(X_train2['totalSF'], y_train2)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('total area SF', fontsize = 18)
plt.title("2.Total area SF")
plt.savefig("graphs/scatter totalSF")

plt.figure()
plt.scatter(all_data['Gr Liv Area'], all_data['SalePrice'])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Gr Liv Area', fontsize = 18)
plt.title("2.Living area")
plt.savefig("graphs/scatter GrLivArea")

#scattermix
attributes2 = [
    "Lot Area",      #original columns used
    "Overall Qual", #categorical
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr", #categorical
'Year Built',
'Fence',
'Pool QC',
'Utilities',   #ordinal values
'House Style',
'Garage Qual',
'Garage Cond',
    'SalePrice']

plt.figure()
scatter_matrix(all_data[attributes2], figsize=(12,8))
plt.savefig("graphs/scatter_matrix_plot_data2")


#There seems to be a linear relationship between totalSF and Saleprice.
# It looks even better than the GR Liv Area against SalePrice.

y_train2 = y_train2.values.reshape(-1,1)

#models
reg_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
elastic_model = ElasticNet()

#Ridge Model
cross_val_scores_ridge = [] #this stores average cross validation scores
alpha_ridge = []
for i in range(1,9):
    ridgemodel = Ridge(alpha = i*0.25)
    scores = cross_val_score(ridgemodel, X_train2, y_train2, cv=8)
    average_cross_val_score = mean(scores)*100 # as a percentage
    cross_val_scores_ridge.append(average_cross_val_score)
    alpha_ridge.append(i*0.25)

for i in range(0, len(alpha_ridge)):
    print(str(alpha_ridge[i])+ " : " + str(cross_val_scores_ridge[i]))

plt.figure()
ax = plt.gca()
ax.plot(alpha_ridge, scores)
plt.xlabel('alpha_ridge')
plt.ylabel('score')
plt.title('Ridge Model')
plt.savefig("graphs/RidgeModel_data2")

#Ridge Regressor L1, using more hyperparameters

ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(X_train2, y_train2)
print(f"best parameter Ridge: {ridge_regressor.best_params_}")
print(f"best score Ridge: {ridge_regressor.best_score_}")
#plot learning curve
plt.figure()
plot_learning_curves(Ridge( alpha = 200, fit_intercept = True, solver = 'cholesky'), X_train2, y_train2)
plt.savefig("graphs/learningcurve_ridge_data2")

#Lasso Regression
cross_val_scores_lasso = []
Lambda = []

for i in range (1,9):
    lassoModel = Lasso(i*0.25, tol = 0.0925)
    lassoModel.fit(X_train2, y_train2)
    scores = cross_val_score(lassoModel, X_train2, y_train2, cv=8)
    average_cross_val_score = mean(scores)*100
    cross_val_scores_lasso.append(average_cross_val_score)
    Lambda.append(i*0.25)

for i in range(0, len(Lambda)):
    print(str(Lambda[i]) + ' : '+str(cross_val_scores_lasso[i]))

plt.figure()
ax = plt.gca()
ax.plot(Lambda, scores)
plt.xlabel('Lambda')
plt.ylabel('score')
plt.title('Lasso Model_data2')
plt.savefig("graphs/ Lamdas_Lasso_Model_data2")

# Lasso Regressor L2, looking are more hyperparameters
lasso_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10, 20, 15, 12, 8, 25, 30, 200],
                "fit_intercept": [True, False],
                "copy_X": [True, False],
                "selection": ['cyclic', 'random']
                }
lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train2, y_train2)

print(f"best parameter Lasso:{lasso_regressor.best_params_}")
print(f"best score Lasso:{lasso_regressor.best_score_}")

#plot learning curve
plt.figure()
plot_learning_curves(Lasso(alpha = 200, copy_X = True, fit_intercept = True, selection = 'random'), X_train2, y_train2)
plt.savefig("graphs/ lasso learning curve_data2")

# Elastic Net
alpha=[]
scoresArr=[]

for i in range(1, 9):
    ElasticNetModel = ElasticNet(alpha=i*0.25, l1_ratio=0.5)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
   # evaluate model
    scores = cross_val_score(ElasticNetModel, X_train2, y_train2, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
   # force scores to be positive
    scores = absolute(scores) #to have the positive values
    scoresArr.append(scores)
    alpha.append(i*0.25)

plt.figure()
ax = plt.gca()
ax.plot(alpha, scoresArr)
plt.xlabel('alpha')
plt.ylabel('score')
plt.title('ElasticNet Model')
plt.savefig("graphs/ElasticNet Model_data2")


#Gridsearch Elastic Net
elastic_params = {
    'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265,
              270, 275, 290, 300, 500],
'l1_ratio' : [0.4, 0.5 0.6, 0.7, 0.8, 0.9]}
elastic_regressor = GridSearchCV(elastic_model, elastic_params, scoring='neg_mean_squared_error', cv=5)
elastic_regressor.fit(X_train2, y_train2)
print(f"best parameter Elastic Net:  {elastic_regressor.best_params_}")
print(f"best score Elastic Net: {elastic_regressor.best_score_}")

#plot learning curve
plt.figure()
plot_learning_curves(ElasticNet(alpha = 0.03), X_train2, y_train2)
plt.savefig("graphs/learningcurve elasticnet_data2")

chosen_model = ElasticNet(alpha = 0.03)
X_test2 = X_test2.fillna(0) #error with NaN values, how to fix?
#Elastic Net Model
chosen_model.fit(X_train2, y_train2)

yFit_elastic = chosen_model.predict(X_train2)
y_pred_elastic = chosen_model.predict(X_test2)

# Plot predictions chosen model
plt.figure()
plt.scatter(yFit_elastic, y_train2, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_pred_elastic, y_test2, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Elastic Net regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.savefig("graphs/predictedvaluesElasticNet_data2")


