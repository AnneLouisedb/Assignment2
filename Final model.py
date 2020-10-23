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

data_dir = Path("data/")
img_dir = Path("../img")

all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
target_column = "SalePrice"

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

# find categorical variables
categorical = [var for var in all_data.columns if all_data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
# find numerical variables
numerical = [var for var in all_data.columns if all_data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# Append Ridge to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('Ridge', Ridge())])


allcolumns = all_data.columns
#functions in preprocessor
print(all_data.isna().sum())
for columns in all_data.columns:
    all_data[columns] = all_data[columns].fillna(0)

print(all_data.isna().sum())

X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

#look at how much data is missing
for var in all_data.columns:
    if all_data[var].isnull().sum()>0:
        print(var, all_data[var].isnull().mean())

#the features for whicha lot of data is missing
for var in all_data.columns:
    if all_data[var].isnull().mean()>0.70:
        print(f"more than 0.70 missing values: {var, all_data[var].unique()}")

all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['Pool QC'] = all_data['Pool QC'].fillna('None')
all_data['Mics Feature'] = all_data['Misc Feature'].fillna('None')

#numerical values:  discrete or continuous
discrete = []
for var in numerical:
    if len(all_data[var].unique())<20:
        print(var, ' values: ', all_data[var].unique())
        discrete.append(var)

continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]

#creating new features
X_train["totalSF"] = X_train['Gr Liv Area'] + X_train['Total Bsmt SF']
X_train["AllfloorSF"] = X_train["1st Flr SF"] + X_train["2nd Flr SF"]
X_train["Qual+Cond"] = X_train["Overall Qual"] + X_train["Overall Cond"]

X_test["totalSF"] = X_test['Gr Liv Area'] + X_test['Total Bsmt SF']
X_test["AllfloorSF"] = X_test["1st Flr SF"] + X_test["2nd Flr SF"]
X_test["Qual+Cond"] = X_test["Overall Qual"] + X_test["Overall Cond"]

X_train = X_train[more_columns]
X_test = X_test[more_columns]

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


print(X_train['Overall Qual'].unique())

#orginal encoding
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

X_train = ordinal_encode(X_train, ordinal_col_dicts) #numerical values assigned to feature class
X_test = ordinal_encode(X_test, ordinal_col_dicts)

#again filling NaN values. (don't understand the error here)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

#Ridge Model
cross_val_scores_ridge = [] #this stores average cross validation scores
alpha_ridge = []
for i in range(1,9):
    ridgemodel = Ridge(alpha = i*0.25)
    scores = cross_val_score(ridgemodel, X_train, y_train, cv=8)
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
ridge_regressor.fit(X_train, y_train)
print(f"beste parameter Ridge: {ridge_regressor.best_params_}")
print(f"best score Ridge: {ridge_regressor.best_score_}")

#plot learning curves
plot_learning_curves(Ridge( alpha = 5, fit_intercept = True, solver = 'svd'), X_train, y_Train)

#Lasso Regression
cross_val_scores_lasso = []
Lambda = []

for i in range (1,9):
    lassoModel = Lasso(i*0.25, tol = 0.0925)
    lassoModel.fit(X_train, y_train)
    scores = cross_val_score(lassoModel, X_train, y_Train, cv=8)
    average_cross_val_score = mean(scores)*100
    cross_val_scores_lasso.append(average_cross_val_score)
    Lambda.append(i*0.25)

for i in range(0, len(Lambda)):
    print(str(Lambda[i]) + ' : '+str(cross_val_scores_lasso[i]))

#example Lasso
#example_lasso = Lasso(alpha = 1)
#example_lasso.fit(X_train, y_train)
#print(example_lasso.score(X_test, y_test))

plt.figure()
ax = plt.gca()
ax.plot(Lambda, scores)
plt.xlabel('Lambda')
plt.ylabel('score')
plt.title('Lasso Model')
plt.savefig("graphs/ Lamdas Lasso Model")

# Lasso Regressor L2, looking are more hyperparameters
lasso_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10, 20, 15, 12, 8, 25, 30, 200],
                "fit_intercept": [True, False],
                "copy_X": [True, False],
                "selection": ['cyclic', 'random']
                }
lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, y_train)

print(f"beste parameter Lasso:{lasso_regressor.best_params_}")
print(f"best score Lasso:{lasso_regressor.best_score_}")

#plot learning curve
#plt.figure()
#plot_learning_curves(Lasso(alpha = 25, copy_X = False, fit_intercept = True, selection = 'random'), X_train, y_Train)
#plt.savefig("graphs/ lasso learning curve")

# Elastic Net
alpha=[]
scoresArr=[]

for i in range(1, 9):
    ElasticNetModel = ElasticNet(alpha=i*0.25, l1_ratio=0.5)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
   # evaluate model
    scores = cross_val_score(ElasticNetModel, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
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
plt.savefig("graphs/ElasticNet Model")


#Gridsearch Elastic Net
elastic_params = {
    'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265,
              270, 275, 290, 300, 500]}
elastic_regressor = GridSearchCV(elastic_model, elastic_params, scoring='neg_mean_squared_error', cv=5)
elastic_regressor.fit(X_train, y_train)
print(f"beste parameter Elastic Net:  {elastic_regressor.best_params_}")
print(f"best score Elastic Net: {elastic_regressor.best_score_}")

#plot learning curve
#plt.figure()
#plot_learning_curves(ElasticNet(alpha = 0.01), X_train, y_Train)
#plt.savefig("graphs/learningcurve elastic net")
#chosen_model = ElasticNet(alpha = 0.01)


#Elastic Net Model
#chosen_model.fit(X_train, y_Train)
#yFit_elastic = chosen_model.predict(X_train)
#y_pred_elastic = chosen_model.predict(X_test)

# Plot predictions chosen model
#plt.figure()
#plt.scatter(yFit_elastic, y_train, c = "blue", marker = "s", label = "Training data")
#plt.scatter(y_pred_elastic, y_test, c = "lightgreen", marker = "s", label = "Validation data")
#plt.title("Elastic Net regression")
#plt.xlabel("Predicted values")
#plt.ylabel("Real values")
#plt.legend(loc = "upper left")
#plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
#lt.savefig("graphs/predictedvaluesElasticNet")
