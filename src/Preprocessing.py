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

