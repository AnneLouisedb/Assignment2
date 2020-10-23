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
from NewModelling import X_train2
from src.extra_functions import plot_model, plot_learning_curves


data_dir = Path("data/")
img_dir = Path("../img")
columns_to_use = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
]

all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
target_column = "SalePrice"

#splitting data
X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train = X_train[columns_to_use]
X_test = X_test[columns_to_use]

#functions in preprocessor

print(all_data[columns_to_use].isna().sum())
for columns in columns_to_use:
    all_data[columns] = all_data[columns].fillna(0)

print(all_data[columns_to_use].isna().sum())

#scattermix - data 1
attributes = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
    "SalePrice"
]
plt.figure()
scatter_matrix(all_data[attributes], figsize=(12,8))
plt.savefig("graphs/scatter_matrix_plot")


#looking at data
plt.figure()
sns.distplot(all_data['SalePrice'], bins=30)
plt.savefig('graphs/ saleprice positive skewness') #its peak deviates from normal distribution

plt.figure()
plt.scatter(all_data['Overall Qual'], all_data["SalePrice"])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Overall Quality', fontsize = 18)
plt.savefig("graphs/OverallQual")

plt.figure()
plt.scatter(all_data["Total Bsmt SF"], all_data["SalePrice"])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('total Bsmt (square feet)', fontsize = 18)
plt.savefig("graphs/Total_Bsmt_SF")

plt.figure()
plt.scatter(all_data['Garage Area'], all_data["SalePrice"])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.savefig("graphs/garage_area")

plt.figure()
plt.scatter(all_data["Lot Area"], all_data["SalePrice"])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Lot Area (square feet)', fontsize = 18)
plt.savefig("graphs/lot_area")

plt.figure()
plt.scatter(all_data["Bedroom AbvGr"], all_data["SalePrice"])
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel("Bedroom AbvGr", fontsize = 18)
plt.savefig("graphs/Bedroom_AbvGr")


