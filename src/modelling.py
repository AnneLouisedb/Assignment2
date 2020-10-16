from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


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

X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train = X_train[columns_to_use]
X_test = X_test[columns_to_use]

imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)
linear_model = LinearRegression()

chosen_model = Pipeline([("imputer", imputer), ("model", linear_model)])
ridge_model = Pipeline([("imputer", imputer), ("ridge.model", Ridge())])
#lasso_model = ...
#Fitting regressors to the training set
chosen_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
#lasso_model.fit(X_train), y_train)

#predicting the Test set results



#linear_model.fit(imputer.transform(X_train), y_train)
#y_pred = linear_model.predict(imputer.transform(X_test))
#print(mean_absolute_error(y_test, y_pred))

#Regularisation, L1 (Lasso) and L2 (Ridge)


#Hyper-parameter Tuning

#Early Stopping

#Interpreting Learning Curves