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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet


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
scaler = StandardScaler()
linear_model = LinearRegression()
lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]

chosen_model = Pipeline(steps = [("imputer", imputer), ("scale", scaler), ("model", linear_model)])
ridge_model = Pipeline(steps = [("imputer", imputer), ("scale", scaler), ("ridge.model", Ridge())])
lasso_model = Pipeline(steps =[("imputer", imputer),("scale", scaler),  ("ridge.model", Lasso())])
elastic_model = Pipeline(steps = [("imputer", imputer), ("scale", scaler), ("ridge.model", ElasticNet())])

#Fitting regressors to the training set
chosen_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
elastic_model.fit(X_train, y_train)

#predicting the Test set results
y_pred_linear = chosen_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_elastic = elastic_model.predict(X_test)

print(f"mean_absolute_error linear: {mean_absolute_error(y_test, y_pred_linear)}")
print(f"mean_absolute_error ridge:  {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"mean_absolute_error lasso:  {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"mean_absolute_error net elastic:  {mean_absolute_error(y_test, y_pred_elastic)}")




#Create list of parameters for Ridge Regression
#normalize = [True, False]
#solver = ['auto', 'svd', 'cholesky', 'Isqr', 'sparse_cg','sag','saga' ]
#parameters = dict('ridge__normalize' : normalize, 'ridge__solver' : solver)
#search_ridge = GridSearchCV(ridge_model, parameters)

#best estimator
#search_ridge.fit(X_train, y_train)




#Hyper-parameter Tuning

#Early Stopping

#Interpreting Learning Curves