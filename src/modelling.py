from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNet, SGDRegressor
from copy import deepcopy
from src.extra_functions import plot_model, plot_learning_curves, save_fig

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

X_train = imputer.fit_transform(X_train)

#models
chosen_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
elastic_model = ElasticNet()


#list of models
pipelines = [chosen_model, ridge_model, lasso_model, elastic_model]
pipe_dict = {0: 'linear regression', 1: 'Ridge', 2: 'Lasso', 3: 'NetElastic'}

for pipe in pipelines:
    pipe.fit(X_train, y_train)

for i, model in enumerate(pipelines):
    print('{} Test Accuracy: {}'.format(pipe_dict[i],model.predict(X_test)))

#predicting the Test set results
y_pred_linear = chosen_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_elastic = elastic_model.predict(X_test)

#accuracy testing
print(f"mean_absolute_error linear: {mean_absolute_error(y_test, y_pred_linear)}")
print(f"mean_absolute_error ridge:  {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"mean_absolute_error lasso:  {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"mean_absolute_error net elastic: {mean_absolute_error(y_test, y_pred_elastic)}")

#ridge regression rives the smallest absolute error


#Linear Model
mse = cross_val_score(chosen_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)

#Ridge Regressor L1
ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265, 270, 275, 290, 300, 500 ]}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(X_train, y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#Lasso Regressor L2
lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}
lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring = 'neg_mean_squared_error', cv=5 )
lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#prediction
prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)

sns.distplot(y_test-prediction_ridge).set_title('ridge model')
save_fig("graphs/ridge_model")
sns.distplot(y_test-prediction_lasso).set_title('lasso model')
save_fig("graphs/lasso_model")


#Create list of parameters for Ridge Regression
#normalize = [ True, False]
#solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
#parameters = dict(ridge__alpha = [200, 230, 250,265, 270, 275, 290, 300, 500] ,ridge__normalize = normalize, ridge__solver = solver)
#search_ridge = GridSearchCV(ridge_model, param_grid = parameters, cv = 5, scoring = 'accuracy' )





#Hyper-parameter Tuning


#Early Stopping

#Interpreting Learning Curves




