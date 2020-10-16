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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LassoLarsCV
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

#looking at data
plt.scatter(all_data[target_column], all_data.drop(columns=target_column)['Garage Area'])
save_fig("graphs/garagearea")
plt.scatter(all_data[target_column], all_data.drop(columns=target_column)['Overall Qual'])
save_fig("graphs/overallquality")
plt.scatter(all_data[target_column], all_data.drop(columns=target_column)["Total Bsmt SF"])
save_fig("graphs/totalbsmtsf")
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

#Hyper-parameter Tuning

#Linear Model
mse = cross_val_score(chosen_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)

#Ridge Regressor L1
ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(X_train, y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#Lasso Regressor L2
lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03],
                "fit_intercept": [True, False],
                "copy_X" : [True, False],
                "selection" :['cyclic', 'random']
                }
lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring = 'neg_mean_squared_error', cv=5 )
lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#Elastic Net
elastic_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265, 270, 275, 290, 300, 500 ]}
elastic_regressor = GridSearchCV(elastic_model, elastic_params, scoring = 'neg_mean_squared_error', cv=5 )
elastic_regressor.fit(X_train,y_train)
print(elastic_regressor.best_params_)
print(elastic_regressor.best_score_)

#prediction and plots
prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)
prediction_elastic = elastic_regressor.predict(X_test)

sns.distplot(y_test-prediction_ridge).set_title('ridge model')
save_fig("graphs/ridge_model")

sns.distplot(y_test-prediction_lasso).set_title('lasso model')
save_fig("graphs/lasso_model")

sns.distplot(y_test-prediction_elastic).set_title('elastic model')
save_fig("graphs/elastic_model")

plt.scatter(X_train, y_train)
save_fig("graphs/scatter training")

#Early Stopping
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean()
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
save_fig('graphs/validation_ridge')


#Interpreting Learning Curves
#RidgeRegression = Ridge(alpha= 5, fit_intercept= True, solver= 'svd')
#plot_learning_curves(RidgeRegression, X_test, y_test)
#save_fig("graphs/learningcurve_ridge")


