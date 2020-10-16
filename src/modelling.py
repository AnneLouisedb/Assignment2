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
from src.extra_functions import plot_model

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

#Learning curves
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

#Plot learning curves
plot_learning_curves(chosen_model, X_train, y_train)

plot_learning_curves(ridge_model, X_train, y_train)

plot_learning_curves(lasso_model, X_train, y_train)

plot_learning_curves(elastic_model, X_train, y_train)



#Create list of parameters for Ridge Regression
#normalize = [ True, False]
# solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
#parameters = dict(ridge__alpha = [1,10],ridge__normalize = normalize, ridge__solver = solver)
#search_ridge = GridSearchCV(ridge_model, param_grid = parameters, cv = 5, scoring = 'accuracy' )

#best estimator
#clf = search_ridge.best_params_
#clf.fit(X_train, y_train)
#print(f'best sore and parameter combination: {search_ridge.best_score_} and {search_ridge.best_params_}')



#Hyper-parameter Tuning

#Early Stopping

#Interpreting Learning Curves