from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
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

#splitting data
X_train, X_test, y_train, y_test = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train = X_train[columns_to_use]
X_test = X_test[columns_to_use]


#functions in preprocessor
garage_area = all_data["Garage Area"]
plt.figure()
plt.plot(garage_area)
plt.savefig("graphs/garagearea2")

garage_area_matrix = garage_area.values.reshape(-1,1)
scaled = preprocessing.MinMaxScaler()
scaled_garage_area = scaled.fit_transform(garage_area_matrix)
plt.figure()
plt.plot(scaled_garage_area)
plt.savefig("graphs/scaled_garage_area")

print(all_data[columns_to_use].isna().sum())
for columns in columns_to_use:
    all_data[columns] = all_data[columns].fillna(0)

print(all_data[columns_to_use].isna().sum())

#scaling values using pandas
tmp_garage_area = all_data["Garage Area"] - all_data["Garage Area"].min()
scaled_garage_area1 = tmp_garage_area / all_data["Garage Area"].max()
all_data["scaled Garage Area"] = scaled_garage_area1
print(all_data["scaled Garage Area"])

#using pipeline for preprocessing
imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0)
scaler = MinMaxScaler()
preprocess = Pipeline(steps = [("imp", imputer) , ('minmaxscale', scaler)])

for columns in columns_to_use:
    X_train[columns] = X_train[columns].preprocess()


#looking at data
plt.figure()
plt.scatter(X_train['Overall Qual'], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Overall Quality', fontsize = 18)
plt.savefig("graphs/OverallQual")

plt.figure()
plt.scatter(X_train["Total Bsmt SF"], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('total Bsmt (square feet)', fontsize = 18)
plt.savefig("graphs/Total_Bsmt_SF")

plt.figure()
plt.scatter(X_train['Garage Area'], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.savefig("graphs/garage_area")

plt.figure()
plt.scatter(X_train["Lot Area"], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Lot Area (square feet)', fontsize = 18)
plt.savefig("graphs/lot_area")

plt.figure()
plt.scatter(X_train["Bedroom AbvGr"], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel("Bedroom AbvGr", fontsize = 18)
plt.savefig("graphs/Bedroom_AbvGr")


# format training data
x_Traingarage = X_train['Garage Area'].values.reshape(-1,1)
y_Traingarage = y_train.values.reshape(-1,1)

pipe_garage_area = preprocess.fit(X_Traingarage)

# Transform the input features, without regularization
Poly = PolynomialFeatures(degree = 10, include_bias = False)
xTrainPoly = Poly.fit_transform(x_Traingarage) ##fill empty values!!


# scaler.scale_, scaler.mean_


#models
chosen_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
elastic_model = ElasticNet()

#fit model to polynomial
chosen_model.fit(xTrainPolyStan, yTraingarage)
ridge_model.fit(xTrainPolyStan, yTraingarage)
lasso_model.fit(xTrainPolyStan, yTraingarage)
elastic_model.fit(xTrainPolyStan, yTraingarage)

#predict linear model
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)

yFit_linear = chosen_model.predict(xFitPolyStan)
yFit_ridge = ridge_model.predict(xFitPolyStan)
yFit_lasso = lasso_model.predict(xFitPolyStan)
yFit_elastic = elastic_model.predict(xFitPolyStan)

#plot results linear model (chosen_model)
plt.plot(xFit,yFit_linear, lw=3, color='r', zorder = 2)
plt.scatter(X_train['Garage Area'], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.savefig("graphs/linear_garage_area")

#make predictions on the test set
y_pred_linear = chosen_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_elastic = elastic_model.predict(X_test)

#accuracy testing
print(f"mean_absolute_error linear: {mean_absolute_error(y_test, y_pred_linear)}")
print(f"mean_absolute_error ridge:  {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"mean_absolute_error lasso:  {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"mean_absolute_error net elastic: {mean_absolute_error(y_test, y_pred_elastic)}")

#Ridge Regression on Garage Area
i = 0
ls = ['-', '--', ':']
color = ['r', 'g', 'orange']

for a in [0, 2, 2000]:
    ridgeReg = Ridge(alpha=a)
    ridgeReg.fit(xTrainPolyStan, y_train)

    # predict
    xFit2 = np.linspace(0, 1500, num=200).reshape(-1, 1)
    xFitPoly2 = Poly.transform(xFit2)
    xFitPolyStan2 = scaler.transform(xFitPoly2)
    yFit2 = ridgeReg.predict(xFitPolyStan2)

    # plot ridge - garage area
    plt.figure()
    plt.plot(xFit2, yFit2, lw=3, color=color[i], zorder=2, label="alpha = " + str(a), linestyle=ls[i])
    i = i + 1
    plt.scatter(X_train['Garage Area'], y_train, marker='o', color='b', linestyle='', zorder=1)
    plt.ylabel('Sale Price (dollars)', fontsize = 18)
    plt.xlabel('Garage Area (square feet)', fontsize = 18)
    plt.savefig("graphs/ridge_garage_area")


#Ridge Regressor L1
ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(xTrainPolyStan, y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


#Linear Model
mse = cross_val_score(chosen_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)


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

plt.figure()
sns.distplot(y_test-prediction_ridge).set_title('ridge model')
plt.savefig("graphs/ridge_model")

plt.figure()
sns.distplot(y_test-prediction_lasso).set_title('lasso model')
plt.savefig("graphs/lasso_model")

plt.figure()
sns.distplot(y_test-prediction_elastic).set_title('elastic model')
plt.savefig("graphs/elastic_model")

plt.figure()
plt.scatter(X_train, y_train)
plt.savefig("graphs/scatter training")





#Early Stopping


#Interpreting Learning Curves

#RidgeRegression = Ridge(alpha= 5, fit_intercept= True, solver= 'svd')
#plot_learning_curves(RidgeRegression, X_test, y_test)
#save_fig("graphs/learningcurve_ridge")


