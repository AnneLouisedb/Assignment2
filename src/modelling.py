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
from evaluation import evaluate_model
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

X_train = X_train.fillna(0)



#looking at data
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

#models
reg_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
elastic_model = ElasticNet()

# format training data
x_Traingarage = X_train['Garage Area'].values.reshape(-1,1)
y_Train = y_train.values.reshape(-1,1)


#Linear Model
reg_model.fit(X_train, Y_train)
yFit_linear = reg_model.predict(X_train)
y_pred_linear = reg_model.predict(X_test)

#Ridge Model
ridge_model.fit(X_train, Y_train)
yFit_ridge = ridge_model.predict(X_train)
y_pred_ridge = ridge_model.predict(X_test)

#Lasso Model
lasso_model.fit(X_train, Y_train)
yFit_lasso = lasso_model.predict(X_train)
y_pred_lasso = lasso_model.predict(X_test)

#Elastic Net Model
elastic_model.fit(X_train, Y_train)
yFit_elastic = elastic_model.predict(X_train)
y_pred_elastic = elastic_model.predict(X_test)

#plot predicted values
plt.figure()
plt.scatter(y_test, y_pred_linear)
plt.savefig("graphs/ predicted values linear model")

print(evaluate_model(reg_model, X_test, y_test))
print(evaluate_model(ridge_model, X_test, y_test))
print(evaluate_model(lasso_model, X_test, y_test))
print(evaluate_model(elastic_model, X_test, y_test))

# Polynomial
Poly = PolynomialFeatures(degree = 10, include_bias = False)
xTrainPoly = Poly.fit_transform(X_train)
xTrainPolyscale = scaler.fit_transform(xTrainPoly)


#fitting on polynomial
#Linear
reg_model.fit(xTrainPolyscale, Y_train)
yFit_linear = reg_model.predict(xTrainPolyscale)
y_pred_linear = reg_model.predict(X_test)

#Ridge
ridge_model.fit(xTrainPolyscale, Y_train)
yFit_ridge = ridge_model.predict(xTrainPolyscale)
y_pred_ridge = ridge_model.predict(X_test)

#Lasso
lasso_model.fit(xTrainPolyscale, Y_train)
yFit_lasso = lasso_model.predict(xTrainPolyscale)
y_pred_lasso = lasso_model.predict(X_test)

#ElasticNet
elastic_model.fit(xTrainPolyscale, Y_train)
yFit_elastic = elastic_model.predict(xTrainPolyscale)
y_pred_elastic = elastic_model.predict(X_test)








#predict linear model
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyscale = scaler.transform(xFitPoly)
yFit_reg = reg_model.predict(xFitPolyscale)


#plot results linear model
plt.figure()
plt.plot(xFit,yFit_reg, lw=3, color='r', zorder = 2)
plt.scatter(X_train['Garage Area'], y_train)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.savefig("graphs/linear_garage_area")


#Ridge Regression on Garage Area
i = 0
ls = ['-', '--', ':']
color = ['r', 'g', 'orange']

for a in [0, 2, 2000]:
    ridgeReg = Ridge(alpha=a)
    ridgeReg.fit(xTrainPolyscale, y_train)

    # predict
    xFit2 = np.linspace(0, 1500, num=200).reshape(-1, 1)
    xFitPoly2 = Poly.transform(xFit2)
    xFitPolyscale2 = scaler.transform(xFitPoly2)
    yFit2 = ridgeReg.predict(xFitPolyscale2)

    # plot ridge - garage area
    plt.figure()
    plt.plot(xFit2, yFit2, lw=3, color=color[i], zorder=2, label="alpha = " + str(a), linestyle=ls[i])
    i = i + 1
    plt.scatter(X_train['Garage Area'], y_train)
    plt.plot(xFit, yFit_reg, lw=3, color='r', zorder=2)
    plt.ylabel('Sale Price (dollars)', fontsize = 18)
    plt.xlabel('Garage Area (square feet)', fontsize = 18)
    plt.savefig("graphs/ridge_garage_area")


#Ridge Regressor L1
ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(xTrainPolyscale, y_train)
print(f"beste parameter ridge garage: {ridge_regressor.best_params_}")
print(f"best score ridge garage: {ridge_regressor.best_score_}")


#Lasso Regressor L2
lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03],
                "fit_intercept": [True, False],
                "copy_X" : [True, False],
                "selection" :['cyclic', 'random']
                }
lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring = 'neg_mean_squared_error', cv=5 )
lasso_regressor.fit(xTrainPolyscale, y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#Elastic Net
elastic_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265, 270, 275, 290, 300, 500 ]}
elastic_regressor = GridSearchCV(elastic_model, elastic_params, scoring = 'neg_mean_squared_error', cv=5 )
elastic_regressor.fit(xTrainPolyscale,y_train)
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



#Early Stopping

#Interpreting Learning Curves



