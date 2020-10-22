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

# find categorical variables
categorical = [var for var in all_data.columns if all_data[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
# find numerical variables
numerical = [var for var in all_data.columns if all_data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))

#look at how much data is missing
for var in all_data.columns:
    if all_data[var].isnull().sum()>0:
        print(var, all_data[var].isnull().mean())
#the features for whicha lot of data is missing
for var in all_data.columns:
    if all_data[var].isnull().mean()>0.70:
        print(f"more than 0.70 missing values: {var, all_data[var].unique()}")
#numerical values are discrete or continuous
discrete = []
for var in numerical:
    if len(all_data[var].unique())<20:
        print(var, ' values: ', all_data[var].unique())
        discrete.append(var)

continuous = [var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice']]


#classifier = LabelEncoder()
#all_data["Garage Qual"] = classifier.fit_transform(all_data["Garage Qual"])

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    all_data.drop(columns=target_column), all_data[target_column]
)

X_train2["totalSF"] = X_train2['Gr Liv Area'] + X_train2['Total Bsmt SF']
X_train2["AllfloorSF"] = X_train2["1st Flr SF"] + X_train2["2nd Flr SF"]
X_train2["Qual+Cond"] = X_train2["Overall Qual"] + X_train2["Overall Cond"]
more_columns = [
    "Lot Area",
    "Overall Qual",
    "Total Bsmt SF",
    "Garage Area",
    "Bedroom AbvGr",
    "AllfloorSF",
    "totalSF",
    "Qual+Cond",
'Year Built']
X_train2 = X_train2[more_columns]
X_train2 = X_train2.fillna(0)

print(X_train2)
#looking at new columns
sns.distplot(X_train2["totalSF"])
sns.distplot(X_train2["Qual+Cond"])
sns.distplot(X_train2["AllfloorSF"])

#scattermix
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
scaler = MinMaxScaler() #normalization
preprocess = Pipeline(steps = [("imp", imputer) , ('minmaxscale', scaler)])
#X_train = X_train.preprocess.fit_transform(X_train)
X_train = X_train.fillna(0)

# format training data
y_Train = y_train.values.reshape(-1,1)

#models
reg_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()
elastic_model = ElasticNet()

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


#Ridge Model
cross_val_scores_ridge = [] #this stores average cross validation scores
alpha_ridge = []
for i in range(1,9):
    ridgemodel = Ridge(alpha = i*0.25)
    scores = cross_val_score(ridgemodel, X_train, y_Train, cv=8)
    average_cross_val_score = mean(scores)*100 # as a percentage
    cross_val_scores_ridge.append(average_cross_val_score)
    alpha_ridge.append(i*0.25)

for i in range(0, len(alpha_ridge)):
    print(str(alpha_ridge[i])+ " : " + str(cross_val_scores_ridge[i]))

#fit ridge regression
example_ridge = Ridge(alpha = 1)
example_ridge.fit(X_train, y_train)
#evaluate ridge model
print(example_ridge.score(X_test, y_test))

plt.figure()
ax = plt.gca()
ax.plot(alpha_ridge, scores)
plt.xlabel('alpha_ridge')
plt.ylabel('score')
plt.title('Ridge Model')
plt.savefig("graphs/RidgeModel")

#Ridge Regressor L1, using more hyperparameters
ridge_params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20,
                         200, 230, 250, 265, 270, 275, 290, 300, 500 ],
                "fit_intercept": [True, False],
                "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_regressor = GridSearchCV(ridge_model, ridge_params, scoring = 'neg_mean_squared_error', cv=5 )
ridge_regressor.fit(X_train, y_train)
print(f"beste parameter Ridge: {ridge_regressor.best_params_}")
print(f"best score Ridge: {ridge_regressor.best_score_}")

#plot model and  learning curves
plot_learning_curves(Ridge( alpha = 5, fit_intercept = True, solver = 'svd'), X_train, y_Train)

#plt.figure()
#plot_model(Ridge, X_train, y_Train, polynomial = False, alphas =(0, 10, 100), random_state=42)
#plt.title("ridge model linear")
#plt.savefig('graphs/ridge learning curve, linear')

#plt.figure()
#lot_model(Ridge, X_train, y_Train, polynomial = True, alphas =(0, 5, 10, 100), random_state=42)
#plt.title("ridge model polynomial")
#plt.savefig('graphs/ridge polynomial')



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
example_lasso = Lasso(alpha = 1)
example_lasso.fit(X_train, y_train)
print(example_lasso.score(X_test, y_test))

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
plt.figure()
plot_learning_curves(Lasso(alpha = 25, copy_X = False, fit_intercept = True, selection = 'random'), X_train, y_Train)
plt.savefig("graphs/ lasso learning curve")

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
plt.figure()
plot_learning_curves(ElasticNet(alpha = 0.01), X_train, y_Train)
plt.savefig("graphs/learningcurve elastic net")
chosen_model = ElasticNet(alpha = 0.01)

# Polynomial
#Poly = PolynomialFeatures(degree = 10, include_bias = False)
#xTrainPoly = Poly.fit_transform(X_train)
#xTrainPolyscaled = scaler.fit_transform(xTrainPoly) #do i need to scale again if the training data is already scaled?

#Ridge Model
#ridge_model.fit(X_train, y_Train)
#yFit_ridge = ridge_model.predict(X_train)
#y_pred_ridge = ridge_model.predict(X_test)

#Lasso Model
#lasso_model.fit(X_train, y_Train)
#yFit_lasso = lasso_model.predict(X_train)
#y_pred_lasso = lasso_model.predict(X_test)

#Elastic Net Model
chosen_model.fit(X_train, y_Train)
yFit_elastic = chosen_model.predict(X_train)
y_pred_elastic = chosen_model.predict(X_test
                                       )
# Plot predictions chosen model
plt.figure()
plt.scatter(yFit_elastic, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_pred_elastic, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Elastic Net regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.savefig("graphs/predictedvaluesElasticNet")

