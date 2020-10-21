n1 = (input("kindly enter any number"))
n2 = (input("kindly enter any number"))

def addition(number1, number2):
    c = number1 + number2
    return c


result = addition(n1,n2)
print(result)



# Polynomial
Poly = PolynomialFeatures(degree = 10, include_bias = False)
xTrainPoly = Poly.fit_transform(X_train)
xTrainPolyscale = scaler.fit_transform(xTrainPoly)







#Ridge Regression 2.0
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
    plt.xlabel('columns_to_use', fontsize = 18)
    plt.savefig("graphs/ridge_2.0")

    # Lasso Regressor L2
    lasso_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03],
                    "fit_intercept": [True, False],
                    "copy_X": [True, False],
                    "selection": ['cyclic', 'random']
                    }
    lasso_regressor = GridSearchCV(lasso_model, lasso_params, scoring='neg_mean_squared_error', cv=5)
    lasso_regressor.fit(xTrainPolyscale, y_train)
    print(lasso_regressor.best_params_)
    print(lasso_regressor.best_score_)

    # Elastic Net
    elastic_params = {
        'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-2, 0.02, 0.024, 0.025, 0.026, 0.03, 1, 5, 10, 20, 200, 230, 250, 265,
                  270, 275, 290, 300, 500]}
    elastic_regressor = GridSearchCV(elastic_model, elastic_params, scoring='neg_mean_squared_error', cv=5)
    elastic_regressor.fit(xTrainPolyscale, y_train)
    print(elastic_regressor.best_params_)
    print(elastic_regressor.best_score_)

#Linear Model
reg_model.fit(X_train, y_Train)
yFit_linear = reg_model.predict(X_train)
y_pred_linear = reg_model.predict(X_test)

#Ridge Model
ridge_model.fit(X_train, y_Train)
yFit_ridge = ridge_model.predict(X_train)
y_pred_ridge = ridge_model.predict(X_test)

#Lasso Model
lasso_model.fit(X_train, y_Train)
yFit_lasso = lasso_model.predict(X_train)
y_pred_lasso = lasso_model.predict(X_test)

#Elastic Net Model
elastic_model.fit(X_train, y_Train)
yFit_elastic = elastic_model.predict(X_train)
y_pred_elastic = elastic_model.predict(X_test)

#plot predicted values
plt.figure()
plt.scatter(y_test, y_pred_linear)
plt.savefig("graphs/ predicted values linear model")



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
