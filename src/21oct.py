n1 = (input("kindly enter any number"))
n2 = (input("kindly enter any number"))

def addition(number1, number2):
    c = number1 + number2
    return c


result = addition(n1,n2)
print(result)




#Ridge Regression 2.0
i = 0
ls = ['-', '--', ':']
color = ['r', 'g', 'orange']

for a in [0, 2, 2000]:
    ridgeReg = Ridge(alpha=a)
    ridgeReg.fit(xTrainPolyscaled, y_train)


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




#Linear Model
lasso_model.fit(xTrainPolyscaled, y_Train)
yFit_lasso = lasso_model.predict(xTrainPolyscaled)
plt.figure()
plt.plot(
    X_train,
    yFit_lasso)
plt.savefig("graphs/graph10")

y_pred_linear = reg_model.predict(X_test)

