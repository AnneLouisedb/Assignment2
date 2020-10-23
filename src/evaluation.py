#from modelling import chosen_model, ridge_model

data_dir = Path("data/")
all_data = pd.read_csv(data_dir / "housing-data.csv", index_col="Order")
test_data = pd.read_csv(data_dir / "hold-out.csv", index_col="Order")



def evaluate_model(
    estimator, Xtest, ytest, train=False, Xtrain=None, ytrain=None
):
    """Evaluate an estimator object using Xtest and ytest as evaluation data. If
    train==True, use Xtrain and ytrain to train the model first. Evaluates model
    based on MAE, MSE, R^2.

    :param estimator: an estimator object with a 'fit' and 'predict' method
    :param Xtest: data to be predicted for
    :param ytest: true target values to be used in evaluation
    :param train: Default == False. If true, fit estimator using Xtrain, ytrain
    :param Xtrain: Default == None. Array or DataFrame used when training the
    model
    :param ytrain: Default == None. Array or DataFrame used when training the
    model
    :returns: evaluation metrics
    :rtype: string
    """
    if train == True:
        estimator.fit(Xtrain, ytrain)
    ypred = estimator.predict(Xtest)
    r2 = r2_score(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    return {"r2": r2, "mae": mae, "mse": mse}

chosen_model = ElasticNet(alpha = 0.03)

print(
    evaluate_model(
        chosen_model,
        test_data.drop(columns="SalePrice"),
        test_data["SalePrice"],
        train=True,
        Xtrain=all_data.drop(columns="SalePrice"),
        ytrain=all_data["SalePrice"],
    )
)

