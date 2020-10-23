from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from pathlib import Path
from urlpath import URL
from requests import get
import matplotlib.pyplot as plt



np.random.seed(42)

m = 20
X = 3 * np.random.rand(m, 1) #this is training data x
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5 #training data y
X_new = np.linspace(0, 3, 100).reshape(100, 1)

def plot_model(model_class, xtrain, ytrain, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = (
            model_class(alpha, **model_kargs)
            if alpha > 0
            else LinearRegression()
        )
        if polynomial:
            model = Pipeline(
                [
                    (
                        "poly_features",
                        PolynomialFeatures(degree=10, include_bias=False),
                    ),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ]
            )
        model.fit(xtrain, ytrain)                 #reg_model.fit(polynomialscaled, ytrain)
        y_new_regul = model.predict(xtrain)        #y>regul = reg_model.predict(polynomialscaled)
        lw = 2 if alpha > 0 else 1
        plt.plot(                                 #plot(x_train, yregul)
            X_new,
            y_new_regul,
            style,
            linewidth=lw,
            label=r"$\alpha = {}$".format(alpha),
        )
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


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




