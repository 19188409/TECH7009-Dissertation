from final_file import load_data, remove_outliers, encode_features, adjust_maxmintemps, split_data, scale_train_test, \
    chi_squared_analysis, chi_squared2, chi_squared3, pca_dataset, chi_attendance_analysis

import math
import statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


def linear_regression(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    r_sq = model.score(x_train, y_train)
    print(f"coefficient of determination: {r_sq}")
    y_pred = model.predict(x_test)
    x = np.arange(len(y_test))
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.legend()
    plt.show()
    print(f"predicted response:\n{y_pred}")
    print("Actuals: ", y_test)
    print("coefficient for test results: ", metrics.r2_score(y_test, y_pred))
    print("MSE for test results: ", mean_squared_error(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))

    var = statistics.variance(y_pred)
    print("\nVariance of predictions: ", var, "\n")

    stdv = math.sqrt(var)
    print("\nSquare root of variance of predictions: ", stdv, "\n")

    print("Second method for bias and variance...")
    sigma = np.mean(y_pred)
    bias = y_pred - y_test

    var = np.mean(np.abs(y_pred - sigma), 0)
    print("bias: ", bias)
    print("var: ", var)

    tot = 0.0
    for i in y_pred:
        tot = tot + (i - sigma) ** 2

    print("Variance loop calc: ", (tot / len(y_pred)))











"""Load Data"""
print("Loading data... \n")
file = 'final_dataset.csv'
df = load_data(file)

print("Loading data for survey days... \n")
file2 = 'survey-days-data.csv'
df_new_data = load_data(file2)

print("Data files loaded successfully!")

print("\nShape of dataframe is:")
print(df.shape)

"""Scale the temperature values so all above 0"""
print("Preprocessing data...\n")
df = adjust_maxmintemps(df)

"""Remove Outlier values and prints boxplot"""
df = remove_outliers(df)

"""Encode features"""
df = encode_features(df)

"""Chi analysis for attendance"""
chi_attendance_analysis(df)

"""Chi 2 analysis"""
print("Chi2 Analysis...\n")
chi_df = chi_squared_analysis(df)
chi_df2 = chi_squared2(df)
chi_df3 = chi_squared3(df)

"""Split data"""
print("Splitting data...\n")
x_train, x_test, y_train, y_test = split_data(df)
x_train, x_test = scale_train_test(x_train, x_test)

x_train_chi, x_test_chi, y_train_chi, y_test_chi = split_data(chi_df)
x_train_chi, x_test_chi = scale_train_test(x_train_chi, x_test_chi)

x_train_chi2, x_test_chi2, y_train_chi2, y_test_chi2 = split_data(chi_df2)
x_train_chi2, x_test_chi2 = scale_train_test(x_train_chi2, x_test_chi2)

x_train_chi3, x_test_chi3, y_train_chi3, y_test_chi3 = split_data(chi_df3)
x_train_chi3, x_test_chi3 = scale_train_test(x_train_chi3, x_test_chi3)

pca_data, pca_test = pca_dataset(x_train, x_test)


"""Linear Regression Model"""

print("\n")
print("Linear Regression with all features in dataset:")
linear_regression(x_train, x_test, y_train, y_test)
print("\n")
print("Linear Regression with reduced dataset 1:")
linear_regression(x_train_chi, x_test_chi, y_train_chi, y_test_chi)
print("\n")
print("Linear Regression with reduced dataset 2:")
linear_regression(x_train_chi2, x_test_chi2, y_train_chi2, y_test_chi2)
print("\n")
print("Linear Regression with reduced dataset 3:")
linear_regression(x_train_chi3, x_test_chi3, y_train_chi3, y_test_chi3)
print("\n")
print("Linear Regression with PCA dataset:")
linear_regression(pca_data, pca_test, y_train, y_test)
