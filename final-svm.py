from final_file import load_data, remove_outliers, encode_features, adjust_maxmintemps, split_data, scale_train_test, \
    chi_squared_analysis, chi_squared2, chi_squared3, pca_dataset, chi_attendance_analysis

import math
import random
import statistics

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV

from sklearn import metrics, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error



def svm_model_function(C, gamma, kernel):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)

    model = clf.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    x = np.arange(len(y_test))
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.legend()
    title = f"SVM Model with C {C}, gamma {gamma} and kernel {kernel}"
    plt.title(title)
    plt.show()

    print("optimal r2 score for test results: ", metrics.r2_score(y_test, y_pred))
    print("MAE for test results: ", mean_absolute_error(y_test, y_pred))

    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    print("RMSLE: ", mean_squared_log_error(y_test, y_pred, squared=False))


def svm_model(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel="linear", C=1)

    model = clf.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    x = np.arange(len(y_test))
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.legend()
    plt.title("Initial linear SVM Model with C=1")
    plt.show()

    print("Predictions: ", y_pred, '\n')
    print("Actuals: ", y_test, '\n')
    print("intial r2 score for test results: ", metrics.r2_score(y_test, y_pred))
    print("MAE for test results: ", mean_absolute_error(y_test, y_pred))
    print("Conducting grid search...")
    Cs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
    gammas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
    kernel = ['linear', 'rbf', 'poly', 'sigmoid']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}

    print("\nMAE scoring attribute...")
    grid_search = GridSearchCV(clf, param_grid, cv=4, scoring='neg_mean_absolute_error')
    grid_search.fit(x_train, y_train)
    temp = grid_search.best_params_
    print("Optimal parameters: ", temp)

    svm_model_function(temp['C'], temp['gamma'], temp['kernel'])

    print("\nr2 score attribute...")
    grid_search = GridSearchCV(clf, param_grid, cv=4, scoring='r2')  # explained_variance
    grid_search.fit(x_train, y_train)
    temp = grid_search.best_params_
    print("Optimal parameters: ", temp)
    print(temp['C'])

    svm_model_function(temp['C'], temp['gamma'], temp['kernel'])


def svm_model2(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel="linear", C=0.01, gamma=0.01)
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    x = np.arange(len(y_test))
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.legend()
    plt.title("SVM Model Predictions")
    plt.show()

    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    print("RMSLE: ", mean_squared_log_error(y_test, y_pred, squared=False))

    var = statistics.variance(y_pred)
    print("\nVariance of predictions: ", var, "\n")

    stdv = math.sqrt(var)
    print("\nSquare root of variance of predictions: ", stdv, "\n")




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



print("\n")
print("SVM with all features in dataset:")
svm_model(x_train, x_test, y_train, y_test)
svm_model2(x_train, x_test, y_train, y_test)
print("\n")
print("SVM with reduced dataset 1:")
svm_model2(x_train_chi, x_test_chi, y_train_chi, y_test_chi)
print("\n")
print("SVM with reduced dataset 2:")
svm_model2(x_train_chi2, x_test_chi2, y_train_chi2, y_test_chi2)
print("\n")
print("SVM with reduced dataset 3:")
svm_model2(x_train_chi3, x_test_chi3, y_train_chi3, y_test_chi3)
print("\n")
print("SVM with PCA dataset:")
svm_model2(pca_data, pca_test, y_train, y_test)