import math
import statistics

from final_file import load_data, remove_outliers, encode_features, adjust_maxmintemps, split_data, scale_train_test, \
    chi_squared_analysis, chi_squared2, chi_squared3, pca_dataset, chi_attendance_analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def optimal_rf(x_train, x_test, y_train, y_test):
    no_of_est = list(range(10, 40, 1))
    mae = []
    mse = []
    for i in no_of_est:
        model = RandomForestRegressor(n_estimators=i, random_state=0)
        history = model.fit(x_train, y_train)
        y_pred = model.predict((x_test))
        mae.append(metrics.mean_absolute_error(y_test, y_pred))
        mse.append(metrics.mean_squared_error(y_test, y_pred))

    x = np.arange(len(no_of_est))
    plt.plot(x, mae, label="MAE")
    plt.xticks(x, no_of_est)
    plt.ylabel("MAE")
    plt.xlabel("No. of estimators")
    plt.title("MAE as estimators increases")
    plt.legend()
    plt.show()

    plt.plot(x, mse, label="MSE")
    plt.xticks(x, no_of_est)
    plt.ylabel("MSE")
    plt.xlabel("No. of estimators")
    plt.title("MSE as estimators increases")
    plt.legend()
    plt.show()


def random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=23, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict((x_test))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))
    print('Root Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred, squared=False))

    print(y_pred)
    print(y_test)

    x = np.arange(18)
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.xlabel("Training Data")
    plt.title("Predicted and Actual Sales Figures")
    plt.legend()
    plt.show()

    list_y = y_test.tolist()
    errors = []
    for i in list_y:
        err = i * 0.1
        errors.append(err)

    x = np.arange(len(y_test))
    plt.plot(x, y_test, label="Actual", marker='x')
    plt.plot(x, y_pred, label="Predictions", marker='+')
    plt.title("Predictions against the 10% boundary")
    plt.fill_between(x, y_test - errors, y_test + errors, color='grey', alpha=0.5)
    plt.ylabel("Sandwich Sale Figures")
    plt.legend()
    plt.show()
    count = 0
    for i in range(len(y_pred)):
        if (list_y[i] - errors[i]) < y_pred[i] < (list_y[i] + errors[i]):
            count = count + 1

    print(count / len(y_pred), "% of predictions fall within the plus/minus criteria")

    for i in range(len(y_pred)):
        if (list_y[i]) < y_pred[i] < (list_y[i] + errors[i]):
            count = count + 1

    print(count / len(y_pred), "% of predictions fall within the criteria")

    errors = []
    for i in list_y:
        err = i * 0.2
        errors.append(err)

    for i in range(len(y_pred)):
        if (list_y[i] - errors[i]) < y_pred[i] < (list_y[i] + errors[i]):
            count = count + 1

    print(count / len(y_pred), "% of predictions fall within the 20% criteria")

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



"""For determing the optimal RF estimator value"""
optimal_rf(x_train, x_test, y_train, y_test)

"""Random Frorest"""

print("\n")
print("Random Forest with all features in dataset:")
random_forest(x_train, x_test, y_train, y_test)
print("\n")
print("Random Forest with reduced dataset 1:")
random_forest(x_train_chi, x_test_chi, y_train_chi, y_test_chi)
print("\n")
print("SVM with reduced dataset 2:")
random_forest(x_train_chi2, x_test_chi2, y_train_chi2, y_test_chi2)
print("\n")
print("SVM with reduced dataset 3:")
random_forest(x_train_chi3, x_test_chi3, y_train_chi3, y_test_chi3)
print("\n")
print("Random Forest with PCA reduced dataset:")
random_forest(pca_data, pca_test, y_train, y_test)

