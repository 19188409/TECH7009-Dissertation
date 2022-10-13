import math
import statistics

import numpy

from final_file import load_data, remove_outliers, encode_features, adjust_maxmintemps, split_data, scale_train_test, \
    chi_squared_analysis, chi_squared2, chi_squared3, pca_dataset, chi_attendance_analysis


import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error




def neural_net_one(x_train, x_test, y_train, y_test):
    """
First MLP Neural Network model, using Relu layers, also plots MSE over epochs, and Actual/Predictions for sandwich figures
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    """
    print("size of testing set is: ", len(x_test))
    print("Length of training set: ", x_train.shape[1])

    layer_input = x_train.shape[1]

    model = Sequential()
    model.add(Dense(layer_input, input_shape=(layer_input,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))

    epoch = 200

    # compile the keras model
    #    model.compile(loss='mean_squared_error', optimizer='AdaGrad', metrics=[keras.metrics.MeanSquaredError()])
    model.compile(loss='mean_squared_error', optimizer='AdaGrad', metrics=['mse', 'mae'])
    # fit the keras model on the dataset
    print("xtype: ", type(x_train))
    print("ytype: ", type(y_train))
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=5)
    # evaluate the keras model

    y_pred = model.predict(x_test)
    x = np.arange(18)
    plt.plot(x, y_pred, label="Predictions")
    plt.plot(x, y_test, label="Actual")
    plt.ylabel("Sandwich Sale Figures")
    plt.xlabel("Training Data")
    plt.title("Predicted and Actual Sales Figures")
    plt.legend()
    plt.show()

    # model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    plt.plot(history.history['mse'], label="Mean Squared Error")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.title("MSE for Neural Network")
    plt.legend()
    plt.show()
    print("Errors for this model are: \n")
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    print("RMSLE: ", mean_squared_log_error(y_test, y_pred, squared=False))

    print("prediction object types:...")

    var = numpy.var(y_pred)
    print("\nVariance of predictions: ", var, "\n")
    std = math.sqrt(var)
    print("\nSquare root of variance of predictions: ", std, "\n")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmsle = mean_squared_log_error(y_test, y_pred, squared=False)
    rows = [mse, mae, rmsle]







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
print("NN with all features in dataset:")
neural_net_one(x_train, x_test, y_train, y_test)
print("\n")
print("NN with reduced dataset:")
neural_net_one(x_train_chi, x_test_chi, y_train_chi, y_test_chi)
print("\n")
print("NN with reduced dataset 2:")
neural_net_one(x_train_chi2, x_test_chi2, y_train_chi2, y_test_chi2)
print("\n")
print("NN with reduced dataset 3:")
neural_net_one(x_train_chi3, x_test_chi3, y_train_chi3, y_test_chi3)
print("\n")
print("NN with PCA dataset:")
neural_net_one(pca_data, pca_test, y_train, y_test)