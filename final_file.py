import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


def load_data(x):
    """
Loads data from a csv file
    :param x:
    :return:
    """
    data = pd.read_csv(x, header=0)
    return data


def remove_outliers(data):
    """This produces a boxplot to determine outlier entries.
    It then removes all zero value rows and outlier rows, and produces a final boxplot
    """

    box_data = data['SandwichesSold']
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(box_data)
    fig1.show()

    q1 = np.quantile(box_data, 0.25)
    q3 = np.quantile(box_data, 0.75)
    med = np.median(box_data)

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (1.5 * iqr)
    print(iqr, upper_bound)

    new_df = data
    # Remove the rows with zero sales figures recorded, and outlier sales figures
    for i in range(len(data)):
        row = data.iloc[i]
        if row["SandwichesSold"] == 0:
            print("found zero value sandwich sales at row ", i)
            new_df = new_df.drop([i])
        if row["SandwichesSold"] > upper_bound:
            print("found outlier value at row ", i, " with value: ", row["SandwichesSold"])
            new_df = new_df.drop([i])

    # Plot the new boxplot with the removed outliers
    print(new_df.columns)
    box_data = new_df['SandwichesSold']
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot 2')
    ax1.boxplot(box_data)
    fig1.show()
    return new_df


def encode_features(data):
    """Through the use of encoding, the columns that contain String types are vectorised, dramatically increasing the number of features"""
    # df2 = pd.get_dummies(data, columns=["Day", 'Weather1', 'Weather2', 'Weather3', 'Weather4', 'Weather5', 'Weather6'])
    df2 = pd.get_dummies(data, columns=['Weather0000', 'Weather0030', 'Weather0100', 'Weather0130', 'Weather0200',
                                        'Weather0230', 'Weather0300',
                                        'Weather0330', 'Weather0400', 'Weather0430', 'Weather0500', 'Weather0530',
                                        'Weather0600', 'Weather0630', 'Weather0700',
                                        'Weather0730', 'Weather0800', 'Weather0830', 'Weather0900', 'Weather0930',
                                        'Weather1000', 'Weather1030', 'Weather1100',
                                        'Weather1130', 'Weather1200', 'Weather1230', 'Weather1300', 'Weather1330',
                                        'Weather1400', 'Weather1430', 'Weather1500',
                                        'Weather1530', 'Weather1600', 'Weather1630', 'Weather1700', 'Weather1730',
                                        'Weather1800', 'Weather1830', 'Weather1900',
                                        'Weather1930', 'Weather2000', 'Weather2030', 'Weather2100', 'Weather2130',
                                        'Weather2200', 'Weather2230', 'Weather2300', 'Weather2330'])

    """Split the date into day, month, year and quarter features"""
    dates = df2['Date']
    print("dates: \n", dates)
    dates = pd.to_datetime(dates, format='%d/%m/%Y')
    print("new dates: \n", dates)
    df2['Date'] = dates
    df2['Date_Day'] = df2["Date"].dt.day
    df2['Date_Month'] = df2["Date"].dt.month
    df2['Date_Year'] = df2["Date"].dt.year
    df2['Date_Quarter'] = df2["Date"].dt.quarter

    df2 = pd.get_dummies(df2, columns=['Date_Day', 'Date_Month', 'Date_Year', 'Date_Quarter'])
    df2 = df2.drop(['Date'], axis=1)
    #    print(df2.head())
    print(df2.columns)
    return df2


def adjust_maxmintemps(data):
    """
This function adjusts the temperature values so that they are all positive, this is done by scaling them all between 1 and the max temp value
    :param data:
    :return:
    """
    temp_df = data[['AvgDailyMin', 'AvgDailyHigh', 'DailyMin', 'DailyMax']]
    minval = min(temp_df.min())
    adjustval = abs(minval) + 1
    temp_df = temp_df + adjustval
    data = data.drop(['AvgDailyMin', 'AvgDailyHigh', 'DailyMin', 'DailyMax'], axis=1)
    data = pd.concat([data, temp_df], axis=1)
    print(data)
    return data


def split_data(data):
    """Splits data into test/train, and also normalises them before returning"""
    label = data['SandwichesSold']
    new_df = data.drop(['SandwichesSold'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(new_df, label, test_size=0.2, random_state=57)

    norm = Normalizer()
    norm_x_train = norm.fit_transform(x_train)
    norm_x_test = norm.transform(x_test)

    return norm_x_train, norm_x_test, y_train, y_test


def scale_train_test(train, test):
    """ Method for scaling the test and train features datasets
    :param train:
    :param test:
    :return:
    """
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)
    return scaled_train, scaled_test


def chi_attendance_analysis(new_df):
    """
Conduct chi squared analysis on the dataset, with TotalAttendance as the target variable. This helps in determining which features are most independent from the outputs,
thus providing a basis for the feature selection.
    :param new_df:
    """
    print("New chi df created for attendance analysis...")
    label = new_df['TotalAttendance']
    chi_df = new_df.drop(['TotalAttendance'], axis=1)
    chi_scores = chi2(chi_df, label)
    p_values = pd.Series(chi_scores[1], index=chi_df.columns)  # p values
    f_values = pd.Series(chi_scores[0], index=chi_df.columns)  # chi square values
    print("p values from chi test:\n", p_values)
    p_values.sort_values(ascending=False, inplace=True)
    print("p values from chi test after sorting:\n", p_values)
    print("\n The most dependent features:")
    for i in range(1, 16):
        print("Entry ", i, " has p_value: ", p_values[-i], ", at feature: ", p_values.index[-i])


def chi_squared_analysis(new_df):
    """
Conduct chi squared analysis on the dataset, with SandwichesSold as the target variable. This helps in determining which features are most independent from the outputs,
thus providing a basis for the feature selection.
    :param new_df:
    """
    print(new_df)
    label = new_df['SandwichesSold']
    chi_df = new_df.drop(['SandwichesSold'], axis=1)

    # chi_df = new_df[['Date_Day','Date_Month', 'Date_Year', 'Date_Quarter', 'PALACE_PARK_AND_GARDENS', 'PARK_AND_GARDENS', 'PARK_ONLY']]
    print("New chi temp df created")
    chi_scores = chi2(chi_df, label)
    # print("chi test: ", chi_scores)
    p_values = pd.Series(chi_scores[1], index=chi_df.columns)  # p values
    f_values = pd.Series(chi_scores[0], index=chi_df.columns)  # chi square values
    print("p values from chi test:\n", p_values)
    p_values.sort_values(ascending=False, inplace=True)
    print("p values from chi test after sorting:\n", p_values)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title("P Values for Chi Test")
    # ax1.plot(p_values)
    # fig1.autofmt_xdate(rotation=45)
    # fig1.show()
    features = []

    for i in range(len(p_values)):
        if p_values[i] > 0.9:  # empty lists could be the cause of the indexing errors
            # print(i)
            feat = p_values.index[i]

            features.append(feat)
            chi_df = chi_df.drop([feat], axis=1)

    print("\n The most dependent features:")
    for i in range(1, 16):
        print("Entry ", i, " has p_value: ", p_values[-i], ", at feature: ", p_values.index[-i])

    """for i in range(len(p_values)):
        if p_values[i] > 0.9:  # empty lists could be the cause of the indexing errors
            # print(i)
            feat = chi_df.columns[i]

            features.append(feat)
            chi_df = chi_df.drop([feat], axis=1)
    """
    print(features, "\n")

    for i in range(5):
        random.seed(i)
        x = random.randrange(0, len(p_values))
        print("Entry ", x, ": ", p_values[x], ", at feature: ", p_values.index[x])

    print(len(features))
    # print(len(feat))
    chi_df = pd.concat([chi_df, label], axis=1)

    return chi_df


def chi_squared2(new_df):
    """
Conduct chi squared analysis on the dataset. This helps in determining which features are most independent from the outputs,
thus providing a basis for the feature selection.
    :param new_df:
    """
    print(new_df)
    label = new_df['SandwichesSold']
    chi_df = new_df.drop(['SandwichesSold'], axis=1)

    # chi_df = new_df[['Date_Day','Date_Month', 'Date_Year', 'Date_Quarter', 'PALACE_PARK_AND_GARDENS', 'PARK_AND_GARDENS', 'PARK_ONLY']]
    print("New chi temp df created")
    chi_scores = chi2(chi_df, label)
    # print("chi test: ", chi_scores)
    p_values = pd.Series(chi_scores[1], index=chi_df.columns)  # p values
    f_values = pd.Series(chi_scores[0], index=chi_df.columns)  # chi square values
    print("p values from chi test:\n", p_values)
    p_values.sort_values(ascending=False, inplace=True)
    print("p values from chi test after sorting:\n", p_values)

    features = []

    for i in range(len(p_values)):
        if p_values[i] > 0.5:  # empty lists could be the cause of the indexing errors
            # print(i)
            feat = p_values.index[i]

            features.append(feat)
            chi_df = chi_df.drop([feat], axis=1)

    print(len(features))
    chi_df = pd.concat([chi_df, label], axis=1)

    return chi_df


def chi_squared3(new_df):
    """
Conduct chi squared analysis on the dataset. This helps in determining which features are most independent from the outputs,
thus providing a basis for the feature selection.
    :param new_df:
    """
    print(new_df)
    label = new_df['SandwichesSold']
    chi_df = new_df.drop(['SandwichesSold'], axis=1)

    # chi_df = new_df[['Date_Day','Date_Month', 'Date_Year', 'Date_Quarter', 'PALACE_PARK_AND_GARDENS', 'PARK_AND_GARDENS', 'PARK_ONLY']]
    print("New chi temp df created")
    chi_scores = chi2(chi_df, label)
    # print("chi test: ", chi_scores)
    p_values = pd.Series(chi_scores[1], index=chi_df.columns)  # p values
    f_values = pd.Series(chi_scores[0], index=chi_df.columns)  # chi square values
    print("p values from chi test:\n", p_values)
    p_values.sort_values(ascending=False, inplace=True)
    print("p values from chi test after sorting:\n", p_values)

    features = []

    for i in range(len(p_values)):
        if p_values[i] > 0.05:  # empty lists could be the cause of the indexing errors
            # print(i)
            feat = p_values.index[i]

            features.append(feat)
            chi_df = chi_df.drop([feat], axis=1)

    print(len(features))
    chi_df = pd.concat([chi_df, label], axis=1)

    return chi_df


def pca_dataset(data, test):
    """
    Conduct PCA on the dataset
    :param data:
    :param test:
    :return:
    """
    pca = PCA(n_components=68)
    pca.fit_transform(data)
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    #
    # Create the visualization plot
    #
    plt.bar(range(0, len(exp_var)), exp_var, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.axhline(y=0.95, color='r', ls='-')
    plt.grid()
    plt.show()

    pca2 = PCA(n_components=40)
    pca_data = pca2.fit_transform(data)
    pca_test = pca2.transform(test)
    return pca_data, pca_test


def neural_net_three(x_train, x_test, y_train, y_test):
    """
    Unused, untested ANN model which has been left here for reference.

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    print("size of testing set is: ", len(x_test))
    print("Length of training set: ", x_train.shape[1])

    layer_input = x_train.shape[1]

    model = Sequential()
    model.add(Dense(layer_input, input_shape=(layer_input,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    print(model.summary())
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

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmsle = mean_squared_log_error(y_test, y_pred, squared=False)
    rows = [mse, mae, rmsle]


