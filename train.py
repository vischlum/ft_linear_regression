#!/usr/bin/env python3

import json
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean

from predict import estimate

def normalize(array):
    max_value = max(array)
    result = [x / max_value for x in array]
    return result

def denormalize(t0, t1, x, y):
    return float(t0 * max(y)), float(t1 * max(y) / max(x))

def t0_sigma(t0, t1, x, y, size):
    t0_sum = 0
    for i in range(size):
        t0_sum += (estimate(t0, t1, x[i]) - y[i])
    return t0_sum

def t1_sigma(t0, t1, x, y, size):
    t1_sum = 0
    for i in range(size):
        t1_sum += (estimate(t0, t1, x[i]) - y[i]) * x[i]
    return t1_sum

# Linear regression using gradient descent algorithm
def linear_regression(x, y, learning_rate):
    t0 = t1 = sum_tmp0 = sum_tmp1 = 0
    size = len(x)
    while (sum_tmp0 != t0_sigma(t0, t1, x, y, size) and sum_tmp1 != t1_sigma(t0, t1, x, y, size)):
        t0_tmp = t0 - (learning_rate * (t0_sigma(t0, t1, x, y, size)/size))
        t1_tmp = t1 - (learning_rate * (t1_sigma(t0, t1, x, y, size)/size))
        sum_tmp0 = t0_sigma(t0, t1, x, y, size)
        sum_tmp1 = t1_sigma(t0, t1, x, y, size)
        t0 = t0_tmp
        t1 = t1_tmp
    return t0, t1

def dump_to_json(theta0, theta1, xlabel, ylabel):
    try:
        json_data = {
            'theta0': theta0,
            'theta1': theta1,
            'xlabel': xlabel,
            'ylabel': ylabel
        }
        with open('values.json', 'w') as outfile:
            json.dump(json_data, outfile)
    except Exception as e:
        print(f"Error when writing the file: {e}", file=sys.stderr)
        sys.exit(1)

def display_plot(xlabel, ylabel, X, Y, theta0, theta1):
    plt.figure('ft_linear_regression')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(X, Y, color='blue')
    plt.plot(X, theta1*X + theta0, color='orange')
    plt.show()

"""
- MAE (Mean Absolute Error) represents the difference between the original and
predicted values extracted by averaged the absolute difference over the data set.
- MSE (Mean Squared Error) represents the difference between the original and
predicted values extracted by squared the average difference over the data set.
- RMSE (Root Mean Squared Error) is the error rate by the square root of MSE.
- R-squared (Coefficient of Determination) represents the coefficient of how well
the values fit compared to the original values. The value from 0 to 1 interpreted as percentages.
The higher the value is, the better the model is.
Source: https://www.datatechnotes.com/2019/02/regression-model-accuracy-mae-mse-rmse.html
"""
def compute_accuracy(t0, t1, x, y):
    predicted = []
    for i in range(len(x)):
        predicted.append(estimate(t0, t1, x[i]))
    delta = y - predicted
    mse = mean(delta**2)
    mae = mean(abs(delta))
    r2 = 1 - (sum(delta**2) / sum((y - mean(y))**2))
    print(f"""Here are the various metrics for accuracy:
    \tMAE (Mean Absolute Error) = {round(mae, 2)}
    \tMSE (Mean Square Error) = {round(mse, 2)}
    \tRMSE (Root Mean Squared Error) = {round(sqrt(mse), 2)}
    \tR-squared (Coefficient of determination) = {round(r2, 2)}""")

def main():
    learning_rate = 0.1

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", help="the .csv dataset to be used (default is data.csv)",
            nargs='?', default="data.csv", type=open)
        parser.add_argument("-v", "--verbose", help="display more calculation details",
            action="store_true")
        parser.add_argument("-p", "--plot", help="display data plot",
            action="store_true")
        args = parser.parse_args()
    except Exception as e:
        print(f"Error when parsing the command-line: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Import the dataset
        dataset = pd.read_csv(args.dataset)
        # Split each column into its own array
        X = dataset.iloc[:,:-1].values
        Y = dataset.iloc[:,1].values
        # Normalize the values to avoid overflows
        x_norm = normalize(X)
        y_norm = normalize(Y)
    except Exception as e:
        print(f"Error when reading the dataset: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        theta0, theta1 = linear_regression(x_norm, y_norm, learning_rate)
        theta0, theta1 = denormalize(theta0, theta1, X, Y)

        if args.verbose:
            print(f"With the given dataset, our two coefficients are:\
                \n\ttheta0 = {round(theta0, 2)}\n\ttheta1 = {round(theta1, 2)}")
            compute_accuracy(theta0, theta1, X, Y)

        dump_to_json(theta0, theta1, dataset.columns[0], dataset.columns[1])

        if args.plot:
            display_plot(dataset.columns[0], dataset.columns[1], X, Y, theta0, theta1)
        
        print("Training successful, you can now use predict.py")
    except Exception as e:
        print(f"Error when processing the dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__" : 
	main()
