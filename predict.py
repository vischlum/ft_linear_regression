#!/usr/bin/env python3

import json
import sys

def estimate(t0, t1, x):
    return float((x * t1) + t0)

def read_from_json():
    try:
        with open('values.json') as json_file:
            data = json.load(json_file)
        return data['theta0'], data['theta1'], data['xlabel'], data['ylabel']
    except FileNotFoundError:
        # When the predictor is launched before the trainer, it must return 0
        print("\x1b[31m\x1b[1mWARNING: values.json not found! Make sure to launch train.py first.\x1b[0m")
        print("\x1b[31mWithout training, predict.py will just return 0 no matter your input.\x1b[0m")
        return 0, 0, "xlabel", "ylabel"
    except Exception as e:
        print(f"Error when reading the file: {e}", file=sys.stderr)
        sys.exit(1)

def receive_user_input(xlabel):
    try:
        x = float(input(f"Please, enter the value of x ({xlabel}) you want to test: "))
        return x
    except Exception as e:
        print(f"Error when reading user input: {e}", file=sys.stderr)
        sys.exit(1)

def prediction(theta0, theta1, x, ylabel):
    estimated_y = estimate(theta0, theta1, x)
    print(f"For the given value of x, y ({ylabel}) would be: {round(estimated_y, 2)}")

def main():
    print("Welcome to the ft_linear_regression predictor")
    theta0, theta1, xlabel, ylabel = read_from_json()

    while True:
        user_x = receive_user_input(xlabel)
        prediction(theta0, theta1, user_x, ylabel)

if __name__ == "__main__" : 
	main()
