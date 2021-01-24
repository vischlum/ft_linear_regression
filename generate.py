#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
from sklearn.datasets import make_regression

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", help="the CSV output file (default is random.csv)",
            nargs='?', default="random.csv", type=argparse.FileType('w', encoding='UTF-8'))
        parser.add_argument("-s", "--samples", help="the number of samples to be generated (default is 100)",
            default=100, type=int)
        parser.add_argument("-n", "--noise", help="the standard deviation applied to the output (default is 10)",
            default=10, type=float)
        args = parser.parse_args()
    except Exception as e:
        print(f"Error when parsing the command-line: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # More details on make_regression can be found in the sklearn documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
        X, Y = make_regression(n_samples=args.samples, n_features=1, noise=args.noise)
        df = pd.DataFrame({'random x': X.flatten().tolist(), 'random y': Y.tolist()},
            columns=['random x', 'random y'])
        df.to_csv(args.dataset, index=None, header=['random x', 'random y'])
        print(f"The dataset has been generated and stored in the file {args.dataset.name}")
    except Exception as e:
        print(f"Error when generating the dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__" : 
	main()
