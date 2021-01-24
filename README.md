# ft_linear_regression
This a [School 42](https://www.42.fr/) project to learn the basics of linear regression. The PDF of the subject is [here](https://cdn.intra.42.fr/pdf/pdf/13331/en.subject.pdf).   
You have three Python 3 scripts:
- [`train.py`](train.py) to process the dataset and compute the linear regression
- [`predict.py`](predict.py) to use the linear regression to compute any given value
- [`generate.py`](generate.py) to randomly generate datasets

## How to run
1. `pip3 install -r requirements.txt` to install the necessary dependencies
2. `./train.py` to train our linear regression
3. `./predict.py` to compute values based on our linear regression

Both `train.py` and `generate.py` offer commandline arguments to customize the program (use `--help` for more details).

## Source of the `tests` datasets
- `swedish_insurance_data.csv` taken from [this tutorial](https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/)
- `salary_data.csv` taken from [this tutorial](https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/)
- `noise0.csv` and `noise100.csv` were created with `generate.py`
