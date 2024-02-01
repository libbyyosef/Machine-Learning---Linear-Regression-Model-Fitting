from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from utils import *
from IMLearn.metrics.loss_functions import mean_square_error

global list_col_x
global mean_x
GRADE = "grade"
SQFT_LIVING = "sqft_living"
pio.templates.default = "simple_white"
columns_to_drop = ["id", "date", "lat", "long", "sqft_above", "sqft_lot15",
                   "sqft_living15", "sqft_lot"]


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a
    single
    DataFrame or a Tuple[DataFrame, Series]
    """
    global list_col_x
    global mean_x
    if y is None:
        return test_preprocess_data(X)
    X["price"] = y
    X = filter_x(X)
    y = X["price"]
    X = X.drop("price", axis=1)
    mean_x = X.mean(axis=0)
    list_col_x = list(X.columns)
    return X, y


def filter_x(X):
    X = X.dropna()
    X = X.drop_duplicates()
    X = X.drop(columns_to_drop, axis=1)
    X = X[X["price"] > 0]
    for col in ["bedrooms", "bathrooms", "sqft_basement"]:
        X = X[X[col] >= 0]
    for col in ["price", "sqft_living", "floors"]:
        X = X[X[col] > 0]
    X = X[X["waterfront"].isin([0, 1])]
    X = X[X["view"].isin(range(5))]
    X = X[X["condition"].isin(range(1, 6))]
    X = X[X["grade"].isin(range(1, 14))]
    X = X[((X["yr_renovated"] >= X["yr_built"]) | (X["yr_renovated"] == 0))]
    X = pd.get_dummies(X, columns=['zipcode'])
    X = add_col_recently_renovated(X)
    return X


def test_preprocess_data(X: pd.DataFrame):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    Returns
    -------
    Post-processed design matrix and response vector (prices) - as a single
    DataFrame
    """
    X = X.drop(columns_to_drop, axis=1)
    X = pd.get_dummies(X, columns=['zipcode'])
    X = add_col_recently_renovated(X)
    X = X.reindex(columns=list_col_x, fill_value=0)
    for col in X.columns:
        X[col].fillna(mean_x[col], inplace=True)
    return X


def add_col_recently_renovated(X):
    X["recently_renovated"] = np.where(
        X["yr_renovated"] >= np.percentile(X.yr_renovated.unique(), 75), 1,
        0)
    X = X.drop("yr_renovated", axis=1)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False))]
    sqft_living_corr = np.cov(X[SQFT_LIVING], y)[0, 1] / (np.std(X[
                                                                     SQFT_LIVING]) *
                                                          np.std(y))
    grade_corr = np.cov(X[GRADE], y)[0, 1] / (np.std(X[GRADE]) *
                                              np.std(y))

    graph1 = px.scatter(pd.DataFrame({'x': X[SQFT_LIVING], 'y': y}), x="x",
                        y="y")
    graph1.update_layout(xaxis_title="feature name",
                         yaxis_title=f"Pearson Correlation between"
                                     f" {SQFT_LIVING} "
                                     "and "
                                     "response",
                         title=f"Pearson Correlation between {SQFT_LIVING} "
                               f"and "
                               f"response Person Correlation"
                               f" {sqft_living_corr}")
    graph1.write_image(output_path + f"/Pearson_Correlation_{SQFT_LIVING}.png")
    graph2 = px.scatter(pd.DataFrame({'x': X[GRADE], 'y': y}), x="x", y="y")
    graph2.update_layout(xaxis_title="feature name",
                         yaxis_title=f"Pearson Correlation between {GRADE} "
                                     f"and "
                                     "response",
                         title=f"Pearson Correlation between {GRADE} and "
                               "response Person Correlation"
                               f" {grade_corr}")
    graph2.write_image(output_path + f"/Pearson_Correlation_{GRADE}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    y = df["price"]
    X = df.drop("price", axis=1)
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)
    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the
    # following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon
    # of size (mean-2*std, mean+2*std)
    outcomes_mtx = np.zeros((91, 10))
    percentages = list(range(10, 101))
    for p in range(10, 101):
        for j in range(10):
            temp_x = train_X.sample(frac=p / 100.0)
            temp_y = train_y.loc[temp_x.index]
            outcomes_mtx[p - 10, j] = LinearRegression(
                include_intercept=True).fit(
                temp_x, temp_y).loss(test_X, test_y.values)
    mean = np.mean(outcomes_mtx, axis=1)
    std = np.std(outcomes_mtx, axis=1)
    upper_bound, lower_bound = mean + 2 * std, mean - 2 * std
    graph = go.Figure([go.Scatter(x=percentages, y=upper_bound,
                                  line=dict(color="lightblue"), mode="lines"),
                       go.Scatter(x=percentages, y=lower_bound, fill='tonexty',
                                  line=dict(color="lightblue"), mode="lines"),
                       go.Scatter(x=percentages, y=mean,
                                  marker=dict(color="blue"),
                                  mode="markers+lines")
                       ],
                      layout=go.Layout(
                          title="Average Loss As Function Of Training Set",
                          xaxis=dict(title="Percentage of Training Set"),
                          yaxis=dict(title="Loss Function"),
                          showlegend=False))
    graph.write_image("Avg_Loss_As_Function_Of_Training_Size.png")
