import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df[df["Month"].isin(range(1, 13))]
    df = df[df["Day"].isin(range(1, 32))]  # day 0?
    df = df[df["Year"] > 0]
    df = df[df["Temp"] > 0]
    df['DayOfYear'] = df["Date"].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df["Country"] == "Israel"]
    temp_day_of_year_graph = px.scatter(df_israel, x="DayOfYear", y="Temp",
                                        color="Year")
    temp_day_of_year_graph.write_image("Israel_Daily_Temperature.png")
    grouped_by_month = df_israel.groupby(["Month"], as_index=False).agg(
        std=("Temp", "std"))
    graph_bar = px.bar(grouped_by_month,
                       title="Temperature Standard Deviation Over Years",
                       x="Month",
                       y="std")
    graph_bar.write_image("Israel_monthly_Average_Temperature.png")

    # Question 3 - Exploring differences between countries
    grouped_by_country_month = df.groupby(["Country", "Month"],
                                          as_index=False).agg(
        mean=("Temp", "mean"), std=("Temp", "std"))
    graph_line = px.line(grouped_by_country_month,
                         x="Month", y="mean", error_y="std", color="Country") \
        .update_layout(title="Average Monthly Temperatures",
                       xaxis_title="Month",
                       yaxis_title="Mean Temperature")
    graph_line.write_image("Mean_Temp_Different_Countries.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_israel["DayOfYear"],
                                                        df_israel["Temp"])
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(

    ), \
        test_X.to_numpy(), test_y.to_numpy()

    deg = list(range(1, 11))
    loss_of_the_models = np.zeros(len(deg), dtype=float)

    for k in deg:
        model = PolynomialFitting(k=k).fit(train_X,
                                           train_y)
        model_loss = model.loss(test_X, test_y)
        loss_of_the_models[deg.index(k)] = np.round(model_loss, 2)

    loss_df = pd.DataFrame({'k': deg, 'loss': loss_of_the_models})
    graph_bar2 = px.bar(loss_df, x='k', y='loss', text='loss',
                        title=r'$\text{Test Error For Different Values of }k$')
    graph_bar2.write_image('Israel_Different_k.png')
    print(loss_df)

    # Question 5 - Evaluating fitted model on different countries
    countries = ["Jordan", "South Africa", "The Netherlands"]
    poly_smallest_loss = PolynomialFitting(k=5)
    model = poly_smallest_loss.fit(df_israel.DayOfYear.to_numpy(),
                                   df_israel.Temp.to_numpy())
    loss_list = []
    for country in countries:
        loss_value = round(model.loss(df[df.Country == country].DayOfYear,
                                      df[df.Country == country].Temp), 2)
        loss_dict = {"country": country, "loss": loss_value}
        loss_list.append(loss_dict)
    df_country_loss = pd.DataFrame(loss_list)
    graph_bar3 = px.bar(df_country_loss, x="country", y="loss", text="loss",
                        color="country",
                        title="Loss Over Countries For Model Fitted Over "
                              "Israel")
    graph_bar3.write_image("Other_Countries_Loss.png")
