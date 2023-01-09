import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def read_dataset(file):
    try:
        df = pd.read_csv(file)
        df.drop_duplicates(inplace=True)
    except Exception:
        df = pd.read_csv("/streamlit_app/src/data/nba_logreg.csv")

    df["3P%"].fillna(df["3P%"].mean(), inplace=True)

    df.columns = [
        "Name",
        "Games Played",
        "Minutes Played",
        "Points Per Game",
        "Field Goals Made",
        "Field Goal attempts",
        "Field Goal Percent",
        "3 Points Made",
        "3 Points Attempts",
        "3 Points Percent",
        "Free Throw Made",
        "Free Throw Attempts",
        "Free Throw Percent",
        "Offensive Rebounds",
        "Defensive Rebounds",
        "Rebounds",
        "Assists",
        "Steals",
        "Blocks",
        "Turnovers",
        "Outcome Career Length",
    ]

    features = df.drop(["Outcome Career Length", "Name"], axis=1)
    target = df["Outcome Career Length"]
    return df, features, target


def pearson_corr(df, col1, col2):
    """
    Calculate Pearson correlation between two columns in a dataframe

    :param df: Pandas dataframe
    :param col1: First column
    :param col2: Second column

    :return: Pearson correlation
    """
    # Check if both columns are numeric
    if df[col1].dtype.kind not in "iufc" or df[col2].dtype.kind not in "iufc":
        raise ValueError("Both columns must be numeric")

    # Calculate Pearson correlation
    corr, _ = pearsonr(df[col1], df[col2])
    return corr


def feature_importance_logistic_regression(features, target, metric="f1", penalty="l2"):
    """
    this function takes the features and the target and the metric to maximize and the penalty to use and returns a bar chart of the feature importance

    :param features: the features of the dataset
    :param target: the target of the dataset
    :param metric: the metric to maximize
    :param penalty: the penalty to use

    :return: a bar chart of the feature importance
    """

    # Create a pipeline that scales the features and trains a logistic regression model with the specified penalty
    model = make_pipeline(
        MinMaxScaler(),
        LogisticRegressionCV(
            penalty=penalty,
            Cs=np.logspace(-5, 5, 11),
            scoring=metric,
            solver="liblinear",
            cv=10,
            refit=True,
        ),
    )
    # Fit the model to the training data
    model.fit(features, target)

    # Create a dataframe of feature coefficients and feature names
    coefficients = pd.DataFrame(model[1].coef_.reshape(-1), columns=["Coefficient"])
    coefficients["Feature"] = features.columns

    # Create a bar chart of the feature importances
    return px.bar(
        coefficients,
        x="Feature",
        y="Coefficient",
        title=f"Feature Selection by {penalty.upper()} Penalized Logistic Regression (maximizing {metric})",
    )

    
