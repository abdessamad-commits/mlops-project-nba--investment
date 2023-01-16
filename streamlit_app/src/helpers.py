import numpy as np
import pandas as pd
import plotly.express as px
from minio import Minio
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def read_csv_from_minio(client, bucket_name, object_name):
    """
    The function read a csv file from a minio bucket
    
    :param client: the minio client
    :param bucket_name: the name of the bucket
    :param object_name: the name of the object
    
    :return: DataFrame
    """
    try:
        # Get the object data
        data = client.get_object(bucket_name, object_name)
        return pd.read_csv(data)
    except Exception as e:
        print(e)


def read_dataset():
    """
    Read the dataset from a csv file

    :param file: path to the csv file

    :return: a dataframe, the features and the target
    """

    minio_client = Minio(
        "minio:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    df = read_csv_from_minio(
        minio_client, "nba-investment-data", "nba_logreg_preprocessed.csv"
    )

    # df.drop("TARGET_5Yrs", axis=1, inplace=True)

    df["3P%"].fillna(df["3P%"].mean(), inplace=True)

    df.columns = [
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

    features = df.drop(["Outcome Career Length"], axis=1)
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


def feature_importance_tree_model(features, target, tree_model):
    """
    this function takes the features and the target and the metric to maximize and returns a bar chart of the feature importance

    :param features: the features of the dataset
    :param target: the target of the dataset
    :param metric: the metric to maximize (default: accuracy_score)

    :return: a bar chart of the feature importance
    """
    # Fit the model to the training data
    tree_model.fit(features, target)

    # Create a dataframe of feature coefficients and feature names
    feature_importance = pd.DataFrame(
        {"Feature": features.columns, "Importance": tree_model.feature_importances_}
    )
    feature_importance.sort_values(by=["Importance"], inplace=True)

    # Create a bar chart of the feature importances
    return px.bar(
        feature_importance,
        x="Feature",
        y="Importance",
        title=f"Feature Selection by default {tree_model.__class__.__name__}",
    )
