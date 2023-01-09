import pandas as pd
from scipy.stats import pearsonr

def read_dataset(file):
    try:
        df = pd.read_csv(file)
        df.drop_duplicates(inplace=True)
    except Exception:
        df = pd.read_csv("/streamlit_app/src/data/nba_logreg.csv")

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
    if not df[col1].dtype.kind in 'iufc' or not df[col2].dtype.kind in 'iufc':
        raise ValueError("Both columns must be numeric")
    
    # Calculate Pearson correlation
    corr, _ = pearsonr(df[col1], df[col2])
    return corr    