import pandas as pd

def convert_numerical_columns_to_float(df):
    """
    this function convert all numerical columns of a DataFrame to float
    
    :param df: DataFrame
    
    ;return: DataFrame
    """
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df

