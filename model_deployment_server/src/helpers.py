import os

import pandas as pd
from minio import Minio


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


def upload_file_to_minio(client, file_path, bucket_name, object_name):
    """
    The function upload a file to a minio bucket
    
    :param client: the minio client
    :param file_path: the path of the file to upload
    :param bucket_name: the name of the bucket
    :param object_name: the name of the object
    
    :return: True if the file is uploaded successfully
    """
    file_size = os.stat(file_path).st_size
    with open(file_path, "rb") as file:
        client.put_object(bucket_name, object_name, file, file_size)
    return True


def convert_numerical_columns_to_float(df):
    """
    this function convert all numerical columns of a DataFrame to float

    :param df: DataFrame

    ;return: DataFrame
    """
    numerical_cols = df.select_dtypes(include=["int", "float"]).columns
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df


def preprocessing_data(df):
    """
    this function preprocess the data
    :param df: DataFrame
    :return: DataFrame
    """
    df = convert_numerical_columns_to_float(df)

    try:
        # dropping the name column
        df.drop(["Name"], axis=1, inplace=True)
    except Exception:
        pass

    return df
