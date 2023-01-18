import contextlib
import os
import pandas as pd
import mlflow
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import Union

# Create a class which inherits from the BaseModel class of pydantic and define the data types of the attributes
class Player(BaseModel):
    gp: Union[int, float]
    min: Union[int, float]
    pts: Union[int, float]
    fgm: Union[int, float]
    fga: Union[int, float]
    fgpercent: Union[int, float]
    treepmade: Union[int, float]
    treepa: Union[int, float]
    treeppercent: Union[int, float]
    ftm: Union[int, float]
    fta: Union[int, float]
    ftpercent: Union[int, float]
    oreb: Union[int, float]
    dreb: Union[int, float]
    reb: Union[int, float]
    ast: Union[int, float]
    stl: Union[int, float]
    blk: Union[int, float]
    tov: Union[int, float]
    model_name: str


def convert_numerical_columns_to_float(df):
    """
    this function convert all numerical columns of a DataFrame to float

    :param df: DataFrame

    ;return: DataFrame
    """
    numerical_cols = df.select_dtypes(include=["int", "float"]).columns
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df


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


def preprocessing_data(df):
    """
    this function preprocess the data
    :param df: DataFrame
    :return: DataFrame
    """
    df = convert_numerical_columns_to_float(df)

    with contextlib.suppress(Exception):
        # dropping the name column
        df.drop(["Name"], axis=1, inplace=True)
    return df


def import_model_from_registry(model_name, stage="Production"):
    """
    This function import the model from the registry

    :param model_name: name of the model to import from the registry
    :param stage: stage of the model

    :return: the model as a pyfunc model
    """
    return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")


def save_to_elasticsearch(player, prediction, client=None):
    """
    This function save the prediction to elasticsearch

    :param player: the player data as a dictionary
    :param prediction: the prediction as a dictionary
    :param client: the elasticsearch client (optional)

    :return: None
    """
    # create the elasticsearch client
    if client is None:
        elasticsearch_client = Elasticsearch(
            "http://elasticsearch:9200",
        )

    # verify if the index exists
    if not elasticsearch_client.indices.exists(index="nba-players-investment"):
        elasticsearch_client.indices.create(index="nba-players-investment", ignore=400)

    # create a new document
    elasticsearch_client.index(
        index="nba-players-investment",
        body={
            "prediction": prediction,
            "player": player,
            "timestamp": pd.Timestamp.now(),
        },
    )
