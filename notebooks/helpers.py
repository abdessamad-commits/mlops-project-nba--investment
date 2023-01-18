import contextlib
import os

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def view_files_in_bucket(client, bucket_name):
    """
    The function will list all the files in a bucket

    :param client: The minio client
    :param bucket_name: The name of the bucket

    :return: None
    """
    objects = client.list_objects(bucket_name)
    for obj in objects:
        print(obj.object_name)


def list_objects_in_minio_bucket(client, bucket_name):
    """
    The function list all the objects in a minio bucket

    :param client: the minio client
    :param bucket_name: the name of the bucket

    :return: list of objects
    """
    try:
        # List all objects in the bucket
        objects = client.list_objects(bucket_name)
        return [obj.object_name for obj in objects]
    except Exception as e:
        print(e)


def read_data_from_minio(client, bucket_name, object_name):
    """
    The function read a csv file from a minio bucket

    :param client: the minio client
    :param bucket_name: the name of the bucket
    :param object_name: the name of the object

    :return: DataFrame
    """
    try:
        # Get the object data
        df = pd.read_csv(client.get_object(bucket_name, object_name))
        df["TARGET_5Yrs"] = df["TARGET_5Yrs"].astype(int)
        features = convert_numerical_columns_to_float(df.drop("TARGET_5Yrs", axis=1))
        target_column = df["TARGET_5Yrs"]
        return pd.concat([features, target_column], axis=1)
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
    This function convert all numerical columns of a DataFrame to float

    :param df: DataFrame

    ;return: DataFrame
    """
    numerical_cols = df.select_dtypes(include=["int", "float"]).columns
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df


def read_data(data_path):
    """
    This function reads the data

    :param data_path: path of the data
    :param target_column: target column of the data

    :return: DataFrame with the target column
    """
    df = pd.read_csv(data_path)
    df["TARGET_5Yrs"] = df["TARGET_5Yrs"].astype(int)
    features = convert_numerical_columns_to_float(df.drop("TARGET_5Yrs", axis=1))
    target_column = df["TARGET_5Yrs"]
    return pd.concat([features, target_column], axis=1)


def preprocessing_data(df):
    """
    This function preprocess the data

    :param df: DataFrame

    :return: DataFrame
    """
    with contextlib.suppress(Exception):
        # dropping the name column
        df.drop(["Name"], axis=1, inplace=True)
    # replacing the missing values with the mean of the column "3P%"
    df["3P%"].fillna(df["3P%"].mean(), inplace=True)
    return df


def test_model_by_run_id(run_id, X_test, y_test):
    """
    This function tests the model based on it's run id on the test data and prints the evaluation metrics 

    :param name: name of the model
    :param stage: stage of the model
    :param X_test: test features
    :param y_test: test target

    :return: dictionary with the evaluation metrics
    """
    with mlflow.start_run():

        mlflow.set_tag("holdout_set", "testing set")

        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        y_test_pred = loaded_model.predict(X_test).round()

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_test_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_test_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_test_pred))
        mlflow.set_tag("run_id", run_id)

        return {
            "run_id": run_id,
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
        }


def test_model_by_registry_version(model_name, model_version, X_test, y_test):
    """
    This function tests the models that have been promoted to the mlflow registry on the test data and prints the evaluation metrics in order to go to productionn,the model version is the version of the model in the registry

    :param model_name: name of the model
    :param model_version: version of the model
    :param X_test: test features
    :param y_test: test target

    :return: dictionary with the evaluation metrics
    """
    with mlflow.start_run():

        mlflow.set_tag("holdout_set", "testing set")

        loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        y_test_pred = loaded_model.predict(X_test).round()

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_test_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_test_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_test_pred))
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_version", model_version)

        return {
            "model_name": model_name,
            "model_version": model_version,
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
        }


def test_model_by_registry_all_versions(model_name, X_test, y_test, sort_by="f1"):
    """
    The function tests all the versions of the model present in the mflow registry and returns a DataFrame with the evaluation metrics

    :param model_name: name of the model
    :param X_test: test features
    :param y_test: test target
    :param sort_by: column to sort the DataFrame by

    :return: DataFrame with the evaluation metrics
    """
    i = 0
    res = []
    while True:
        try:
            res.append(
                test_model_by_registry_version(model_name, i + 1, X_test, y_test)
            )
        except Exception:
            break
        i += 1
    return pd.DataFrame(res).sort_values(by=sort_by, ascending=False)
