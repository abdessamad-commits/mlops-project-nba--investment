import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def convert_numerical_columns_to_float(df):
    """
    this function convert all numerical columns of a DataFrame to float

    :param df: DataFrame

    ;return: DataFrame
    """
    numerical_cols = df.select_dtypes(include=["int", "float"]).columns
    df[numerical_cols] = df[numerical_cols].astype(float)
    return df


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


def test_model_by_registry_version(model_name, model_version, X_test, y_test):
    """
    this function tests the model on the test data and prints the evaluation metrics in order to go to production

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


def transition_best_model_version_to_prod(
    X_test, y_test, model_name, metric, mlfow_client
):
    """
    The function transitions the best model version on the model registry to production by testing it on the test data
    :param X_test: test features
    :param y_test: test target
    :param model_name: name of the model
    :param metric: metric to use for model selection
    :param mlfow_client: mlflow client

    :return: None
    """

    def test_model_by_registry_all_versions(X_test, y_test, model_name):
        """
        the function tests all the versions of the model and returns a DataFrame with the evaluation metrics

        :param model_name: name of the model on the registry
        :param X_test: test features
        :param y_test: test target

        :return: dictionary with the evaluation metrics
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
        return res

    testing_metrics = test_model_by_registry_all_versions(
        X_test, y_test, "nba-investment-prediction-model"
    )

    max_version = None
    max_metric = float("-inf")
    for data in testing_metrics:
        if data[metric] > max_metric:
            max_metric = data[metric]
            max_version = data["model_version"]

    mlfow_client.transition_model_version_stage(
        name=model_name, version=max_version, stage="Production"
    )
