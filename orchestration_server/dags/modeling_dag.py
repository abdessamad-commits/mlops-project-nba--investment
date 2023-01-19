# import libraries
import contextlib
import os
from datetime import datetime, timedelta

import mlflow
import pandas as pd
import xgboost as xgb
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from helpers import *
from helpers import (
    transition_best_model_version_to_prod,
)
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from minio import Minio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def read_data_preprocess_data(**kwargs):
    """
    The function read the data from minio server, preprocess it and save it to minio server so that it can be used by the following tasks in the DAG

    :param kwargs "bucket_name": the name of the bucket
    :param kwargs "object_name": the name of the object

    :return: None
    """

    # create a minio client
    minio_client = Minio(
        endpoint="20.224.70.229:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # read the data from minio server
    data = minio_client.get_object(kwargs["bucket_name"], kwargs["object_name"])
    df = pd.read_csv(data)

    # dropping the name column
    with contextlib.suppress(Exception):
        df.drop(["Name"], axis=1, inplace=True)

    # replacing the missing values with the mean of the column "3P%"
    df["3P%"].fillna(df["3P%"].mean(), inplace=True)

    # saving the preprocessed data
    df.to_csv("nba_logreg_processed.csv", index=False)
    print(f"these are the actual files in the local storage{str(os.listdir())}")

    # upload the preprocessed data to minio server
    minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_preprocessed.csv",
        file_path="nba_logreg_processed.csv",
    )

    # remove the preprocessed data from the local storage
    # os.remove("nba_logreg_processed.csv")

    # splitting the data into train and test
    train_set, test_set = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["TARGET_5Yrs"]
    )

    # save the train and test sets
    train_set.to_csv("nba_logreg_processed_train.csv", index=False)
    test_set.to_csv("nba_logreg_processed_test.csv", index=False)

    # upload the train to minio server
    minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_processed_train.csv",
        file_path="nba_logreg_processed_train.csv",
    )

    # upload the test to minio server
    minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_processed_test.csv",
        file_path="nba_logreg_processed_test.csv",
    )

    # remove the train and test sets from the local storage
    # os.remove("nba_logreg_processed_train.csv")
    # os.remove("nba_logreg_processed_test.csv")


def search_best_hyper_params(**kwargs):
    """
    This function trains the model using hyperparameter tuning

    :param kwargs "bucket_name": the name of the bucket
    :param kwargs "object_name": the name of the object
    :param kwargs "metric": the metric to optimize

    :return: best hyperparameters, it's stored in xcom
    """

    # set tracking uri
    mlflow.set_tracking_uri("http://20.224.70.229:5000/")
    # set experiment name
    mlflow.set_experiment("nba-investment-experiment")
    # initialize mlflow client
    mlflow_client = mlflow.tracking.MlflowClient()

    # create a minio client
    minio_client = Minio(
        endpoint="20.224.70.229:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # read the data from minio server
    train = pd.read_csv(
        minio_client.get_object(kwargs["bucket_name"], kwargs["object_name"])
    )

    # define features and target
    X_train = train.drop("TARGET_5Yrs", axis=1)
    y_train = train["TARGET_5Yrs"]

    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # create the training and validation data for the model in the xgboost
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # define the objective function for the hyperparameter optimization
    def objective(params, metric=kwargs["metric"]):
        with mlflow.start_run():
            # Set the model and the search space in the run metadata
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("holdout_set", "validation set")
            mlflow.log_params(params)

            # Train the XGBoost model using the specified hyperparameters
            booster = xgb.train(
                params=params,  # Hyperparameters
                dtrain=train,  # Training data
                num_boost_round=50,  # Train for 1000 rounds
                evals=[
                    (valid, "validation")
                ],  # Evaluate on the validation data at each iteration of training
                early_stopping_rounds=50,
                # Stop training if the validation score does not improve for 50
                # rounds
            )

            # Make predictions on the validation data
            y_pred = booster.predict(valid).round()

            # Calculate the evaluation scores
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            # define the metric to optimize
            if metric in ["f1_score", "f1"]:
                metric = f1
            elif metric == "accuracy":
                metric = accuracy
            elif metric == "precision":
                metric = precision
            elif metric == "recall":
                metric = recall

            # Log the evaluation scores to MLFlow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

        return {
            "loss": 1 - metric,
            "status": STATUS_OK,
        }  # Minimize the negative F1 score

    # Define the search space for the hyperparameters
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 200, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "binary:logistic",
        "seed": 42,
    }

    # Perform the hyperparameter optimization using the Tree Parzen Estimator
    # algorithm
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )

    best_result["max_depth"] = int(best_result["max_depth"])
    best_result["objective"] = "binary:logistic"
    best_result["seed"] = 42

    # upload the best hyperparameters to xcom to be used in the next task
    kwargs["ti"].xcom_push(key="best_hyper_params", value=best_result)


def train_model_with_best_parameter(**kwargs):
    """
    This function trains the model with the best hyperparameters

    :param kwargs "bucket_name": the name of the bucket
    :param kwargs "object_name": the name of the object
    :param kwargs "model_name": the name of the model on the mlflow registry

    :return: None
    """

    # set tracking uri
    mlflow.set_tracking_uri("http://20.224.70.229:5000/")
    # set experiment name
    mlflow.set_experiment("nba-investment-experiment")
    # initialize mlflow client
    mlflow_client = mlflow.tracking.MlflowClient()

    # create a minio client
    minio_client = Minio(
        endpoint="20.224.70.229:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # read the training data from minio storage
    train = pd.read_csv(
        minio_client.get_object(kwargs["bucket_name"], kwargs["object_name"])
    )

    # define features and target
    X_train = train.drop("TARGET_5Yrs", axis=1)
    y_train = train["TARGET_5Yrs"]

    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # create the training and validation data for the model in the xgboost
    # format
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    with mlflow.start_run():
        # Log the best hyperparameters to MLFlow
        best_result = kwargs["ti"].xcom_pull(
            key="best_hyper_params", task_ids="search_best_hyper_params_task"
        )
        # these are the best hyperparameters
        print("this is the best hyperparameters: ", best_result)
        mlflow.log_params(best_result)
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("holdout_set", "validation set")

        # Train the XGBoost model using the specified hyperparameters
        booster = xgb.train(
            params=best_result,  # Hyperparameters
            dtrain=train,  # Training data
            num_boost_round=100,  # Train for 1000 rounds
            evals=[
                (valid, "validation")
            ],  # Evaluate on the validation data at each iteration of training
            early_stopping_rounds=50,
            # Stop training if the validation score does not improve for 50
            # rounds
        )

        # Make predictions on the validation data
        y_pred = booster.predict(valid).round()
        print("this is the y_pred: ", y_pred)

        # Calculate the evaluation scores
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Log the evaluation scores to MLFlow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model to MLFlow registry
        with contextlib.suppress(Exception):
            mlflow.xgboost.log_model(
                booster,
                artifact_path="model",
                registered_model_name=kwargs["model_name"],
            )


def transition_best_model_version_to_prod(**kwargs):
    """
    The function transitions the best model version on the model registry to production by testing it on the test data

    :param kwargs "bucket_name": the name of the bucket
    :param kwargs "object_name": the name of the object
    :param kwargs "model_name": the name of the model on the mlflow registry

    :return: None
    """

    # set tracking uri
    mlflow.set_tracking_uri("http://20.224.70.229:5000/")
    # set experiment name
    mlflow.set_experiment("nba-investment-experiment")
    # initialize mlflow client
    mlflow_client = mlflow.tracking.MlflowClient()

    # create a minio client
    minio_client = Minio(
        endpoint="20.224.70.229:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # read the training data from minio storage
    test = pd.read_csv(
        minio_client.get_object(kwargs["bucket_name"], kwargs["object_name"])
    )

    # define features and target
    X_test = test.drop("TARGET_5Yrs", axis=1)
    y_test = test["TARGET_5Yrs"]

    # get the model name of the registry
    model_name = kwargs["model_name"]
    # get the metric to be used for the testing
    metric = kwargs["metric"]

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

    testing_metrics = test_model_by_registry_all_versions(X_test, y_test, model_name)

    max_version = None
    max_metric = float("-inf")
    for data in testing_metrics:
        if data[metric] > max_metric:
            max_metric = data[metric]
            max_version = data["model_version"]

    # transition the best model version on the registry to production
    with contextlib.suppress(Exception):
        mlflow_client.transition_model_version_stage(
            name=model_name, version=max_version, stage="Production"
        )


# define the default arguments for the DAG
default_args = {
    "owner": "abdessamad",  # the owner of the DAG
    "start_date": datetime.now(),  # the start date of the DAG
    "depends_on_past": True,  # the DAG depends on the past
    "retries": 2,  # the number of retries
    "retry_delay": timedelta(hours=1),  # the delay between retries
    "catchup": False,  # the DAG does not catch up with the past
}


# define the DAG with the default arguments
with DAG(
    "ml-workflow", default_args=default_args, schedule_interval=timedelta(hours=1)
) as dag:

    # define the tasks of the DAG

    # task to read the data from minio storage and preprocess it and save it
    # to minio storage again
    read_data_preprocess_data_task = PythonOperator(
        task_id="read_data_preprocess_data_task",
        python_callable=read_data_preprocess_data,
        provide_context=True,
        op_kwargs={
            "bucket_name": "nba-investment-data",
            "object_name": "nba_logreg_raw.csv",
        },
        trigger_rule="all_success"
        # xcom_push=True,
    )

    # task to search for the best hyper parameters
    search_best_hyper_params_task = PythonOperator(
        task_id="search_best_hyper_params_task",
        python_callable=search_best_hyper_params,
        provide_context=True,
        op_kwargs={
            "bucket_name": "nba-investment-data",
            "object_name": "nba_logreg_processed_train.csv",
            "metric": "f1_score",
        },
        trigger_rule="all_success",
    )

    # task to train the model with the best hyper parameters
    train_model_with_best_parameter_task = PythonOperator(
        task_id="train_model_with_best_parameter_task",
        python_callable=train_model_with_best_parameter,
        provide_context=True,
        op_kwargs={
            "bucket_name": "nba-investment-data",
            "object_name": "nba_logreg_processed_train.csv",
            "model_name": '"nba-investment-model-f1"',
        },
        trigger_rule="all_success",
    )

    # task to test the models' versions on the defined regisrry with the best
    # performance on the test data and transition it the best model to
    # production
    transition_best_model_version_to_prod_task = PythonOperator(
        task_id="transition_best_model_version_to_prod_task",
        python_callable=transition_best_model_version_to_prod,
        provide_context=True,
        op_kwargs={
            "bucket_name": "nba-investment-data",
            "object_name": "nba_logreg_processed_test.csv",
            "model_name": '"nba-investment-model-f1"',
            "metric": "f1_score",
        },
        trigger_rule="all_success",
    )

    (
        read_data_preprocess_data_task
        >> search_best_hyper_params_task
        >> train_model_with_best_parameter_task
        >> transition_best_model_version_to_prod_task
    )
