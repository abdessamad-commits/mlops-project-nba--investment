# import libraries
import contextlib
import mlflow
import pandas as pd
import xgboost as xgb
from helpers import (
    convert_numerical_columns_to_float,
    transition_best_model_version_to_prod,
)
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from minio import Minio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging


def read_data(data_path, target_column):
    """
    This function reads the data

    :param data_path: path of the data
    :param target_column: target column of the data

    :return: DataFrame with the target column
    """
    df = pd.read_csv(data_path)
    df["TARGET_5Yrs"] = df["TARGET_5Yrs"].astype(int)
    features = convert_numerical_columns_to_float(df.drop(target_column, axis=1))
    target_column = df[target_column]
    return pd.concat([features, target_column], axis=1)


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
        logging.exception(e)


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

    # define features and target
    X_train = df.drop("TARGET_5Yrs", axis=1)
    y_train = df["TARGET_5Yrs"]

    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    return X_train, X_val, y_train, y_val


def converting_to_xgb_matrix(X_train, X_val, y_train, y_val):
    """
    This function converts the data to xgb matrix

    :param X_train: training features
    :param X_val: validation features
    :param y_train: training target
    :param y_val: validation target

    :return: xgb matrix
    """
    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val)


def search_best_hyper_params(train, valid, y_val, metric):
    """
    This function looks for the best hyperparameters for the XGBoost model while logging the results to MLFlow and maximizing a chosen metric on the validation set

    :param train: training data
    :param valid: validation data
    :param y_val: target column of the validation data

    :return: best hyperparameters
    """
    # Define the objective function for the hyperparameter optimization
    def objective(params, metric=metric):
        with mlflow.start_run():
            # Set the model and the search space in the run metadata
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("holdout_set", "validation set")
            mlflow.log_params(params)

            # Train the XGBoost model using the specified hyperparameters
            booster = xgb.train(
                params=params,  # Hyperparameters
                dtrain=train,  # Training data
                num_boost_round=800,  # Train for 1000 rounds
                evals=[
                    (valid, "validation")
                ],  # Evaluate on the validation data at each iteration of training
                early_stopping_rounds=40,
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

            # defining the metric to maximize
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
        }

    # Define the search space for the hyperparameters
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 300, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "binary:logistic",
        "seed": 42,
    }

    # Perform the hyperparameter optimization
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )

    # Convert the best hyperparameters to the correct data type with the
    # correct format
    best_result["max_depth"] = int(best_result["max_depth"])
    best_result["objective"] = "binary:logistic"
    best_result["seed"] = 42

    return best_result


def train_model_with_best_parameter(
    train, valid, y_val, best_result, registered_model_name
):
    """
    This function trains xgboost model with the best hyperparameters found in the previous step and logs the model to MLFlow model registry

    :param train: training data
    :param valid: validation data
    :param y_val: target column of the validation data
    :param best_result: best hyperparameters found in the hyperparameter tuning step
    :param registered_model_name: name of the registered model in the MLFlow model registry

    :return: None
    """
    with mlflow.start_run():
        # Log the best hyperparameters to MLFlow
        mlflow.log_params(best_result)
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("holdout_set", "validation set")

        # Train the XGBoost model using the specified hyperparameters
        booster = xgb.train(
            params=best_result,  # Hyperparameters
            dtrain=train,  # Training data
            num_boost_round=1000,  # Train for 1000 rounds
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

        # Log the evaluation scores to MLFlow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model to MLFlow registry
        mlflow.xgboost.log_model(
            booster, artifact_path="model", registered_model_name=registered_model_name
        )


def main_workflow():

    # set tracking uri
    mlflow.set_tracking_uri("http://20.224.70.229:5000/")
    # set experiment name
    mlflow.set_experiment("nba-investment-experiment")
    # initialize mlflow client
    mlflow_client = mlflow.tracking.MlflowClient()

    # initialize minio client
    minio_client = Minio(
        "20.224.70.229:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # reading train, valid and test data from minio storage
    df_test = read_data_from_minio(
        minio_client, "nba-investment-data", "nba_logreg_processed_test.csv"
    )
    df_train_val = read_data_from_minio(
        minio_client, "nba-investment-data", "nba_logreg_processed_train.csv"
    )

    # define the features and target for train and validation data for test
    # data
    X_test = df_test.drop("TARGET_5Yrs", axis=1)
    y_test = df_test["TARGET_5Yrs"]

    # define the features and target for train and validation data
    X_train, X_val, y_train, y_val = preprocessing_data(df_train_val)

    # create the training and validation data for the model in the xgboost
    # format
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    # search for the best hyperparameters that maximize the accuracy score
    best_result = search_best_hyper_params(train, valid, y_val, metric="accuracy")
    # train the xgboost model with the best hyperparameters
    train_model_with_best_parameter(
        train,
        valid,
        y_val,
        best_result,
        registered_model_name="nba-investment-model-accuracy",
    )

    # search for the best hyperparameters that maximize the f1 score
    best_result = search_best_hyper_params(train, valid, y_val, metric="f1")
    # train the xgboost model with the best hyperparameters
    train_model_with_best_parameter(
        train,
        valid,
        y_val,
        best_result,
        registered_model_name="nba-investment-model-f1",
    )
    # transition the best version of the model to production
    transition_best_model_version_to_prod(
        X_test, y_test, "nba-investment-model-f1", "f1", mlflow_client
    )

    # search for the best hyperparameters that maximize the recall score
    best_result = search_best_hyper_params(train, valid, y_val, metric="recall")
    # train the xgboost model with the best hyperparameters
    train_model_with_best_parameter(
        train,
        valid,
        y_val,
        best_result,
        registered_model_name="nba-investment-model-recall",
    )
    # transition the best version of the model to production
    transition_best_model_version_to_prod(
        X_test, y_test, "nba-investment-model-recall", "recall", mlflow_client
    )

    # search for the best hyperparameters that maximize the precision score
    best_result = search_best_hyper_params(train, valid, y_val, metric="precision")
    # train the xgboost model with the best hyperparameters
    train_model_with_best_parameter(
        train,
        valid,
        y_val,
        best_result,
        registered_model_name="nba-investment-model-precision",
    )
    # transition the best version of the model to production
    transition_best_model_version_to_prod(
        X_test, y_test, "nba-investment-model-precision", "precision", mlflow_client
    )


if __name__ == "__main__":
    main_workflow()
