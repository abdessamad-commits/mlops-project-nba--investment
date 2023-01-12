# import libraries
import os

import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from helpers import convert_numerical_columns_to_float

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

@task
def read_data(data_path, target_column):
    """
    this function reads the data

    :param data_path: path of the data
    :param target_column: target column of the data

    :return: DataFrame with the target column
    """
    df = pd.read_csv(data_path)
    df["TARGET_5Yrs"] = df["TARGET_5Yrs"].astype(int)
    features = convert_numerical_columns_to_float(df.drop(target_column, axis=1))
    target_column = df[target_column]
    return pd.concat([features, target_column], axis=1)

@task
def preprocessing_data(df):
    """
    this function preprocess the data
    :param df: DataFrame
    :return: DataFrame
    """
    try:
        # dropping the name column
        df.drop(["Name"], axis=1, inplace=True)
    except Exception:
        pass

    # replacing the missing values with the mean of the column "3P%"
    df["3P%"].fillna(df["3P%"].mean(), inplace=True)

    # define features and target
    X_train = df.drop("TARGET_5Yrs", axis=1)
    y_train = df["TARGET_5Yrs"]

    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # X_train = xgb.DMatrix(X_train, label=y_train)
    # X_val = xgb.DMatrix(X_val, label=y_val)

    return X_train, X_val, y_train, y_val

@task
def converting_to_xgb_matrix(X_train, X_val, y_train, y_val):
    """
    this function converts the data to xgb matrix

    :return: xgb matrix
    """
    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val)

@task
def train_model_hyper_param_tuning(train, valid, y_val):
    """
    this function trains the model using hyperparameter tuning

    :param train: training data
    :param valid: validation data
    :param y_val: target column of the validation data

    :return: best hyperparameters
    """
    # Define the objective function for the hyperparameter optimization
    def objective(params):
        with mlflow.start_run():
            # Set the model and the search space in the run metadata
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("holdout_set", "validation set")
            mlflow.log_params(params)

            # Train the XGBoost model using the specified hyperparameters
            booster = xgb.train(
                params=params,  # Hyperparameters
                dtrain=train,  # Training data
                num_boost_round=1000,  # Train for 1000 rounds
                evals=[
                    (valid, "validation")
                ],  # Evaluate on the validation data at each iteration of training
                early_stopping_rounds=50,  # Stop training if the validation score does not improve for 50 rounds
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

        return {"loss": 1 - f1, "status": STATUS_OK}  # Minimize the negative F1 score

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

    # Perform the hyperparameter optimization using the Tree Parzen Estimator algorithm
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

    return best_result

@task
def train_model_with_best_parameter(train, valid, y_val, best_result):
    """
    this function trains the model with the best hyperparameters

    :param train: training data
    :param valid: validation data
    :param y_val: target column of the validation data

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
            early_stopping_rounds=50,  # Stop training if the validation score does not improve for 50 rounds
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

        # mlflow.log_artifact("mymodel", artifact_path="model")
        # mlflow.framework.log_model(model_object, artifact_path="model")
        mlflow.xgboost.log_model(booster, "model")
        
@flow(task_runner=SequentialTaskRunner())
def main_workflow(data_relative_path):
    """ """
    
    # set tracking uri
    mlflow.set_tracking_uri("http://20.224.70.229:5000/")
    # set experiment name
    mlflow.set_experiment("nba-investment-experiment")
    
    # read data
    df = read_data(data_path + data_relative_path, "TARGET_5Yrs")
    
    # define the features and target
    X_train, X_val, y_train, y_val = preprocessing_data(df).result()
    
    # create the training and validation data for the model in the xgboost format
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    # search for the best hyperparameters
    best_result = train_model_hyper_param_tuning(train, valid, y_val)
    
    # train the xgboost model with the best hyperparameters
    train_model_with_best_parameter(train, valid, y_val, best_result)


main_workflow("/data/nba_logreg_train.csv")
