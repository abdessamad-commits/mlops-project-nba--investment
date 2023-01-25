import contextlib
import logging

import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from minio import Minio
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split


class Pipeline:
    def __init__(
        self,
        minio_bucket_name="nba-investment-data",
        minio_object_name=None,
        mlflow_tracking_uri="http://20.224.70.229:5000/",
        mlflow_experiment_name="nba-investment-experiment",
    ):

        """
        This function initialize the pipeline class by setting the tracking uri and the experiment name and initializing the mlflow client and the minio client and reading the data from the minio bucket
        
        :param minio_object_name: the name of the object in the minio bucket 
        ;param minio_bucket_name: the name of the bucket
        :param mlflow_tracking_uri: the tracking uri
        :param mlflow_experiment_name: the experiment name

        :return: None
        """

        self.minio_object_name = minio_object_name
        self.minio_bucket_name = minio_bucket_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        # set tracking uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # set experiment name
        mlflow.set_experiment(self.mlflow_experiment_name)
        # initialize mlflow client
        self.mlflow_client = mlflow.tracking.MlflowClient()

        # initialize minio client
        self.minio_client = Minio(
            "20.224.70.229:9000",
            access_key="abdessamadbaahmed",
            secret_key="baahmedabdessamad",
            secure=False,
        )

        # read data from minio bucket if the object name is not None
        if self.minio_object_name != None:
            self.data = self.read_data_from_minio(
                self.minio_bucket_name, self.minio_object_name
            )


    def __convert_numerical_columns_to_float(self, df):
        """
        This function convert all numerical columns of a DataFrame to float

        :param df: DataFrame

        ;return: DataFrame
        """
        numerical_cols = df.select_dtypes(include=["int", "float"]).columns
        df[numerical_cols] = df[numerical_cols].astype(float)
        return df

    def read_data_from_minio(self, bucket_name, object_name):
        """
        The function read a csv file from a minio bucket

        :param bucket_name: the name of the bucket
        :param object_name: the name of the object

        :return: DataFrame
        """
        try:
            # Get the object data
            df = pd.read_csv(self.minio_client.get_object(bucket_name, object_name))
            df["TARGET_5Yrs"] = df["TARGET_5Yrs"].astype(int)
            features = self.__convert_numerical_columns_to_float(
                df.drop("TARGET_5Yrs", axis=1)
            )
            target_column = df["TARGET_5Yrs"]
            return pd.concat([features, target_column], axis=1)
        except Exception as e:
            logging.exception(e)

    def preprocessing_data(self, df):
        """
        This function preprocess the data
        
        :param df: DataFrame to preprocess 
        
        :return: preprocessed DataFrame
        """

        with contextlib.suppress(Exception):
            # dropping the name column if it exists
            df.drop(["Name"], axis=1, inplace=True)

        # replacing the missing values with the mean of the column "3P%"
        df["3P%"].fillna(df["3P%"].mean(), inplace=True)

        return df

    def search_best_hyper_params(self, train, valid, y_val, metric):
        """
        This function looks for the best hyperparameters for the XGBoost model while logging the results to MLFlow and maximizing a chosen metric on the validation set

        :param train: training data
        :param valid: validation data
        :param y_val: target column of the validation data
        :param metric: the metric to maximize

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
        self, train, best_result, registered_model_name
    ):
        """
        This function trains xgboost model with the best hyperparameters found in the previous step and logs the model to MLFlow model registry

        :param train: training data
        :param best_result: best hyperparameters found in the hyperparameter tuning step
        :param registered_model_name: name of the registered model in the MLFlow model registry

        :return: None
        """
        with mlflow.start_run():
            # Log the best hyperparameters to MLFlow
            mlflow.log_params(best_result)
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("holdout_set", "none")

            # Train the XGBoost model using the specified hyperparameters
            booster = xgb.train(
                params=best_result,  # Hyperparameters
                dtrain=train,  # Training data
                num_boost_round=800,  # Train for 1000 rounds
            )

            # Log the model to MLFlow registry
            mlflow.xgboost.log_model(
                booster,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )

    def test_model_by_registry_version(self, model_name, model_version, X_test, y_test):
        """
        This function tests the model on the test data and prints the evaluation metrics in order to go to production

        :param model_name: name of the model in the MLFlow model registry
        :param model_version: version of the model in the MLFlow model registry
        :param X_test: test features data
        :param y_test: test target data

        :return: dictionary with the evaluation metrics
        """
        with mlflow.start_run():

            # Specify the model and the holdout set in the run metadata
            mlflow.set_tag("holdout_set", "testing set")

            # Load the model from the MLFlow model registry
            loaded_model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{model_version}"
            )

            # Make predictions on the test data
            y_test_pred = loaded_model.predict(X_test).round()

            # Calculate the evaluation scores
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)

            # Log the evaluation scores to MLFlow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", model_version)

            return {
                "model_name": model_name,
                "model_version": model_version,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    def test_model_by_registry_all_versions(self, X_test, y_test, model_name):
        """
        The function tests all the versions of the model and returns a DataFrame with the evaluation metrics

        :param X_test: test features
        :param y_test: test target
        :param model_name: name of the model on the registry to test all the versions of it

        :return: None
        """
        i = 0
        res = []
        while True:
            try:
                res.append(
                    self.test_model_by_registry_version(
                        model_name, i + 1, X_test, y_test
                    )
                )
            except Exception:
                break
            i += 1
        return res

    def transition_best_model_version_to_prod(
        self, X_test, y_test, model_name, metric,
    ):
        """
        The function transitions the best model version on the model registry to production by testing it on the test data
        
        :param X_test: test features
        :param y_test: test target
        :param model_name: name of the model on the registry to test all the versions of it
        
        :param metric: metric to use for model selection

        :return: None
        """
        # get the model name of the registry
        model_name = model_name
        # get the metric to be used for the testing
        metric = metric

        # test all the versions of the model
        testing_metrics = self.test_model_by_registry_all_versions(
            X_test, y_test, model_name
        )

        # select the best model version based on the metric
        max_version = None
        max_metric = float("-inf")
        for data in testing_metrics:
            if data[metric] > max_metric:
                max_metric = data[metric]
                max_version = data["model_version"]

        # transition the best model version to production
        self.mlflow_client.transition_model_version_stage(
            name=model_name, version=max_version, stage="Production"
        )


if __name__ == "__main__":
    # initialize the pipeline
    pipeline = Pipeline(minio_object_name="nba_logreg_raw.csv")
    # get the raw data
    raw_data = pipeline.data
    # preprocess the data
    processed_data = pipeline.preprocessing_data(raw_data)
    # split the data into train, validation and test
    train_data, test_data = train_test_split(
        processed_data,
        test_size=0.2,
        random_state=42,
        stratify=processed_data["TARGET_5Yrs"],
    )
    # define the features and target
    X_train = train_data.drop("TARGET_5Yrs", axis=1)
    y_train = train_data["TARGET_5Yrs"]
    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    # convert the data to xgboost format
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    # hyperparameter tuning
    best_result = pipeline.search_best_hyper_params(train, valid, y_val, "f1")
    # train the model with the best hyperparameters
    pipeline.train_model_with_best_parameter(
        train, best_result, "nba-investment-model-f1"
    )
    # define the test features and target
    X_test = test_data.drop("TARGET_5Yrs", axis=1)
    y_test = test_data["TARGET_5Yrs"]
    # transition the best model version to production
    pipeline.transition_best_model_version_to_prod(
        X_test, y_test, "nba-investment-model-f1", "f1"
    )
