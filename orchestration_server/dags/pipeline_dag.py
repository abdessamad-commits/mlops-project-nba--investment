import os
from datetime import datetime, timedelta

import xgboost as xgb
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.model_selection import train_test_split

from pipeline import Pipeline


def read_raw_data_preprocess_raw_data():
    """
    This function reads the raw data from the minio server, preprocess, 
    split the data into train and test and upload the train and test sets to the minio server
    """
    # initialize the pipeline, the pipeline is an encapsulation of the minio
    # client, the mlfow client and the data retrieved from the minio client
    pipeline = Pipeline(minio_object_name="nba_logreg_raw.csv")
    # get the raw data from the minio client
    raw_data = pipeline.data
    # preprocess the raw data
    processed_data = pipeline.preprocessing_data(raw_data)
    # save the processed data to the minio client
    processed_data.to_csv("nba_logreg_processed.csv", index=False)
    # upload the preprocessed data to minio server
    pipeline.minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_preprocessed.csv",
        file_path="nba_logreg_processed.csv",
    )
    # splitting the data into train and test
    train_set, test_set = train_test_split(
        processed_data,
        test_size=0.2,
        random_state=42,
        stratify=processed_data["TARGET_5Yrs"],
    )
    # save the train and test sets
    train_set.to_csv("nba_logreg_processed_train.csv", index=False)
    test_set.to_csv("nba_logreg_processed_test.csv", index=False)
    # upload the train to minio server
    pipeline.minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_processed_train.csv",
        file_path="nba_logreg_processed_train.csv",
    )
    # upload the test to minio server
    pipeline.minio_client.fput_object(
        bucket_name="nba-investment-data",
        object_name="nba_logreg_processed_test.csv",
        file_path="nba_logreg_processed_test.csv",
    )

    # delete the local files created
    os.remove("nba_logreg_processed.csv")
    os.remove("nba_logreg_processed_train.csv")
    os.remove("nba_logreg_processed_test.csv")


def search_best_hyperparameters(**kwargs):
    """
    This function search for the best hyperparameters for the model using hyperopt
    """
    metric = kwargs["metric"]
    # initialize the pipeline, the pipeline is an encapsulation of the minio
    # client, the mlfow client and the data retrieved from the minio client
    pipeline = Pipeline(minio_object_name="nba_logreg_processed_train.csv")
    # get the train data from the minio client
    train_data = pipeline.data
    # define features and target
    X_train = train_data.drop("TARGET_5Yrs", axis=1)
    y_train = train_data["TARGET_5Yrs"]
    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    # convert the data to DMatrix
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    # search for the best hyperparameters
    best_result = pipeline.search_best_hyper_params(train, valid, y_val, metric)
    # saving the best hyperparameters to the xcom
    kwargs["ti"].xcom_push(key="best_hyperparameters", value=best_result)


def train_model_with_best_hyperparameters(**kwargs):
    """
    This function trains the model with the best hyperparameters found
    
    :param registered_model_name: the name of the registered model in mlflow
    
    :return: None
    """
    # get the best hyperparameters from the xcom
    best_hyperparameters = kwargs["ti"].xcom_pull(
        key="best_hyperparameters", task_ids=kwargs["task_id"]
    )
    # initialize the pipeline, the pipeline is an encapsulation of the minio
    # client, the mlfow client and the data retrieved from the minio client
    pipeline = Pipeline(minio_object_name="nba_logreg_processed_train.csv")
    # get the train data from the minio client
    train_data = pipeline.data
    # define features and target
    X_train = train_data.drop("TARGET_5Yrs", axis=1)
    y_train = train_data["TARGET_5Yrs"]
    # convert the data to DMatrix
    train = xgb.DMatrix(X_train, label=y_train)
    # train the model with the best hyperparameters
    pipeline.train_model_with_best_parameter(
        train, best_hyperparameters, kwargs["registered_model_name"]
    )


def transition_best_model_version_to_prod(**kwargs):
    """
    This function transition the best model version to production
    
    :param registered_model_name: the name of the registered model in mlflow
    :param metric: the metric used to search for the best hyperparameters
    
    :return: None
    """
    # initialize the pipeline, the pipeline is an encapsulation of the minio
    # client, the mlfow client and the data retrieved from the minio client
    pipeline = Pipeline(minio_object_name="nba_logreg_processed_test.csv")
    # get the test data from the minio client
    test_data = pipeline.data
    # define features and target
    X_test = test_data.drop("TARGET_5Yrs", axis=1)
    y_test = test_data["TARGET_5Yrs"]
    # transition the best model version to production
    pipeline.transition_best_model_version_to_prod(
        X_test, y_test, kwargs["registered_model_name"], kwargs["metric"]
    )


# define the default arguments for the DAG
default_args = {
    "owner": "abdessamad",  # the owner of the DAG
    "start_date": datetime.now(),  # the start date of the DAG
    "depends_on_past": True,  # the DAG depends on the past
    "retries": 3,  # the number of retries
    "retry_delay": timedelta(hours=1),  # the delay between retries
    "catchup": False,  # the DAG does not catch up with the past
    "schedule_interval": "@daily",  # the schedule interval of the DAG
    "email_on_failure": True,  # send an email on failure
    "email": "baahmedabdessamad@gmail.com",
}


# define the DAG with the default arguments
with DAG("better_workflow_3", default_args=default_args) as dag:

    # define the tasks of the DAG

    # task to read the data from minio storage and preprocess it and save it
    # to minio storage again
    read_raw_data_preprocess_raw_data_task = PythonOperator(
        task_id="read_data_preprocess_data_task",
        python_callable=read_raw_data_preprocess_raw_data,
        provide_context=True,
        trigger_rule="all_success",
    )

    # task to search for the best hyperparameters for the model using hyperopt, maximizing the f1 score
    search_best_hyperparameters_f1_task = PythonOperator(
        task_id="search_best_hyperparameters_f1_task",
        python_callable=search_best_hyperparameters,
        provide_context=True,
        op_kwargs={"metric": "f1",},
        trigger_rule="all_success",
    )
    # task to train the model with the best hyperparameters found using the f1 score
    train_model_with_best_hyperparameters_f1_task = PythonOperator(
        task_id="train_model_with_best_hyperparameters_f1_task",
        python_callable=train_model_with_best_hyperparameters,
        provide_context=True,
        op_kwargs={
            "registered_model_name": "nba-investment-model-f1",
            "task_id": "search_best_hyperparameters_f1_task",
        },
        trigger_rule="all_success",
    )
    # task to transition the best model version to production
    transition_best_model_version_to_prod_f1_task = PythonOperator(
        task_id="transition_best_model_version_to_prod_f1_task",
        python_callable=transition_best_model_version_to_prod,
        provide_context=True,
        op_kwargs={"registered_model_name": "nba-investment-model-f1", "metric": "f1"},
        trigger_rule="all_success",
    )

    # task to search for the best hyperparameters for the model using hyperopt, maximizing the recall score
    search_best_hyperparameters_recall_task = PythonOperator(
        task_id="search_best_hyperparameters_recall_task",
        python_callable=search_best_hyperparameters,
        provide_context=True,
        op_kwargs={"metric": "recall",},
        trigger_rule="all_success",
    )
    # task to train the model with the best hyperparameters found using the recall score
    train_model_with_best_hyperparameters_recall_task = PythonOperator(
        task_id="train_model_with_best_hyperparameters_recall_task",
        python_callable=train_model_with_best_hyperparameters,
        provide_context=True,
        op_kwargs={
            "registered_model_name": "nba-investment-model-recall",
            "task_id": "search_best_hyperparameters_recall_task",
        },
        trigger_rule="all_success",
    )
    # task to transition the best model version to production, the best model version is the model version with the best metric
    # if the current model in production is not the best model version then
    # the best model version is transitioned to production else nothing is
    transition_best_model_version_to_prod_recall_task = PythonOperator(
        task_id="transition_best_model_version_to_prod_recall_task",
        python_callable=transition_best_model_version_to_prod,
        provide_context=True,
        op_kwargs={
            "registered_model_name": "nba-investment-model-recall",
            "metric": "recall",
        },
        trigger_rule="all_success",
    )

    # task to search for the best hyperparameters for the model using hyperopt, maximizing the precision score
    search_best_hyperparameters_precision_task = PythonOperator(
        task_id="search_best_hyperparameters_precision_task",
        python_callable=search_best_hyperparameters,
        provide_context=True,
        op_kwargs={"metric": "precision",},
        trigger_rule="all_success",
    )
    # task to train the model with the best hyperparameters found using the precision score
    train_model_with_best_hyperparameters_precision_task = PythonOperator(
        task_id="train_model_with_best_hyperparameters_precision_task",
        python_callable=train_model_with_best_hyperparameters,
        provide_context=True,
        op_kwargs={
            "registered_model_name": "nba-investment-model-precision",
            "task_id": "search_best_hyperparameters_precision_task",
        },
        trigger_rule="all_success",
    )
    # task to transition the best model version to production, the best model version is the model version with the best metric
    # if the current model in production is not the best model version then
    # the best model version is transitioned to production else nothing is
    transition_best_model_version_to_prod_precision_task = PythonOperator(
        task_id="transition_best_model_version_to_prod_precision_task",
        python_callable=transition_best_model_version_to_prod,
        provide_context=True,
        op_kwargs={
            "registered_model_name": "nba-investment-model-precision",
            "metric": "precision",
        },
        trigger_rule="all_success",
    )

    # define the dependencies between the tasks
    read_raw_data_preprocess_raw_data_task >> search_best_hyperparameters_f1_task >> train_model_with_best_hyperparameters_f1_task >> transition_best_model_version_to_prod_f1_task
    read_raw_data_preprocess_raw_data_task >> search_best_hyperparameters_recall_task >> train_model_with_best_hyperparameters_recall_task >> transition_best_model_version_to_prod_recall_task
    read_raw_data_preprocess_raw_data_task >> search_best_hyperparameters_precision_task >> train_model_with_best_hyperparameters_precision_task >> transition_best_model_version_to_prod_precision_task
