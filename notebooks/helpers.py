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

def read_data(data_path, target_column):
    """
    this function reads the data

    :param data_path: path of the data
    :param target_column: target column of the data

    :return: DataFrame with the target column
    """
    df = pd.read_csv(data_path)
    df['TARGET_5Yrs'] = df['TARGET_5Yrs'].astype(int)
    features = convert_numerical_columns_to_float(df.drop(target_column, axis=1))
    target_column = df[target_column]
    return pd.concat([features, target_column], axis=1)

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
    return df

def test_model(run_id, X_test, y_test):
    """
    this function tests the model on the test data and prints the evaluation metrics in order to go to production

    :param name: name of the model
    :param stage: stage of the model
    :param X_test: test features
    :param y_test: test target

    :return: None
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
