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

def test_model_by_run_id(run_id, X_test, y_test):
    """
    this function tests the model on the test data and prints the evaluation metrics in order to go to production

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
        
def test_model_by_regsitry_version(model_name, model_version, X_test, y_test):
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
            model_uri=f"models:/{model_name}/{model_version}")
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
        
def test_model_by_regsitry_all_versions(model_name, X_test, y_test, sort_by="f1"):
    """
    the function tests all the versions of the model and returns a DataFrame with the evaluation metrics
    
    :param model_name: name of the model
    :param X_test: test features
    :param y_test: test target
    
    :return: DataFrame with the evaluation metrics
    """
    i = 0
    res = []
    while True:
        try:
            res.append(test_model_by_regsitry_version(model_name, i+1, X_test, y_test))
        except Exception:
            break
        i+=1
    return pd.DataFrame(res).sort_values(by=sort_by, ascending=False)
        
        
