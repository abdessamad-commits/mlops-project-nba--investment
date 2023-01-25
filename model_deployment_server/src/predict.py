import logging
import os
import shutil
import uuid
from minio import Minio
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from .helpers import (
    Player,
    import_model_from_registry,
    preprocessing_data,
    save_to_elasticsearch,
    upload_file_to_minio,
)
from fastapi import FastAPI, HTTPException, UploadFile


# Constants for status codes
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400


# logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename="model_deployment.log",
    filemode="a",
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Create a new instance of the FastAPI class
app = FastAPI()

# Define a route for the root path
@app.get("/")
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {"message": "Welcome to the NBA players investment API"}


# Define a route for the prediction model
@app.post("/online_predict")
def online_predict(player: Player):
    """
    The function takes a player as input and returns a prediction, the model_name parameter is the name of the model you want to use for prediction, you can choose between nba-investment-model-precision, nba-investment-model-recall, nba-investment-model-f1"

    :param player: a player object
    
    :return: a prediction of the model as a json object
    """
    try:

        # set tracking uri
        mlflow.set_tracking_uri("http://20.224.70.229:5000/")
        # set experiment name
        mlflow.set_experiment("nba-investment-experiment")

        # Create a pandas dataframe and select the features the current model expects
        input_data = player.dict()

        # Convert the values to float if they are numerical values else we keep them in their original format
        values = [
            float(val) if isinstance(val, (int, float)) else val
            for val in input_data.values()
        ]

        # check if all the values are non-negative
        if any(val < 0 for val in values if isinstance(val, (int, float))):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid entry, all values must be non-negative",
            )

        # set the columns names for the entry data
        columns = [
            "GP",
            "MIN",
            "PTS",
            "FGM",
            "FGA",
            "FG%",
            "3P Made",
            "3PA",
            "3P%",
            "FTM",
            "FTA",
            "FT%",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "model_name",
        ]

        # Create a pandas dataframe from the input data
        input_data = pd.DataFrame([values], columns=columns)
        # Get the model name to be used from the input data
        model_name = input_data["model_name"].values[0]
        # Drop the model name to be used from the input data
        input_data.drop(columns=["model_name"], inplace=True)

        # Load model as a PyFuncModel using the model registry
        loaded_model = import_model_from_registry(model_name, stage="Production")

        # Predict on a Pandas DataFrame.
        prediction = loaded_model.predict(input_data)

        # Cast the prediction to int
        prediction = int(prediction.round())

        # If the prediction is 1, the player is a good investment
        message = "Good investment" if prediction == 1 else "Bad investment"

        # Convert the input data to a dictionary
        input_data_dict = input_data.to_dict(orient="records")[0]

        # Logging input data and model name
        logging.info(
            {
                "input_data": input_data_dict,
                "model_name": model_name,
                "prediction": prediction,
            }
        )

        # Save the prediction to elasticsearch
        save_to_elasticsearch(input_data_dict, prediction)

        # Return the api response
        return {
            "prediction": prediction,
            "status": HTTP_200_OK,
            "message": message,
            "model_name": model_name,
        }

    except Exception as e:
        # Log the error
        logging.exception(e)
        raise e


@app.post("/batch_predict")
def batch_predict(file: UploadFile, model_name: str):
    """
    The function takes a csv file as input and returns a csv file with the predictions as output, the model_name parameter is the name of the model you want to use for prediction, you can choose between nba-investment-model-precision, nba-investment-model-recall, nba-investment-model-f1"


    :param file: csv file
    :param model_name: name of the model to be used from the model registry

    :return: csv file with predictions
    """

    try:

        # initialize minio client
        minio_client = Minio(
            "minio:9000",
            access_key="abdessamadbaahmed",
            secret_key="baahmedabdessamad",
            secure=False,
        )

        # set tracking uri
        mlflow.set_tracking_uri("http://20.224.70.229:5000/")

        # set experiment name
        mlflow.set_experiment("nba-investment-experiment")

        # download the file to the local directory
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # read the csv file
        df = pd.read_csv(file.filename)

        # Preprocess the data
        df = preprocessing_data(df)
        df.drop(columns=["TARGET_5Yrs"], inplace=True)

        # Check if all the values are non-negative
        if (df < 0).any().any() < 0:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="Invalid entry"
            )

        # Load model as a PyFuncModel.
        loaded_model = import_model_from_registry(model_name, stage="Production")

        # Predict on a Pandas DataFrame.
        predictions = loaded_model.predict(df)

        # Create a new column in the DataFrame to store the predictions
        df["predictions"] = predictions

        # Cast the predictions to int
        predictions = predictions.astype(int)

        # Generate a unique file name
        file_name = f"predictions_{str(uuid.uuid4())}.csv"

        # Save the DataFrame to a csv file
        df.to_csv(file_name, index=False)

        # upload the file to minios
        upload_file_to_minio(
            minio_client,
            f"{str(file_name)}",
            "nba-investment-data",
            file_name,
        )

        # delete the file from the local directory
        os.remove(file_name)
        os.remove(file.filename)

        # Read the saved file and return it as a response
        return {"file_name": file_name, "status": HTTP_200_OK, "model_name": model_name}

    except Exception as e:
        # delete the file from the local directory
        os.remove(file_name)
        os.remove(file.filename)
        logging.exception(e)
        raise e
