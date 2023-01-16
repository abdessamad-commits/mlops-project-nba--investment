import logging
import os
import shutil
import uuid
from typing import Union

import joblib
import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .helpers import *

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename="model_deployment.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Constants for status codes
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_500_INTERNAL_SERVER_ERROR = 500


# Create a class which inherits from the BaseModel class of pydantic and define the data types of the attributes
class Player(BaseModel):
    gp: Union[int, float]
    min: Union[int, float]
    pts: Union[int, float]
    fgm: Union[int, float]
    fga: Union[int, float]
    fgpercent: Union[int, float]
    treepmade: Union[int, float]
    treepa: Union[int, float]
    treeppercent: Union[int, float]
    ftm: Union[int, float]
    fta: Union[int, float]
    ftpercent: Union[int, float]
    oreb: Union[int, float]
    dreb: Union[int, float]
    reb: Union[int, float]
    ast: Union[int, float]
    stl: Union[int, float]
    blk: Union[int, float]
    tov: Union[int, float]


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
    try:
        # set tracking uri
        mlflow.set_tracking_uri("http://20.224.70.229:5000/")

        # set experiment name
        mlflow.set_experiment("nba-investment-experiment")

        # Create a pandas dataframe and select the features the current model expects
        input_data = player.dict()

        # Convert the values to float
        values = [float(value) for value in list(input_data.values())]

        # check if all the values are non-negative
        if any(val < 0 for val in values):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="Invalid entry"
            )

        # Create a pandas dataframe and select the features the current model expects
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
        ]
        input_data = pd.DataFrame([values], columns=columns)

        # defining the model from mlfow registry
        logged_model = "runs:/92ba1717068040aba1137975e81e4b81/model"

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on a Pandas DataFrame.
        prediction = loaded_model.predict(input_data)

        # Cast the prediction to int
        prediction = int(prediction.round())

        # If the prediction is 1, the player is a good investment
        message = "Good investment" if prediction == 1 else "Bad investment"

        # Return the api response
        return {"prediction": prediction, "status": HTTP_200_OK, "message": message}

    except Exception as e:
        logging.exception(e)
        raise e


@app.post("/batch_predict")
def batch_predict(file: UploadFile):
    """ 
    the function takes a csv file as input and returns a csv file with the predictions as output
    
    :param file: csv file
    
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

        # Check if all the values are non-negative
        if (df < 0).any().any() < 0:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="Invalid entry"
            )

        # defining the model from mlfow registry
        logged_model = "runs:/92ba1717068040aba1137975e81e4b81/model"

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

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
            minio_client, f"{str(file_name)}", "nba-investment-data", file_name,
        )

        # delete the file from the local directory
        os.remove(file_name)
        os.remove(file.filename)

        # Read the saved file and return it as a response
        return {"file_name": file_name, "status": HTTP_200_OK}

    except Exception as e:
        logging.exception(e)
        raise e
