import json
import logging
import os

import pandas as pd
import requests
import streamlit as st
from helpers import read_csv_from_minio
from minio import Minio

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename="streamlit.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

tab1, tab2 = st.tabs(["Batch Prediction", "Online Prediction"])

with tab1:

    minio_client = Minio(
        "minio:9000",
        access_key="abdessamadbaahmed",
        secret_key="baahmedabdessamad",
        secure=False,
    )

    # csv file upload
    st.title("NBA Player Investment Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    model_name = st.selectbox(
        "What type of investors are you ?",
        (
            "Risk Taker",
            "Cautious Investor",
            "Balanced between Risk Taker and Cautious Investor",
        ),
        key=3,
    )

    if model_name == "Cautious Investor":
        st.write(
            "You are the Cautious type of investors so the model used for prediction maximizes precision"
        )
        model_name = "nba-investment-model-precision"

    elif model_name == "Risk Taker":
        st.write(
            "You are the Risk Taker type of investors so the model used for prediction maximizes recall"
        )
        model_name = "nba-investment-model-recall"

    else:
        st.write(
            "You are the Balanced type of investors so the model used for prediction maximizes f1-score"
        )
        model_name = "nba-investment-model-f1"

    if st.button("Predict", key=1) and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv("uploaded_data_temp.csv", index=False)

        url = (
            f"http://model_deployment_server:4100/batch_predict?model_name={model_name}"
        )

        files = [
            (
                "file",
                (
                    "uploaded_data_temp.csv",
                    open("uploaded_data_temp.csv", "rb"),
                    "text/csv",
                ),
            )
        ]

        response = requests.request(
            "POST", url, headers={}, data={}, files=files
        ).json()

        os.remove("uploaded_data_temp.csv")

        st.write(
            "The file with the predictions have been successfully stored on the bucket nba-investment-data with the following name: "
            + str(response["file_name"])
        )

        st.dataframe(
            read_csv_from_minio(
                minio_client, "nba-investment-data", response["file_name"]
            )
        )


with tab2:
    st.title("NBA Player Investment Prediction")
    gp = st.number_input("GP", min_value=0)
    minn = st.number_input("MIN", min_value=0)
    pts = st.number_input("PTS", min_value=0)
    fgm = st.number_input("FGM", min_value=0)
    fga = st.number_input("FGA", min_value=0)
    fgpercent = st.number_input("FG%", min_value=0)
    treepmade = st.number_input("3P Made", min_value=0)
    treepa = st.number_input("3PA", min_value=0)
    treeppercent = st.number_input("3P%", min_value=0)
    ftm = st.number_input("FTM", min_value=0)
    fta = st.number_input("FTA", min_value=0)
    ftpercent = st.number_input("FT%", min_value=0)
    oreb = st.number_input("OREB", min_value=0)
    dreb = st.number_input("DREB", min_value=0)
    reb = st.number_input("REB", min_value=0)
    ast = st.number_input("AST", min_value=0)
    stl = st.number_input("STL", min_value=0)
    blk = st.number_input("BLK", min_value=0)
    tov = st.number_input("TOV", min_value=0)

    option_10 = st.selectbox(
        "What type of investors are you ?",
        (
            "Risk Taker",
            "Cautious Investor",
            "Balanced between Risk Taker and Cautious Investor",
        ),
    )

    if option_10 == "Cautious Investor":
        st.write(
            "You are the Cautious type of investors so the model used for prediction maximizes precision"
        )
        model_name = "nba-investment-model-precision"

    elif option_10 == "Risk Taker":
        st.write(
            "You are the Risk Taker type of investors so the model used for prediction maximizes recall"
        )
        model_name = "nba-investment-model-recall"

    else:
        st.write(
            "You are the Balanced type of investors so the model used for prediction maximizes f1-score"
        )
        model_name = "nba-investment-model-f1"

    player = {
        "gp": gp,
        "min": minn,
        "pts": pts,
        "fgm": fgm,
        "fga": fga,
        "fgpercent": fgpercent,
        "treepmade": treepmade,
        "treepa": treepa,
        "treeppercent": treeppercent,
        "ftm": ftm,
        "fta": fta,
        "ftpercent": ftpercent,
        "oreb": oreb,
        "dreb": dreb,
        "reb": reb,
        "ast": ast,
        "stl": stl,
        "blk": blk,
        "tov": tov,
        "model_name": model_name,
    }

    if st.button("Predict", key=2):
        # The url of the endpoint that the request will be sent to
        url = "http://model_deployment_server:4100/online_predict"

        # The headers of the request, in this case it's setting the accept and Content-Type
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        try:
            # Send the request
            response = requests.post(url, headers=headers, json=player)

            # Get the response as a json
            response = response.json()

            # Get the prediction from the response
            prediction = response["prediction"]

            # Get the message from the response
            message = response["message"]

            # Display the prediction
            st.success(
                f"The prediction of the model is: {prediction}, This is a {message}"
            )

        except Exception as e:
            logging.exception(e)
            st.error(f"there was an error: {e}")
