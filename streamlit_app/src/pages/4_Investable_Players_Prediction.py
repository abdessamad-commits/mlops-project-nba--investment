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
    filemode="w",
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
    if st.button("Predict", key=1) and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv("uploaded_data_temp.csv", index=False)

        url = "http://model_deployment_server:4100/batch_predict"

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
    gp = st.number_input("gp", min_value=0)
    minn = st.number_input("min", min_value=0)
    pts = st.number_input("pts", min_value=0)
    fgm = st.number_input("fgm", min_value=0)
    fga = st.number_input("fga", min_value=0)
    fgpercent = st.number_input("fgpercent", min_value=0)
    treepmade = st.number_input("treepmade", min_value=0)
    treepa = st.number_input("treepa", min_value=0)
    treeppercent = st.number_input("treeppercent", min_value=0)
    ftm = st.number_input("ftm", min_value=0)
    fta = st.number_input("fta", min_value=0)
    ftpercent = st.number_input("ftpercent", min_value=0)
    oreb = st.number_input("oreb", min_value=0)
    dreb = st.number_input("dreb", min_value=0)
    reb = st.number_input("reb", min_value=0)
    ast = st.number_input("ast", min_value=0)
    stl = st.number_input("stl", min_value=0)
    blk = st.number_input("blk", min_value=0)
    tov = st.number_input("tov", min_value=0)

    option_10 = st.selectbox(
        "What type of investors are you ?", ("Risk Taker", "Cautious Investor")
    )
    if option_10 == "Cautious Investor":
        st.write(
            "You are the Cautious type of investors so the model used for prediction maximises precision"
        )
    else:
        st.write(
            "You are the Risk Taker type of investors so the model used for prediction maximises recall"
        )

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
