import json

import requests
import streamlit as st



tab1, tab2 = st.tabs(["Batch Prediction", "Online Prediction"])

with tab1:
    # upload file to predict on
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


with tab2:
    st.title("NBA Player Investment Prediction")
    gp = st.number_input("Games Played", min_value=0)
    minn = st.number_input("Minutes Played", min_value=0)
    pts = st.number_input("Points Per Game", min_value=0)
    fgm = st.number_input("Field Goals Made", min_value=0)
    fga = st.number_input("Field Goal attempts", min_value=0)
    fgpercent = st.number_input("Field Goal Percent", min_value=0)
    ftm = st.number_input("Free Throw Made", min_value=0)
    ftpercent = st.number_input("Free Throw Percent", min_value=0)
    reb = st.number_input("Rebounds", min_value=0)

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

    df = {
        "gp": gp,
        "minn": minn,
        "pts": pts,
        "fgm": fgm,
        "fga": fga,
        "fgpercent": fgpercent,
        "ftm": ftm,
        "ftpercent": ftpercent,
        "reb": reb,
        "option_10": option_10,
    }

    if st.button("Predict"):
        response = requests.post("http://0.0.0.0:8000/predict", json=df)
        # response = requests.post('http://localhost:8000/predict', json=df)
        prediction = response.text
        st.success(f"The prediction of the model is: {prediction}")

