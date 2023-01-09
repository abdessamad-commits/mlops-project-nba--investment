import time

import plotly.express as px
import streamlit as st
from helpers import feature_importance_logistic_regression, pearson_corr, read_dataset

df, features, target = read_dataset(
    "/Users/abdessamadbaahmed/Desktop/livrable_mp_data/data/nba_logreg.csv"
)

metric = st.selectbox(
    "Choose the metric of interest", ("accuracy", "recall", "precision", "f1")
)

penalty = st.selectbox("Choose the penalty of interest", ("l1", "l2"))

st.plotly_chart(feature_importance_logistic_regression(features, target, metric=metric, penalty=penalty))
