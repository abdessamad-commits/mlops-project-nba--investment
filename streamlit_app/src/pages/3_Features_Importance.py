import time

import plotly.express as px
import streamlit as st
import xgboost as xgb
from helpers import (
    feature_importance_logistic_regression,
    feature_importance_tree_model,
    read_dataset,
)
from sklearn.ensemble import RandomForestClassifier

df, features, target = read_dataset()

tab1, tab2 = st.tabs(
    ["Logistic regression feature importance", "tree models feature importance"]
)


with tab1:
    metric = st.selectbox(
        "Choose the metric of interest", ("accuracy", "recall", "precision", "f1")
    )
    penalty = st.selectbox("Choose the penalty of interest", ("l1", "l2"))

    st.plotly_chart(
        feature_importance_logistic_regression(
            features, target, metric=metric, penalty=penalty
        )
    )

with tab2:
    st.plotly_chart(
        feature_importance_tree_model(features, target, RandomForestClassifier())
    )
    st.plotly_chart(
        feature_importance_tree_model(features, target, xgb.XGBClassifier())
    )
