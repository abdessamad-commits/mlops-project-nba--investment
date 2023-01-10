import time
import plotly.express as px
import streamlit as st
from helpers import feature_importance_logistic_regression, pearson_corr, read_dataset, feature_importance_tree_model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df, features, target = read_dataset(
    "/Users/abdessamadbaahmed/Desktop/livrable_mp_data/data/nba_logreg.csv"
)

tab1, tab2 = st.tabs(["Logistic regression feature importance", "tree models feature importance"])


with tab1:
    metric = st.selectbox(
    "Choose the metric of interest", ("accuracy", "recall", "precision", "f1")
    )
    penalty = st.selectbox("Choose the penalty of interest", ("l1", "l2"))
    
    st.plotly_chart(feature_importance_logistic_regression(features, target, metric=metric, penalty=penalty))

with tab2:
    st.plotly_chart(feature_importance_tree_model(features, target, RandomForestClassifier()))
    st.plotly_chart(feature_importance_tree_model(features, target, xgb.XGBClassifier()))