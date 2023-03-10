import time

import numpy as np
import plotly.express as px
import streamlit as st
from helpers import pearson_corr, read_dataset

df, features, target = read_dataset()

tab1, tab2 = st.tabs(["Scatter Plot", "Correlation Matrix"])

with tab1:
    option_1 = st.selectbox(
        "Please select the variable to vizualize on the x-axis", list(features)
    )

    option_2 = st.selectbox(
        "Please select the variable to vizualize on the y-axis", list(features), index=1
    )

    option_3 = st.selectbox(
        "Please select the variable representing the data points size", list(features)
    )

    choice3 = st.selectbox(
        "Please select the marginal distribution plot", ("box", "violin", "histogram")
    )
    choice2 = st.selectbox("Labeled Data", ("Yes", "No"))

    df["label"] = df["Outcome Career Length"].astype(str)

    if choice2 == "Yes":
        # compute the pearson coefficent
        with st.container():
            st.write(
                f"Pearson correlation between the variables '{option_1}' and '{option_2} is': {pearson_corr(df, option_1, option_2):.2f}"
            )
            fig3 = px.scatter(
                df,
                x=option_1,
                y=option_2,
                color="label",
                marginal_x=choice3,
                marginal_y=choice3,
                width=800,
                height=800,
                size=option_3,
                title="Scatter Plot",
                trendline="ols",
            )
            st.plotly_chart(fig3)

    else:
        with st.container():
            st.write(
                f"Pearson correlation between the variables '{option_1}' and '{option_2} is': {pearson_corr(df, option_1, option_2):.2f}"
            )
            fig3 = px.scatter(
                df,
                x=option_1,
                y=option_2,
                marginal_x=choice3,
                marginal_y=choice3,
                width=800,
                height=800,
                size=option_3,
                title="Log-transformed fit on linear axes",
                trendline="ols",
            )
            st.plotly_chart(fig3)


with tab2:
    fig4 = px.imshow(features.corr(), width=800, height=800, text_auto=True)
    st.plotly_chart(fig4)
