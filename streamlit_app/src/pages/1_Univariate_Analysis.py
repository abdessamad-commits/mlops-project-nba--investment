import pandas as pd
import plotly.express as px
import streamlit as st
from helpers import read_dataset

df, features, target = read_dataset()

tab1, tab2 = st.tabs(["Summary Statistics", "Viz"])


with tab1:
    if agree1 := st.checkbox("Plot the dataset", value=True):
        st.dataframe(data=df, width=None, height=None)
        st.write(df.shape)

    if agree2 := st.checkbox("Plot summary statistics", value=False):
        st.dataframe(features.describe())

    if agree3 := st.checkbox("Plot class distribution", value=False):
        dist = px.histogram(
            df, x="Outcome Career Length", color="Outcome Career Length"
        )
        st.plotly_chart(dist)


with tab2:
    option = st.selectbox(
        "Please select one Numerical variable to vizualize", list(features)
    )

    choice = st.selectbox(
        "Please select the chosen plot", ("Box Plot", "Histogram", "Violin Plot")
    )
    if choice == "Histogram":
        with st.container():
            # st.write('First column')
            fig1 = px.histogram(
                df, x=option, title="Distribution of " + option, width=800
            )
            st.plotly_chart(fig1)
    elif choice == "Box Plot":
        with st.container():
            # st.write('First column')
            fig1 = px.box(df, y=option, title="Distribution of " + option, width=800)
            st.plotly_chart(fig1)

    if choice == "Histogram":
        with st.container():
            # st.write('First column')
            fig2 = px.histogram(
                df,
                x=option,
                color="Outcome Career Length",
                width=800,
                title="Distribution of "
                + option
                + " grouped by the career length larger then five years",
            )
            st.plotly_chart(fig2)
    elif choice == "Box Plot":
        with st.container():
            # st.write('First column')
            fig2 = px.box(
                df,
                y=option,
                color="Outcome Career Length",
                width=800,
                title="Distribution of "
                + option
                + " grouped by the career length being larger than five years",
            )
            st.plotly_chart(fig2)
    if choice == "Violin Plot":
        with st.container():
            # st.write('First column')
            fig2 = px.violin(
                df,
                y=option,
                color="Outcome Career Length",
                width=800,
                title="Distribution of "
                + option
                + " grouped by the career length being larger than five years",
            )
            st.plotly_chart(fig2)
