import streamlit as st
from helpers import read_dataset

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the basketball investment APP")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This web APP containes the EDA of the basketball investment data. you can
    also try the model to predict if a basketball player will be a good investment or not 
    by providing it's stats.
    **👈 Select from the sidebar** to see have a look at the data and the model.

"""
)
