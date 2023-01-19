import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to the basketball investment APP")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This web APP containes the EDA of the basketball investment data. you can
    also try the model to predict if a basketball player will be a good investment or not 
    by providing it's stats.
    - To view the tracking and details about the deployed model visit [mlflow service](http://20.224.70.229:5000/)
    - To vizualize the data inputs and the predictions of the models visit [kibana service](http://20.224.70.229:5601/)
    - To view the running DAGs visit [airflow service](http://20.224.70.229:8080/), you will need a password to login, username: airflow, password: airflow
    - To view the fastapi endpoints visit [fastapi service](http://20.224.70.229:4100/docs)
    - To view the data stored in the minio bucket visit [minio service](http://20.224.70.229:9001/) you will need a password to login, username: abdessamadbaahmed, password: baahmedabdessamad
"""
)
