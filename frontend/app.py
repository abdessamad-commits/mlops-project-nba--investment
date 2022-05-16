import app1
import app2
import app3
import streamlit as st
import pandas as pd

st.set_page_config(layout='wide')

PAGES = {
    "Univariate Anlysis": app1,
    "Multivariate Analysis": app2,
   "Investable Players Prediction": app3
}

def read_dataset(file):
    df = pd.read_csv(file)
    df.drop_duplicates(inplace=True)

    df.columns = ["Name", "Games Played", "Minutes Played", "Points Per Game", "Field Goals Made", "Field Goal attempts",
               "Field Goal Percent", "3 Points Made", "3 Points Attempts", "3 Points Percent",
              "Free Throw Made", "Free Throw Attempts", "Free Throw Percent", "Offensive Rebounds", "Defensive Rebounds","Rebounds",
              "Assists", "Steals", "Blocks", "Turnovers", "Outcome Career Length"]

    features = df.drop(["Outcome Career Length","Name"], axis=1)
    target = df["Outcome Career Length"]
    return df, features, target

df, features, target = read_dataset('data/nba_logreg.csv')


st.sidebar.title('Tabs')
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app(df, features, target)


