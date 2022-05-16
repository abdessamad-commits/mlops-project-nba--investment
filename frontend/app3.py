import streamlit as st
import requests
import json


# ["Name", "Games Played", "Minutes Played", "Points Per Game", "Field Goals Made", "Field Goal attempts",
#              "Field Goal Percent", "3 Points Made", "3 Points Attempts", "3 Points Percent",
#             "Free Throw Made", "Free Throw Attempts", "Free Throw Percent", "Offensive Rebounds", "Defensive Rebounds","Rebounds",
#             "Assists", "Steals", "Blocks", "Turnovers", "Outcome Career Length"]

# ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', 'FTM', 'FT%', 'REB']

# url = 'http://fastapi:8000'
# endpoint = '/predict'


def app(df, features, target):
    st.title('NBA Player Investment Prediction')
    gp = st.number_input('Games Played', min_value=0)
    minn = st.number_input('Minutes Played', min_value=0)
    pts = st.number_input('Points Per Game', min_value=0)
    fgm = st.number_input('Field Goals Made', min_value=0)
    fga = st.number_input('Field Goal attempts', min_value=0)
    fgpercent = st.number_input('Field Goal Percent', min_value=0)
    ftm = st.number_input('Free Throw Made', min_value=0)
    ftpercent = st.number_input('Free Throw Percent', min_value=0)
    reb = st.number_input('Rebounds', min_value=0)

    option_10 = st.selectbox(
        "What type of investors are you ?",
        ('Risk Taker', 'Cautious Investor'))
    if option_10 == 'Cautious Investor':
        st.write("You are the risk taker type of investors so the model used for prediction maximises precision")
    else:
        st.write("You are the Cautious type of investors so the model used for prediction maximises recall")


    df = {
        'gp': gp,
        'minn': minn,
        'pts': pts,
        'fgm': fgm,
        'fga': fga,
        'fgpercent': fgpercent,
        'ftm': ftm,
        'ftpercent': ftpercent,
        'reb': reb,
        'option_10': option_10
    }



    if st.button("Predict"):
        response = requests.post('http://0.0.0.0:8000/predict', json=df)
        prediction = response.text
        st.success(f"The prediction of the model is: {prediction}")


if __name__ == '__main__':
    app()
