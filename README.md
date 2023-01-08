## NBA Players Investment Prediction
This is a project on the deployement of a machine learning model using FastAPI in order to build the REST API and Streamlit in order to build the frontend page that will allow for EDA and taking input from the user

### Project Steps : 
* Part 1 -  EDA and Data Viz made in the web application for an interactive exploration of the data
* Part 2 -  Train and save the model  
* Part 3 - Build the REST API and connect it to a Streamlit frontend

### Prerequisites
* IDE(best to use pycharm), The IDE is not necessary to run the app as it can be launched only using the terminal but the IDE makes the process easier
* [Scikit Learn](https://scikit-learn.org/stable/)
* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [FastAPI](https://fastapi.tiangolo.com)
* [Streamlit](https://streamlit.io)
* Plotly
* [...]

## Project Structure
This project has three major parts :
1. NBA_Dataset_Analysis.ipynb - Contains the code of our machine learning model to predict the investable player based on the dataset "nba_logreg.csv".
2. main.py - Contains FastAPI's API that receives relevant statistics of a basketball player through GUI built using Streamlit, and predict if the player is investable or not
3. app.py - app1.py - app2.py - app3.py contain all the Streamlit code for the data visualization and also uses requests module to call the API already defined in main.py and dispalys the returned value.


## Run The APP

1. Create a python env (a conda env can also be used)
```
virtualenv env
source env/bin/activate
```

2. Install all of the dependencies
```
pip install -r requirements.txt
```

3. Run the server
```
uvicorn main:app --reload
```

4. Run the Streamlit Web APP

In a new terminal in the root(mandatory) of the project we use the following command:
```
Streamlit run frontend/app.py
```






