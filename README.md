# NBA Players Investment System
This project is a prediction system for basketball player investment which utilizes a microservices architecture. The system comprises of several different services, including a machine learning model that has been trained to predict the investment potential of a basketball player, a frontend service that allows users to interact with the system and make requests, and an MLflow service that is responsible for managing and selecting the most performant model. All these services are Dockerized, ensuring ease of deployment and scalability. The model takes requests from the frontend and returns predictions, while the MLflow service is used to track and retrieve the best-performing model.

## Project Microservices:
1. **Service 1 - FastAPI model**: contains endpoints for making predictions based on input data. 
   Deployed at `http://20.224.70.229:4100/`
2. **Service 2 - Mlflow service**: tracks performance of models, automatically uploads best models to registry, and delivers them to FastAPI service for predictions. 
   Deployed at `http://20.224.70.229:5000/`
3. **Service 3 - Streamlit service**: front end for inspecting and visualizing data and interacting with deployed models by submitting own data or viewing model details. 
   Deployed at `http://20.224.70.229:8501/`
4. **Service 4 - MinIO storage**: responsible for storing data in various formats (raw, processed, training, validation, testing, batch predictions). Allows all containers to share necessary data and write to a bucket. 
   Deployed at `http://20.224.70.229:9001/`, accessible with a password that can be viewed/modified in the `docker-compose` file.
5. **Service 5 - Airflow service**: orchestrates workflow for training models periodically using new incoming data, automatically upgrades the best model to production and updates the FastAPI service by retrieving from the registry. 
   Deployed at `http://20.224.70.229:9001/`, accessible with a user: `airflow` and password: `airflow` which can be viewed/modified in the `docker-compose` file.



### Main technologies
* [Mlflow](https://scikit-learn.org/stable/)
* [Python](https://www.python.org/)
* [Airflow](https://pandas.pydata.org/)
* [FastAPI](https://fastapi.tiangolo.com)
* [Streamlit](https://streamlit.io)
* [Docker](https://streamlit.io)
* [ELK stack](https://streamlit.io)



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






