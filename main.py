# Import Needed Libraries

from fastapi import FastAPI

import joblib
import uvicorn
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
model_saved = joblib.load('model/fine_tuned_svm.joblib')
model_saved_recall = joblib.load('model/fine_tuned_svm_recall.joblib')
model_saved_precision = joblib.load('model/fine_tuned_svm_precision.joblib')


class Data(BaseModel):
    gp : int
    minn : int
    pts : int
    fgm : int
    fga : int
    fgpercent : int
    ftm : int
    ftpercent : int
    reb : int
    option_10 : str

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'Welcome to the NBA players investment API'}



@app.post("/predict")
def predict(data: Data):
    data = data.dict()
    #data_df = pd.DataFrame.from_dict([data_dict])
    gp = data['gp']
    minn = data['minn']
    pts = data['pts']
    fgm = data['fgm']
    fga = data['fga']
    fgpercent = data['fgpercent']
    ftm = data['ftm']
    ftpercent = data['ftpercent']
    reb = data['reb']
    option_10 = data['option_10']

    if option_10 == 'Cautious Investor':
        model = model_saved_precision
    else:
        model = model_saved_recall

    prediction = model.predict([[gp, minn, pts, fgm, fga, fgpercent, ftm, ftpercent, reb]])
    return {'prediction ': prediction.tolist()}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

