import requests

# The URL of the deployed model
URL = "http://0.0.0.0:4100/predict"

# The headers of the request, in this case it's setting the accept and Content-Type
headers = {"accept": "application/json", "Content-Type": "application/json"}

# The data that will be sent in the request body
player = {
    "gp": 0.0,
    "min": 0.0,
    "pts": 0.0,
    "fgm": 0.0,
    "fga": 0.0,
    "fgpercent": 0.0,
    "treepmade": 0.0,
    "treepa": 0.0,
    "treeppercent": 0.0,
    "ftm": 0.0,
    "fta": 0.0,
    "ftpercent": 0.0,
    "oreb": 0.0,
    "dreb": 0.0,
    "reb": 0.0,
    "ast": 0.0,
    "stl": 0.0,
    "blk": 0.0,
    "tov": 0.0,
}

response = requests.post(URL, headers=headers, json=player)
prediction = response.json()

print("the prediction is: ", prediction)
