import random

import requests

# The URL of the deployed model
URL = "http://20.224.70.229:4100/online_predict"

# The headers of the request, in this case it's setting the accept and Content-Type
headers = {"accept": "application/json", "Content-Type": "application/json"}


def generate_random_data(num_points):
    data = []
    for _ in range(num_points):
        player = {
            "gp": random.uniform(1, 200),
            "min": random.uniform(1, 200),
            "pts": random.uniform(1, 200),
            "fgm": random.uniform(1, 200),
            "fga": random.uniform(1, 200),
            "fgpercent": random.uniform(1, 200),
            "treepmade": random.uniform(1, 200),
            "treepa": random.uniform(1, 200),
            "treeppercent": random.uniform(1, 200),
            "ftm": random.uniform(1, 200),
            "fta": random.uniform(1, 200),
            "ftpercent": random.uniform(1, 200),
            "oreb": random.uniform(1, 200),
            "dreb": random.uniform(1, 200),
            "reb": random.uniform(1, 200),
            "ast": random.uniform(1, 200),
            "stl": random.uniform(1, 200),
            "blk": random.uniform(1, 200),
            "tov": random.uniform(1, 200),
            "model_name": "nba-investment-model-f1",
        }
        data.append(player)
    return data


players = generate_random_data(100)
print(players[0])

for player in players:
    # Send the request
    response = requests.post(URL, headers=headers, json=player)
    prediction = response.json()
    print("the prediction is: ", prediction)
