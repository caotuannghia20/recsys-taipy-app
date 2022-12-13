import pandas as pd
from datetime import datetime

data = pd.read_csv("dataset/rating_train.csv")

timestamp = data.timestamp.to_list()
date = []
for time in timestamp:
    date.append(datetime.fromtimestamp(time))
data["timestamp"] = date
data.to_csv("data.csv", index=False)
