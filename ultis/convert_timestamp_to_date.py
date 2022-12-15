import pandas as pd
from datetime import datetime


def convert(data_path: str):
    data = pd.read_csv(data_path)

    timestamp = data.timestamp.to_list()
    date = []
    for time in timestamp:
        date.append(datetime.fromtimestamp(time))
    data["timestamp"] = date
    data.to_csv("data.csv", index=False)
