import pandas as pd
import numpy as np

TEST_RATIO = 0.2
MOVIELENS_DATA_PATH = "u.data"


def convert_data_to_dataframe(data_path: str):
    data = pd.read_table(data_path)
    data.columns = ["u_id", "i_id", "rating", "timestamp"]
    data_sort = data.sort_values(by=["u_id"])
    return data_sort


movies_list = []

print("\nReading the ratings file...")
ratings = convert_data_to_dataframe(MOVIELENS_DATA_PATH)

userIds = ratings.u_id.unique()

trainIds = []
testIds = []

print("Splitting the ratings data...")
for u in userIds:
    rating_of_u = ratings.loc[ratings.u_id == u]

    trainIds_sample = rating_of_u.sample(
        frac=(1-TEST_RATIO), random_state=7)
    testIds_sample = rating_of_u.drop(trainIds_sample.index.tolist())

    for _, rating in trainIds_sample.iterrows():
        if rating.i_id not in movies_list:
            # Append new movie's Id to the movie list
            movies_list.append(rating.u_id)

    trainIds_sample = trainIds_sample.index.values
    trainIds_sample.sort()
    trainIds = np.append(trainIds, trainIds_sample)

    testIds_sample = testIds_sample.index.values
    testIds_sample.sort()
    testIds = np.append(testIds, testIds_sample)
print("Done.")

print("Write ratings to new file...")
train = ratings.loc[ratings.index.isin(trainIds)]
train.to_csv("rating_train.csv", index=False)
test = ratings.loc[ratings.index.isin(testIds)]
test.to_csv("rating_test.csv", index=False)
print("Done.")
