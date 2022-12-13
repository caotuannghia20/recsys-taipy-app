import pandas as pd
import numpy as np

test_ratio = 0.2
val_ratio = 0.1
print("\nReading the ratings file...")
ratings = pd.read_csv("rating_val_date.csv")
dates = ratings.userId.unique()

movies_list = []
trainIds = []
testIds = []
valIds = []

print("Splitting the ratings data...")
for date in dates:
    rating_of_date = ratings.loc[ratings.timestamp == date]

    trainIds_sample = rating_of_date.sample(
        frac=(1 - test_ratio - val_ratio), random_state=7
    )
    valIds_sample = rating_of_date.drop(trainIds_sample.index.tolist()).sample(
        frac=(val_ratio / (val_ratio + test_ratio)), random_state=8
    )
    testIds_sample = rating_of_date.drop(trainIds_sample.index.tolist()).drop(
        valIds_sample.index.tolist()
    )

    for _, rating in trainIds_sample.iterrows():
        if rating.movieId not in movies_list:
            # Append new movie's Id to the movie list
            movies_list.append(rating.movieId)

    trainIds_sample = trainIds_sample.index.values
    trainIds_sample.sort()
    trainIds = np.append(trainIds, trainIds_sample)

    valIds_sample = valIds_sample.index.values
    valIds_sample.sort()
    valIds = np.append(valIds, valIds_sample)

    testIds_sample = testIds_sample.index.values
    testIds_sample.sort()
    testIds = np.append(testIds, testIds_sample)

print("Write ratings to new file...")
train = ratings.loc[ratings.index.isin(trainIds)]
train.to_csv("/rating_train.csv", index=False)
test = ratings.loc[ratings.index.isin(testIds)]
test.to_csv("/rating_test.csv", index=False)
val = ratings.loc[ratings.index.isin(valIds)]
val.to_csv("/rating_val.csv", index=False)
print("Done.")
