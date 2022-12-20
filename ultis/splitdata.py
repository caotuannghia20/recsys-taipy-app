import pandas as pd
import numpy as np

TEST_RATIO = 0.2
MOVIELENS_DATA_PATH = "u.data"
MOVIE_DATA_PATH = "u.item"


def convert_data_to_dataframe(data_path: str, movie_path: str):
    data = pd.read_table(data_path)
    data.columns = ["u_id", "i_id", "rating", "timestamp"]
    data_sort = data.sort_values(by=["u_id"])

    movie_name = pd.read_table(
        movie_path, sep='|', encoding="latin-1", header=None)
    movie_name.drop([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23], inplace=True, axis=1)
    movie_name.rename(columns={0: "movieId", 1: "title"}, inplace=True)

    return data_sort, movie_name


movies_list = []

print("\nReading the ratings file...")
ratings, movie_name = convert_data_to_dataframe(
    MOVIELENS_DATA_PATH, MOVIE_DATA_PATH)

userIds = ratings.u_id.unique()

trainIds = []
testIds = []

print("Splitting the ratings data...")
for u in userIds:
    rating_of_u = ratings.loc[ratings.u_id == u]

    trainIds_sample = rating_of_u.sample(
        frac=(1-test_ratio), random_state=7)
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
movie_name.to_csv("movie.csv", index=False)
print("Done.")
