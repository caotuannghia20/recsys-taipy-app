import pandas as pd
import numpy as np
from ultis.dataloader import DataLoader
from helper.svd_helper import sgd, predict_svd_pair
from helper.knn_helper import predict_pair, compute_similarity_matrix

model = ["kNN", "MF"]
def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, movie_name: pd.DataFrame
):
    """Preprocess data from Pd.DataFrame.

    Args:
        train_data (pd.DataFrame): Train data load from "csv" file.
        test_data (pd.DataFrame): Test data load from "csv" file.
        movie_name (pd.DataFrame): The name of movies to recommend load from  "csv" file.

    Returns:
        train_set (numpy.array): Preprocessed training data.
        test_set (numpy.array): Preprocessed testing data.
        true_testset_movies_id (numpydarray): When convert, the movie_ids are reseting index, list all actual movies_id.
    """
    loader = DataLoader()
    true_testset_movies_id = test_data.i_id.to_numpy()
    train_set, test_set = loader.load_csv2ndarray(
        train_data=train_data,
        test_data=test_data,
        columns=["u_id", "i_id", "rating", "timestamp"],
    )
    return train_set, test_set, true_testset_movies_id


def fit(
    X,
    sim_measure="pcc",
    n_factors=100,
    n_epochs=50,
    learning_rate=0.01,
    algorithm="kNN",
):
    """Train the recommendation model.

    Args:
        X (numpyarray): Training data.
        sim_measure (str, optional):  Similarity measure function. Defaults to "pcc".
        n_factors (int, optional): Number of latent factors. Defaults to 100.
        n_epochs (int, optional): Number of SGD iterations. Defaults to 50.
        learning_rate (float, optional): The common learning rate. Defaults to 0.01.
        algorithm (str, optional): The algorithm used to make recommendations. Possible values are "kNN" or "MF. Defaults to "kNN".

    Returns:
        S (numpyarray): Compute the similarity between all pairs of users.
        x_rated (numpyarray): All users who rated each item are stored in list.
        x_list (numpyarray): All user id in training set.
        y_list (numpyarray): All movie id in training set.
        global_mean (float): Mean ratings in training set.
        pu (numpyarray): Users latent factor matrix.
        qi (numpyarray): Items latent factor matrix.
        bu (numpyarray): Users biases vector.
        bi (numpyarray): Items biases vector.
    """
    if algorithm not in model:
        raise SystemExit(f"{algorithm} is not avalid algorithm ")

    global_mean = np.mean(X[:, 2])
    if algorithm == "kNN":
        S, x_rated, x_list, y_list = compute_similarity_matrix(
            X, sim_measure
        )
        pu, qi, bu, bi = [], [], [], []

    elif algorithm == "MF":

        users_list = np.unique(X[:, 0])
        items_list = np.unique(X[:, 1])
        n_user = users_list.shape[0]
        n_item = items_list.shape[0]

        # Initialize pu, qi, bu, bi
        qi = np.random.normal(0, 0.1, (n_item, n_factors))
        pu = np.random.normal(0, 0.1, (n_user, n_factors))
        bu = np.zeros(n_user)
        bi = np.zeros(n_item)

        lr_pu, lr_qi, lr_bu, lr_bi, reg_pu, reg_qi, reg_bu, reg_bi = (
            learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate, learning_rate,)
        pu, qi, bu, bi, _ = sgd(
            X, pu, qi, bu, bi,
            n_epochs, global_mean, n_factors,
            lr_pu, lr_qi, lr_bu, lr_bi,
            reg_pu, reg_qi, reg_bu, reg_bi,
        )
        S, x_rated, x_list, y_list = [], [], [], []
    else:
        S, x_rated, x_list, y_list, pu, qi, bu, bi = (
            0, 0, 0, 0, 0, 0, 0, 0,)

    return S, x_rated, x_list, y_list, global_mean, pu, qi, bu, bi


def predict(
    test_set_origin,
    x_id, x_rated, S, k, min_k, x_list, y_list,
    global_mean, true_testset_movies_id,
    bu, bi, pu, qi, algorithm="MF",
):
    """Predict the ratings to movies.

    Args:
        test_set_origin (numpyarray): Storing all user/item pairs we want to predict the ratings.
        x_id (int): The user_id we want to recommend to him.
        x_rated (numpyarray): All users who rated each item are stored in list.
        S (numpyarray): Compute the similarity between all pairs of users.
        k (int): Number of neighbors use in prediction.
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        x_list (numpyarray): All user id in training set.
        y_list (numpyarray): All movie id in training set.
        global_mean (float): Mean ratings in training set.
        true_testset_movies_id (numpyarray): List all actual movie id.
        bu (numpyarray): Users biases vector.
        bi (numpyarray): Items biases vector.
        pu (numpyarray): Users latent factor matrix.
        qi (numpyarray): Items latent factor matrix.
        algorithm (str, optional): The algorithm used to make recommendations. Possible values are "kNN" or "MF. Defaults to "MF".

    Returns:
        predictions (numpyarray): Storing all predictions of the given user/item pairs. The first column is user id, the second column is item id, the thirh colums is the actual movie id, the forth column is the observed rating, and the fifth column is the predicted rating.
    """
    if algorithm not in model:
        raise SystemExit(f"{algorithm} is not avalid algorithm ")
    test_items = []
    test_set = np.zeros(
        (test_set_origin.shape[0], test_set_origin.shape[1] + 1))
    test_set[:, :3] = test_set_origin
    test_set[:, 3] = true_testset_movies_id.T
    for index in range(test_set.shape[0]):
        if test_set[index][0] == x_id:
            test_items.append(test_set[index])
    test_items = np.array(test_items)
    n_pairs = test_items.shape[0]

    predictions = np.zeros((n_pairs, test_items.shape[1] + 1))
    predictions[:, :4] = test_items

    if algorithm == "MF":
        for pair in range(n_pairs):
            predictions[pair, 4] = predict_svd_pair(
                test_items[pair, 0].astype(int),
                test_items[pair, 1].astype(int),
                global_mean,
                bu, bi, pu, qi,
            )

    elif algorithm == "kNN":
        for pair in range(n_pairs):
            predictions[pair, 4] = predict_pair(
                test_items[pair, 0].astype(int),
                test_items[pair, 1].astype(int),
                x_rated, S, k, min_k,
                x_list, y_list,
                global_mean,
            )

    np.clip(predictions[:, 4], 0.5, 5, out=predictions[:, 4])
    return predictions
