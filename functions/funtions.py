import pandas as pd
import numpy as np
from ultis.dataloader import DataLoader
from helper.svd_helper import sgd, predict_svd_pair
from helper.knn_helper import predict_pair, compute_similarity_matrix


def take_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, movie_name_cfg: pd.DataFrame
):
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
            learning_rate,
            learning_rate,
            learning_rate,
            learning_rate,
            learning_rate,
            learning_rate,
            learning_rate,
            learning_rate,
        )
        pu, qi, bu, bi, _ = sgd(
            X,
            pu,
            qi,
            bu,
            bi,
            n_epochs,
            global_mean,
            n_factors,
            lr_pu,
            lr_qi,
            lr_bu,
            lr_bi,
            reg_pu,
            reg_qi,
            reg_bu,
            reg_bi,
        )
        S, x_rated, x_list, y_list = [], [], [], []
    else:
        S, x_rated, x_list, y_list, pu, qi, bu, bi = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    return S, x_rated, x_list, y_list, global_mean, pu, qi, bu, bi


def predict(
    test_set_origin,
    x_id,
    x_rated,
    S,
    k,
    min_k,
    x_list,
    y_list,
    global_mean,
    true_testset_movies_id,
    bu,
    bi,
    pu,
    qi,
    algorithm="MF",
):
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
                bu,
                bi,
                pu,
                qi,
            )

    elif algorithm == "kNN":
        for pair in range(n_pairs):
            predictions[pair, 4] = predict_pair(
                test_items[pair, 0].astype(int),
                test_items[pair, 1].astype(int),
                x_rated,
                S,
                k,
                min_k,
                x_list,
                y_list,
                global_mean,
            )

    np.clip(predictions[:, 4], 0.5, 5, out=predictions[:, 4])
    return predictions
