import numpy as np
from numba import njit


def sgd(
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
):
    for epoch_ix in range(n_epochs):
        pu, qi, bu, bi, train_loss = _run_svd_epoch(
            X,
            pu,
            qi,
            bu,
            bi,
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

    return pu, qi, bu, bi, train_loss


@njit
def _run_svd_epoch(
    X,
    pu,
    qi,
    bu,
    bi,
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
):
    """Runs an SVD epoch, updating model weights (pu, qi, bu, bi).

    Args:
        X (ndarray): the training set.
        pu (ndarray): users latent factor matrix.
        qi (ndarray): items latent factor matrix.
        bu (ndarray): users biases vector.
        bi (ndarray): items biases vector.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
        lr_pu (float, optional): Pu's specific learning rate.
        lr_qi (float, optional): Qi's specific learning rate.
        lr_bu (float, optional): bu's specific learning rate.
        lr_bi (float, optional): bi's specific learning rate.
        reg_pu (float, optional): Pu's specific regularization term.
        reg_qi (float, optional): Qi's specific regularization term.
        reg_bu (float, optional): bu's specific regularization term.
        reg_bi (float, optional): bi's specific regularization term.

    Returns:
        pu (ndarray): the updated users latent factor matrix.
        qi (ndarray): the updated items latent factor matrix.
        bu (ndarray): the updated users biases vector.
        bi (ndarray): the updated items biases vector.
        train_loss (float): training loss.
    """
    residuals = []
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        err = rating - pred
        residuals.append(err)

        # Update biases
        bu[user] += lr_bu * (err - reg_bu * bu[user])
        bi[item] += lr_bi * (err - reg_bi * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr_pu * (err * qif - reg_pu * puf)
            qi[item, factor] += lr_qi * (err * puf - reg_qi * qif)

    residuals = np.array(residuals)
    train_loss = np.square(residuals).mean()
    return pu, qi, bu, bi, train_loss


@njit
def predict_svd_pair(u_id, i_id, global_mean, bu, bi, pu, qi):
    """Returns the model rating prediction for a given user/item pair.

    Args:
        u_id (int): a user id.
        i_id (int): an item id.

    Returns:
        pred (float): the estimated rating for the given user/item pair.
    """
    user_known, item_known = False, False
    pred = global_mean

    if u_id != -1:
        user_known = True
        pred += bu[u_id]

    if i_id != -1:
        item_known = True
        pred += bi[i_id]

    if user_known and item_known:
        pred += np.dot(pu[u_id], qi[i_id])

    return pred
