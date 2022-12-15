import numpy as np
from numba import njit


def compute_similarity_matrix(train_set, sim_measure="pcc"):
    x_rated, _, n_x, _, x_list, y_list = list_ur_ir(train_set)
    if sim_measure == "pcc":
        print("Computing similarity matrix as pcc...")
        S = pcc(n_x, x_rated, min_support=1)
    elif sim_measure == "cosine":
        print("Computing similarity matrix as cosine...")
        S = cosine(n_x, x_rated, min_support=1)
    else:
        S = 0
    return S, x_rated, x_list, y_list


def fit_train_set(train_set):
    X = train_set.copy()
    x_list = np.unique(X[:, 0])  # For uuCF, x -> user
    y_list = np.unique(X[:, 1])  # For uuCF, y -> item
    n_x = len(x_list)
    n_y = len(y_list)
    return n_x, n_y, X, x_list, y_list


def list_ur_ir(train_set):
    n_x, n_y, X, x_list, y_list = fit_train_set(train_set)
    x_rated = [
        [] for _ in range(n_y)
    ]  # List where element `i` is ndarray of `(x, rating)` where `x` is all x that rated y, and the ratings.
    y_ratedby = [
        [] for _ in range(n_x)
    ]  # List where element `i` is ndarray of `(y, rating)` where `y` is all y that rated by x, and the ratings.

    for xid, yid, r in X:
        x_rated[int(yid)].append([xid, r])
        y_ratedby[int(xid)].append([yid, r])

    for yid in range(n_y):
        x_rated[yid] = np.array(x_rated[yid])
    for xid in range(n_x):
        y_ratedby[xid] = np.array(y_ratedby[xid])
    return x_rated, y_ratedby, n_x, n_y, x_list, y_list


def pcc(n_x, yr, min_support=1):
    """Compute the Pearson coefficient correlation between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
    """
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    si = np.zeros((n_x, n_x), np.double)
    sj = np.zeros((n_x, n_x), np.double)

    for y_ratings in yr:
        prods, freq, sqi, sqj, si, sj = run_pearson_params(
            prods, freq, sqi, sqj, si, sj, y_ratings
        )

    sim = calculate_pearson_similarity(
        prods, freq, sqi, sqj, si, sj, n_x, min_support)

    return sim


@njit
def run_pearson_params(prods, freq, sqi, sqj, si, sj, y_ratings):
    for xi, ri in y_ratings:
        xi = int(xi)
        for xj, rj in y_ratings:
            xj = int(xj)
            freq[xi, xj] += 1
            prods[xi, xj] += ri * rj
            sqi[xi, xj] += ri**2
            sqj[xi, xj] += rj**2
            si[xi, xj] += ri
            sj[xi, xj] += rj

    return prods, freq, sqi, sqj, si, sj


@njit
def calculate_pearson_similarity(prods, freq, sqi, sqj, si, sj, n_x, min_sprt):
    sim = np.zeros((n_x, n_x), np.double)

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt(
                    (n * sqi[xi, xj] - si[xi, xj] ** 2)
                    * (n * sqj[xi, xj] - sj[xi, xj] ** 2)
                )
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


def predict_knn_pair(x_id, y_id, x_rated, S, k, min_k, x_list, y_list, global_mean):
    x_known, y_known = False, False
    if x_id in x_list:
        x_known = True
    if y_id in y_list:
        y_known = True

    if not (x_known and y_known):
        # if uuCF:
        #     print(f"Can not predict rating of user {x_id} for item {y_id}.")
        # else:
        #     print(f"Can not predict rating of user {y_id} for item {x_id}.")
        return global_mean

    return predict_knn(x_id, y_id, x_rated[y_id], S, k, min_k)


def predict_knn(x_id, y_id, x_rated, S, k, k_min):
    k_neighbors = np.zeros((k, 2))
    k_neighbors[:, 1] = -1  # All similarity degree is default to -1

    for x2, rating in x_rated:
        if int(x2) == x_id:
            continue  # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (sim, r) in k_neighbors:
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * r
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

        return est
    return 0


def cosine(n_x, yr, min_support=1):
    """Compute the cosine similarity between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
    """
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)

    for y_ratings in yr:
        prods, freq, sqi, sqj = run_cosine_params(
            prods, freq, sqi, sqj, y_ratings)

    sim = calculate_cosine_similarity(prods, freq, sqi, sqj, n_x, min_support)

    return sim


@njit
def run_cosine_params(prods, freq, sqi, sqj, y_ratings):
    for xi, ri in y_ratings:
        xi = int(xi)
        for xj, rj in y_ratings:
            xj = int(xj)
            freq[xi, xj] += 1
            prods[xi, xj] += ri * rj
            sqi[xi, xj] += ri**2
            sqj[xi, xj] += rj**2

    return prods, freq, sqi, sqj


@njit
def calculate_cosine_similarity(prods, freq, sqi, sqj, n_x, min_sprt):
    sim = np.zeros((n_x, n_x), np.double)

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


@njit
def calculate_precision_recall(user_ratings, k, threshold):
    """Calculate the precision and recall at k metric for the user based on his/her obversed rating and his/her predicted rating.

    Args:
        user_ratings (ndarray): An array contains the predicted rating in the first column and the obversed rating in the second column.
        k (int): the k metric.
        threshold (float): relevant threshold.

    Returns:
        (precision, recall): the precision and recall score for the user.
    """
    # Sort user ratings by estimated value
    # user_ratings = user_ratings[user_ratings[:, 0].argsort()][::-1]

    # Number of relevant items
    n_rel = 0
    for _, true_r in user_ratings:
        if true_r >= threshold:
            n_rel += 1

    # Number of recommended items in top k
    n_rec_k = 0
    for est, _ in user_ratings[:k]:
        if est >= threshold:
            n_rec_k += 1

    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = 0
    for (est, true_r) in user_ratings[:k]:
        if true_r >= threshold and est >= threshold:
            n_rel_and_rec_k += 1

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    if n_rec_k != 0:
        precision = n_rel_and_rec_k / n_rec_k
    else:
        precision = 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.
    if n_rel != 0:
        recall = n_rel_and_rec_k / n_rel
    else:
        recall = 0

    return precision, recall
