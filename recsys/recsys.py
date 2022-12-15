import taipy as tp
from taipy.config import Config
from taipy.gui import notify, Markdown


import pandas as pd
import numpy as np

from helper.knn_helper import calculate_precision_recall
from config.config import pipeline_cfg

Config.configure_global_app(clean_entities_enabled=True)
tp.clean_all_entities()

scenario_cfg = Config.configure_scenario(
    id="scenario", pipeline_configs=pipeline_cfg)

dataset = pd.read_csv("dataset/data.csv")
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

sim_measure_selector = ["pcc", "cosine"]
selected_sim_measure = sim_measure_selector[0]

model_selector = ["kNN", "MF"]
selected_model = sim_measure_selector[0]
# set up parameter of kNN model
n_min_k, n_k_neighboor, x_id, top_k = 1, 10, 1, 1
# set up parameter of svd model
n_epochs, n_factors, learning_rate = 30, 40, 0.001

results, y_id, y_id_real, results_real, recall, precision = (
    None, None, None, None, None, None,)

all_scenarios = tp.get_scenarios()
[tp.delete(scenario.id) for scenario in all_scenarios if scenario.name is None]

scenario_selector = [(scenario.id, scenario.name)
                     for scenario in tp.get_scenarios()]


def create_scenario(state):
    global selected_scenario

    print("Creating scenario...")

    scenario = tp.create_scenario(scenario_cfg)

    scenario.sim_measure.write(str(state.selected_sim_measure))
    scenario.algorithm.write(str(state.selected_model))
    scenario.n_min_k.write(int(state.n_min_k))
    scenario.n_k_neighboor.write(int(state.n_k_neighboor))
    scenario.x_id.write(int(state.x_id))
    scenario.n_factors.write(int(state.n_factors))
    scenario.n_epochs.write(int(state.n_epochs))
    scenario.learning_rate.write(float(learning_rate))

    selected_scenario = scenario.id
    update_scenario_selector(state, scenario)
    tp.submit(scenario)


def submit_scenario(state):
    (
        state.y_id,
        state.y_id_real,
        state.results_real,
        state.recall,
        state.precision,
        state.results,
    ) = (
        None,
        None,
        [],
        None,
        None,
        [],
    )
    print("Submitting scenario...")
    # Get the selected scenario: in this current step a single scenario is created then modified here.
    scenario = tp.get(selected_scenario)

    # Change the default parameters by writing in the datanodes
    if scenario.sim_measure.read() != state.selected_sim_measure:
        scenario.sim_measure.write(str(state.selected_sim_measure))
    if scenario.n_min_k.read() != state.n_min_k:
        scenario.n_min_k.write(int(state.n_min_k))
    if scenario.n_k_neighboor.read() != state.n_k_neighboor:
        scenario.n_k_neighboor.write(int(state.n_k_neighboor))
    if scenario.x_id.read() != state.x_id:
        scenario.x_id.write(int(state.x_id))
    if scenario.n_factors.read() != state.n_factors:
        scenario.n_factors.write(int(state.n_factors))
    if scenario.n_epochs.read() != state.n_epochs:
        scenario.n_epochs.write(int(state.n_epochs))
    if scenario.learning_rate.read() != state.learning_rate:
        scenario.learning_rate.write(float(learning_rate))
    if scenario.algorithm.read() != state.selected_model:
        scenario.algorithm.write(str(state.selected_model))

    # Execute the pipelines/code
    tp.submit(scenario)


def update_scenario_selector(state, scenario):
    print("Updating scenario selector...")
    # Update the scenario selector
    state.scenario_selector += [(scenario.id, scenario.name)]


def take_all_movies_rated_by_x_id(test_set, x_id):

    test_items = []
    test_set = test_set.copy()
    for index in range(test_set.shape[0]):
        if test_set[index][0] == x_id:
            test_items.append(test_set[index])
    test_items = np.array(test_items)
    return test_items


def predicts(state):
    id, predict, id_real, predict_real, user_ratings = (
        [], [], [], [], [],)
    scenario = tp.get(selected_scenario)
    movie_name = (scenario.movie_name.read()).title.to_numpy()

    print("'Predict' button clicked")
    result = scenario.predictions.read()
    result = result[result[:, 4].argsort()[::-1]]

    if state.top_k > result.shape[0]:
        notify(
            state,
            notification_type="error",
            message="Out of range, top_k max = {}".format(result.shape[0]),
        )
        state.top_k = result.shape[0]
    top_k = state.top_k

    test_set = scenario.testset.read()
    true_testset_movies_id = scenario.true_testset_movies_id.read()

    test_set[:, 1] = true_testset_movies_id.T
    test_items = take_all_movies_rated_by_x_id(test_set, state.x_id)
    test_items = test_items[test_items[:, 2].argsort()[::-1]]

    for i in range(int(top_k)):
        id.append(int(result[i][3]))
        id_real.append(int(test_items[i][1]))

    for i, j in zip(id, id_real):
        predict.append(movie_name[(i - 1)])
        predict_real.append(movie_name[(j - 1)])

    state.y_id = np.array2string(
        np.array(id), precision=2, separator=", ", suppress_small=True)
    state.y_id_real = np.array2string(
        np.array(id_real), precision=2, separator=", ", suppress_small=True)
    state.results = predict
    state.results_real = predict_real

    for _, _, true_r, _, est in result:
        user_ratings.append([est, true_r])
    user_ratings = np.array(user_ratings)
    state.precision, state.recall = calculate_precision_recall(
        user_ratings, top_k, 3)


page_scenario_manager = Markdown("recsys/recsys.md")
