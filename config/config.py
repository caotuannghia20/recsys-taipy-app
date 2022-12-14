from taipy import Config, Scope
from functions.funtions import preprocess_data, fit, predict
from config.svd_config import (
    n_epochs_cfg,
    n_factors_cfg,
    learning_rate_cfg,
    qi_cfg,
    bi_cfg,
    bu_cfg,
    pu_cfg,
)
from config.kNN_config import (
    x_id_cfg,
    n_k_neighboor_cfg,
    n_min_k_cfg,
    sim_measure_cfg,
    list_ur_ir_cfg,
    similarity_matrix_cfg,
    users_id_cfg,
    items_id_cfg,
)
TRAIN_DATA_PATH = "dataset/rating_train.csv"
TEST_DATA_PATH = "dataset/rating_test.csv"
MOVIE_DATA_PATH = "dataset/movie.csv"

train_dataset_cfg = Config.configure_data_node(
    id="train_dataset",
    storage_type="csv",
    path=TRAIN_DATA_PATH,
    scope=Scope.GLOBAL,
    cacheable=True,
)

test_dataset_cfg = Config.configure_data_node(
    id="test_dataset",
    storage_type="csv",
    path=TEST_DATA_PATH,
    scope=Scope.GLOBAL,
    cacheable=True,
)
movie_name_cfg = Config.configure_data_node(
    id="movie_name",
    storage_type="csv",
    path=MOVIE_DATA_PATH,
    scope=Scope.GLOBAL,
    cacheable=True,
)

trainset_cfg = Config.configure_data_node(
    id="trainset", scope=Scope.PIPELINE, cacheable=True)

testset_cfg = Config.configure_data_node(
    id="testset", scope=Scope.PIPELINE, cacheable=True)

true_testset_movies_id_cfg = Config.configure_data_node(
    id="true_testset_movies_id", scope=Scope.PIPELINE, cacheable=True
)
algorithm_cfg = Config.configure_in_memory_data_node(
    id="algorithm", default_data="kNN")
global_mean_cfg = Config.configure_data_node(
    id="global_mean", cope=Scope.GLOBAL, cacheable=True
)
predictions_cfg = Config.configure_data_node(
    id="predictions", scope=Scope.PIPELINE)

# Config task
load_data_task_cfg = Config.configure_task(
    id="load_data",
    function=preprocess_data,
    input=[train_dataset_cfg, test_dataset_cfg, movie_name_cfg],
    output=[trainset_cfg, testset_cfg, true_testset_movies_id_cfg],
)

train_data_task_cfg = Config.configure_task(
    id="train_data",
    function=fit,
    input=[
        trainset_cfg, sim_measure_cfg,
        n_factors_cfg, n_epochs_cfg, learning_rate_cfg,
        algorithm_cfg,
    ],
    output=[
        similarity_matrix_cfg, list_ur_ir_cfg, users_id_cfg, items_id_cfg, global_mean_cfg,
        pu_cfg, qi_cfg, bu_cfg, bi_cfg,
    ],
)

predict_task_cfg = Config.configure_task(
    id="predict_task",
    function=predict,
    input=[
        testset_cfg,
        x_id_cfg, list_ur_ir_cfg, similarity_matrix_cfg, n_k_neighboor_cfg, n_min_k_cfg, users_id_cfg, items_id_cfg,
        global_mean_cfg, true_testset_movies_id_cfg,
        bu_cfg, bi_cfg, pu_cfg, qi_cfg,
        algorithm_cfg,
    ],
    output=predictions_cfg,
)


# Config pipeline
pipeline_cfg = Config.configure_pipeline(
    id="pipeline",
    task_configs=[load_data_task_cfg, train_data_task_cfg, predict_task_cfg],
)
