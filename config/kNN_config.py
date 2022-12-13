from taipy import Config, Scope

x_id_cfg = Config.configure_data_node(id="x_id", default_data=1)
n_min_k_cfg = Config.configure_data_node(id="n_min_k", default_data=10)
n_k_neighboor_cfg = Config.configure_data_node(id="n_k_neighboor", default_data=1)

sim_measure_cfg = Config.configure_in_memory_data_node(
    id="sim_measure", default_data="pcc"
)
list_ur_ir_cfg = Config.configure_data_node(
    id="list_ur_ir", cope=Scope.GLOBAL, cacheable=True
)
similarity_matrix_cfg = Config.configure_data_node(
    id="similarity_matrix", cope=Scope.GLOBAL, cacheable=True
)
items_id_cfg = Config.configure_data_node(
    id="x_list", cope=Scope.GLOBAL, cacheable=True
)
users_id_cfg = Config.configure_data_node(
    id="y_list", cope=Scope.GLOBAL, cacheable=True
)
