from taipy import Config, Scope
n_factors_cfg = Config.configure_data_node(id="n_factors", default_data=30)

n_epochs_cfg = Config.configure_data_node(id="n_epochs", default_data=50)

learning_rate_cfg = Config.configure_data_node(
    id="learning_rate", default_data=0.001)

qi_cfg = Config.configure_data_node(id="qi", cope=Scope.GLOBAL, cacheable=True)

pu_cfg = Config.configure_data_node(id="pu", cope=Scope.GLOBAL, cacheable=True)

bu_cfg = Config.configure_data_node(id="bu", cope=Scope.GLOBAL, cacheable=True)

bi_cfg = Config.configure_data_node(id="bi", cope=Scope.GLOBAL, cacheable=True)
