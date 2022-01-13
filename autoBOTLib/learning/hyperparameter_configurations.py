import numpy as np

torch_sparse_nn_ff_basic = {
    "batch_size": [16],
    "num_epochs": [100],
    "learning_rate": [0.01, 0.001, 0.005],
    "stopping_crit": [5],
    "hidden_layer_size": [64, 128, 256],
    "num_hidden": [1, 2, 4],
    "dropout": [0.01, 0.1, 0.3, 0.4],
    "device": ["cpu"]
}

scikit_default = {
    "loss": ["hinge", "log"],
    "penalty": ["elasticnet"],
    "alpha": [0.01, 0.001, 0.0001],
    "l1_ratio": [0, 0.1, 0.5, 0.9]
}

scikit_intense = {
    "loss": ["log"],
    "penalty": ["elasticnet"],
    "power_t": [0.1, 0.2, 0.3, 0.4, 0.5],
    "class_weight": ["balanced"],
    "n_iter_no_change": [8],
    "alpha": [0.0005],
    "l1_ratio": [0, 0.05, 0.25, 0.3, 0.6, 0.8, 0.95, 1]
}

scikit_intense_final = {
    "loss": ["hinge", "log", "modified_huber"],
    "penalty": ["elasticnet"],
    "power_t": np.arange(0.05, 0.5, 0.05).tolist(),
    "class_weight": ["balanced"],
    "n_iter_no_change": [8, 32],
    "alpha":
    [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001, 0.00005],
    "l1_ratio": np.arange(0, 1, 0.02).tolist()
}

scikit_generic_final = {
    "loss": ["hinge", "log", "modified_huber"],
    "penalty": ["elasticnet"],
    "power_t": [0.1, 0.2, 0.3, 0.4, 0.5],
    "class_weight": ["balanced"],
    "n_iter_no_change": [8, 32],
    "alpha": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
    "l1_ratio": [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
}

scikit_mini_l1 = {
    "loss": ["log"],
    "penalty": ["l1"]
}

scikit_mini_l2 = {
    "loss": ["log"],
    "penalty": ["l2"]
}

scikit_knn = {
    "n_neighbors": list(range(1, 64, 1)),
    "weights": ['uniform', 'distance'],
    "metric": ["euclidean", "manhattan", "minkowski"]
}
