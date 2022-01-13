# Some Torch-based FFNNs - Skrlj 2021

import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from autoBOTLib.learning.hyperparameter_configurations import torch_sparse_nn_ff_basic
import logging
import numpy as np
from scipy import sparse

RANDOM_SEED = 123321
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):
    """
    A generic toch dataframe loader adapted for sparse (CSR) matrices.
    """
    def __init__(self, features, targets=None):
        self.features = sparse.csr_matrix(features)
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
            return instance, target
        else:
            target = None
            return instance


def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1, 1))


class HyperParamNeuralObject:
    """Meta learning object governing hyperparameter optimization"""
    def __init__(self,
                 hyperparam_space,
                 verbose=0,
                 device="cpu",
                 metric="f1_macro"):

        self.verbose = verbose
        self.device = device
        self.metric = metric
        self.hyperparam_space = hyperparam_space
        self.history = []
        hyperparam_space['device'] = [self.device]

    def fit(self, X, Y, refit=False, n_configs=1):
        """A generic fit method, resembling GridSearchCV"""

        score_global = 0
        top_config = None

        for x in range(n_configs):
            config = get_random_config(self.hyperparam_space)
            if self.verbose:
                logging.info(
                    f"Current iteration: {x}, current optimum: {score_global}")

            score = cross_val_score_nn(config, X, Y, self.metric)
            config[self.metric] = score
            self.history.append(config)
            if score > score_global:
                score_global = score
                top_config = config
                if self.verbose:
                    logging.info(
                        f"Found a better config: {config} with score: {score}")

                if score_global == 1:
                    logging.info("Found a perfect fit (beware).")
                    break

        if self.verbose:
            logging.info(f"The best config found: {top_config}")

        if refit:
            if self.verbose:
                logging.info("Refit in progress ..")
            clf = SFNN(**top_config)
            clf.fit(X, Y)
            self.best_estimator_ = clf

        else:
            clf = None

        self.best_score = score_global


class GenericFFNN(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_layer_size,
                 num_hidden=2,
                 dropout=0.02,
                 device="cuda"):
        super(GenericFFNN, self).__init__()

        self.device = device
        self.latent_dim = hidden_layer_size
        self.latent_first = nn.Linear(input_size, hidden_layer_size)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.coefficient_first = nn.Parameter(
            torch.randn(input_size, requires_grad=True))
        self.coefficient_softmax = nn.Softmax()
        self.coefficient_bn = nn.BatchNorm1d(input_size)

        latent_space = []
        for j in range(num_hidden):
            latent_space.append(nn.Linear(hidden_layer_size,
                                          hidden_layer_size))
            latent_space.append(nn.BatchNorm1d(hidden_layer_size))
            latent_space.append(nn.Dropout(dropout))
            latent_space.append(nn.SELU())

        self.latent_layers = nn.ModuleList(latent_space)

        # Final output layer
        self.output_layer = nn.Linear(hidden_layer_size, num_classes)

    def hadamard_act(self, x):

        x = x.reshape(x.shape[0], -1)
        out = self.coefficient_bn(x)
        out = torch.mul(self.coefficient_first, out)
        out = self.coefficient_softmax(out)
        return out

    def forward(self, x):

        out = self.hadamard_act(x)
        out = self.latent_first(x)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.output_layer(out)
        out = self.sigmoid(out)
        return out

    def get_importances(self):

        out = self.coefficient_softmax(self.coefficient_first)
        return out


class SFNN:
    def __init__(self,
                 batch_size=32,
                 num_epochs=32,
                 learning_rate=0.001,
                 stopping_crit=10,
                 hidden_layer_size=64,
                 dropout=0.2,
                 num_hidden=2,
                 device="cpu",
                 verbose=0,
                 *args,
                 **kwargs):
        self.device = device
        self.verbose = verbose
        self.loss = torch.nn.BCELoss()
        self.dropout = dropout
        self.batch_size = int(batch_size)
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None
        self.loss_trace = []

    def __str__(self):
        print(self.__dict__)
        return str(self)

    def fit(self, features, labels):

        nun = len(np.unique(labels))
        one_hot_labels = []
        for j in range(len(labels)):
            lvec = np.zeros(nun)
            lj = labels[j]
            lvec[lj] = 1
            one_hot_labels.append(lvec)
        one_hot_labels = np.matrix(one_hot_labels)
        if self.verbose:
            logging.info("Found {} unique labels.".format(nun))
        train_dataset = E2EDatasetLoader(features, one_hot_labels)
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=1)
        stopping_iteration = 0
        current_loss = np.inf

        if features.shape[1] > self.hidden_layer_size:
            self.hidden_layer_size = int(features.shape[1] * 0.75)

        self.model = GenericFFNN(features.shape[1],
                                 num_classes=nun,
                                 hidden_layer_size=self.hidden_layer_size,
                                 num_hidden=self.num_hidden,
                                 dropout=self.dropout,
                                 device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        if self.verbose:
            logging.info("Number of parameters {}".format(self.num_params))
            logging.info("Starting training for {} epochs".format(
                self.num_epochs))
            logging.info(self.model)
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                if self.verbose:
                    logging.info("Stopping reached!")
                break
            losses_per_batch = []
            self.model.train()
            for i, (features, labels) in enumerate(dataloader):
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                outputs = self.model(features)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))
            mean_loss = np.mean(losses_per_batch)
            if mean_loss < current_loss:
                current_loss = mean_loss
                stopping_iteration = 0
            else:
                stopping_iteration += 1
            if self.verbose:
                logging.info("epoch {}, mean loss per batch {}".format(
                    epoch, mean_loss))
            self.loss_trace.append(mean_loss)

        self.get_importances(features)

    def predict(self, features, return_proba=False):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        with torch.no_grad():
            for features in test_dataset:
                self.model.eval()
                features = features.float().to(self.device)
                representation = self.model(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        if not return_proba:
            a = [np.argmax(a_) for a_ in predictions]
            return np.array(a).flatten()
        else:
            a = [a_ for a_ in predictions]
            return a

    def get_importances(self, features):

        all_importances = self.model.get_importances().cpu().detach().numpy()
        self.coef_ = all_importances

    def predict_proba(self, features):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        a = [a_[1] for a_ in predictions]
        return np.array(a).flatten()


def cross_val_score_nn(config, X, Y, metric="f1_macro"):
    """A method which performs cross-validation and returns top score"""

    folds = StratifiedShuffleSplit(n_splits=2,
                                   test_size=0.1,
                                   random_state=RANDOM_SEED)
    scores = []
    scorer = metrics.get_scorer(metric)
    for train_index, test_index in folds.split(X, Y):
        train_instances = X[train_index]
        test_instances = X[test_index]
        train_targets = Y[train_index].reshape(-1)
        test_targets = Y[test_index].reshape(-1)
        clf = SFNN(**config)
        clf.fit(train_instances, train_targets)
        preds = clf.predict(test_instances)
        score = scorer._score_func(test_targets, preds)
        scores.append(score)
    return np.mean(scores)


def get_random_config(cdict):
    """Select a random hyperparam configuration"""

    conf_dict = {}
    for k, v in cdict.items():
        conf_dict[k] = np.random.choice(v)
    return conf_dict


def hyper_opt_neural(X,
                     Y,
                     refit=True,
                     verbose=1,
                     device="cpu",
                     learner_preset="default",
                     metric="f1_macro"):
    """Generic hyperoptimization routine"""

    hyper_opt_obj = HyperParamNeuralObject(torch_sparse_nn_ff_basic,
                                           verbose=verbose,
                                           device=device,
                                           metric=metric)

    if learner_preset == "default":
        n_configs = 3

    elif learner_preset == "intense":
        n_configs = 10

    else:
        n_configs = 2

    hyper_opt_obj.fit(X, Y, refit, n_configs)
    score_global = hyper_opt_obj.best_score

    return score_global, hyper_opt_obj


def torch_learners(final_run,
                   X,
                   Y,
                   custom_hyperparameters,
                   learner_preset,
                   learner,
                   task,
                   metric,
                   num_folds,
                   validation_percentage,
                   random_seed,
                   verbose,
                   validation_type,
                   num_cpu,
                   device="cpu"):
    """A method for searching the architecture space"""
    print(final_run)
    score, clf = hyper_opt_neural(X,
                                  Y,
                                  refit=final_run,
                                  verbose=verbose,
                                  device=device,
                                  learner_preset=learner_preset,
                                  metric=metric)
    return score, clf


if __name__ == "__main__":

    import numpy as np
    from scipy.sparse import csr_matrix
    from numpy.random import default_rng
    from scipy.sparse import random
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = default_rng()
    rvs = stats.uniform().rvs
    a = random(10000, 30000, density=0.09, random_state=rng,
               data_rvs=rvs).tocsr()
    b = a[:, 0].multiply(a[:, 1])
    b = b.A
    b = np.where(b > 0.5, 0, 1)
    score, clf = hyper_opt_neural(a, b, refit=True, verbose=1)
    print(score, clf.best_estimator_.coef_)
