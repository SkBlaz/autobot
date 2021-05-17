"""
This is the main GA underlying the autoBOT approach.
This file contains, without warranty, the code that performs the optimization.
Made by Blaz Skrlj, Ljubljana 2020, Jozef Stefan Institute
"""

## some generic logging
import logging
import wget  # For conceptnet download if necessary
logging.basicConfig(format = '%(asctime)s - %(message)s',
                    datefmt = '%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

## evolution helpers -> this needs to be global for proper persistence handling. If there is as better way, please open a pull request!
from deap import base, creator, tools
global gcreator
gcreator = creator
gcreator.create("FitnessMulti", base.Fitness, weights = (1.0, ))
gcreator.create("Individual", list, fitness = creator.FitnessMulti)

import operator

## feature space construction
from .feature_constructors import *
from .metrics import *
from collections import defaultdict
from scipy import sparse
import requests  ## for downloading the KG

## modeling
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

## monitoring
import tqdm
import numpy as np
import itertools
import time
import os

## more efficient computation
import multiprocessing as mp

## omit some redundant warnings
from warnings import simplefilter
simplefilter(action = 'ignore')

## relevant for visualization purposes, otherwise can be omitted.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    logging.info("Visualization libraries were found (NetworkX, plt, sns).")

except:

    logging.info("For full visualization, Networkx and Matplotlib are needed!")


class GAlearner:
    """
    The core GA class. It includes methods for evolution of a learner assembly.
    Each instance of autoBOT must be first instantiated.
    In general, the workflow for working with this class is as follows:
    1.) Instantiate the class
    2.) Evolve
    3.) Predict
    """
    def __init__(self,
                 train_sequences_raw,
                 train_targets,
                 time_constraint,
                 num_cpu = "all",
                 task_name = "update:",
                 latent_dim = 512,
                 sparsity = 0.1,
                 hof_size = 3,
                 initial_separate_spaces = True,
                 scoring_metric=None,
                 top_k_importances = 25,
                 representation_type = "neurosymbolic-default",
                 binarize_importances = False,
                 memory_storage = "memory",
                 classifier = None,
                 n_fold_cv = 6,
                 random_seed = 8954,
                 classifier_hyperparameters = None,
                 custom_transformer_pipeline = None,
                 combine_with_existing_representation = False,
                 conceptnet_url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
                 default_importance = 0.05,
                 classifier_preset = "default",
                 include_concept_features = False,
                 verbose = 1):
        
        """The object initialization method

        :param train_sequences_raw: a list of texts
        :param train_targets: a list of natural numbers (targets, multiclass)
        :param time_constraint: Number of hours to evolve.
        :param num_cpu: Number of threads to exploit
        :param task_name: Task identifier for logging
        :param latent_dim: The latent dimension of embeddings
        :param sparsity: The assumed sparsity of the induced space (see paper)
        :param hof_size: Hof many final models to consider?
        :param initial_separate_spaces: Whether to include separate spaces as part of the initial population.
        :param scoring_metric: The type of metric to optimize (sklearn-compatible)
        :param top_k_importances: How many top importances to remember for explanations.
        :param representation_type: symbolic, neurosymbolic or custom
        :param binarize_importances: Feature selection instead of ranking as explanation
        :param memory_storage: The storage of conceptnet.txt.gz-like triplet database
        :param classifier: custom classifier. If none, linear learners are used.
        :param classifier_hyperparameters: The space to be optimized w.r.t. the classifier param.
        :param conceptnet_url: URL of the conceptnet used.
        :param classifier_preset: Type of classification to be considered (default = paper), mini -> very lightweight regression, emphasis on space exploration.
        :param include_concept_features: Whether to include external background knowledge if possible
        :param default_importance: Minimum possible initial weight.

        """
        
        ## Set the random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logo = """

      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWM
      WWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWM
      xokKNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWMMMMMMMMMM
      l'';cdOKNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNMMMMMMMMMM
      d'.''',;ldOXWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
      x'...''''',c0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWMMMWXXMMMMMMMWWMMMMMMMMMM
      k,....''''';kWMMMMMMMMMMMMMMMMMMMMMMMWNXWMN00NMMXk0NMMMMWXKNMMMMMMMMMM
      0;......''';kWNK00000OOOOOOOOOOOKNXK0xx0WN0kONMWX00NMMMMN00XMMMMMWNNMM
      K:........';OWx;;;;;;;;;;:::::ccldxxxxk0NNXXNWMMMMMMMMMMMWNWMMMMMWNNMM
      Xc.........;OWx,',,,,,;;;;;:::::ccoxkOxdkKkxk0NWMMMMMMMMMMMMMMMMMMMMMM
      Nl.........,OWk,''',,,,,;;;;;::::lxkkkOOKNK0KXX00NMMMWWMMMMMMMMMMMMMMM
      Wd.........,OWk,''''',,,,,;;;;;::oddllx0KKNMWKxoxKMMWKKWMMMMMMMMMMMMMM
      Wx.........,OMXxc;''''',,,,,;;;;:okkk0NN0xx0NN0OKWMMNkONMMWXNMMMMMMMMM
      MO'........,OMMWWKkoc,''',,,,,;;cddld0N00NXOkKNNWMMMWXKNMWKk0WMMMMMMMM
      M0, .......,OWOkXWMMN0ko:,,,,,,,;cox0Kxcl0XkoxOXWMMMMMMMMWNXXWMMMMMMMM
      MK;   .....,ONo.;dKWMMMWX0xl:,,,;oOKX0xddk00KNMMMMWWMMMMMMMMMMWNWMMMMM
      MXc     ...'ONl...,lONMMMMMWXOddkOxxkKX0xooONMMMWKxkNMMMMMMMMN0OXMMMMM
      MNl.      .'ONl.....'ckXWMMMMMMWWWNNNWWMWXXWMMMWKdcdXMMMMMMMMWNNWMMMMM
      MWd.       'ONl........;dKWMMMMMMMMMMMMMMMMMMMMWKkx0WMMNKXMMMMMMMMMMMM
      MMx.       'OWOoc;'......,oONMMMMMMMMMWWMW00NWMMMMMMMMW0dxKWMMMMMMMMMM
      MMO.      .;0MMWWNX0kxoc:,',ckXWMMMMWKdlxKOllxKWMWWMMMWXKXNWMMMMMMMMMM
      MM0,    .:xKWMNkdx0XWMMWNXKOxdkKWMWWXkkO0XNK00XWWOkNMMMMMMMMMMMMMMMMMM
      MMX; 'cxKWN0x0WKc..';ldk0XWMMMWWWMMMMMMMMMMMMMMNk:lKMMN0kOKWMMMMMMMMMM
      MMWOOXWN0d;. 'dNXo.  ....,:ldkKXWWXXKO0KXNMMMMMNOxd0WMWOcoKMMMMMMMMMMM
      MMMMW0d;.     .:0Nx'.  .......';oxc,:oxl:OXOkOOXWMWWMMMN0KWMMMMMMMMMMM
      MMMW0c.         'xXO,.  .........:ooloxxclkl,cxXWMN0KWMMMMMMMMMMMMMMMM
      MMMMMNKxl,.      .cK0:.   .......:lc,.cxodOOkXWMMNkccdKWMMMMMMMMMMMMMM
      MMMMMMMMMNKxl;.    ,k0l.    .....'okdodlcxKWWWMMMWNXK00NMMMMMMMMMMMMMM
      MMMMMMMMMMMMMWKkl;. .l0d.     ...:k0xclONWNkldONMMMMMMMMMMMMMMMMMMMMMM
      MMMMMMMMMMMMMMMMMWKkl:lkx,.    ..:x0d:ckNNOdx0NWMMMMMMMMMMMMMMMMMMMMMM
      MMMMMMMMMMMMMMMMMMMMMN0OXO:,,,;;o0K0kocdXNNWMMMMMMMMMMMMMMMMMMMMMMMMMM
        """

        self.verbose = verbose
        if self.verbose: print(logo)
        self.default_importance = default_importance
        self.classifier_preset = classifier_preset
        
        if self.verbose: logging.info("Instantiated the evolution-based learner.")
        self.scoring_metric = scoring_metric

        self.representation_type = representation_type
        self.custom_transformer_pipeline = custom_transformer_pipeline
        self.combine_with_existing_representation = combine_with_existing_representation
        self.initial_separate_spaces = initial_separate_spaces

        if not self.custom_transformer_pipeline is None:
            if self.verbose: logging.info("Using custom feature transformations.")

        self.binarize_importances = binarize_importances
        self.latent_dim = latent_dim
        self.sparsity = sparsity

        ## Dict of labels to int
        self.label_mapping, self.inverse_label_mapping = self.get_label_map(train_targets)

        ## Encoded target space for training purposes
        train_targets = np.array(self.apply_label_map(train_targets))
        counts = np.bincount(train_targets)
        self.majority_class = np.argmax(counts)
        self.classifier = classifier
        self.classifier_hyperparameters = classifier_hyperparameters

        ## parallelism settings
        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
        else:
            self.num_cpu = num_cpu

        if self.verbose: logging.info(f"Using {self.num_cpu} cores.")

        self.task = task_name
        self.topk = top_k_importances

        train_sequences = []

        ## Do some mandatory encoding
        if type(train_sequences_raw) == list:
            for sequence in train_sequences_raw:
                train_sequences.append(
                    sequence.encode("utf-8").decode("utf-8"))
        else:
            for sequence in train_sequences_raw.values:
                train_sequences.append(
                    sequence.encode("utf-8").decode("utf-8"))

        ## build dataframe
        self.train_seq = self.return_dataframe_from_text(train_sequences)
        self.train_targets = train_targets

        self.hof = []  ## The hall of fame

        self.memory_storage = memory_storage  ## Path to the memory storage
        self.include_concept_features = include_concept_features

        if self.include_concept_features:
            
            if not os.path.exists(self.memory_storage):
                
                if self.verbose:
                    logging.info(f"Attempting to download the knowledge graph. Folder: {self.memory_storage}")
                    
                try:

                    if self.verbose: logging.info(
                        "Memory (ConceptNet) not found! Downloading into ./memory folder. Please wait a few minutes (a few hundred MB is being downloaded)."
                    )

                    os.mkdir("./memory")

                    wget.download(conceptnet_url, out="memory")
                    fname = list(
                        os.walk("memory"))[0][2][0]  ## Get the new file name

                    self.memory_storage = f"memory/{fname}"
                    
                    if self.verbose: logging.info(
                        f"Memory storage downloaded: {self.memory_storage}")

                except Exception as es:
                    if self.verbose: logging.info(
                        f"ConceptNet could not be downloaded. Please download it and store it as memory/conceptnet.txt.gz. Omitting this feature type.{es}"
                    )
                    self.include_concept_features = False

            else:

                try:
                    cnames = list(
                        os.walk(self.memory_storage))[0][2][0]
                    
                    self.memory_storage = self.memory_storage+"/"+cnames

                except Exception as es:

                    if self.verbose: logging.info(f"Could not find the knowledge graph memory storage. Please do the following: \n a) Check if there is empty memory folder. \n b) Download manually into the created memory folder via: {conceptnet_url} \n c) Check if the file is not corrupted. \n autoBOT will now continue without the ConceptNet-based features!")
                    
                    self.include_concept_features = False
            
        self.population = None  ## this object gets evolved

        ## establish constraints
        self.max_time = time_constraint
        self.unique_labels = len(set(train_targets))
        self.initial_time = None
        self.subspace_feature_names = None
        self.ensemble_of_learners = []
        self.performance_reports = []
        self.n_fold_cv = n_fold_cv

        if self.verbose: logging.info(
            "Initiating the seed vectorizer instance and initial feature space .."
        )

        ## hyperparameter space. Parameters correspond to weights of subspaces, as well as subsets + regularization of LR.

        #self.weight_params = self.weight_params

        ## other hyperparameters
        self.hof_size = hof_size  ## size of the hall of fame.

        if self.hof_size % 2 == 0:
            if self.verbose: logging.info(
                "HOF size must be odd, adding one member ({}).".format(
                    self.hof_size))
            self.hof_size += 1

        self.fitness_container = []  ## store fitness across evalution

        ## stats
        self.feature_importances = []
        self.fitness_max_trace = []
        self.fitness_mean_trace = []
        self.feat_min_trace = []
        self.feat_mean_trace = []
        self.opt_population = None

        if self.verbose: logging.info(
            "Loaded a dataset of {} texts with {} unique labels.".format(
                self.train_seq.shape[0], len(set(train_targets))))

        if self.scoring_metric is None:
            if self.unique_labels > 2:
                self.scoring_metric = "f1_macro"

            else:
                self.scoring_metric = "f1"


    def get_label_map(self, train_targets):

        """
        Identify unique target labels and remember them.
        
        :param train_targets: The training target space (or any other for that matter)
        :returns label_map, inverse_label_map: Two dicts, mapping to and from encoded space suitable for autoML loopings.

        """
        
        unique_train_target_labels = set(train_targets)
        label_map = {}
        for enx, j in enumerate(unique_train_target_labels):
            label_map[j] = enx

        inverse_label_map = {y:x for x,y in label_map.items()}
        return label_map, inverse_label_map

    def apply_label_map(self, targets, inverse = False):

        """
        A simple mapping back from encoded target space.
        
        :param targets: The target space
        :param inverse: Boolean if map to origin space or not (default encodes into continuum)
        :returns new_targets: Encoded target space

        """
        if inverse:
            new_targets = [self.inverse_label_mapping[x] for x in targets]
            
        else:
            new_targets = [self.label_mapping[x] for x in targets]
            
        return new_targets
                
    def update_global_feature_importances(self):
        """
        Aggregate feature importances across top learners to obtain the final ranking.
        """

        fdict = {}
        self.sparsity_coef = []

        ## get an indicator of global feature space and re-map.
        global_fmaps = defaultdict(list)
        for enx, importance_tuple in enumerate(self.feature_importances):
            subspace_features = importance_tuple[1]
            coefficients = importance_tuple[0]
            assert len(subspace_features) == len(coefficients)
            sparsity_coef = np.count_nonzero(coefficients) / len(coefficients)
            self.sparsity_coef.append(sparsity_coef)
            if self.verbose: logging.info("Importance (learner {}) sparsity of {}".format(
                enx, sparsity_coef))

            for fx, coef in zip(subspace_features, coefficients):
                space_of_the_feature = self.global_feature_name_hash[fx]

                if not fx in fdict:
                    fdict[fx] = np.abs(coef)

                else:
                    fdict[fx] += np.abs(coef)
                global_fmaps[space_of_the_feature].append((fx, coef))

        self.global_feature_map = {}
        for k, v in global_fmaps.items():
            tmp = {}
            for a, b in v:
                tmp[a] = round(b, 2)
            mask = ["x"] * self.topk
            top5 = [
                " : ".join([str(y) for y in x]) for x in sorted(
                    tmp.items(), key = operator.itemgetter(1), reverse = True)
            ][0:self.topk]
            mask[0:len(top5)] = top5
            self.global_feature_map[k] = mask
        self.global_feature_map = pd.DataFrame(self.global_feature_map)
        self.sparsity_coef = np.mean(self.sparsity_coef)
        self._feature_importances = sorted(fdict.items(),
                                           key = operator.itemgetter(1),
                                           reverse = True)
        if self.verbose: logging.info(
            "Feature importances can be accessed by ._feature_importances")

    def compute_time_diff(self):
        """
        A method for approximate time monitoring.
        """

        return ((time.time() - self.initial_time) / 60) / 60

    def prune_redundant_info(self):
        """
        A method for removing redundant additional info which increases the final object's size.                
        """
        
        self.fitness_container = []
        self.feature_importances = []
        if self.verbose: logging.info("Cleaned fitness and importances, the object should be smaller now.")
    
    def parallelize_dataframe(self, df, func):
        """
        A method for parallel traversal of a given dataframe.

        :param df: dataframe of text (Pandas object)
        :param func: function to be executed (a function)
        """

        if self.verbose: logging.info("Computing the seed dataframe ..")

        ## Do a pre-split of the data and compute in parallel.
        df_split = np.array_split(df, self.num_cpu * 10)
        pool = mp.Pool(self.num_cpu)
        df = pd.concat(
            tqdm.tqdm(pool.imap(func, df_split), total = len(df_split)))

        pool.close()
        pool.join()

        return df

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x.

        :param: x: (vector of floats)
        """

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def return_dataframe_from_text(self, text):
        """
        A helper method that returns a given dataframe from text.

        :param text: list of texts.
        :return parsed df: A parsed text (a DataFrame)
        """

        return self.parallelize_dataframe(text, build_dataframe)

    def generate_random_initial_state(self, weights_importances):
        """
        The initialization method, capable of generation of individuals.
        """

        weights = np.random.uniform(low = 0.6, high = 1,
                                    size = self.weight_params).tolist()
        weights[0:len(weights_importances)] = weights_importances
        generic_individual = np.array(weights)

        assert len(generic_individual) == self.weight_params
        return generic_individual

    def custom_initialization(self):
        """
        Custom initialization employs random uniform prior. See the paper for more details.
        """
                
        if self.verbose: logging.info("Performing initial screening on {} subspaces.".format(
            len(self.feature_subspaces)))

        performances = []
        for subspace, name in zip(self.feature_subspaces, self.feature_names):
            f1, _ = self.cross_val_scores(subspace, n_cpu = self.num_cpu)
            performances.append(f1)

        pairs = [
            " -- ".join([str(y) for y in x])
            for x in list(zip(self.feature_names, performances))
        ]

        print("Initial screening report:")
        print("\n".join(pairs))

        weights = np.array(performances) / max(performances)
        generic_individual = self.generate_random_initial_state(weights)
        assert len(generic_individual) == self.weight_params
        for ind in self.population:
            noise = np.random.uniform(low = 0.95,
                                      high = 1.05,
                                      size = self.weight_params)
            generic_individual = generic_individual * noise + self.default_importance
            ind[:] = np.abs(generic_individual)

        ## Separate spaces -- each subspace massively amplified
        if self.initial_separate_spaces:
            for k in range(self.weight_params):
                individual = self.population[0]
                individual[:] = np.zeros(self.weight_params)
                individual[k] = 1 ## amplify particular subspace.
                self.population.append(individual)

    def apply_weights(self,
                      parameters,
                      custom_feature_space = False,
                      custom_feature_matrix = None):
        """
        This method applies weights to individual parts of the feature space.

        :param parameters: a vector of real-valued parameters - solution = an individual
        :param custom_feature_space: Custom feature space, relevant during making of predictions.
        :return tmp_space: Temporary weighted space (individual)
        """

        ## Compute cumulative sum across number of features per feature type.
        indices = self.intermediary_indices

        ## Copy the space as it will be subsetted.
        if not custom_feature_space:
            tmp_space = sparse.csr_matrix(self.train_feature_space.copy())
        else:
            tmp_space = sparse.csr_matrix(custom_feature_matrix)

        indices_pairs = []
        assert len(indices) == self.weight_params + 1

        for k in range(self.weight_params):
            i1 = indices[k]
            i2 = indices[k + 1]
            indices_pairs.append((i1, i2))

        ## subset the core feature matrix -- only consider non-neural features for this.
        for j, pair in enumerate(indices_pairs):
            tmp_space[:,
                      pair[0]:pair[1]] = tmp_space[:,
                                                   pair[0]:pair[1]].multiply(
                                                       parameters[j])

        return tmp_space

    def cross_val_scores(self, tmp_feature_space, final_run = False, n_cpu = None):
        """
        Compute the learnability of the representation.
        
        :param tmp_feature_space: An individual's solution space.
        :param final_run: Last run is more extensive.
        :param n_cpu: Number of CPUs to use.
        :return f1_perf, clf: F1 performance and the learned classifier.
        """

        if self.classifier_hyperparameters is None:

            if final_run:
                ## we can afford this final round to be more rigorous.
                parameters = {
                    "loss": ["hinge", "log"],
                    "penalty": ["elasticnet"],
                    "alpha": [0.01, 0.001, 0.0001, 0.0005],
                    "l1_ratio": [0, 0.05, 0.25, 0.3, 0.6, 0.8, 0.95]
                }

            else:
                ## this is for screening purposes.

                if self.classifier_preset == "default":
                    parameters = {
                        "loss": ["hinge", "log"],
                        "penalty": ["elasticnet"],
                        "alpha": [0.01, 0.001, 0.0001],
                        "l1_ratio": [0, 0.1, 0.5, 0.9]
                    }
                    
                elif self.classifier_preset == "mini-l1":
                     parameters = {
                        "loss": ["log"],
                        "penalty": ["l1"]
                    }

                elif self.classifier_preset == "mini-l2":
                     parameters = {
                        "loss": ["log"],
                        "penalty": ["l2"]
                    }
                    
        else:

            parameters = self.classifier_hyperparameters

        if self.classifier is None:
            svc = SGDClassifier()

        else:
            svc = self.classifier

        performance_score = self.scoring_metric

        if final_run:
            clf = GridSearchCV(svc,
                               parameters,
                               verbose = self.verbose,
                               n_jobs = self.num_cpu,
                               cv = self.n_fold_cv,
                               scoring = performance_score,
                               refit = True)

        else:
            clf = GridSearchCV(svc,
                               parameters,
                               verbose = self.verbose,
                               n_jobs = self.num_cpu,
                               cv = self.n_fold_cv,
                               scoring = performance_score,
                               refit = False)

        clf.fit(tmp_feature_space, self.train_targets)
        f1_perf = max(clf.cv_results_['mean_test_score'])

        if final_run:
            report = clf.cv_results_
            #clf = clf.best_estimator_
            return f1_perf, clf, report

        return f1_perf, clf

    def evaluate_fitness(self,
                         individual,
                         max_num_feat = 1000,
                         return_clf_and_vec = False):
        """
        A helper method for evaluating an individual solution. Given a real-valued vector, this constructs the representations and evaluates a given learner.

        :param individual: an individual (solution)
        :param max_num_feat: maximum number of features that are outputted
        :param return_clf_and_vec: return classifier and vectorizer? This is useful for deployment.
        :return score: The fitness score.
        """
        individual = np.array(individual)
        if np.sum(individual[:]) > self.weight_params:
            return (0, )

        if (np.array(individual) <= 0).any():
            individual[(individual < 0)] = 0

        if self.binarize_importances:
            for k in range(len(self.feature_names)):
                weight = individual[k]
                if weight > 0.5:
                    individual[k] = 1
                else:
                    individual[k] = 0

        if self.vectorizer:

            tmp_feature_space = self.apply_weights(individual[:])
            feature_names = self.all_feature_names

            ## Return the trained classifier.
            if return_clf_and_vec:

                ## fine tune final learner
                if self.verbose: logging.info("Final round of optimization.")
                f1_perf, clf, report = self.cross_val_scores(tmp_feature_space,
                                                             final_run=True)
                return clf, individual[:], f1_perf, feature_names, report

            f1_perf, _ = self.cross_val_scores(tmp_feature_space)
            return (f1_perf, )

        elif return_clf_and_vec:
            return (0, )

        else:
            return (0, )

    def generate_and_update_stats(self, fits):
        """
        A helper method for generating stats.

        :param fits: fitness values of the current population
        """

        f1_scores = []

        for fit in fits:

            f1_scores.append(fit)

        return np.mean(f1_scores)

    def report_performance(self, fits, gen = 0):
        """
        A helper method for performance reports.

        :param fits: fitness values (vector of floats)
        :param gen: generation to be reported (int)
        """

        f1_top = self.generate_and_update_stats(fits)
        if self.verbose: logging.info(r"{} (gen {}) {}: {}, time: {}min".format(
                self.task, gen, self.scoring_metric, np.round(f1_top, 3),
            np.round(self.compute_time_diff(), 2) * 60))
        return f1_top

    def get_feature_space(self):
        """
        Extract final feature space considered for learning purposes.
        """
        transformed_instances, feature_indices = self.apply_weights(
            self.hof[0][1:])
        assert transformed_instances.shape[0] == len(self.train_targets)
        return (transformed_instances, self.train_targets)

    def predict_proba(self, instances):
        """
        Predict on new instances. Note that the prediction is actually a maxvote across the hall-of-fame.

        :param instances: predict labels for new instances = texts.
        """
        
        if self.verbose: logging.info("Obtaining final predictions from {} models.".format(
            len(self.ensemble_of_learners)))
        
        if not self.ensemble_of_learners:
            if self.verbose: logging.info("Please, evolve the model first!")
            return None

        else:

            instances = self.return_dataframe_from_text(instances)
            transformed_instances = self.vectorizer.transform(instances)
            prediction_space = []

            # transformed_instances = self.update_intermediary_feature_space(custom_space = transformed_instances)
            if self.verbose: logging.info("Representation obtained ..")
            for learner_tuple in self.ensemble_of_learners:

                try:

                    ## get the solution.
                    learner, individual, score = learner_tuple
                    learner = learner.best_estimator_

                    ## Subset the matrix.
                    subsetted_space = self.apply_weights(
                        individual,
                        custom_feature_space = True,
                        custom_feature_matrix = transformed_instances)

                    ## obtain the predictions.
                    if not prediction_space is None:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                    else:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                except Exception as es:
                    print(
                        es,
                        "Please, re-check the data you are predicting from!")

            ## generate the prediction matrix by maximum voting scheme.
            pspace = np.matrix(prediction_space).T
            np.nan_to_num(pspace, copy = False, nan = self.majority_class)
            all_predictions = self.probability_extraction(pspace) ## Most common prediction is chosen.
            if self.verbose: logging.info("Predictions obtained")
            return all_predictions

    def probability_extraction(self, pred_matrix):

        """
        Predict probabilities for individual classes. Probabilities are based as proportions of a particular label predicted with a given classifier.
        
        :param pred_matrix: Matrix of predictions.
        :return prob_df: A DataFrame of probabilities for each class.

        """

        ## identify individual class labels
        pred_matrix = np.asarray(pred_matrix)
        unique_values = np.unique(pred_matrix).tolist()
        prediction_matrix_final = []
        for k in range(pred_matrix.shape[0]):
            pred_row = np.asarray(pred_matrix[k,:])
            assert len(pred_row) == pred_matrix.shape[1]
            counts = np.bincount(pred_row)
            probability_vector = []                
            for p in range(len(unique_values)):
                if p+1 <= len(counts):
                    prob = counts[p]
                else:
                    prob = 0
                probability_vector.append(prob)
            assert len(probability_vector) == len(unique_values)
            
            prediction_matrix_final.append(probability_vector)
        final_matrix = np.array(prediction_matrix_final)
        prob_df = pd.DataFrame(final_matrix)        
        prob_df.columns = self.apply_label_map(unique_values, inverse=True)

        ## It's possible some labels are never predicted!
        all_possible_labels = list(self.label_mapping.keys())
        for l in all_possible_labels:
            if not l in prob_df.columns:
                prob_df[l] = 0.0

        ## Normalization
        prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
        csum = prob_df.sum(axis = 1).values
        zero_index = np.where(csum == 0)[0]        
        for j in zero_index:
            prob_df.iloc[j,self.majority_class] = 1
        prob_df = prob_df.fillna(0)        
        assert len(np.where(prob_df.sum(axis = 1) < 1)[0]) == 0
        
        return prob_df
        
    def predict(self, instances):
        """
        Predict on new instances. Note that the prediction is actually a maxvote across the hall-of-fame.

        :param instances: predict labels for new instances = texts.
        :returns all_predictions: Vector of predictions (decoded)

        """
        
        if self.verbose: logging.info("Obtaining final predictions from {} models.".format(
            len(self.ensemble_of_learners)))

        if not self.ensemble_of_learners:
            if self.verbose: logging.info("Please, evolve the model first!")
            return None

        else:

            instances = self.return_dataframe_from_text(instances)
            transformed_instances = self.vectorizer.transform(instances)
            prediction_space = []

            # transformed_instances = self.update_intermediary_feature_space(custom_space = transformed_instances)
            if self.verbose: logging.info("Representation obtained ..")
            for learner_tuple in self.ensemble_of_learners:

                try:

                    ## get the solution.
                    learner, individual, score = learner_tuple
                    learner = learner.best_estimator_

                    ## Subset the matrix.
                    subsetted_space = self.apply_weights(
                        individual,
                        custom_feature_space = True,
                        custom_feature_matrix = transformed_instances)

                    ## obtain the predictions.
                    if not prediction_space is None:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                    else:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                except Exception as es:
                    print(
                        es,
                        "Please, re-check the data you are predicting from!")

            ## generate the prediction matrix by maximum voting scheme.
            pspace = np.matrix(prediction_space).T
            np.nan_to_num(pspace, copy = False, nan = self.majority_class)
            all_predictions = self.mode_pred(pspace) ## Most common prediction is chosen.
            if self.verbose: logging.info("Predictions obtained")

            ## Transform back to origin space
            all_predictions = self.apply_label_map(all_predictions,
                                                   inverse = True)
            return all_predictions

    def mode_pred(self, prediction_matrix):

        """        
        Obtain most frequent elements for each row.
        
        :param prediction_matrix: Matrix of predictions.
        :return prediction_vector: Vector of aggregate predictions.

        """
        
        if prediction_matrix.ndim == 1:
            return prediction_matrix.reshape(-1).tolist()
        
        prediction_vector = []
        for k in range(len(prediction_matrix)):
            counts = np.bincount(np.asarray(prediction_matrix[k,:])[0])
            prediction = np.argmax(counts)
            prediction_vector.append(prediction)
            
        return prediction_vector

    def summarise_final_learners(self):

        performances = []
        for enx, top_learner in enumerate(self.ensemble_of_learners):
            top_learner, individual, scores = top_learner
            performance = top_learner.cv_results_
            performance_df = pd.DataFrame.from_dict(performance)
            performance_df['learner ID'] = f"Learner_{enx}"
            performances.append(performance_df)
        return pd.concat(performances, axis = 0)

    def generate_id_intervals(self):
        """
        Generate independent intervals.
        """

        reg_range = [0.1, 1, 10, 100]
        self.weight_params
        ks = [2]
        for k in ks:
            if k == 2:

                interval = [0, 1]
                layer_combs = list(
                    itertools.product(interval, repeat = self.weight_params - 1))

                np.random.shuffle(layer_combs)
                if self.verbose: logging.info(
                    "Ready to evaluate {} solutions at resolution: {}".format(
                        len(layer_combs) * len(reg_range), k))

                for comb in layer_combs:
                    for reg_val in reg_range:
                        otpt = np.array([reg_val] + list(comb))
                        yield otpt

    def get_feature_importance_report(self, individual, fitnesses):
        """Report feature importances.

        :param individual: an individual solution (a vector of floats)
        :param fitnesses: fitness space (list of reals)
        :return report: A prinout of current performance.

        """

        f1_scores = []

        if self.binarize_importances:
            for k in range(len(self.feature_names)):
                weight = individual[k]
                if weight > 0.5:
                    individual[k] = 1
                else:
                    individual[k] = 0

        for fit in fitnesses:
            f1_scores.append(fit[0])

        try:
            max_f1 = np.max(f1_scores)
        except:
            max_f1 = 0

        try:
            importances = list(
                zip(self.feature_names,
                    individual[0:self.weight_params].tolist()))

        except:
            importances = list(
                zip(self.feature_names, individual[0:self.weight_params]))

        report = ["-" * 60, "|| Feature type   Importance ||", "-" * 60]
        cnt = -1

        for fn, imp in importances:
            cnt += 1
            if len(str(fn)) < 17:
                fn = str(fn) + (17 - len(str(fn))) * " "
            report.append(str(fn) + "  " + str(np.round(imp, 2)))

        report.append("-" * 60)
        report.append("Max {}: {}".format(self.scoring_metric, max_f1))

        print("\n".join(report))

    def mutReg(self, individual, p = 1):
        """
        Custom mutation operator used for regularization optimization.

        :param individual: individual (vector of floats)
        :return individual: An individual solution.
        """

        individual[0] += np.random.random() * self.reg_constant
        return individual,

    def update_intermediary_feature_space(self, custom_space = None):
        """
        Create the subset of the origin feature space based on the starting_feature_numbers vector that gets evolved.
        """

        index_pairs = []
        for enx in range(len(self.initial_indices) - 1):
            diff1 = self.initial_indices[enx + 1] - self.initial_indices[enx]

            prop_diff = diff1
            i1 = int(self.initial_indices[enx])
            i2 = int(self.initial_indices[enx] + prop_diff)
            index_pairs.append((i1, i2))

        submatrices = []
        assert len(self.feature_names) == len(index_pairs)

        if not custom_space is None:
            considered_space = custom_space

        else:
            considered_space = self.train_feature_space

        fnames = []
        assert len(index_pairs) == len(self.feature_names)
        self.intermediary_indices = []

        for enx, el in enumerate(index_pairs):
            mx = considered_space[:, el[0]:el[1]]
            self.intermediary_indices.append(mx.shape[1])
            fnames += self.global_all_feature_names[el[0]:el[1]]
            submatrices.append(sparse.csr_matrix(mx))

        self.intermediary_indices = [0] + np.cumsum(
            self.intermediary_indices).tolist()
        assert len(submatrices) == len(self.feature_names)
        self.all_feature_names = fnames  ## this is the new set of features.
        output_matrix = sparse.hstack(submatrices).tocsr()
        if self.verbose: logging.info("Space update finished. {}, {} matrices joined.".format(
            output_matrix.shape, len(submatrices)))
        assert len(self.all_feature_names) == output_matrix.shape[1]

        if not custom_space is None:
            return output_matrix

        else:
            self.intermediary_feature_space = output_matrix

        del submatrices

    def instantiate_validation_env(self):
        """
        This method refreshes the feature space. This is needed to maximize efficiency.
        """

        self.vectorizer, self.feature_names, self.train_feature_space = get_features(
            self.train_seq,
            representation_type = self.representation_type,
            sparsity = self.sparsity,
            embedding_dim = self.latent_dim,
            targets = self.train_targets,
            random_seed = self.random_seed,
            memory_location = self.memory_storage,
            custom_pipeline = self.custom_transformer_pipeline,
            concept_features = self.include_concept_features,
            combine_with_existing_representation = self.combine_with_existing_representation)

        self.all_feature_names = []
        if self.verbose: logging.info("Initialized training matrix of dimension {}".format(
            self.train_feature_space.shape))

        self.feature_space_tuples = []
        self.global_feature_name_hash = {}

        ## This information is used to efficiently subset and index the sparse representation
        self.feature_subspaces = []
        current_fnum = 0
        for transformer in self.vectorizer.named_steps[
                'union'].transformer_list:
            features = transformer[1].steps[1][1].get_feature_names()
            self.feature_subspaces.append(
                self.train_feature_space[:, current_fnum:(current_fnum +
                                                          len(features))])
            current_fnum += len(features)
            self.all_feature_names += features
            num_feat = len(features)
            for f in features:
                self.global_feature_name_hash[f] = transformer[0]
            self.feature_space_tuples.append((transformer[0], num_feat))

        self.global_all_feature_names = self.all_feature_names
        self.intermediary_indices = [0] + np.cumsum(
            np.array([x[1] for x in self.feature_space_tuples])).tolist()

    def feature_type_importances(self, solution_index = 0):
        """
        A method which prints feature type importances as a pandas df.

        :param solution_index: Which consequent individual to inspect.
        :return feature_ranking: Final table of rankings
        """
        feature_importances = self.hof[
            solution_index]  ## global importances .feature_names rabi
        struct = []
        for a, b in zip(feature_importances, self.feature_names):
            struct.append((str(a), str(b)))
        dfx = pd.DataFrame(struct)  # Create a Pandas dataframe
        dfx.columns = ['Importance', 'Feature subspace']
        ## store global top features
        feature_ranking = self.global_feature_map  ## all features
        return feature_ranking, dfx

    def visualize_fitness(self, image_path = "fitnessExample.png"):
        """
        A method for visualizing fitness.

        :param image_path: Path to file, ending denotes file type. If set to None, only DataFrame of statistics is returned.
        :return dfx: DataFrame of evolution evaluations
        """

        try:
            fitness_string = self.fitness_container
            fdf = []
            ids = []
            for enx, el in enumerate(fitness_string):
                ids.append(enx)
                parts = [str(x[0]) for x in el]
                fdf.append(parts)
            dfx = pd.DataFrame(fdf)
            dfx = dfx.astype(float)

            if image_path is None:
                return dfx

            mean_fitness = dfx.mean(axis = 1).values.tolist()
            sns.lineplot(list(range(len(mean_fitness))), mean_fitness)
            plt.xlabel("Generation")
            plt.ylabel(f"Mean fitness ({self.scoring_metric})")
            plt.tight_layout()
            plt.savefig(image_path, dpi = 300)

        except Exception as es:
            if self.verbose: logging.info(es)

        return dfx

    def evolve(self,
               nind = 10,
               crossover_proba = 0.4,
               mutpb = 0.15,
               stopping_interval = 20,
               strategy = "evolution",
               validation_type = "cv"):
        """The core evolution method. First constrain the maximum number of features to be taken into account by lowering the bound w.r.t performance.
        next, evolve.

        :param nind: number of individuals (int)
        :param crossover_proba: crossover probability (float)
        :param mutpb: mutation probability (float)
        :param stopping_interval: stopping interval -> for how long no improvement is tolerated before a hard reset (int)
        :param strategy: type of evolution (str)
        :param validation_type: type of validation, either train_val or cv (cross validation or train-val split)
        """

        self.validation_type = validation_type
        self.initial_time = time.time()
        self.popsize = nind
        self.instantiate_validation_env()

        if self.verbose: logging.info("Evolution will last for ~{}h ..".format(self.max_time))

        if self.verbose: logging.info("Selected strategy is evolution.")

        self.toolbox = base.Toolbox()
        self.weight_params = len(self.feature_names)
        self.toolbox.register("attr_float", np.random.uniform, 0.00001, 0.999999)
        self.toolbox.register("individual",
                              tools.initRepeat,
                              gcreator.Individual,
                              self.toolbox.attr_float,
                              n = self.weight_params)

        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual,
                              n = nind)

        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("mate", tools.cxUniform, indpb = 0.5)
        self.toolbox.register("mutate",
                              tools.mutGaussian,
                              mu = 0,
                              sigma = 0.2,
                              indpb = 0.2)

        self.toolbox.register("mutReg", self.mutReg)
        self.toolbox.register("select", tools.selTournament)

        ## Keep the best-performing individuals
        self.hof = tools.HallOfFame(self.hof_size)
        if self.verbose: logging.info("Total number of subspace importance parameters {}".format(
            self.weight_params))

        ## Population initialization
        if self.population == None:

            self.population = self.toolbox.population()
            self.custom_initialization()  ## works on self.population
            if self.verbose: logging.info("Initialized population of size {}".format(
                len(self.population)))
            if self.verbose: logging.info("Computing initial fitness ..")

        ## Gather fitness values.
        fits = list(map(self.toolbox.evaluate, self.population))

        for fit, ind in zip(fits, self.population):
            ind.fitness.values = fit

        self.report_performance(fits)
        self.hof.update(self.population)
        gen = 0
        if self.verbose: logging.info("Initiating evaluation ..")

        stopping = 1
        cf1 = 0

        ## Start the evolution.
        while True:

            gen += 1
            tdiff = self.compute_time_diff()

            if tdiff >= self.max_time:
                break

            offspring = list(map(self.toolbox.clone, self.population))

            ## Perform crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if np.random.random() < crossover_proba:

                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            ## Perform mutation
            for mutant in offspring:

                if np.random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            fits = list(map(self.toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                if isinstance(fit, int) and not isinstance(fit, tuple):
                    fit = (fit, )
                ind.fitness.values = fit

            self.hof.update(offspring)

            ## append to overall fitness container.
            self.fitness_container.append(fits)

            self.get_feature_importance_report(self.hof[0], fits)

            f1 = self.report_performance(fits, gen = gen)

            if f1 == cf1:
                stopping += 1

            else:
                cf1 = f1

            self.population = self.toolbox.select(self.population + offspring,
                                                  k = nind,
                                                  tournsize = int(nind / 3))

        try:
            selections = self.hof

        except:
            selections = self.population

        self.selections = [np.array(x).tolist() for x in selections]

        ## Ensemble of learners is finally filled and used for prediction.
        for enx, top_individual in enumerate(selections):

            if len(top_individual) == 1:
                top_individual = top_individual[0]

            try:
                learner, individual, score, feature_names, report = self.evaluate_fitness(
                    top_individual, return_clf_and_vec = True)
                self.performance_reports.append(report)

            except Exception as es:
                if self.verbose: logging.info(
                    f"Evaluation of individual {top_individual} did not produce a viable learner. Increase time!")

            coefficients = learner.best_estimator_.coef_

            ## coefficients are given for each class. We take maximum one (abs val)
            coefficients = np.asarray(np.abs(np.max(coefficients,
                                                    axis = 0))).reshape(-1)

            if self.verbose: logging.info("Coefficients and indices: {}".format(
                len(coefficients)))
            if self.verbose: logging.info(
                "Adding importances of shape {} for learner {} with score {}".
                format(coefficients.shape, enx, score))

            self.feature_importances.append((coefficients, feature_names))

            single_learner = (learner, individual, score)
            self.ensemble_of_learners.append(single_learner)

        ## Update the final importance space.
        self.update_global_feature_importances()
        return self
