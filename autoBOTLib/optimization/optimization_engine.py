# some generic logging
from warnings import simplefilter
import pickle
import multiprocessing as mp
import os
import time
import itertools
import numpy as np
from scipy import sparse
from collections import defaultdict, Counter
from autoBOTLib.optimization.optimization_metrics import *
from autoBOTLib.optimization.optimization_feature_constructors import *
from autoBOTLib.learning.scikit_based import scikit_learners
from autoBOTLib.learning.torch_sparse_nn import torch_learners
import operator
import copy
import gc
from deap import base, creator, tools
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)

# evolution helpers -> this needs to be global for proper persistence handling.
# If there is as better way, please open a pull request!

global gcreator
gcreator = creator
gcreator.create("FitnessMulti", base.Fitness, weights=(1.0, ))
gcreator.create("Individual", list, fitness=creator.FitnessMulti)

# omit some redundant warnings
simplefilter(action='ignore')

# relevant for visualization purposes, otherwise can be omitted.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

except Exception:    
    pass


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
                 num_cpu="all",
                 device="cpu",
                 task_name="Super cool task.",
                 latent_dim=512,
                 sparsity=0.1,
                 hof_size=1,
                 initial_separate_spaces=True,
                 scoring_metric=None,
                 top_k_importances=15,
                 representation_type="neurosymbolic",
                 binarize_importances=False,
                 memory_storage="memory",
                 learner=None,
                 n_fold_cv=5,
                 random_seed=8954,
                 learner_hyperparameters=None,
                 use_checkpoints=True,
                 visualize_progress=False,
                 custom_transformer_pipeline=None,
                 combine_with_existing_representation=False,
                 default_importance=0.05,
                 learner_preset="default",
                 task="classification",
                 contextual_model="all-mpnet-base-v2",
                 upsample=False,
                 verbose=1,
                 framework="scikit",
                 normalization_norm="l2",
                 validation_percentage=0.2,
                 validation_type="cv"):
        """The object initialization method; specify the core optimization
           parameter with this method.

        :param list/PandasSeries train_sequences_raw: a list of texts
        :param list/np.array train_targets: a list of natural numbers (targets, multiclass), a list of lists (multilabel)
        :param str device: Specification of the computation backend device
        :param int time_constraint: Number of hours to evolve.
        :param int/str num_cpu: Number of threads to exploit
        :param str task_name: Task identifier for logging
        :param int latent_dim: The latent dimension of embeddings
        :param float sparsity: The assumed sparsity of the induced space (see paper)
        :param int hof_size: Hof many final models to consider?
        :param bool initial_separate_spaces: Whether to include separate spaces as part of the initial population.
        :param str scoring_metric: The type of metric to optimize (sklearn-compatible)
        :param int top_k_importances: How many top importances to remember for explanations.
        :param str representation_type: "symbolic", "neural", "neurosymbolic", "neurosymbolic-default", "neurosymbolic-lite" or "custom". The "symbolic" feature space will only include feature types that we humans directly comprehend. The "neural" will include the embedding-based ones. The "neurosymbolic-default" will include the ones based on the origin MLJ paper, the "neurosymbolic" is the current alpha version with some new additions (constantly updated/developed). The "neurosymbolic-lite" version includes language-agnostic features but does not consider document graphs (due to space constraints)
        :param str framework: The framework used for obtaining the final models (torch, scikit)
        :param bool binarize_importances: Feature selection instead of ranking as explanation
        :param str memory_storage: The storage of the gzipped (TSV) triplets (SPO).
        :param obj learner: custom learner. If none, linear learners are used.
        :param obj learner_hyperparameters: The space to be optimized w.r.t. the learner param.
        :param int random_seed: The random seed used.
        :param str contextual_model: The language model string compatible with sentence-transformers library (this is in beta)
        :param bool visualize_progress: Progress visualization (progress.pdf, reqires MPL).
        :param str task: Either "classification" - SGDClassifier, or "regression" - SGDRegressor
        :param int n_fold_cv: The number of folds to be used for model evaluation.
        :param str learner_preset: Type of classification to be considered (default=paper), ""mini-l1"" or ""mini-l2" -> very lightweight regression, emphasis on space exploration.
        :param float default_importance: Minimum possible initial weight.
        :param bool upsample: Whether to equalize the number of instances by upsampling.
        :param float validation_percentage: The percentage of data to used as test set if validation_type="train_test"
        :param str validation_type: type of validation, either train_val or cv (cross validation or train-val split)
        """

        # Set the random seed
        self.random_seed = random_seed
        self.framework = framework
        self.upsample = upsample
        self.visualize_progress = visualize_progress
        self.validation_type = validation_type
        self.device = device
        self.validation_percentage = validation_percentage
        self.task = task
        self.use_checkpoints = use_checkpoints
        self.contextual_model = contextual_model
        self.multimodal_input = False
        np.random.seed(random_seed)
        self.verbose = verbose
        self.mlc_flag = False

        if self.upsample:
            train_sequences_raw, train_targets = self.upsample_dataset(
                train_sequences_raw, train_targets)
        
        if isinstance(train_sequences_raw[0], str):
            train_sequences_raw = [{"text_a": x.encode("utf-8")\
                                    .decode("utf-8")} for x in train_sequences_raw]
            
        else:
            for el in train_sequences_raw:
                el['text_a'] = el['text_a'].encode("utf-8").decode("utf-8")

        assert isinstance(train_sequences_raw[0], dict)

        if len(train_sequences_raw[0]) > 1:
            if self.verbose:
                logging.info(f"Considered multimodal autoBOT! Input types found: {train_sequences_raw[0].keys()}")
            self.multimodal = True

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

        if self.verbose:
            print(logo)
            logging.info(f"Considering preset: {representation_type}")
            logging.info(f"Considering learning framework: {self.framework}")
            
        self.default_importance = default_importance
        self.learner_preset = learner_preset
        self.scoring_metric = scoring_metric
        self.normalization_norm = normalization_norm
        self.representation_type = representation_type
        self.custom_transformer_pipeline = custom_transformer_pipeline
        self.combine_with_existing_representation \
            = combine_with_existing_representation
        self.initial_separate_spaces = initial_separate_spaces

        if self.custom_transformer_pipeline is not None:
            if self.verbose:
                logging.info("Using custom feature transformations.")

        self.binarize_importances = binarize_importances
        self.latent_dim = latent_dim
        self.sparsity = sparsity

        # Dict of labels to int
        self.label_mapping, self.inverse_label_mapping = self.get_label_map(
            train_targets)

        if self.verbose:
            logging.info("Instantiated the evolution-based learner.")
            self.summarise_dataset(train_sequences_raw, train_targets)

        if not isinstance(train_targets, list):

            try:
                train_targets = train_targets.tolist()

            except Exception as es:
                logging.info(
                    "Please make the targets either a list or a np.array!", es)

        # Encoded target space for training purposes
        if self.task == "classification":
            train_targets = np.array(self.apply_label_map(train_targets))
            counts = np.bincount(train_targets)
            self.majority_class = np.argmax(counts)

        else:
            train_targets = np.array(train_targets, dtype=np.float64)

        self.learner = learner
        self.learner_hyperparameters = learner_hyperparameters

        # parallelism settings
        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()

        else:
            self.num_cpu = num_cpu

        if self.verbose:
            logging.info(f"Using {self.num_cpu} cores.")

        self.task_name = task_name
        self.topk = top_k_importances

        self.train_seq = self.return_dataframe_from_text(train_sequences_raw)
        self.train_targets = train_targets

        self.hof = []  # The hall of fame

        self.memory_storage = memory_storage  # Path to the memory storage

        self.population = None  # this object gets evolved

        # establish constraints
        self.max_time = time_constraint
        self.unique_labels = len(set(train_targets))
        self.initial_time = None
        self.subspace_feature_names = None
        self.ensemble_of_learners = []
        self.n_fold_cv = n_fold_cv

        if self.verbose:
            logging.info(
                "Initiating the seed vectorizer instance and initial feature \
space ..")

        # hyperparameter space. Parameters correspond to weights of subspaces,
        # as well as subsets + regularization of LR.
        # other hyperparameters
        self.hof_size = hof_size  # size of the hall of fame.

        if self.hof_size % 2 == 0:
            if self.verbose:
                logging.info(
                    "HOF size must be odd, adding one member ({}).".format(
                        self.hof_size))
            self.hof_size += 1

        self.fitness_container = []  # store fitness across evalution

        # stats
        self.feature_importances = []
        self.fitness_max_trace = []
        self.fitness_mean_trace = []
        self.feat_min_trace = []
        self.feat_mean_trace = []
        self.opt_population = None

        if self.verbose:
            logging.info(
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

        :param list/np.array train_targets: The training target space (or any other for that matter)
        :return label_map, inverse_label_map: Two dicts, mapping to and from encoded space suitable for autoML loopings.

        """

        # Primitive MLC -> each subset is a possible label
        if isinstance(train_targets[0], list):

            self.mlc_flag = True
            train_targets = [str(x) for x in train_targets]

        unique_train_target_labels = set(train_targets)
        label_map = {}
        for enx, j in enumerate(unique_train_target_labels):
            label_map[j] = enx

        inverse_label_map = {y: x for x, y in label_map.items()}
        return label_map, inverse_label_map

    def apply_label_map(self, targets, inverse=False):
        """
        A simple mapping back from encoded target space.

        :param list/np.array targets: The target space
        :param bool inverse: Boolean if map to origin space or not (default encodes into continuum)
        :return list new_targets: Encoded target space

        """

        if inverse:
            new_targets = [self.inverse_label_mapping[x] for x in targets]

        else:

            if self.mlc_flag:
                targets = [str(x) for x in targets]

            new_targets = [self.label_mapping[x] for x in targets]

        return new_targets

    def update_global_feature_importances(self):
        """
        Aggregate feature importances across top learners to obtain the final ranking.
        """

        fdict = {}
        self.sparsity_coef = []

        # get an indicator of global feature space and re-map.
        global_fmaps = defaultdict(list)
        for enx, importance_tuple in enumerate(self.feature_importances):
            subspace_features = importance_tuple[1]
            coefficients = importance_tuple[0]
            assert len(subspace_features) == len(coefficients)
            sparsity_coef = np.count_nonzero(coefficients) / len(coefficients)
            self.sparsity_coef.append(sparsity_coef)
            if self.verbose:
                logging.info("Importance (learner {}) sparsity of {}".format(
                    enx, sparsity_coef))

            for fx, coef in zip(subspace_features, coefficients):
                space_of_the_feature = self.global_feature_name_hash[fx]

                if fx not in fdict:
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
                    tmp.items(), key=operator.itemgetter(1), reverse=True)
            ][0:self.topk]
            mask[0:len(top5)] = top5
            self.global_feature_map[k] = mask
        self.global_feature_map = pd.DataFrame(self.global_feature_map)
        self.sparsity_coef = np.mean(self.sparsity_coef)
        self._feature_importances = sorted(fdict.items(),
                                           key=operator.itemgetter(1),
                                           reverse=True)
        if self.verbose:
            logging.info(
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
        if self.verbose:
            logging.info(
                "Cleaned fitness and importances, the object should be smaller now."
            )

    def parallelize_dataframe(self, df, func):
        """
        A method for parallel traversal of a given dataframe.

        :param pd.DataFrame df: dataframe of text (Pandas object)
        :param obj func: function to be executed (a function)
        """

        if self.verbose:
            logging.info("Computing the seed dataframe ..")

        df = pd.concat(map(func, df))
        
        # Do a pre-split of the data and compute in parallel.
        # df_split = np.array_split(df, self.num_cpu * 10)
        # pool = mp.Pool(self.num_cpu)
        # df = pd.concat(
        #     tqdm.tqdm(pool.imap(func, df_split), total=len(df_split)))
        # pool.close()
        # pool.join()
        return df

    def upsample_dataset(self, X, Y):
        """
        Perform very basic upsampling of less-present classes.

        :param list X: Input list of documents
        :param np.array/list Y: Targets
        :return X,Y: Return upsampled data.
        """

        if self.verbose:
            logging.info("Performing upsampling ..")

        if not isinstance(X, list):
            X = X.values.tolist()

        if not isinstance(Y, list):
            Y = Y.values.tolist()

        extra_targets = []
        extra_instances = []
        counter_for_classes = Counter(Y)
        if self.verbose:
            for k, v in counter_for_classes.items():
                logging.info(f"Presence of class {k}; {v/len(Y)}")

        class_counts = {
            k: v
            for k, v in sorted(dict(counter_for_classes).items(),
                               key=lambda item: item[1])
        }
        classes = list(class_counts.keys())[::-1]
        most_frequent = classes[0]
        most_frequent_count = class_counts[most_frequent]
        for cname in classes[1:]:
            difference = most_frequent_count - class_counts[cname]
            if difference == 0:
                continue
            if self.verbose:
                logging.info(f"Upsampling for: {cname}; samples: {difference}")
            indices = [enx for enx, x in enumerate(Y) if x == cname]
            random_subspace = np.random.choice(indices, difference)
            for sample in random_subspace:
                extra_targets.append(Y[sample])
                extra_instances.append(X[sample])
        if self.verbose:
            logging.info(
                f"Generated {len(extra_instances)} new instances to balance the data."
            )
        X = X + extra_instances
        Y = Y + extra_targets
        return X, Y

    def return_dataframe_from_text(self, text):
        """
        A helper method that return a given dataframe from text.

        :param list/pd.Series text: list of texts.
        :return parsed df: A parsed text (a DataFrame)
        """

        return build_dataframe(text)

    def generate_random_initial_state(self, weights_importances):
        """
        The initialization method, capable of generation of individuals.
        """

        weights = np.random.uniform(low=0.6, high=1,
                                    size=self.weight_params).tolist()
        weights[0:len(weights_importances)] = weights_importances
        generic_individual = np.array(weights)

        assert len(generic_individual) == self.weight_params
        return generic_individual

    def summarise_dataset(self, list_of_texts, targets):

        list_of_texts = [x['text_a'] for x in list_of_texts]
        if not isinstance(targets, list):
            targets = targets.tolist()

        lengths = []
        unique_tokens = set()
        targets = [str(x) for x in targets]
        
        for x in list_of_texts:
            lengths.append(len(x))
            parts = x.strip().split()

            for part in parts:
                unique_tokens.add(part)

        logging.info(f"Number of documents: {len(list_of_texts)}")
        logging.info(f"Average document length: {np.mean(lengths)}")
        logging.info(f"Number of unique tokens: {len(unique_tokens)}")

        if len(set(targets)) < 200:
            logging.info(f"Unique target values: {set(targets)}")

    def custom_initialization(self):
        """
        Custom initialization employs random uniform prior. See the paper for more details.
        """

        if self.verbose:
            logging.info(
                "Performing initial screening on {} subspaces.".format(
                    len(self.feature_subspaces)))

        performances = []
        self.subspace_performance = {}
        for subspace, name in zip(self.feature_subspaces, self.feature_names):
            f1, _ = self.cross_val_scores(subspace)
            self.subspace_performance[name] = f1
            performances.append(f1)

        pairs = [
            " -- ".join([str(y) for y in x])
            for x in list(zip(self.feature_names, performances))
        ]

        if self.verbose:
            logging.info("Initial screening report follows.")

        for pair in pairs:
            if self.verbose:
                logging.info(pair)

        weights = np.array(performances) / max(performances) if len(performances) > 0 and max(performances) > 0 else np.ones(len(performances))
        generic_individual = self.generate_random_initial_state(weights)
        assert len(generic_individual) == self.weight_params
        for ind in self.population:
            noise = np.random.uniform(low=0.95,
                                      high=1.05,
                                      size=self.weight_params)
            generic_individual = generic_individual * noise \
                + self.default_importance
            ind[:] = np.abs(generic_individual)

        # Separate spaces -- each subspace massively amplified
        self.separate_individual_spaces = []

        # All weights set to one (this is the naive learning setting)
        unweighted = copy.deepcopy(self.population[0])
        unweighted[:] = np.ones(self.weight_params)
        self.separate_individual_spaces.append(unweighted)

        # Add separate spaces as solutions too
        if self.initial_separate_spaces:
            for k in range(self.weight_params):
                individual = copy.deepcopy(self.population[0])
                individual[:] = np.zeros(self.weight_params)
                individual[k] = 1  # amplify particular subspace.
                self.separate_individual_spaces.append(individual)

    def apply_weights(self,
                      parameters,
                      custom_feature_space=False,
                      custom_feature_matrix=None):
        """
        This method applies weights to individual parts of the feature space.

        :param np.array parameters: a vector of real-valued parameters - solution=an individual
        :param bool custom_feature_space: Custom feature space, relevant during making of predictions.
        :return np.array tmp_space: Temporary weighted space (individual)
        """

        # Compute cumulative sum across number of features per feature type.
        indices = self.intermediary_indices

        # Copy the space as it will be subsetted.
        if not custom_feature_space:
            # Use a more memory-efficient copy approach
            tmp_space = self.train_feature_space.copy()
            if sparse.issparse(tmp_space):
                tmp_space = sparse.csr_matrix(tmp_space)
            else:
                tmp_space = sparse.csr_matrix(tmp_space)

        else:
            tmp_space = sparse.csr_matrix(custom_feature_matrix)

        indices_pairs = []
        assert len(indices) == self.weight_params + 1

        for k in range(self.weight_params):
            i1 = indices[k]
            i2 = indices[k + 1]
            indices_pairs.append((i1, i2))

        # subset the core feature matrix -- only consider non-neural features for this.
        for j, pair in enumerate(indices_pairs):
            tmp_space[:,
                      pair[0]:pair[1]] = tmp_space[:,
                                                   pair[0]:pair[1]].multiply(
                                                       parameters[j])

        return tmp_space

    def cross_val_scores(self, tmp_feature_space, final_run=False):
        """
        Compute the learnability of the representation.

        :param np.array tmp_feature_space: An individual's solution space.
        :param bool final_run: Last run is more extensive.
        :return float performance_score, clf: F1 performance and the learned learner.
        """

        # Scikit-based learners
        if self.framework == "scikit":
            performance_score, clf = scikit_learners(
                final_run, tmp_feature_space, self.train_targets,
                self.learner_hyperparameters, self.learner_preset,
                self.learner, self.task, self.scoring_metric, self.n_fold_cv,
                self.validation_percentage, self.random_seed, self.verbose,
                self.validation_type, self.num_cpu)

        elif self.framework == "torch":
            performance_score, clf = torch_learners(
                final_run, tmp_feature_space, self.train_targets,
                self.learner_hyperparameters, self.learner_preset,
                self.learner, self.task, self.scoring_metric, self.n_fold_cv,
                self.validation_percentage, self.random_seed, self.verbose,
                self.validation_type, self.num_cpu, self.device)

        else:
            raise NotImplementedError(
                "Select either `torch` or `scikit` as the framework used.")

        return performance_score, clf

    def evaluate_fitness(self,
                         individual,
                         max_num_feat=1000,
                         return_clf_and_vec=False):
        """
        A helper method for evaluating an individual solution. Given a real-valued vector, this constructs the representations and evaluates a given learner.

        :param np.array individual: an individual (solution)
        :param int max_num_feat: maximum number of features that are outputted
        :param bool return_clf_and_vec: return learner and vectorizer? This is useful for deployment.
        :return float score: The fitness score.

        """
        individual = np.array(individual)
        if self.task == "classification":
            if np.sum(individual[:]) > self.weight_params:
                return (0, )

            if (np.array(individual) <= 0).any():
                individual[(individual < 0)] = 0

        else:
            individual = np.abs(individual)

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

            # Return the trained learner.
            if return_clf_and_vec:

                # fine tune final learner
                if self.verbose:
                    logging.info("Final round of optimization.")
                performance_score, clf = self.cross_val_scores(
                    tmp_feature_space, final_run=True)

                return clf, individual[:], performance_score, feature_names

            performance_score, _ = self.cross_val_scores(tmp_feature_space)

            return (performance_score, )

        elif return_clf_and_vec:

            return (0, )

        else:
            return (0, )

    def generate_and_update_stats(self, fits):
        """
        A helper method for generating stats.

        :param list fits: fitness values of the current population
        :return float meanScore: The mean of the fitnesses
        """

        f1_scores = []

        for fit in fits:

            f1_scores.append(fit)

        return np.mean(f1_scores)

    def report_performance(self, fits, gen=0):
        """
        A helper method for performance reports.

        :param np.array fits: fitness values (vector of floats)
        :param int gen: generation to be reported (int)
        """

        f1_top = self.generate_and_update_stats(fits)
        if self.verbose:
            logging.info(r"{} (gen {}) {}: {}, time: {}min".format(
                self.task_name, gen, self.scoring_metric, np.round(f1_top, 3),
                np.round(self.compute_time_diff(), 2) * 60))

        return f1_top

    def get_feature_space(self):
        """
        Extract final feature space considered for learning purposes.
        """

        transformed_instances, feature_indices = self.apply_weights(
            self.hof[0])
        assert transformed_instances.shape[0] == len(self.train_targets)
        return (transformed_instances, self.train_targets)

    def predict_proba(self, instances):
        """
        Predict on new instances. Note that the prediction is actually a maxvote across the hall-of-fame.

        :param list/pd.Series instances: predict labels for new instances=texts.
        """

        if self.verbose:
            logging.info("Obtaining final predictions from {} models.".format(
                len(self.ensemble_of_learners)))

        if not self.ensemble_of_learners:
            if self.verbose:
                logging.info("Please, evolve the model first!")
            return None

        else:

            instances = self.return_dataframe_from_text(instances)
            transformed_instances = self.vectorizer.transform(instances)
            prediction_space = []

            # transformed_instances=self.update_intermediary_feature_space(custom_space=transformed_instances)
            if self.verbose:
                logging.info("Representation obtained ..")
            for learner_tuple in self.ensemble_of_learners:

                try:

                    # get the solution.
                    learner, individual, score = learner_tuple
                    learner = learner.best_estimator_

                    # Subset the matrix.
                    subsetted_space = self.apply_weights(
                        individual,
                        custom_feature_space=True,
                        custom_feature_matrix=transformed_instances)

                    # obtain the predictions.
                    if prediction_space is not None:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                    else:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                except Exception as es:
                    print(
                        es,
                        "Please, re-check the data you are predicting from!")

            # generate the prediction matrix by maximum voting scheme.
            pspace = np.matrix(prediction_space).T
            np.nan_to_num(pspace, copy=False, nan=self.majority_class)
            all_predictions = self.probability_extraction(
                pspace)  # Most common prediction is chosen.
            if self.verbose:
                logging.info("Predictions obtained")
            return all_predictions

    def probability_extraction(self, pred_matrix):
        """
        Predict probabilities for individual classes. Probabilities are based as proportions of a particular label predicted with a given learner.

        :param np.array pred_matrix: Matrix of predictions.
        :return pd.DataFrame prob_df: A DataFrame of probabilities for each class.

        """

        # identify individual class labels
        pred_matrix = np.asarray(pred_matrix)
        unique_values = np.unique(pred_matrix).tolist()
        prediction_matrix_final = []

        for k in range(pred_matrix.shape[0]):

            pred_row = np.asarray(pred_matrix[k, :])
            assert len(pred_row) == pred_matrix.shape[1]
            counts = np.bincount(pred_row)
            probability_vector = []

            for p in range(len(unique_values)):

                if p + 1 <= len(counts):
                    prob = counts[p]

                else:
                    prob = 0

                probability_vector.append(prob)

            assert len(probability_vector) == len(unique_values)

            prediction_matrix_final.append(probability_vector)

        final_matrix = np.array(prediction_matrix_final)
        prob_df = pd.DataFrame(final_matrix)
        prob_df.columns = self.apply_label_map(unique_values, inverse=True)

        # It's possible some labels are never predicted!
        all_possible_labels = list(self.label_mapping.keys())
        for i in all_possible_labels:
            if i not in prob_df.columns:
                prob_df[i] = 0.0

        # Normalization
        prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)
        csum = prob_df.sum(axis=1).values
        zero_index = np.where(csum == 0)[0]

        for j in zero_index:
            prob_df.iloc[j, self.majority_class] = 1

        prob_df = prob_df.fillna(0)
        assert len(np.where(prob_df.sum(axis=1) < 1)[0]) == 0

        # Clean up temporary matrices
        if 'prediction_matrix_final' in locals():
            del prediction_matrix_final
        if 'transformed_instances' in locals():
            del transformed_instances
        gc.collect()

        return prob_df

    def transform(self, instances):
        """
        Generate only the representations (obtain a feature matrix subject to evolution in autoBOT)

        :param list/pd.DataFrame instances: A collection of instances to be transformed into feature matrix.
        :return sparseMatrix output_representation: Representation of the documents.

        """

        if self.vectorizer is None:
            if self.verbose:
                logging.info(
                    "Please call evolution() first to learn the representation\
 mappings.")

        instances = self.return_dataframe_from_text(instances)
        output_representation = self.vectorizer.transform(instances)

        return output_representation

    def predict(self, instances):
        """
        Predict on new instances. Note that the prediction is actually a maxvote across the hall-of-fame.

        :param list/pd.Series instances: predict labels for new instances=texts.
        :return np.array all_predictions: Vector of predictions (decoded)

        """

        if self.verbose:
            logging.info("Obtaining final predictions from {} models.".format(
                len(self.ensemble_of_learners)))

        if not self.ensemble_of_learners:
            if self.verbose:
                logging.info("Please, evolve the model first!")
            return None

        else:

            instances = self.return_dataframe_from_text(instances)
            transformed_instances = self.vectorizer.transform(instances)
            prediction_space = []

            # transformed_instances=self.update_intermediary_feature_space(custom_space=transformed_instances)
            if self.verbose:
                logging.info("Representation obtained ..")
            for learner_tuple in self.ensemble_of_learners:

                try:

                    # get the solution.
                    learner, individual, score = learner_tuple
                    learner = learner.best_estimator_

                    # Subset the matrix.
                    subsetted_space = self.apply_weights(
                        individual,
                        custom_feature_space=True,
                        custom_feature_matrix=transformed_instances)

                    # obtain the predictions.
                    if prediction_space is not None:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                    else:
                        prediction_space.append(
                            learner.predict(subsetted_space).tolist())

                except Exception as es:
                    print(
                        es,
                        "Please, re-check the data you are predicting from!")

            # generate the prediction matrix by maximum voting scheme.
            pspace = np.matrix(prediction_space).T
            if self.task == "classification":

                converged_predictions = np.where(
                    ~np.isnan(pspace).any(axis=0) == True)[0]
                pspace = pspace[:, converged_predictions]
                all_predictions = self.mode_pred(
                    pspace)  # Most common prediction is chosen.

                # Transform back to the origin space
                all_predictions = self.apply_label_map(all_predictions,
                                                       inverse=True)

            else:
                all_predictions = np.mean(pspace, axis=1).reshape(-1).tolist()

            if self.verbose:
                logging.info("Predictions obtained")

            # Clean up temporary matrices
            del transformed_instances
            if 'pspace' in locals():
                del pspace
            if 'subsetted_space' in locals():
                del subsetted_space
            gc.collect()

            return all_predictions

    def mode_pred(self, prediction_matrix):
        """
        Obtain most frequent elements for each row.

        :param np.array prediction_matrix: Matrix of predictions.
        :return np.array prediction_vector: Vector of aggregate predictions.

        """

        if prediction_matrix.ndim == 1:
            return prediction_matrix.reshape(-1).tolist()

        prediction_vector = []
        for k in range(len(prediction_matrix)):
            counts = np.bincount(np.asarray(prediction_matrix[k, :])[0])
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
        return pd.concat(performances, axis=0)

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
                    itertools.product(interval, repeat=self.weight_params - 1))

                np.random.shuffle(layer_combs)
                if self.verbose:
                    logging.info(
                        "Ready to evaluate {} solutions at resolution: {}".
                        format(len(layer_combs) * len(reg_range), k))

                for comb in layer_combs:
                    for reg_val in reg_range:
                        otpt = np.array([reg_val] + list(comb))
                        yield otpt

    def get_feature_importance_report(self, individual, fitnesses):
        """Report feature importances.

        :param np.array individual: an individual solution (a vector of floats)
        :param list fitnesses: fitness space (list of reals)

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
        except Exception:
            max_f1 = 0

        try:
            importances = list(
                zip(self.feature_names,
                    individual[0:self.weight_params].tolist()))

        except Exception:
            importances = list(
                zip(self.feature_names, individual[0:self.weight_params]))

        dfx = pd.DataFrame(importances)
        dfx.columns = ['Feature type', 'Importance']

        print(dfx.to_markdown())
        logging.info("Max {} in generation: {}\n".format(
            self.scoring_metric, round(max_f1, 3)))

    def mutReg(self, individual, p=1):
        """
        Custom mutation operator used for regularization optimization.

        :param individual: individual (vector of floats)
        :return individual: An individual solution.
        """

        individual[0] += np.random.random() * self.reg_constant
        return individual,

    def update_intermediary_feature_space(self, custom_space=None):
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

        if custom_space is not None:
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
        self.all_feature_names = fnames  # this is the new set of features.
        output_matrix = sparse.hstack(submatrices).tocsr()
        if self.verbose:
            logging.info(
                "Space update finished. {}, {} matrices joined.".format(
                    output_matrix.shape, len(submatrices)))
        assert len(self.all_feature_names) == output_matrix.shape[1]

        if custom_space is not None:
            return output_matrix

        else:
            self.intermediary_feature_space = output_matrix

        del submatrices

    def visualize_learners(self, learner_dataframe, image_path):
        """A generic hyperparameter visualization method. This helps the user with understanding of the overall optimization.

        :param learner_dataframe pd.DataFrame: The learner dataframe.
        :param image_path str: The output file's path.
        :return: None

        """

        for cname in learner_dataframe.columns:
            if "param_" in cname:

                try:
                    if self.verbose:
                        logging.info(f"Visualizing hyperparameter: {cname}")
                    sns.lineplot(x=learner_dataframe[cname],
                                 y=learner_dataframe.mean_test_score,
                                 color="black")
                    plt.ylabel(
                        f"Cross validation score ({self.scoring_metric})")

                    plt.xlabel(cname)
                    plt.savefig(image_path.replace("PARAM", cname), dpi=300)
                    plt.clf()
                    plt.cla()

                except Exception as es:
                    logging.info(es)

    def visualize_global_importances(self, importances_object, job_id,
                                     output_folder):

        try:

            importances_object['Importance'] = importances_object[
                'Importance'].astype(float)
            importances_object = importances_object.sort_values(
                by=['Importance'])
            sns.barplot(x='Importance', 
                        y='Feature subspace', 
                        data=importances_object, 
                        palette="coolwarm")
            #sns.barplot(importances_object.Importance,
            #            importances_object['Feature subspace'],
            #            palette="coolwarm")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder,
                                     f"{job_id}_barplot_global.pdf"),
                        dpi=300)
            plt.clf()
            plt.cla()

        except Exception as es:

            logging.info(es)

    def generate_report(self, output_folder="./report", job_id="genericJobId"):
        """An auxilliary method for creating a report

        :param string output_folder: The folder containing the report
        :param string job_id: The identifier of a given job
        :return: None

        """
        os.makedirs(output_folder, exist_ok=True)

        importances_local, importances_global = self.feature_type_importances()

        importances_local.to_csv(output_folder + f"{job_id}_local.tsv",
                                 sep="\t",
                                 index=False)

        self.visualize_global_importances(importances_global, job_id,
                                          output_folder)
        importances_global.to_csv(output_folder + f"{job_id}_global.tsv",
                                  sep="\t",
                                  index=False)

        learners = self.summarise_final_learners()
        learners = learners.sort_values(by=["mean_test_score"])
        learners.to_csv(output_folder + f"{job_id}_learners.tsv",
                        sep="\t",
                        index=False)

        self.visualize_learners(learners,
                                image_path=os.path.join(
                                    output_folder,
                                    f"{job_id}_learners_PARAM.pdf"))

        fitness = self.visualize_fitness(
            image_path=os.path.join(output_folder, f"{job_id}_fitness.pdf"))

        fitness.to_csv(output_folder + f"{job_id}_fitness.tsv",
                       sep="\t",
                       index=False)

        topics = self.get_topic_explanation()

        if topics is not None:
            topics.to_csv(output_folder + f"{job_id}_topics.tsv",
                          sep="\t",
                          index=False)

        if self.verbose:
            logging.info(f"Report generated! Check: {output_folder} folder.")

    def instantiate_validation_env(self):
        """
        This method refreshes the feature space. This is needed to maximize efficiency.
        """

        self.vectorizer, self.feature_names, self.train_feature_space = get_features(
            self.train_seq,
            representation_type=self.representation_type,
            sparsity=self.sparsity,
            embedding_dim=self.latent_dim,
            targets=self.train_targets,
            random_seed=self.random_seed,
            normalization_norm=self.normalization_norm,
            memory_location=self.memory_storage,
            custom_pipeline=self.custom_transformer_pipeline,
            contextual_model=self.contextual_model,
            combine_with_existing_representation=self.
            combine_with_existing_representation)

        # Check if feature construction failed
        if self.train_feature_space is None:
            raise RuntimeError("Feature construction failed - unable to create feature matrix. "
                             "This might be due to insufficient samples or incompatible data.")

        self.all_feature_names = []
        if self.verbose:
            logging.info("Initialized training matrix of dimension {}".format(
                self.train_feature_space.shape))

        self.feature_space_tuples = []
        self.global_feature_name_hash = {}

        # This information is used to efficiently subset and index the sparse representation
        self.feature_subspaces = []
        current_fnum = 0
        for transformer in self.vectorizer.named_steps[
                'union'].transformer_list:
            features = transformer[1].steps[1][1].get_feature_names_out()
            # Store only metadata instead of the actual subspace data to save memory
            # The subspace can be recreated when needed from the main feature space
            current_fnum += len(features)
            self.all_feature_names += list(features)
            num_feat = len(features)
            for f in features:
                self.global_feature_name_hash[f] = transformer[0]
            self.feature_space_tuples.append((transformer[0], num_feat))

        self.global_all_feature_names = self.all_feature_names
        self.intermediary_indices = [0] + np.cumsum(
            np.array([x[1] for x in self.feature_space_tuples])).tolist()

    def feature_type_importances(self, solution_index=0):
        """
        A method which prints feature type importances as a pandas df.

        :param solution_index: Which consequent individual to inspect.
        :return feature_ranking: Final table of rankings
        """
        feature_importances = self.hof[
            solution_index]  # global importances .feature_names
        struct = []
        for a, b in zip(feature_importances, self.feature_names):
            struct.append((str(a), str(b)))
        dfx = pd.DataFrame(struct)  # Create a Pandas dataframe
        dfx.columns = ['Importance', 'Feature subspace']

        # store global top features
        try:
            feature_ranking = self.global_feature_map  # all features

        except:            
            feature_ranking = pd.DataFrame(pd.Series({k: 1.0 for k in self.feature_names})).T

        return feature_ranking, dfx

    def get_topic_explanation(self):
        """
        A method for extracting the key topics.
        :return pd.DataFrame topicList: A list of topic-id tuples.
        """

        out_df = None

        try:
            feature_ranking = self.global_feature_map
            topic_transformer_trained = [
                x
                for x in self.vectorizer.named_steps['union'].transformer_list
                if x[0] == "topic_features"
            ][0][1]

            topic_feature_space = topic_transformer_trained[
                'topic_features'].topic_features

            top_topics = [
                int(x.split(":")[0].replace(" ", "").split("_")[1])
                for x in feature_ranking['topic_features'].values.tolist()
            ]

            importances = [
                float(x.split(":")[1].replace(" ", ""))
                for x in feature_ranking['topic_features'].values.tolist()
            ]

            ordered_topics = []

            for top_topic in top_topics:
                topic = " AND ".join(topic_feature_space[top_topic])
                ordered_topics.append(topic)

            out_df = pd.DataFrame()
            out_df['topic cluster'] = ordered_topics
            out_df['importances'] = importances

        except Exception:
            logging.info("Topics were not computed.")

        return out_df

    def visualize_fitness(self, image_path="fitnessExample.png"):
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

            mean_fitness = dfx.mean(axis=1)
            max_fitness = dfx.max(axis=1)
            min_fitness = dfx.min(axis=1)
            generations = list(range(dfx.shape[0]))
            sns.lineplot(x=generations, y=mean_fitness, color="black", label="mean")
            sns.lineplot(x=generations, y=max_fitness, color="green", label="max")
            sns.lineplot(x=generations, y=min_fitness, color="red", label="min")

            plt.xlabel("Generation")
            plt.ylabel(f"Fitness ({self.scoring_metric})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(image_path, dpi=300)
            plt.clf()
            plt.cla()

        except Exception as es:
            if self.verbose:
                logging.info(es)

        return dfx

    def store_top_solutions(self):
        """A method for storing the HOF"""

        try:
            file_to_store = open("hof_checkpoint.pickle", "wb")
            pickle.dump(self.hof, file_to_store)
            file_to_store.close()
        except Exception as es:
            if self.verbose:
                logging.info(
                    f"Could not store the hall of fame as a pickle: {es}")

    def load_top_solutions(self):
        """Load the top solutions as HOF"""

        if os.path.isfile("hof_checkpoint.pickle"):
            try:
                file_to_store = open("hof_checkpoint.pickle", "rb")
                self.hof = pickle.load(file_to_store)
                if self.verbose:
                    logging.info(
                        "Loaded the checkpoint file (hof_checkpoint.pickle)!")
                file_to_store.close()
            except Exception:
                if self.verbose:
                    logging.info("Could not load the checkpoint.")

    def evolve(self,
               nind=10,
               crossover_proba=0.4,
               mutpb=0.15,
               stopping_interval=20,
               strategy="evolution",
               representation_step_only=False):
        """The core evolution method. First constrain the maximum number of features to be taken into account by lowering the bound w.r.t performance.
        next, evolve.

        :param int nind: number of individuals (int)
        :param float crossover_proba: crossover probability (float)
        :param float mutpb: mutation probability (float)
        :param int stopping_interval: stopping interval -> for how long no improvement is tolerated before a hard reset (int)
        :param str strategy: type of evolution (str)
        :param bool representation_step_only: Learn only the feature transformations, skip the evolution. Suitable for custom experiments with transform()
        """

        self.initial_time = time.time()
        self.popsize = nind
        self.instantiate_validation_env()

        if representation_step_only:  # Skip the remainder
            return self

        self.weight_params = len(self.feature_names)

        if self.use_checkpoints:
            self.load_top_solutions()

        if strategy == "direct-learning":
            if self.verbose:
                logging.info("Training a learner without evolution.")
            top_individual = np.ones(self.weight_params)
            learner, individual, score, feature_names = self.evaluate_fitness(
                top_individual, return_clf_and_vec=True)
            coefficients = learner.best_estimator_.coef_
            coefficients = np.asarray(np.abs(np.max(coefficients,
                                                    axis=0))).reshape(-1)
            self.feature_importances.append((coefficients, feature_names))
            single_learner = (learner, individual, score)
            self.ensemble_of_learners.append(single_learner)
            self.hof = [top_individual]
            self.update_global_feature_importances()

        if strategy == "evolution":
            if self.verbose:
                logging.info("Evolution will last for ~{}h ..".format(
                    self.max_time))

            if self.verbose:
                logging.info("Selected strategy is evolution.")

            toolbox = base.Toolbox()
            toolbox.register("attr_float", np.random.uniform, 0.00001,
                             0.999999)
            toolbox.register("individual",
                             tools.initRepeat,
                             gcreator.Individual,
                             toolbox.attr_float,
                             n=self.weight_params)

            toolbox.register("population",
                             tools.initRepeat,
                             list,
                             toolbox.individual,
                             n=nind)

            toolbox.register("evaluate", self.evaluate_fitness)
            toolbox.register("mate", tools.cxUniform, indpb=0.5)
            toolbox.register("mutate",
                             tools.mutGaussian,
                             mu=0,
                             sigma=0.2,
                             indpb=0.2)

            toolbox.register("mutReg", self.mutReg)
            toolbox.register("select", tools.selTournament)

            if self.validation_type == "train_test":

                pool_tmp = mp.Pool(self.num_cpu)
                if self.verbose:
                    logging.info(
                        f"Instantiating parallel pool of individuals. Parallelization at the population level ({self.num_cpu} jobs)."
                    )
                toolbox.register("map", pool_tmp.map)

            else:
                toolbox.register("map", map)

            # Keep the best-performing individuals
            self.hof = tools.HallOfFame(self.hof_size)
            if self.verbose:
                logging.info(
                    "Total number of subspace importance parameters {}".format(
                        self.weight_params))

            # Population initialization
            if self.population == None:
                self.population = toolbox.population()
                self.custom_initialization()  # works on self.population
                if self.verbose:
                    logging.info("Initialized population of size {}".format(
                        len(self.population)))
                if self.verbose:
                    logging.info("Computing initial fitness ..")

            # Gather fitness values.
            fits = list(toolbox.map(toolbox.evaluate, self.population))

            for fit, ind in zip(fits, self.population):
                ind.fitness.values = fit

            # Update HOF
            self.hof.update(self.population)

            # Report performance
            self.report_performance(fits)

            gen = 0
            if self.verbose:
                logging.info("Initiating evaluation ..")

            stopping = 1
            cf1 = 0

            # Start the evolution.
            while True:

                gen += 1
                tdiff = self.compute_time_diff()

                if tdiff >= self.max_time:
                    break

                offspring = list(toolbox.map(toolbox.clone, self.population))

                # Perform crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):

                    if np.random.random() < crossover_proba:

                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Perform mutation
                for mutant in offspring:

                    if np.random.random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # In the first population, include isolated spaces
                if gen == 1:
                    offspring = offspring + self.separate_individual_spaces

                fits = list(toolbox.map(toolbox.evaluate, offspring))
                for ind, fit in zip(offspring, fits):
                    if isinstance(fit, int) and not isinstance(fit, tuple):
                        fit = (fit, )
                    ind.fitness.values = fit

                self.hof.update(offspring)  # Update HOF

                # append to overall fitness container.
                self.fitness_container.append(fits)

                self.get_feature_importance_report(self.hof[0], fits)

                f1 = self.report_performance(fits, gen=gen)

                if f1 == cf1:
                    stopping += 1

                else:
                    cf1 = f1

                self.population = toolbox.select(self.population + offspring,
                                                 k=nind,
                                                 tournsize=int(nind / 3))

            try:
                selections = self.hof

            except:
                selections = self.population

            if self.use_checkpoints:
                self.store_top_solutions()

            if self.visualize_progress:
                self.visualize_fitness(image_path=f"PROGRESS_gen_{gen}.pdf")

            self.selections = [np.array(x).tolist() for x in selections]

            # Ensemble of learners is finally filled and used for prediction.
            for enx, top_individual in enumerate(selections):

                if len(top_individual) == 1:
                    top_individual = top_individual[0]

                try:
                    learner, individual, score, feature_names = self.evaluate_fitness(
                        top_individual, return_clf_and_vec=True)

                except Exception as es:
                    if self.verbose:
                        logging.info(
                            f"Evaluation of individual {top_individual} did not produce a viable learner. Increase time! {es}"
                        )

                try:
                    coefficients = learner.best_estimator_.coef_

                    # coefficients are given for each class. We take maxima  (abs val)
                    coefficients = np.asarray(
                        np.abs(np.max(coefficients, axis=0))).reshape(-1)

                    if self.verbose:
                        logging.info("Coefficients and indices: {}".format(
                            len(coefficients)))

                    if self.verbose:
                        logging.info(
                            "Adding importances of shape {} for learner {} with score {}"
                            .format(coefficients.shape, enx, score))

                    self.feature_importances.append(
                        (coefficients, feature_names))

                    # Update the final importance space.
                    if self.task == "classification":
                        self.update_global_feature_importances()

                except Exception as es:
                    logging.info(
                        f"The considered classifier cannot produce feature importances. {es}"
                    )

                single_learner = (learner, individual, score)
                self.ensemble_of_learners.append(single_learner)

        # Clean up memory after evolution
        if hasattr(self, 'population'):
            del self.population
        if hasattr(self, 'fitness_container'):
            # Keep only the most recent fitness values, clear older ones
            if len(self.fitness_container) > 10:
                self.fitness_container = self.fitness_container[-10:]
        
        # Force garbage collection to free up memory
        gc.collect()
        
        return self
