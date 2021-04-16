import autoBOTLib
import pandas as pd

## Load example data frame
dataframe = pd.read_csv("../data/spanish/train.tsv", sep="\t")
train_sequences = dataframe['text_a'].values.tolist()
train_targets = dataframe['label'].values
print(train_sequences[0:3])
print(train_targets[0:3])

#Possible metrics: ['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'accuracy', 'top_k_accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'balanced_accuracy', 'average_precision', 'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'rand_score', 'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted']

autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
                                     train_targets,
                                     scoring_metric = "accuracy",
                                     representation_type = "neurosymbolic",
                                     time_constraint=0.1).evolve()
