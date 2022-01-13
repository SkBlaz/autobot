from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from autoBOTLib.learning.hyperparameter_configurations import scikit_default, scikit_intense, scikit_mini_l1, scikit_mini_l2, scikit_intense_final, scikit_generic_final, scikit_knn


def scikit_learners(final_run, tmp_feature_space, train_targets, learner_hyperparameters,
                    learner_preset, learner,
                    task, scoring_metric, n_fold_cv, validation_percentage,
                    random_seed, verbose, validation_type, num_cpu):
    """An auxilliary method which conducts sklearn-based learning"""
    
    if learner_hyperparameters is None:

        # this is for screening purposes.
        if learner_preset == "default":
            parameters = scikit_default
            
        elif learner_preset == "intense":
            parameters = scikit_intense
            
        elif learner_preset == "mini-l1":
            parameters = scikit_mini_l1
            
        elif learner_preset == "mini-l2":
            parameters = scikit_mini_l2

        elif learner_preset == "test":
            parameters = scikit_mini_l2
            
        elif learner_preset == "knn":
            parameters = scikit_knn

        if final_run and learner_preset != "knn":

            # we can afford this final round to be more extensive.
            if learner_preset == "intense":

                parameters = scikit_intense_final

            elif learner_preset == "test":
                
                parameters = scikit_mini_l2
                
            else:

                parameters = scikit_generic_final
    else:

        parameters = learner_hyperparameters

    if learner is None:

        if task == "classification":

            if learner_preset == "knn":
                svc = KNeighborsClassifier()
            else:
                svc = SGDClassifier(max_iter=1000000)

        else:
            if learner_preset == "knn":
                svc = KNeighborsClassifier()
            else:
                svc = SGDRegressor(max_iter=1000000)
                parameters['loss'] = ['squared_loss']

    else:
        svc = learner

    performance_score = scoring_metric

    if validation_type == "train_test":
        cv = ShuffleSplit(n_splits=1,
                          test_size=validation_percentage,
                          random_state=random_seed)
        num_cpu = 1

    else:
        cv = n_fold_cv

    if verbose == 0:
        verbosity_factor = 0
    else:
        verbosity_factor = 0 if not final_run else 10

    if final_run:
        refit = True
        
    else:
        refit = False

    clf = GridSearchCV(svc,
                       parameters,
                       verbose=verbose + verbosity_factor,
                       n_jobs=num_cpu,
                       cv=cv,
                       scoring=performance_score,
                       refit=refit)

    clf.fit(tmp_feature_space, train_targets)
    f1_perf = max(clf.cv_results_['mean_test_score'])

    return f1_perf, clf
