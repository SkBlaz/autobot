import autoBOTLib
import pandas as pd
from sklearn.linear_model import SGDClassifier


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    ## The syntax for specifying a learner and the hyperparameter space!
    ## These are the hyperparameters to be explored for each representation.
    classifier_hyperparameters = {
        "loss": ["hinge"],
        "penalty": ["elasticnet"],
        "alpha": [0.01, 0.001],
        "l1_ratio": [0, 0.001, 1]
    }

    ## This is the classifier compatible with the hyperparameters.
    custom_classifier = SGDClassifier()

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,  # input sequences
        train_targets,  # target space 
        time_constraint=1,  # time in hours
        num_cpu=4,  # number of CPUs to use
        task_name="example test",  # task identifier
        hof_size=3,  # size of the hall of fame
        top_k_importances=25,  # how many top features to output as final ranking
        memory_storage="./memory",
        representation_type="symbolic",
        learner=custom_classifier,
        learner_hyperparameters=classifier_hyperparameters
    )  # or neurosymbolic or neural

    autoBOTLibObj.evolve(
        nind=10,  ## population size
        strategy="evolution",  ## optimization strategy
        crossover_proba=0.6,  ## crossover rate
        mutpb=0.4)  ## mutation rate


if __name__ == "__main__":
    run()
