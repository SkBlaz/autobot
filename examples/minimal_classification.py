## A simple example showcasing the minimal usecase of autoBOTLib on an insults classification data.

import autoBOTLib
import pandas as pd


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].values.tolist()
    train_targets = dataframe['label'].values

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,  # input sequences
        train_targets,  # target space 
        time_constraint=3,  # time in hoursc
        num_cpu="all",  # number of CPUs to use
        latent_dim=768,  ## latent dim for neural representations
        sparsity=0.1,  ## latent_dim/sparsity dim for sparse representations
        task_name="example test",  # task identifier
        scoring_metric="f1",  # sklearn-compatible scoring metric as the fitness.
        hof_size=3,  # size of the hall of fame
        top_k_importances=25,  # how many top features to output as final ranking
        memory_storage="./memory",  # tripled base for concept features
        representation_type="neurosymbolic")  # or symbolic or neural

    autoBOTLibObj.evolve(
        nind=8,  ## population size
        strategy="evolution",  ## optimization strategy
        crossover_proba=0.6,  ## crossover rate
        mutpb=0.4)  ## mutation rate

    ## Persistence demonstration (how to store models for further use?)
    autoBOTLib.store_autobot_model(
        autoBOTLibObj, "../stored_models/example_insults_model.pickle")
    autoBOTLibObj = autoBOTLib.load_autobot_model(
        "../stored_models/example_insults_model.pickle")

    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a'].values.tolist()
    test_targets = dataframe2['label'].values
    predictions = autoBOTLibObj.predict(test_sequences)
    print(predictions)
    performance = autoBOTLib.compute_metrics(
        "first_run_task_name", predictions,
        test_targets)  ## compute F1, acc and F1_acc (as in GLUE)

    ## visualize performance
    print(performance)

    ## Visualize importances (global -> type, local -> individual features)
    importances_local, importances_global = autoBOTLibObj.feature_type_importances(
    )
    print(importances_global)
    print(importances_local)

    final_learners = autoBOTLibObj.summarise_final_learners()
    print(final_learners)

    ## Visualize the fitness trace
    fitness_summary = autoBOTLibObj.visualize_fitness(
        image_path="./fitness_new.png")
    print(fitness_summary)


if __name__ == "__main__":
    run()
