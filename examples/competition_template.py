## A simple example showcasing the minimal usecase of autoBOT on an insults classification data.

import autoBOTLib
import pandas as pd
import secrets
import csv

def run():
    jid = secrets.token_hex(nbytes=16)
    df_path = None
    
    ## Load example data frame
    dataframe = pd.read_csv(df_path, sep="\t")
    train_sequences = None
    train_sequences = None
    train_targets = None

    print(len(train_sequences))
    print(len(train_targets))

    for classx in possible_classes:
    
        autoBOTObj = autoBOTLib.GAlearner(
            train_sequences,  # input sequences
            train_targets,  # target space
            time_constraint = 1,  # time in hoursc
            num_cpu = 32,  # number of CPUs to use
            sparsity = 0.1,
            task_name="example test",  # task identifier
            scoring_metric = "f1", # sklearn-compatible scoring metric as the fitness.
            hof_size = 3,  # size of the hall of fame
            top_k_importances = 25,  # how many top features to output as final ranking
            memory_storage = "./memory",  # tripled base for concept features
            representation_type = "neurosymbolic")  # or symbolic or neural

        autoBOTObj.evolve(
            nind = 8,  ## population size
            strategy = "evolution",  ## optimization strategy
            crossover_proba = 0.6,  ## crossover rate
            mutpb = 0.4)  ## mutation rate

        autoBOTLib.store_autobot_model(autoBOTObj,f"./models/{jid}_{classx}_model.pickle")

    for classx in possible_classes:

        test_data = None
        autoBOTObj = autoBOTLib.load_autobot_model(f"./models/{jid}_{classx}_model.pickle")
        predictions = autoBOTObj.predict(test_sequences)
                    
if __name__ == "__main__":

    import argparse
    run()
