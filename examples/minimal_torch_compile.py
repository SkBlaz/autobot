## A simple example showcasing autoBOTLib with torch.compile support

import autoBOTLib
import pandas as pd
from cluster_utils import output_classification_results
import os

#TOKENIZERS_PARALLELISM=(true | false)
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def run():
    ## Load example data frame
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t").iloc[:]
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']
    reptype = "neurosymbolic"
    
    # Example 1: Run without torch.compile (default behavior)
    print("=== Running without torch.compile ===")
    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        representation_type=reptype,
        n_fold_cv=3,
        framework="torch",
        memory_storage="memory",
        learner_preset="default",
        verbose=1,
        sparsity=0.1,
        upsample=False,
        time_constraint=1).evolve(
            strategy="evolution",
            nind=3)
    
    # Make predictions
    dataframe2 = pd.read_csv("../data/insults/test.tsv", sep="\t")
    test_sequences = dataframe2['text_a']
    predictions = autoBOTLibObj.predict(test_sequences)
    test_classes = dataframe2['label'].values.tolist()
    
    output_classification_results(predictions,
                                  test_classes,
                                  f"./predictions/TORCH_regular.json",
                                  model_spec={"compile": False})
    
    # Example 2: Test with torch.compile enabled (custom hyperparameters)
    print("\n=== Running with torch.compile enabled ===")
    # Custom hyperparameters that include torch.compile
    custom_torch_hyperparams = {
        "batch_size": [16],
        "num_epochs": [50],
        "learning_rate": [0.001],
        "stopping_crit": [5],
        "hidden_layer_size": [128],
        "num_hidden": [2],
        "dropout": [0.1],
        "device": ["cpu"],
        "compile_model": [True]  # Enable torch.compile
    }
    
    autoBOTLibObj_compiled = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        representation_type=reptype,
        n_fold_cv=3,
        framework="torch",
        custom_hyperparameters=custom_torch_hyperparams,
        memory_storage="memory",
        learner_preset="default",
        verbose=1,
        sparsity=0.1,
        upsample=False,
        time_constraint=1).evolve(
            strategy="evolution",
            nind=3)
    
    # Make predictions with compiled model
    predictions_compiled = autoBOTLibObj_compiled.predict(test_sequences)
    
    output_classification_results(predictions_compiled,
                                  test_classes,
                                  f"./predictions/TORCH_compiled.json",
                                  model_spec={"compile": True})
    
    print("\n=== Comparison ===")
    print(f"Regular model predictions: {len(predictions)} samples")
    print(f"Compiled model predictions: {len(predictions_compiled)} samples")
    print("Both models completed successfully!")


if __name__ == "__main__":
    run()