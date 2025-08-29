#!/usr/bin/env python3
"""
Debug the specific indexing error
"""

import autoBOTLib
import pandas as pd
import traceback

def debug_test():
    """Debug the exact issue"""
    
    print("Debug test...")
    try:
        dataframe = pd.read_csv("data/insults/train.tsv", sep="\t").head(50)  # Even smaller
        train_sequences = dataframe['text_a']
        train_targets = dataframe['label']
        
        print(f"Data shape: {len(train_sequences)}")
        print(f"Targets: {set(train_targets)}")
        
        autoBOTLibObj = autoBOTLib.GAlearner(
            train_sequences,
            train_targets,
            representation_type="symbolic",
            n_fold_cv=2,  # Smaller CV
            sparsity=0.8,  # Higher sparsity
            time_constraint=0.005,
            hof_size=1,
            verbose=1  # Enable verbose for debugging
        )
        
        print("Training...")
        autoBOTLibObj.evolve(strategy="direct-learning")
        
        print("Testing prediction with 1 sample...")
        predictions = autoBOTLibObj.predict([train_sequences.iloc[0]])
        print(f"Prediction successful: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_test()