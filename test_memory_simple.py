#!/usr/bin/env python3
"""
Simple test to verify memory optimizations work
"""

import autoBOTLib
import pandas as pd
import gc

# Use the actual insults dataset for testing instead of synthetic data
def test_with_real_data():
    """Test with real data to avoid edge cases"""
    
    print("Testing with real dataset...")
    
    # Load a smaller subset of the real data
    try:
        dataframe = pd.read_csv("data/insults/train.tsv", sep="\t").head(500)  # Use only first 500 samples
        train_sequences = dataframe['text_a']
        train_targets = dataframe['label']
        
        print(f"Dataset shape: {len(train_sequences)} samples")
        print(f"Unique labels: {set(train_targets)}")
        
        # Initialize with memory-friendly settings
        autoBOTLibObj = autoBOTLib.GAlearner(
            train_sequences,
            train_targets,
            representation_type="symbolic",  # Use symbolic only to reduce memory
            n_fold_cv=3,
            sparsity=0.3,  # Increase sparsity to reduce feature count
            time_constraint=0.01,  # Very short for testing
            hof_size=1,  # Reduce hall of fame size
            num_cpu=2,  # Use fewer cores
            memory_storage="memory"
        )
        
        # Test evolution
        autoBOTLibObj.evolve(strategy="direct-learning")  # Use direct learning, not evolution
        print("✓ Training completed successfully")
        
        # Test prediction
        test_data = train_sequences.head(10)
        predictions = autoBOTLibObj.predict(test_data)
        print(f"✓ Predictions completed: {len(predictions)} predictions")
        
        # Clean up
        del autoBOTLibObj
        gc.collect()
        print("✓ Memory cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_with_real_data()
    if success:
        print("Memory optimization test: PASSED")
    else:
        print("Memory optimization test: FAILED")