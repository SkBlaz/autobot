#!/usr/bin/env python3
"""
Test script to reproduce memory issues with larger datasets
"""

import autoBOTLib
import pandas as pd
import psutil
import os
import gc
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_large_dataset(n_samples=5000, n_chars_per_sample=200):
    """Create a large synthetic dataset for testing memory usage"""
    np.random.seed(42)
    texts = []
    for i in range(n_samples):
        # Create random text samples
        text = ' '.join([f'word{j}' for j in range(n_chars_per_sample // 10)])
        texts.append(text)
    
    # Create random labels
    labels = np.random.randint(0, 2, n_samples).tolist()
    
    return texts, labels

def test_memory_with_different_sizes():
    """Test memory usage with different dataset sizes"""
    sizes = [1000, 2000, 3000, 4000, 5000]
    
    for size in sizes:
        print(f"\n=== Testing with {size} samples ===")
        
        # Initial memory
        gc.collect()
        initial_memory = get_memory_usage()
        print(f"Initial memory: {initial_memory:.2f} MB")
        
        try:
            # Create dataset
            train_sequences, train_targets = create_large_dataset(size)
            after_data_memory = get_memory_usage()
            print(f"After creating data: {after_data_memory:.2f} MB (+{after_data_memory - initial_memory:.2f} MB)")
            
            # Initialize autoBOT
            autoBOTLibObj = autoBOTLib.GAlearner(
                train_sequences,
                train_targets,
                representation_type="symbolic",
                n_fold_cv=3,
                sparsity=0.1,
                time_constraint=0.05,  # Very short time for testing
                memory_storage="memory"
            )
            
            after_init_memory = get_memory_usage()
            print(f"After autoBOT init: {after_init_memory:.2f} MB (+{after_init_memory - after_data_memory:.2f} MB)")
            
            # Try to evolve (this is where memory issues typically occur)
            autoBOTLibObj.evolve(strategy="evolution")
            
            after_evolve_memory = get_memory_usage()
            print(f"After evolution: {after_evolve_memory:.2f} MB (+{after_evolve_memory - after_init_memory:.2f} MB)")
            print(f"Total memory increase: {after_evolve_memory - initial_memory:.2f} MB")
            
        except Exception as e:
            error_memory = get_memory_usage()
            print(f"ERROR at {size} samples: {e}")
            print(f"Memory at error: {error_memory:.2f} MB")
            break
        
        finally:
            # Cleanup
            del train_sequences, train_targets
            if 'autoBOTLibObj' in locals():
                del autoBOTLibObj
            gc.collect()

if __name__ == "__main__":
    print("Testing memory usage with different dataset sizes...")
    test_memory_with_different_sizes()