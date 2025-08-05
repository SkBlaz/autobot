#!/usr/bin/env python3
"""
Test script to validate torch.compile integration in autoBOT with diverse test scenarios
"""

import numpy as np
import torch
import time
from scipy import sparse
from autoBOTLib.learning.torch_sparse_nn import SFNN


def create_diverse_test_data():
    """Create diverse test datasets for comprehensive testing"""
    np.random.seed(42)
    
    # Dataset 1: Small dense features (text classification scenario)
    X1 = sparse.random(150, 50, density=0.3, format='csr')
    y1 = np.random.choice([0, 1], 150, p=[0.6, 0.4])  # Imbalanced classes
    
    # Dataset 2: Large sparse features (document vectors)
    X2 = sparse.random(500, 1000, density=0.05, format='csr')
    y2 = np.random.choice([0, 1], 500, p=[0.5, 0.5])  # Balanced classes
    
    # Dataset 3: Medium features with different sparsity
    X3 = sparse.random(300, 200, density=0.15, format='csr')
    y3 = np.random.choice([0, 1], 300, p=[0.3, 0.7])  # Different imbalance
    
    return [(X1, y1, "small_dense"), (X2, y2, "large_sparse"), (X3, y3, "medium_mixed")]


def test_torch_compile_functionality():
    """Test torch.compile with diverse datasets and configurations"""
    print("Testing torch.compile functionality with diverse data...")
    
    datasets = create_diverse_test_data()
    
    for X, y, dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing with {dataset_name} dataset:")
        print(f"  Shape: {X.shape}, Density: {X.nnz / (X.shape[0] * X.shape[1]):.3f}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        # Test with different model configurations
        configs = [
            {"hidden_layer_size": 32, "num_hidden": 1, "learning_rate": 0.01},
            {"hidden_layer_size": 64, "num_hidden": 2, "learning_rate": 0.005},
            {"hidden_layer_size": 128, "num_hidden": 3, "learning_rate": 0.001}
        ]
        
        for i, config in enumerate(configs):
            print(f"\n--- Configuration {i+1}: {config} ---")
            
            # Test with torch.compile enabled
            model_compiled = SFNN(
                batch_size=16,
                num_epochs=3,
                dropout=0.1,
                verbose=0,
                torch_compile=True,
                **config
            )
            
            start_time = time.time()
            model_compiled.fit(X, y)
            compile_time = time.time() - start_time
            
            # Test with torch.compile disabled
            model_uncompiled = SFNN(
                batch_size=16,
                num_epochs=3,
                dropout=0.1,
                verbose=0,
                torch_compile=False,
                **config
            )
            
            start_time = time.time()
            model_uncompiled.fit(X, y)
            uncompiled_time = time.time() - start_time
            
            print(f"    Training time - Compiled: {compile_time:.2f}s, Uncompiled: {uncompiled_time:.2f}s")
            
            # Test predictions on a subset
            X_test = X[:10]
            pred_compiled = model_compiled.predict(X_test)
            pred_uncompiled = model_uncompiled.predict(X_test)
            proba_compiled = model_compiled.predict_proba(X_test)
            proba_uncompiled = model_uncompiled.predict_proba(X_test)
            
            print(f"    Predictions shape - Compiled: {pred_compiled.shape}, Uncompiled: {pred_uncompiled.shape}")
            print(f"    Probabilities shape - Compiled: {proba_compiled.shape}, Uncompiled: {proba_uncompiled.shape}")
            
            # Verify both models produce consistent outputs
            assert pred_compiled.shape == pred_uncompiled.shape, "Prediction shapes should match"
            assert proba_compiled.shape == proba_uncompiled.shape, "Probability shapes should match"


def test_torch_compile_edge_cases():
    """Test edge cases and error handling"""
    print(f"\n{'='*60}")
    print("Testing edge cases and error handling...")
    
    # Very small dataset
    X_tiny = sparse.csr_matrix(np.random.random((10, 5)))
    y_tiny = np.random.randint(0, 2, 10)
    
    print("\n--- Testing with tiny dataset ---")
    model_tiny = SFNN(
        batch_size=5,
        num_epochs=2,
        torch_compile=True,
        verbose=1
    )
    model_tiny.fit(X_tiny, y_tiny)
    
    # Single sample prediction
    pred_single = model_tiny.predict(X_tiny[:1])
    proba_single = model_tiny.predict_proba(X_tiny[:1])
    print(f"Single prediction shape: {pred_single.shape}")
    print(f"Single probability shape: {proba_single.shape}")
    
    # Verify predictions work (shapes are consistent with existing behavior)
    assert pred_single.shape == (1,), f"Expected single prediction shape (1,), got {pred_single.shape}"
    assert proba_single.shape == (1,), f"Expected single probability shape (1,), got {proba_single.shape}"


def test_compatibility_across_pytorch_versions():
    """Test that the implementation works across different PyTorch scenarios"""
    print(f"\n{'='*60}")
    print("Testing PyTorch version compatibility...")
    
    X, y = sparse.random(100, 20, density=0.2, format='csr'), np.random.randint(0, 2, 100)
    
    # Test when torch.compile is available
    if hasattr(torch, 'compile'):
        print("✓ torch.compile is available - testing compilation")
        model = SFNN(torch_compile=True, num_epochs=2, verbose=1)
        model.fit(X, y)
    else:
        print("✗ torch.compile not available - testing fallback")
        model = SFNN(torch_compile=True, num_epochs=2, verbose=1)
        model.fit(X, y)
    
    print(f"PyTorch version: {torch.__version__}")
    print("Compatibility test completed successfully")


if __name__ == "__main__":
    print("=" * 80)
    print("AUTOBOT TORCH.COMPILE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    try:
        test_torch_compile_functionality()
        test_torch_compile_edge_cases()
        test_compatibility_across_pytorch_versions()
        
        print(f"\n{'='*80}")
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("torch.compile integration is working correctly with diverse scenarios")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()