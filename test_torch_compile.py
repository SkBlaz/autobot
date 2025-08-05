#!/usr/bin/env python3
"""
Test script to validate torch.compile integration in autoBOT
"""

import numpy as np
import torch
import time
from scipy import sparse
from autoBOTLib.learning.torch_sparse_nn import SFNN


def create_test_data(n_samples=1000, n_features=100, density=0.1):
    """Create sample sparse data for testing"""
    np.random.seed(42)
    X = sparse.random(n_samples, n_features, density=density, format='csr')
    y = np.random.randint(0, 2, n_samples)
    return X, y


def test_torch_compile_functionality():
    """Test that torch.compile works correctly"""
    print("Testing torch.compile functionality...")
    
    X, y = create_test_data(n_samples=200, n_features=50)
    
    # Test with torch.compile enabled
    model_compiled = SFNN(
        batch_size=32,
        num_epochs=5,
        learning_rate=0.01,
        hidden_layer_size=64,
        num_hidden=2,
        verbose=1,
        torch_compile=True
    )
    
    print("\n=== Training with torch.compile=True ===")
    start_time = time.time()
    model_compiled.fit(X, y)
    compile_time = time.time() - start_time
    
    # Test with torch.compile disabled
    model_uncompiled = SFNN(
        batch_size=32,
        num_epochs=5,
        learning_rate=0.01,
        hidden_layer_size=64,
        num_hidden=2,
        verbose=1,
        torch_compile=False
    )
    
    print("\n=== Training with torch.compile=False ===")
    start_time = time.time()
    model_uncompiled.fit(X, y)
    uncompiled_time = time.time() - start_time
    
    print(f"\nTraining time comparison:")
    print(f"  With torch.compile: {compile_time:.2f}s")
    print(f"  Without torch.compile: {uncompiled_time:.2f}s")
    print(f"  Speedup: {uncompiled_time/compile_time:.2f}x")
    
    # Test predictions
    print("\n=== Testing Predictions ===")
    X_test = X[:20]
    
    pred_compiled = model_compiled.predict(X_test)
    pred_uncompiled = model_uncompiled.predict(X_test)
    
    print(f"Compiled model predictions shape: {pred_compiled.shape}")
    print(f"Uncompiled model predictions shape: {pred_uncompiled.shape}")
    
    # Test probability predictions
    proba_compiled = model_compiled.predict_proba(X_test)
    proba_uncompiled = model_uncompiled.predict_proba(X_test)
    
    print(f"Compiled model probabilities shape: {proba_compiled.shape}")
    print(f"Uncompiled model probabilities shape: {proba_uncompiled.shape}")
    
    print("\n✓ torch.compile functionality test passed!")
    return True


def test_torch_compile_availability():
    """Test torch.compile availability detection"""
    print("Testing torch.compile availability detection...")
    
    has_compile = hasattr(torch, 'compile')
    torch_version = torch.__version__
    
    print(f"PyTorch version: {torch_version}")
    print(f"torch.compile available: {has_compile}")
    
    if has_compile:
        print("✓ torch.compile is available")
    else:
        print("⚠ torch.compile is not available (requires PyTorch >= 2.0)")
    
    return has_compile


def test_backward_compatibility():
    """Test that the new torch_compile parameter doesn't break existing code"""
    print("\nTesting backward compatibility...")
    
    X, y = create_test_data(n_samples=100, n_features=30)
    
    # Test creating SFNN without torch_compile parameter (should default to True)
    model_default = SFNN(
        batch_size=16,
        num_epochs=2,
        learning_rate=0.01,
        hidden_layer_size=32,
        verbose=0
    )
    
    model_default.fit(X, y)
    predictions = model_default.predict(X[:10])
    
    print(f"Default model (torch_compile should be True): predictions shape {predictions.shape}")
    print("✓ Backward compatibility test passed!")
    
    return True


if __name__ == "__main__":
    print("="*50)
    print("autoBOT torch.compile Integration Test")
    print("="*50)
    
    # Run all tests
    compile_available = test_torch_compile_availability()
    
    if compile_available:
        test_torch_compile_functionality()
    else:
        print("Skipping functionality test (torch.compile not available)")
    
    test_backward_compatibility()
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)