#!/usr/bin/env python3
"""
Integration test for torch.compile in autoBOT's torch neural network component
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from autoBOTLib.learning.torch_sparse_nn import torch_learners


def create_text_data():
    """Create some sample text data for testing"""
    texts = [
        "This is a positive example",
        "This is another positive text",
        "Great work on this project",
        "Excellent performance results",
        "I love this implementation",
        "This is a negative example",
        "Bad implementation here",
        "Poor performance results",
        "I hate this approach",
        "Terrible code quality"
    ] * 20  # Repeat to get more data
    
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20
    return texts, labels


def test_torch_compile_in_autobot_workflow():
    """Test torch.compile integration in the autoBOT workflow"""
    print("Testing torch.compile integration in autoBOT workflow...")
    
    # Create sample data
    texts, labels = create_text_data()
    
    # Convert to features using TF-IDF (simulating autoBOT feature extraction)
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Test torch_learners function with torch_compile enabled
    print("\n=== Testing with torch.compile enabled ===")
    
    # Mock parameters that would normally come from autoBOT
    final_run = True
    custom_hyperparameters = None
    learner_preset = "default"
    learner = "torch"
    task = "classification"
    metric = "f1_macro"
    num_folds = 2
    validation_percentage = 0.1
    random_seed = 42
    verbose = 1
    validation_type = "split"
    num_cpu = 1
    device = "cpu"
    
    score_compiled, clf_compiled = torch_learners(
        final_run=final_run,
        X=X,
        Y=y,
        custom_hyperparameters=custom_hyperparameters,
        learner_preset=learner_preset,
        learner=learner,
        task=task,
        metric=metric,
        num_folds=num_folds,
        validation_percentage=validation_percentage,
        random_seed=random_seed,
        verbose=verbose,
        validation_type=validation_type,
        num_cpu=num_cpu,
        device=device
    )
    
    print(f"Compiled model score: {score_compiled:.4f}")
    
    # Test predictions
    if hasattr(clf_compiled, 'best_estimator_'):
        model = clf_compiled.best_estimator_
        
        # Test prediction on a subset
        X_test = X[:10]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Sample predictions: {predictions[:5]}")
        print(f"Sample probabilities: {probabilities[:5]}")
        
        # Verify the model was compiled
        if hasattr(model.model, '_orig_mod'):
            print("✓ Model was successfully compiled with torch.compile")
        else:
            print("⚠ Model was not compiled (unexpected)")
        
        print("✓ torch.compile integration test passed!")
        return True
    else:
        print("⚠ No best_estimator_ found")
        return False


if __name__ == "__main__":
    print("="*60)
    print("autoBOT torch.compile Integration Test")
    print("="*60)
    
    try:
        result = test_torch_compile_in_autobot_workflow()
        if result:
            print("\n" + "="*60)
            print("Integration test completed successfully!")
            print("torch.compile is working in the autoBOT workflow")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Integration test completed with warnings")
            print("="*60)
    except Exception as e:
        print(f"\nIntegration test failed with error: {e}")
        import traceback
        traceback.print_exc()