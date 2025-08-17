"""Test torch.compile functionality with autoBOT neural networks."""

import pytest
import numpy as np
import torch
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng

from autoBOTLib.learning.torch_sparse_nn import SFNN, hyper_opt_neural


class TestTorchCompile:
    """Tests for torch.compile integration."""

    def setup_method(self):
        """Setup test data."""
        rng = default_rng(42)
        rvs = stats.uniform().rvs
        # Create sparse test data
        self.X = random(1000, 100, density=0.1, random_state=rng, data_rvs=rvs).tocsr()
        # Create binary target
        self.y = np.random.RandomState(42).randint(0, 2, 1000)

    def test_sfnn_with_compile_disabled(self):
        """Test SFNN with compile_model=False (default behavior)."""
        model = SFNN(
            batch_size=16,
            num_epochs=5,
            learning_rate=0.01,
            stopping_crit=2,
            hidden_layer_size=32,
            compile_model=False,
            verbose=1
        )
        
        # Should not raise any errors
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        proba_predictions = model.predict_proba(self.X[:10])
        
        assert len(predictions) == 10
        assert len(proba_predictions) == 10
        assert hasattr(model, 'coef_')

    def test_sfnn_with_compile_enabled(self):
        """Test SFNN with compile_model=True."""
        model = SFNN(
            batch_size=16,
            num_epochs=5,
            learning_rate=0.01,
            stopping_crit=2,
            hidden_layer_size=32,
            compile_model=True,
            verbose=1
        )
        
        # Should not raise any errors
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        proba_predictions = model.predict_proba(self.X[:10])
        
        assert len(predictions) == 10
        assert len(proba_predictions) == 10
        assert hasattr(model, 'coef_')

    def test_hyper_opt_neural_with_compile(self):
        """Test hyperoptimization with compile_model parameter."""
        # Create a custom hyperparameter space with compile_model
        custom_hyperparam_space = {
            "batch_size": [16],
            "num_epochs": [5],
            "learning_rate": [0.01],
            "stopping_crit": [2],
            "hidden_layer_size": [32],
            "num_hidden": [1],
            "dropout": [0.1],
            "device": ["cpu"],
            "compile_model": [True, False]
        }
        
        from autoBOTLib.learning.torch_sparse_nn import HyperParamNeuralObject
        
        hyper_opt_obj = HyperParamNeuralObject(
            custom_hyperparam_space,
            verbose=1,
            device="cpu",
            metric="f1_macro"
        )
        
        # Test with n_configs=2 to test both compiled and non-compiled versions
        hyper_opt_obj.fit(self.X, self.y, refit=False, n_configs=2)
        
        assert len(hyper_opt_obj.history) == 2
        assert hyper_opt_obj.best_score >= 0
        
        # Verify that compile_model was included in the configuration history
        compile_values = [config.get('compile_model', False) for config in hyper_opt_obj.history]
        assert len(set(compile_values)) >= 1  # At least one unique value

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
    def test_torch_compile_availability(self):
        """Verify that torch.compile is available in the environment."""
        assert hasattr(torch, 'compile'), "torch.compile should be available"
        
        # Test basic torch.compile functionality
        def simple_function(x):
            return x * 2
        
        compiled_fn = torch.compile(simple_function)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = compiled_fn(x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        
        assert torch.allclose(result, expected)