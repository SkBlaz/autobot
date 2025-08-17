# torch.compile Support in autoBOT

autoBOT now supports PyTorch's `torch.compile` feature for optimized neural network training and inference.

## What is torch.compile?

`torch.compile` is PyTorch's JIT compilation feature introduced in PyTorch 2.0 that can provide significant speedups by compiling PyTorch models into optimized code.

## Usage

### Through Hyperparameter Configuration

The `compile_model` parameter is now available in the torch hyperparameter configuration:

```python
import autoBOTLib

custom_hyperparams = {
    "batch_size": [16],
    "num_epochs": [100],
    "learning_rate": [0.001],
    "compile_model": [True]  # Enable torch.compile
}

autobot = autoBOTLib.GAlearner(
    texts,
    labels,
    framework="torch",
    custom_hyperparameters=custom_hyperparams
)
```

### Direct Usage with SFNN

```python
from autoBOTLib.learning.torch_sparse_nn import SFNN

# Create model with torch.compile enabled
model = SFNN(
    batch_size=16,
    num_epochs=50,
    compile_model=True  # Enable compilation
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Benefits

- **Performance**: Can provide significant speedup during training and inference
- **Automatic**: Works transparently with existing code
- **Backward Compatible**: Default behavior unchanged (`compile_model=False`)

## Requirements

- PyTorch 2.0 or later
- Compatible hardware (some hardware may not support all compilation features)

## Example

See `examples/minimal_torch_compile.py` for a complete example comparing compiled vs non-compiled models.