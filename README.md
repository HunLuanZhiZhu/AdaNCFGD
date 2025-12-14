# Adaptive Fractional Gradient Descent Optimizers (AdaFGD & AdaNCFGD)

[![PyPI version](https://badge.fury.io/py/adancfgd.svg)](https://badge.fury.io/py/adancfgd)
[![GitHub license](https://img.shields.io/github/license/HunLuanZhiZhu/AdaNCFGD.svg)](https://github.com/HunLuanZhiZhu/AdaNCFGD/blob/main/LICENSE)

## Description

The `adancfgd` package implements two advanced optimizers combining fractional gradient descent with adaptive learning rates, along with a comprehensive Spiking Neural Network (SNN) framework. Building on PyTorch's SGD, these algorithms enhance convergence and performance for machine learning tasks using fractional calculus-based gradient adjustments.

## Overview

This package provides two novel optimization algorithms and a complete SNN implementation:

### Optimizers
1. **AdaFGD** - Adaptive Fractional Gradient Descent: Uses fractional derivatives for gradient adjustment with adaptive learning rates.
2. **AdaNCFGD** - Adaptive Non-Causal Fractional Gradient Descent: Extends AdaFGD by considering multiple previous parameter values.

### Spiking Neural Network (SNN) Components
- **Core Layers**: SNNLinear, SNNConv2d, SNNDropout
- **Batch Normalization**: SNNBatchNorm1d, SNNBatchNorm2d
- **Composite Layers**: SNNLinearWithBatchNorm, SNNConv2dWithBatchNorm
- **Complete Models**: SNN, SNNCNN

## Installation

The package has been successfully uploaded to PyPI. You can install it directly from PyPI or from the source code.

### From PyPI

```bash
pip install adancfgd
```

### From Source

```bash
git clone https://github.com/HunLuanZhiZhu/AdaNCFGD.git
cd AdaNCFGD
pip install -e .
```

## Dependencies

- PyTorch >= 1.7.0
- NumPy
- Python >= 3.6

## Key Features

### AdaFGD
- Utilizes fractional derivatives for gradient adjustment
- Incorporates adaptive learning rates using first and second moment estimates
- Maintains parameter history for fractional gradient calculation
- Parameter updates handled by parent SGD class
- Supports AMSGrad variant
- Compatible with all PyTorch models

### AdaNCFGD
- Extends AdaFGD with non-causal fractional gradient descent
- Considers multiple previous parameter values (two steps back)
- Adapts gradient calculation based on parameter direction changes
- Maintains richer parameter history
- Supports all AdaFGD features
- Enhanced convergence for complex models

### SNN Components
- Spike-based neural network implementation
- Surrogate gradient approach for training
- Support for both fully connected and convolutional architectures
- Batch normalization for spike data
- Dropout regularization for SNNs
- Complete SNN and SNNCNN models

## API Documentation

### Optimizers

#### AdaFGD

```python
AdaFGD(params, lr=0.001, alpha=1.0, epsilon=1e-4, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, maximize=False, foreach=None, differentiable=False, fused=None)
```

**Parameters:**
- `params`: Iterable of parameters to optimize or dicts defining parameter groups
- `lr`: Learning rate (default: 0.001)
- `alpha`: Fractional order (must satisfy 0 < alpha < 2, default: 1.0)
- `epsilon`: Small positive constant to avoid division by zero (default: 1e-4)
- `momentum`: Momentum factor (default: 0.0)
- `dampening`: Dampening for momentum (default: 0.0)
- `weight_decay`: Weight decay (L2 penalty, default: 0.0)
- `nesterov`: Enables Nesterov momentum (default: False)
- `betas`: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
- `eps`: Term added to the denominator to improve numerical stability (default: 1e-8)
- `amsgrad`: Whether to use the AMSGrad variant (default: False)

#### AdaNCFGD

```python
AdaNCFGD(params, lr=0.001, alpha=1.0, epsilon=1e-4, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, maximize=False, foreach=None, differentiable=False, fused=None)
```

**Parameters:**
- Same as AdaFGD, with the addition of non-causal fractional gradient calculation

### SNN Components

#### SNNLinear

```python
SNNLinear(in_features, out_features, bias=True, window_t=100, threshold_voltage=15, initial_potential_ratio=0.5, w_mean=0, w_std=1, device=None, dtype=None)
```

#### SNNConv2d

```python
SNNConv2d(input_shape, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, window_t=100, threshold_voltage=15, initial_potential_ratio=0.5, w_mean=0, w_std=1, device=None, dtype=None)
```

#### SNNCNN

```python
SNNCNN(device=None, dtype=None)
```

## Usage Examples

### Example 1: Using AdaFGD with a Simple Model

```python
import torch
import torch.nn as nn
from adancfgd import AdaFGD

# Create a simple model
model = nn.Linear(10, 1)

# Initialize optimizer with default parameters
optimizer = AdaFGD(model.parameters(), lr=0.001, alpha=1.0)

# Training loop
inputs = torch.randn(32, 10)
targets = torch.randn(32, 1)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Example 2: Using AdaNCFGD with a CNN

```python
import torch
import torch.nn as nn
from adancfgd import AdaNCFGD

# Create a simple CNN
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 10)
)

# Initialize optimizer
optimizer = AdaNCFGD(model.parameters(), lr=0.0001, alpha=1.0, betas=(0.9, 0.999))

# Training loop (simplified)
inputs = torch.randn(32, 3, 32, 32)
targets = torch.randint(0, 10, (32,))
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Example 3: Using SNN Components

```python
import torch
from adancfgd import SNNLinear, SNNConv2d, SNNDropout, SNNSequential

# Create a simple SNN model using SNNSequential
snn_model = SNNSequential(
    SNNLinear(784, 256),
    SNNDropout(p=0.5),
    SNNLinear(256, 10)
)

# Create test data
spikes = torch.randn(10, 32, 784)  # (time_steps, batch, features)

# Forward pass
output = snn_model.step(spikes)
print(f"Output shape: {output.shape}")
```

### Example 4: Training SNNCNN on MNIST

```python
import torch
from adancfgd import SNNCNN
from adancfgd.train_mnist import train

# Create SNNCNN model
model = SNNCNN()

# Train on MNIST dataset
train(model, 'snncnn_test', epochs=10, batch_size=50)
```

## Performance Comparison

### Convergence Speed

AdaFGD and AdaNCFGD have been shown to converge faster than traditional optimizers like SGD and Adam for certain tasks, especially for models with complex loss landscapes.

### Accuracy

For image classification tasks, these optimizers can achieve higher final accuracy compared to standard optimizers when used with appropriate hyperparameters.

### SNN Performance

The SNN implementation supports efficient training of spiking neural networks using surrogate gradients, achieving competitive performance with traditional ANN models on MNIST and other datasets.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Yihe Zhu** - *Initial work* - [HunLuanZhiZhu](https://github.com/HunLuanZhiZhu)

## Acknowledgments

- This work was inspired by recent advances in fractional calculus and adaptive optimization algorithms.
- Built on PyTorch's robust deep learning framework.

## Contact

- Email: zhu.yihe@qq.com
- GitHub: [https://github.com/HunLuanZhiZhu/AdaNCFGD](https://github.com/HunLuanZhiZhu/AdaNCFGD)

## Version History

- **0.1.5** - Enhanced package stability, improved documentation, fixed minor issues, verified uninstall-reinstall functionality
- **0.1.4** - Complete SNN implementation, fixed import issues, improved documentation
- **0.1.3** - Fixed package structure, updated __init__.py
- **0.1.2** - Initial release with AdaFGD and AdaNCFGD optimizers

## Citation

If you use this package in your research, please consider citing:

```
@software{adancfgd2025,
  author = {Yihe Zhu},
  title = {adancfgd: Adaptive Fractional Gradient Descent Optimizers and SNN Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HunLuanZhiZhu/AdaNCFGD}},
}
```
