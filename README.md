# Adaptive Fractional Gradient Descent Optimizers (AdaFGD & AdaNCFGD)

## Description

This module implements two advanced optimizers combining fractional gradient descent with adaptive learning rates. Building on PyTorch's SGD, these algorithms enhance convergence and performance for machine learning tasks using fractional calculus-based gradient adjustments.

## Overview

This module implements two novel optimization algorithms that combine fractional gradient descent with adaptive learning rate adjustment:

1. **AdaFGD** - Adaptive Fractional Gradient Descent: Uses fractional derivatives for gradient adjustment with adaptive learning rates.
2. **AdaNCFGD** - Adaptive Non-Causal Fractional Gradient Descent: Extends AdaFGD by considering multiple previous parameter values.

Both optimizers build upon PyTorch's SGD optimizer, incorporating adaptive learning rate mechanisms similar to Adam but with distinct implementations to differentiate from the original Adam.

## Key Features

### AdaFGD
- Utilizes fractional derivatives for gradient adjustment
- Incorporates adaptive learning rates using first and second moment estimates
- Maintains parameter history for fractional gradient calculation
- Parameter updates handled by parent SGD class
- Supports AMSGrad variant

### AdaNCFGD
- Extends AdaFGD with non-causal fractional gradient descent
- Considers multiple previous parameter values (two steps back)
- Adapts gradient calculation based on parameter direction changes
- Maintains richer parameter history
- Supports all AdaFGD features

## Installation

This module is part of the scientificProject311 package. Ensure you have the following dependencies:
- PyTorch >= 1.7.0
- Python >= 3.6

## Usage

### Basic Import

To use these optimizers, ensure the `adancfgd.py` file is in your current working directory or added to your Python path, then import directly:

```python
from adancfgd import AdaFGD, AdaNCFGD
```

### Example Usage with a Simple Model

```python
import torch
import torch.nn as nn
from adancfgd import AdaFGD, AdaNCFGD

# Create a simple model
model = nn.Linear(10, 1)

# Initialize optimizer with default parameters
optimizer = AdaFGD(model.parameters(), lr=0.001, alpha=1.0)
# or
# optimizer = AdaNCFGD(model.parameters(), lr=0.001, alpha=1.0)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
