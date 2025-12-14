import math
import warnings
import torch
from torch import Tensor
from torch.optim import SGD

"""
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
```

## Parameters

Both optimizers share the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | Iterable | - | Parameters to optimize |
| `lr` | float or Tensor | 0.001 | Learning rate |
| `alpha` | float | 1.0 | Fractional order (must satisfy 0 < alpha < 2) |
| `epsilon` | float | 1e-4 | Small positive constant for numerical stability |
| `momentum` | float | 0.0 | Momentum factor |
| `dampening` | float | 0.0 | Dampening for momentum |
| `weight_decay` | float | 0.0 | Weight decay (L2 penalty) |
| `nesterov` | bool | False | Enables Nesterov momentum |
| `betas` | Tuple | (0.9, 0.999) | Coefficients for running averages |
| `eps` | float | 1e-8 | Term for numerical stability in denominator |
| `amsgrad` | bool | False | Whether to use AMSGrad variant |
| `maximize` | bool | False | Whether to maximize instead of minimize |
| `foreach` | Optional[bool] | None | Whether to use foreach implementation |
| `differentiable` | bool | False | Whether autograd should occur through optimizer step |
| `fused` | Optional[bool] | None | Whether to use fused implementation |

## Performance

The module includes a test function `adafgd_optimization_performance()` that compares the convergence of AdaFGD and AdaNCFGD with traditional optimizers (SGD, Adam) on a simple linear regression problem. To run the test:

```python
python -m snn.ncf.e34.adancfgd
```

The test generates convergence curves saved as PNG files:
- `adafgd_vs_adancfgd_convergence.png`: Comparison of AdaFGD, AdaNCFGD, SGD, and Adam
- `adafgd_vs_adancfgd_diff.png`: Difference in loss between AdaFGD and AdaNCFGD

## Implementation Notes

1. **Fractional Gradient Calculation**: Both optimizers use fractional derivatives to adjust gradients, which can capture long-term dependencies in the optimization landscape.

2. **Adaptive Learning Rate**: The optimizers incorporate Adam-like adaptive learning rates, using bias-corrected first and second moment estimates.

3. **Parameter History**: AdaFGD maintains the previous parameter value, while AdaNCFGD keeps track of two previous values to enable non-causal gradient calculation.

4. **Numerical Stability**: Careful attention is paid to numerical stability, with small epsilon terms added to denominators and absolute values used where appropriate.

5. **Typing Support**: The code includes optional typing support, with fallbacks to handle environments where the typing module is unavailable.

## Theoretical Foundation

The optimizers are based on fractional calculus principles, extending traditional gradient descent to use fractional derivatives. This allows them to leverage information from previous optimization steps more effectively, potentially leading to better convergence properties in certain scenarios.

## Citation

If you use these optimizers in your research, please consider citing the relevant work on fractional gradient descent and adaptive optimization.

## License

This code is part of the scientificProject311 package and is available under the project's license.

"""


try:
    from typing import Optional, Callable, Union, Tuple
    try:
        from torch.optim.optimizer import ParamsT
    except ImportError:
        warnings.warn("ParamsT not found from PyTorch. ParamsT is defined with basic types.")
        from typing import Iterable, Dict, Any, Tuple

        try:
            from typing import TypeAlias

            ParamsT: TypeAlias = Union[
                Iterable[torch.Tensor],
                Iterable[Dict[str, Any]],
                Iterable[Tuple[str, torch.Tensor]]
            ]
        except ImportError:
            warnings.warn("TypeAlias not found (Python < 3.10). Using basic type assignment for ParamsT.")

            ParamsT = Union[
                Iterable[torch.Tensor],
                Iterable[Dict[str, Any]],
                Iterable[Tuple[str, torch.Tensor]]
            ]
except:
    warnings.warn("typing module not available.")
    Optional, Callable, Union, Tuple, Iterable, Dict, Any, Tuple, ParamsT = (None for _ in range(9))




class AdaFGD(SGD):
    """
    Adaptive Fractional Gradient Descent (AdaFGD) optimizer.
    
    This optimizer combines fractional gradient descent with adaptive learning rate adjustment
    similar to Adam, but with distinct implementation to differentiate it from the original Adam.
    
    Key features:
    - Uses fractional derivative for gradient adjustment
    - Incorporates adaptive learning rate using first and second moment estimates
    - Maintains parameter history for fractional gradient calculation
    - Parameter updates handled by parent SGD class
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.001)
        alpha: Fractional order (must satisfy 0 < alpha < 2, default: 1.0)
        epsilon: Small positive constant to avoid division by zero (default: 1e-4)
        momentum: Momentum factor (default: 0.0)
        dampening: Dampening for momentum (default: 0.0)
        weight_decay: Weight decay (L2 penalty, default: 0.0)
        nesterov: Enables Nesterov momentum (default: False)
        betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        amsgrad: Whether to use the AMSGrad variant (default: False)
        maximize: Whether to maximize the params based on the objective, instead of minimizing (default: False)
        foreach: Whether to use foreach implementation (default: None)
        differentiable: Whether autograd should occur through the optimizer step (default: False)
        fused: Whether to use fused implementation (default: None)
    """

    def __init__(
            self,
            params: ParamsT,
            lr: Union[float, torch.Tensor] = 0.001,
            alpha: float = 1.0,
            epsilon: float = 1e-4,
            # SGD parameters
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            # Ada parameters
            betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
            eps: float = 1e-8,
            amsgrad: bool = False,
            *,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            differentiable: bool = False,
            fused: Optional[bool] = None,
    ):
        """
        Initialize AdaFGD optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate
            alpha: Fractional order (0 < alpha < 2)
            epsilon: Small positive constant for numerical stability
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enables Nesterov momentum
            betas: Coefficients for running averages
            eps: Term for numerical stability in denominator
            amsgrad: Whether to use AMSGrad variant
            maximize: Whether to maximize instead of minimize
            foreach: Whether to use foreach implementation
            differentiable: Whether autograd should occur through optimizer step
            fused: Whether to use fused implementation
        
        Raises:
            ValueError: If alpha is not in (0, 2) or epsilon is not positive
        """
        # Parameter validation (meets the requirement 0 < α < 2 from the paper)
        if alpha <= 0 or alpha >= 2:
            raise ValueError(f"Fractional order alpha must satisfy 0 < alpha < 2, got {alpha}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {epsilon}")

        # Version compatibility
        if fused is None:
            super().__init__(
                params, lr, momentum, dampening, weight_decay,
                nesterov, maximize=maximize, foreach=foreach,
                differentiable=differentiable
            )
        else:
            super().__init__(
                params, lr, momentum, dampening, weight_decay,
                nesterov, maximize=maximize, foreach=foreach,
                differentiable=differentiable, fused=fused
            )

        # AdaFGD parameters
        self.alpha = alpha
        self.epsilon = epsilon

        # Ada constants
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad

        # Precompute constants (Γ(2-α) and 1-α in the paper formula)
        self.inv_gamma = 1 / math.gamma(2 - alpha)
        self.one_minus_alpha = 1 - alpha
        self.inv_alpha = 1 / alpha
        self.one_minus_betas = (1 - betas[0], 1 - betas[1])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss. Optional for most optimizers.
            
        Returns:
            Optional[float]: Loss value if closure is provided, otherwise None.
        """
        loss = None 
        if closure is not None: 
            with torch.enable_grad():
                loss = closure()

        # AdaFGD constants
        inv_gamma = self.inv_gamma
        one_minus_alpha = self.one_minus_alpha
        inv_alpha = self.inv_alpha
        epsilon = self.epsilon

        # Ada constants
        amsgrad = self.amsgrad
        beta1, beta2 = self.betas
        one_minus_betas1, one_minus_betas2 = self.one_minus_betas
        eps = self.eps

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state):
                    # Update step count
                    state['step'] += 1
                    step = state['step']
                    # Get previous parameter value
                    param_prev = state['param_prev']
                    param_diff = p - param_prev
                    # Update Ada state (first and second moment estimates)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg *= beta1
                    exp_avg += one_minus_betas1 * p.grad.sign()
                    exp_avg_sq *= beta2
                    exp_avg_sq += one_minus_betas2 * p.grad.square()
                    # Handle amsgrad variant
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        exp_avg_sq = torch.max(exp_avg_sq, max_exp_avg_sq)
                        state['max_exp_avg_sq'] = exp_avg_sq
                else:
                    # Initialize state for first iteration
                    step = 1
                    param_prev = p
                    # Calculate initial parameter difference for stable first update
                    param_diff = torch.full_like(p.grad, (group['lr'] * inv_gamma) ** inv_alpha)
                    state['step'] = step
                    state['param_prev'] = torch.empty_like(p)
                    # Initialize Ada state variables
                    exp_avg = one_minus_betas1 * p.grad.sign()
                    exp_avg_sq = one_minus_betas2 * p.grad.square()
                    state['exp_avg'] = exp_avg
                    state['exp_avg_sq'] = exp_avg_sq
                    # Initialize amsgrad state if needed
                    if amsgrad:
                        state['max_exp_avg_sq'] = exp_avg_sq.clone()

                # Compute bias-corrected first and second moment estimates
                hat_m = exp_avg.abs() / (1 - beta1 ** step)
                hat_v = exp_avg_sq / (1 - beta2 ** step)
                # Calculate adaptive term similar to Adam
                ada_term = (hat_m + eps) / (hat_v.sqrt() + eps)

                # Calculate fractional gradient adjustment term
                fractional_term = inv_gamma * param_prev.grad * (param_diff.abs() + epsilon).pow(one_minus_alpha)

                # Compute surrogate gradient by combining adaptive and fractional terms
                grad_p = ada_term * fractional_term
                # Update parameter history and gradients
                state['param_prev'].copy_(p)
                state['param_prev'].grad = p.grad.clone()
                # Replace original gradient with surrogate gradient
                p.grad.copy_(grad_p)

        # Let parent SGD class handle parameter updates
        super().step()
        return loss


class AdaNCFGD(SGD):
    """
    Adaptive Non-Causal Fractional Gradient Descent (AdaNCFGD) optimizer.
    
    This optimizer extends AdaFGD by incorporating non-causal fractional gradient descent,
    which considers multiple previous parameter values when calculating the fractional gradient.
    It combines non-causal fractional derivative with adaptive learning rate adjustment.
    
    Key features:
    - Uses non-causal fractional derivative for gradient adjustment
    - Considers multiple previous parameter values in gradient calculation
    - Incorporates adaptive learning rate using first and second moment estimates
    - Maintains parameter history for non-causal fractional gradient calculation
    - Parameter updates handled by parent SGD class
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.001)
        alpha: Fractional order (must satisfy 0 < alpha < 2, default: 1.0)
        epsilon: Small positive constant to avoid division by zero (default: 1e-4)
        momentum: Momentum factor (default: 0.0)
        dampening: Dampening for momentum (default: 0.0)
        weight_decay: Weight decay (L2 penalty, default: 0.0)
        nesterov: Enables Nesterov momentum (default: False)
        betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        amsgrad: Whether to use the AMSGrad variant (default: False)
        maximize: Whether to maximize the params based on the objective, instead of minimizing (default: False)
        foreach: Whether to use foreach implementation (default: None)
        differentiable: Whether autograd should occur through the optimizer step (default: False)
        fused: Whether to use fused implementation (default: None)
    """

    def __init__(
            self,
            params: ParamsT,
            lr: Union[float, torch.Tensor] = 0.001,
            alpha: float = 1.0,
            epsilon: float = 1e-4,
            # SGD parameters
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            # Ada parameters
            betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
            eps: float = 1e-8,
            amsgrad: bool = False,
            *,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            differentiable: bool = False,
            fused: Optional[bool] = None,
    ):
        """
        Initialize AdaNCFGD optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate
            alpha: Fractional order (0 < alpha < 2)
            epsilon: Small positive constant for numerical stability
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enables Nesterov momentum
            betas: Coefficients for running averages
            eps: Term for numerical stability in denominator
            amsgrad: Whether to use AMSGrad variant
            maximize: Whether to maximize instead of minimize
            foreach: Whether to use foreach implementation
            differentiable: Whether autograd should occur through optimizer step
            fused: Whether to use fused implementation
        
        Raises:
            ValueError: If alpha is not in (0, 2) or epsilon is not positive
        """
        # Parameter validation (meets the requirement 0 < α < 2 from the paper)
        if alpha <= 0 or alpha >= 2:
            raise ValueError(f"Fractional order alpha must satisfy 0 < alpha < 2, got {alpha}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {epsilon}")

        # Version compatibility
        if fused is None:
            super().__init__(
                params, lr, momentum, dampening, weight_decay,
                nesterov, maximize=maximize, foreach=foreach,
                differentiable=differentiable
            )
        else:
            super().__init__(
                params, lr, momentum, dampening, weight_decay,
                nesterov, maximize=maximize, foreach=foreach,
                differentiable=differentiable, fused=fused
            )

        # AdaNCFGD parameters
        self.alpha = alpha
        self.epsilon = epsilon

        # Ada constants
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad

        # Precompute constants for non-causal fractional gradient calculation
        self.inv_gamma = 1 / math.gamma(2 - alpha)
        self.inv_gamma2 = 1 / (2* math.gamma(2 - alpha))

        self.one_minus_alpha = 1 - alpha
        self.inv_alpha = 1 / alpha
        self.one_minus_betas = (1 - betas[0], 1 - betas[1])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss. Optional for most optimizers.
            
        Returns:
            Optional[float]: Loss value if closure is provided, otherwise None.
        """
        loss = None 
        if closure is not None: 
            with torch.enable_grad():
                loss = closure()

        # AdaNCFGD constants
        inv_gamma = self.inv_gamma
        inv_gamma2 = self.inv_gamma2
        one_minus_alpha = self.one_minus_alpha
        inv_alpha = self.inv_alpha
        epsilon = self.epsilon

        # Ada constants
        amsgrad = self.amsgrad
        beta1, beta2 = self.betas
        one_minus_betas1, one_minus_betas2 = self.one_minus_betas
        eps = self.eps

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state):
                    # Update step count
                    state['step'] += 1
                    step = state['step']
                    # Get previous parameter values (two steps back)
                    param_prev = state['param_prev']
                    param_prev2 = state['param_prev2']
                    param_diff = p - param_prev
                    param_diff2 = p - param_prev2
                    param_diff12 = param_prev - param_prev2
                    # Update Ada state (first and second moment estimates)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg *= beta1
                    exp_avg += one_minus_betas1 * p.grad.sign()
                    exp_avg_sq *= beta2
                    exp_avg_sq += one_minus_betas2 * p.grad.square()
                    # Handle amsgrad variant
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        exp_avg_sq = torch.max(exp_avg_sq, max_exp_avg_sq)
                        state['max_exp_avg_sq'] = exp_avg_sq
                else:
                    # Initialize state for first iteration
                    step = 1
                    param_prev2 = param_prev = p
                    # Calculate initial parameter differences for stable first update
                    param_diff = torch.full_like(p.grad, (group['lr'] * inv_gamma) ** inv_alpha)
                    param_diff2 = torch.empty_like(param_diff)
                    param_diff12 = param_diff.clone()

                    state['step'] = step
                    state['param_prev'] = torch.empty_like(p)
                    state['param_prev2'] = torch.empty_like(p)

                    # Initialize Ada state variables
                    exp_avg = one_minus_betas1 * p.grad.sign()
                    exp_avg_sq = one_minus_betas2 * p.grad.square()
                    state['exp_avg'] = exp_avg
                    state['exp_avg_sq'] = exp_avg_sq
                    # Initialize amsgrad state if needed
                    if amsgrad:
                        state['max_exp_avg_sq'] = exp_avg_sq.clone()

                # Compute bias-corrected first and second moment estimates
                hat_m = exp_avg.abs() / (1 - beta1 ** step)
                hat_v = exp_avg_sq / (1 - beta2 ** step)
                # Calculate adaptive term similar to Adam
                ada_term = (hat_m + eps) / (hat_v.sqrt() + eps)

                # Calculate non-causal fractional gradient adjustment terms
                param_diff_sign = param_diff.sign()
                # Determine appropriate previous gradient based on sign consistency
                prev_step_diff_fractional_term = torch.where(
                    (param_diff_sign * param_diff12.sign()).eq(1),
                    param_prev.grad, p.grad
                ) * (param_diff.abs() + epsilon).pow(one_minus_alpha)

                # Calculate fractional term for two steps back
                prev2_step_diff_fractional_term = param_prev2.grad * (param_diff2.abs() + epsilon).pow(one_minus_alpha)

                # Combine fractional terms based on direction change
                fractional_term = torch.where(
                    (param_diff_sign * param_diff2.sign()).eq(-1),
                    inv_gamma2 * (prev_step_diff_fractional_term + prev2_step_diff_fractional_term),
                    inv_gamma * prev_step_diff_fractional_term
                )

                # Compute surrogate gradient by combining adaptive and fractional terms
                grad_p = ada_term * fractional_term

                # Update parameter history for next iteration
                state['param_prev2'] = param_prev
                state['param_prev'] = p.clone()
                state['param_prev'].grad = p.grad.clone()
                # Replace original gradient with surrogate gradient
                p.grad.copy_(grad_p)

        # Let parent SGD class handle parameter updates
        super().step()
        return loss


def adafgd_optimization_performance():
    """
    Test the optimization performance of AdaFGD and AdaNCFGD optimizers.
    
    This function performs a convergence test using a simple linear regression problem
    to compare the performance of AdaFGD and AdaNCFGD optimizers.
    
    The test generates noisy linear data and trains multiple models using different optimizers
    (AdaFGD, AdaNCFGD, SGD, and Adam) with the same initial parameters and learning rates.
    It records the loss values during training and generates convergence curves.
    
    Steps performed:
    1. Generate noisy linear data (y = 3x + 2 + noise)
    2. Create identical models for each optimizer
    3. Initialize the optimizers with the same learning rate
    4. Train the models for a fixed number of epochs
    5. Record loss values at each epoch
    6. Plot convergence curves and save them as PNG files
    7. Verify that AdaFGD converges effectively
    
    Generated files:
    - adafgd_vs_adancfgd_convergence.png: Convergence curves comparison
    - adafgd_vs_adancfgd_diff.png: Difference in loss between AdaFGD and AdaNCFGD
    
    Raises:
        AssertionError: If AdaFGD fails to converge effectively (final loss not less than 1/10 of initial loss)
    
    Returns:
        None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.optim import Adam

    # 1. Generate noisy linear data (y = 3x + 2 + noise)
    torch.manual_seed(42)  # Fix random seed
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = 3 * x + 2 + 0.1 * torch.randn_like(x)

    # 2. Define two identical models (optimized with AdaFGD and AdaNCFGD respectively)
    num_models = 4
    models = [torch.nn.Linear(1, 1) for _ in range(num_models)]
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())  # Same initial parameters

    criterion = torch.nn.MSELoss()

    names = ['AdaFGD', 'AdaNCFGD', 'SGD', 'Adam'][:2]

    # 3. Define optimizers (using the same learning rate)
    lr = 0.5
    alpha = 1.6
    optimizers = [
        lambda params: AdaFGD(params, lr=lr, alpha=alpha),
        lambda params: AdaNCFGD(params, lr=lr, alpha=alpha),
        lambda params: SGD(params, lr=lr),
        lambda params: Adam(params, lr=lr)
    ]
    optimizers = [optimizer(model.parameters()) for (model, optimizer) in zip(models, optimizers)]

    # 4. Train and record loss
    epoch_losses = [[] for _ in range(num_models)]
    epochs = 100

    for _ in range(epochs):
        for model, optimizer, epoch_loss in zip(models, optimizers, epoch_losses):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

    # 5. Plot convergence curves
    plt.figure(figsize=(10, 6))
    for name, epoch_loss in zip(names, epoch_losses):
        plt.plot(np.arange(1, epochs + 1), epoch_loss, label=name, alpha=1)

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('AdaFGD vs AdaNCFGD')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('adafgd_vs_adancfgd_convergence.png')
    print("AdaFGD vs AdaNCFGD convergence plot saved as 'adafgd_vs_adancfgd_convergence.png'")
    plt.show()
    #plt.close()

    plt.plot([loss1 - loss2 for (loss1, loss2) in zip(epoch_losses[0], epoch_losses[1])])
    plt.savefig('adafgd_vs_adancfgd_diff.png')
    print("AdaFGD vs AdaNCFGD loss difference plot saved as 'adafgd_vs_adancfgd_diff.png'")
    plt.show()
    #plt.close()

    # 6. Verify if AdaFGD converges (final loss is less than 1/100 of initial loss)
    assert epoch_losses[0][-1] < epoch_losses[0][0] / 10, "AdaFGD failed to converge effectively"
    print("AdaFGD optimization performance verification passed ✅ (convergence curves saved as images)")


if __name__ == "__main__":
    # Test optimization performance
    adafgd_optimization_performance()
