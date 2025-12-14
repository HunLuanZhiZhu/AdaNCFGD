"""Convolutional Spiking Neural Network (SNN-CNN) implementation.

This module implements a convolutional spiking neural network based on the components from the snn module.
It combines convolutional layers, pooling layers, and fully connected layers to create a complete SNN-CNN model.

The implementation follows a surrogate gradient approach for training spiking neural networks.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
from snn import SNNDropout, SNNConv2d, SNNLinear, SNNLinearWithBatchNorm, SNNConv2dWithBatchNorm  # Import basic components from snn module

# Reuse global parameters (consistent with snn file)
WINDOW_T = 100
AVERAGE_TAU = 50
THRESHOLD_VOLTAGE = 15
INITIAL_POTENTIAL_RATIO = 0.5
WEIGHT_MEAN = 0
WEIGHT_STD = 1


def pool_spikes(spikes: Tensor, hat_spk: Optional[Tensor], pool: nn.Module) -> [Tensor, Optional[Tensor]]:
    """Pooling layer forward propagation, applies pooling to input spike sequences"""
    if hat_spk is not None:
        hat_spk = pool(hat_spk)
    steps, batch_size, in_features, w, h = spikes.shape
    spikes = spikes.view(-1, in_features, w, h)
    spikes = pool(spikes)
    _, out_features, w, h = spikes.shape
    spikes = spikes.view(steps, batch_size, out_features, w, h)
    return spikes, hat_spk


class SNNCNN(nn.Module):
    """Convolutional model based on spiking neural network, strictly reproducing the target architecture"""

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.window_t = WINDOW_T
        self.average_tau = AVERAGE_TAU
        # ========== Convolutional layer module (matching original model's Conv2D structure) ==========
        self.conv1 = SNNConv2d(
            [28, 28],
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            padding=2,  # 5x5 convolution maintains 28x28 size
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        self.conv2 = SNNConv2d(
            [28, 28],
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            padding=2,  # Maintain 28x28 size
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        self.conv3 = SNNConv2dWithBatchNorm(
            [14, 14],
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,  # 3x3 convolution maintains 14x14 size
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        self.conv4 = SNNConv2dWithBatchNorm(
            [14, 14],
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,  # Maintain 14x14 size
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        # ========== Pooling and Dropout layers ==========
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 14x14 → 7x7
        self.dropout1 = SNNDropout(p=0.5)

        # ========== Fully connected layer module (matching original model's Dense structure) ==========
        self.fc1 = SNNLinearWithBatchNorm(
            in_features=7 * 7 * 64,  # Flattened 7x7 feature map dimension
            out_features=512,
            bias=True,
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        self.fc2 = SNNLinear(
            in_features=512,
            out_features=10,  # Number of output classes
            bias=True,  # With bias
            window_t=WINDOW_T,
            threshold_voltage=THRESHOLD_VOLTAGE,
            initial_potential_ratio=INITIAL_POTENTIAL_RATIO,
            w_mean=WEIGHT_MEAN,
            w_std=WEIGHT_STD,
            device=device,
            dtype=dtype
        )

        # Membrane potential storage (maintained during training/inference)
        self.mems: Optional[List[Tensor]] = None

    def reset(self):
        self.mems = None

    def forward(
            self,
            spikes: Tensor,  # Input spike sequence, shape: (time_steps, batch, input_features),
            mems: List[Tensor],
            hat_spk: Optional[Tensor] = None  # Input spike sequence average, shape: (batch, input_features)
    ) -> (Tensor, Optional[Tensor]):

        steps, batch_size, in_features, w, h = spikes.shape
        # 1
        spikes, men0, hat_spk = self.conv1(spikes, mems[0], hat_spk)

        # # 2
        spikes, men1, hat_spk = self.conv2(spikes, mems[1], hat_spk)
        spikes, hat_spk = pool_spikes(spikes, hat_spk, self.pool1)

        # 3
        spikes, men2, hat_spk = self.conv3(spikes, mems[2], hat_spk)

        # 4
        spikes, men3, hat_spk = self.conv4(spikes, mems[3], hat_spk)


        spikes, hat_spk = pool_spikes(spikes, hat_spk, self.pool2)

        spikes = spikes.view(steps, batch_size, -1)
        if hat_spk is not None:
            hat_spk = hat_spk.view(batch_size, -1)

        # 5
        spikes, men4, hat_spk = self.fc1(spikes, mems[4], hat_spk)
        spikes, hat_spk = self.dropout1(spikes, hat_spk)

        # 6
        spikes, men5, hat_spk = self.fc2(spikes, mems[5], hat_spk)

        mems[:] = (men0,
                   men1,
                   men2,
                   men3,
                   men4,
                   men5,
                   )

        return spikes, hat_spk

    def step(
            self,
            spikes: Tensor,  # Input spike sequence, shape: (time_steps, batch, input_features),
            hat_spk: Optional[Tensor] = None  # Input spike sequence average, shape: (batch, input_features)
    ) -> Optional[Tensor]:
        steps, batch_size, in_features, w, h = spikes.shape

        if self.mems is None:
            self.mems = [
                self.conv1.init_if(batch_size),
                self.conv2.init_if(batch_size),
                self.conv3.init_if(batch_size),
                self.conv4.init_if(batch_size),
                self.fc1.init_if(batch_size),
                self.fc2.init_if(batch_size),
            ]

        return self(spikes, self.mems, hat_spk)


# Verify total number of model parameters (should match original model: 887,530)
if __name__ == "__main__":
    model = SNNCNN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")  
    # Test input shape compatibility
    test_spikes = torch.randn(10, 50, 1, 28, 28)  # (time_steps=10, batch=50, channels=1, height=28, width=28)
    output, _ = model.step(test_spikes, test_spikes.mean(dim=0))
    print(f"Output shape: {output.shape}")  # Should output (10, 50, 10)
    model.eval()
    output, _ = model.step(test_spikes)
    print(f"Output shape: {output.shape}")  # Should output (10, 50, 10)

























































