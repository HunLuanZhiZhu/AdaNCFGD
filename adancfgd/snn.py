"""Spiking Neural Network (SNN) implementation with various components.

This module provides core components for building spiking neural networks, including:
- SNNDropout: Dropout layer for spike trains
- SNNBatchNorm1d/2d: Batch normalization for spike data
- SNNLinear: Fully connected layer for SNNs
- SNNLinearWithBatchNorm: Fully connected layer with batch normalization
- SNNConv2d: Convolutional layer for SNNs
- SNNConv2dWithBatchNorm: Convolutional layer with batch normalization
- SNN: Complete spiking neural network model

The implementation uses a surrogate gradient approach for training spiking neural networks.
"""
from typing import Optional, Union, Sequence, List
import torch
import torch.nn as nn
from torch import Tensor

WINDOW_T = 100
AVERAGE_TAU = 50
THRESHOLD_VOLTAGE = 15
WEIGHT_MEAN = 0
WEIGHT_STD = 1
INITIAL_POTENTIAL_RATIO = 0.5


# INITIAL_POTENTIAL_RATIO = 0

class ForwardFirstBackwardSecond(torch.autograd.Function):
    @staticmethod
    def forward(_, input1: Tensor, input2: Tensor) -> Tensor:
        return input1

    @staticmethod
    def backward(_, *grad_output: Tensor):
        grad_output = grad_output[0]
        return torch.zeros_like(grad_output), grad_output


class SNNDropout(nn.Module):
    def __init__(
            self,
            p: float = 0.5,
            inplace: bool = False,
    ):
        super().__init__()
        self.p = p
        self.inplace = inplace

        if self.p < 0 or self.p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, "
                "but got {}".format(self.p)
            )

        # Internal constants
        self._C1 = 1 / (1 - self.p)
        self._C2 = 1 - p

    def forward(
            self,
            spikes: Tensor,
            hat_spikes: Optional[Tensor] = None
    ) -> (Tensor, Optional[Tensor]):

        if not self.training or self.p == 0:
            return spikes, hat_spikes
        C1, C2 = self._C1, self._C2

        # Apply dropout in training mode
        if self.inplace:
            # In-place operation
            mask = torch.empty_like(spikes[0]).bernoulli_(C2)
            spikes.mul_(mask.unsqueeze(0) * C1)

            if hat_spikes is not None:
                hat_spikes.mul_(mask * C1)
        else:
            # Non-in-place operation
            mask = torch.empty_like(spikes[0]).bernoulli_(C2)
            spikes = spikes * mask.unsqueeze(0) * C1

            if hat_spikes is not None:
                hat_spikes = hat_spikes * mask * C1

        return spikes, hat_spikes


class SNNBatchNorm1d(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(
            self,
            spikes: Tensor,
    ) -> Tensor:

        dim = spikes.dim()

        if dim <= 2:
            spikes = self.bn(spikes)
        elif dim == 3:
            with torch.no_grad():
                eps = self.bn.eps
                weight, bias = self.bn.weight[None, None, :], self.bn.bias[None, None, :]
                mean, var = self.bn.running_mean[None, None, :], self.bn.running_var[None, None, :]
                spikes = (spikes - mean) / (var + eps).sqrt() * weight + bias
        else:
            raise ValueError(
                "SNNBatchNorm1d does not support dim > 3, "
                "but got dim {}".format(dim)
            )

        return spikes


class SNNBatchNorm2d(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(
            self,
            spikes: Tensor,
    ) -> Tensor:

        dim = spikes.dim()
        if dim <= 4:
            spikes = self.bn(spikes)
        elif dim == 5:
            with torch.no_grad():
                eps = self.bn.eps
                weight, bias = self.bn.weight[None, None, :, None, None], self.bn.bias[None, None, :, None, None]
                mean, var = self.bn.running_mean[None, None, :, None, None], self.bn.running_var[None, None, :, None,
                                                                             None]
                spikes = (spikes - mean) / (var + eps).sqrt() * weight + bias
        else:
            raise ValueError(
                "SNNBatchNorm2d does not support dim >5, "
                "but got dim {}".format(dim)
            )

        return spikes


class SNNLinear(nn.Module):
    def __init__(
            self,
            in_features: int, out_features: int,
            bias: bool = True, device=None, dtype=None,
            window_t: int = WINDOW_T,
            threshold_voltage: Union[int, float] = THRESHOLD_VOLTAGE,
            initial_potential_ratio: Union[int, float] = INITIAL_POTENTIAL_RATIO,
            w_mean: Union[int, float] = WEIGHT_MEAN,
            w_std: Union[int, float] = WEIGHT_STD
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias, device, dtype)
        self.window_t = window_t
        self.threshold_voltage = threshold_voltage
        self.initial_potential_ratio = initial_potential_ratio

        # Initialize weights
        self.fc.weight.data.normal_(mean=w_mean, std=w_std)
        if bias:
            self.fc.bias.data.normal_(mean=w_mean, std=w_std)

        # Internal constants
        self._C1 = initial_potential_ratio * threshold_voltage
        self._C2 = 1 / self.threshold_voltage
        self._C3 = (1 - initial_potential_ratio) / window_t

    def init_if(self, batch_size: int) -> Tensor:
        return torch.full(
            (batch_size, self.fc.out_features),
            fill_value=self._C1,
            device=self.fc.weight.device, dtype=self.fc.weight.dtype
        )

    def forward(
            self,
            spikes: Tensor,
            mem: Tensor,
            hat_spikes: Optional[Tensor] = None
    ) -> (Tensor, Tensor, Optional[Tensor]):
        steps, batch_size, in_features = spikes.shape
        out_features = self.fc.out_features
        device, dtype = spikes.device, spikes.dtype
        ss = torch.empty(
            (steps, batch_size, out_features),
            device=device, dtype=dtype
        )
        delta_us: Tensor = self.fc(spikes)

        if self.training:
            with torch.no_grad():

                sum_u = torch.zeros(
                    (batch_size, out_features),
                    device=device, dtype=dtype
                )
                for t in range(steps):
                    # Update membrane potential
                    mem += delta_us[t]
                    # Generate spikes
                    s = mem.gt(self.threshold_voltage).to(dtype)
                    ss[t] = s
                    # Reset membrane potential
                    mem.clamp_(0, self.threshold_voltage)
                    sum_u += mem
                    mem -= mem * s

                hat_s = ss.mean(dim=0)
                hat_u = sum_u / steps

            delta_u = self.fc(hat_spikes)  # .clamp_max(self.threshold_voltage)

            # Gradient calculation
            hat_u: Tensor = ForwardFirstBackwardSecond.apply(
                hat_u, delta_u
            )
            hat_s = ForwardFirstBackwardSecond.apply(
                hat_s,
                # (hat_u * self._C2).clamp(self._C3,1.5)
                (hat_u * self._C2).clamp_min(self._C3)
                # hat_u * self._C2
            )
        else:
            for t in range(steps):
                # Update membrane potential
                mem += delta_us[t]
                # Generate spikes
                s = mem.gt(self.threshold_voltage).to(dtype)
                ss[t] = s
                # Reset membrane potential
                mem.relu_()
                mem -= mem * s
            hat_s = None

        return ss, mem, hat_s


class SNNLinearWithBatchNorm(nn.Module):
    def __init__(
            self,
            in_features: int, out_features: int,
            bias: bool = True, device=None, dtype=None,
            window_t: int = WINDOW_T,
            threshold_voltage: Union[int, float] = THRESHOLD_VOLTAGE,
            initial_potential_ratio: Union[int, float] = INITIAL_POTENTIAL_RATIO,
            w_mean: Union[int, float] = WEIGHT_MEAN,
            w_std: Union[int, float] = WEIGHT_STD
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias, device, dtype)
        self.bn = SNNBatchNorm1d(out_features, device=device, dtype=dtype)
        self.window_t = window_t
        self.threshold_voltage = threshold_voltage
        self.initial_potential_ratio = initial_potential_ratio

        # Initialize weights
        self.fc.weight.data.normal_(mean=w_mean, std=w_std)
        if bias:
            self.fc.bias.data.normal_(mean=w_mean, std=w_std)

        # Internal constants
        self._C1 = initial_potential_ratio * threshold_voltage
        self._C2 = 1 / self.threshold_voltage
        self._C3 = (1 - initial_potential_ratio) / window_t

    def init_if(self, batch_size: int) -> Tensor:
        return torch.full(
            (batch_size, self.fc.out_features),
            fill_value=self._C1,
            device=self.fc.weight.device, dtype=self.fc.weight.dtype
        )

    def forward(
            self,
            spikes: Tensor,
            mem: Tensor,
            hat_spikes: Optional[Tensor] = None
    ) -> (Tensor, Tensor, Optional[Tensor]):
        steps, batch_size, in_features = spikes.shape
        out_features = self.fc.out_features
        device, dtype = spikes.device, spikes.dtype
        ss = torch.empty(
            (steps, batch_size, out_features),
            device=device, dtype=dtype
        )
        delta_us: Tensor = self.bn(self.fc(spikes))

        if self.training:
            with torch.no_grad():
                sum_u = torch.zeros(
                    (batch_size, out_features),
                    device=device, dtype=dtype
                )
                for t in range(steps):
                    # Update membrane potential
                    mem += delta_us[t]
                    # Generate spikes
                    s = mem.gt(self.threshold_voltage).to(dtype)
                    ss[t] = s
                    # Reset membrane potential
                    mem.clamp_(0, self.threshold_voltage)
                    sum_u += mem
                    mem -= mem * s

                hat_s = ss.mean(dim=0)
                hat_u = sum_u / steps

            delta_u = self.bn(self.fc(hat_spikes))  # .clamp_max(self.threshold_voltage)

            # Gradient calculation
            hat_u: Tensor = ForwardFirstBackwardSecond.apply(
                hat_u, delta_u
            )
            hat_s = ForwardFirstBackwardSecond.apply(
                hat_s,
                 (hat_u * self._C2).clamp(self._C3, 1.5)
                #(hat_u * self._C2).clamp_min(self._C3)
                # hat_u * self._C2
            )
        else:
            for t in range(steps):
                # Update membrane potential
                mem += delta_us[t]
                # Generate spikes
                s = mem.gt(self.threshold_voltage).to(dtype)
                ss[t] = s
                # Reset membrane potential
                mem.relu_()
                mem -= mem * s
            hat_s = None

        return ss, mem, hat_s


class SNNConv2d(nn.Module):
    def __init__(
            self,
            out_features,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # TODO: refine this type
            device=None,
            dtype=None,
            window_t: int = WINDOW_T,
            threshold_voltage: Union[int, float] = THRESHOLD_VOLTAGE,
            initial_potential_ratio: Union[int, float] = INITIAL_POTENTIAL_RATIO,
            w_mean: Union[int, float] = WEIGHT_MEAN,
            w_std: Union[int, float] = WEIGHT_STD
    ):
        super().__init__()
        self.out_features = out_features
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)
        self.window_t = window_t
        self.threshold_voltage = threshold_voltage
        self.initial_potential_ratio = initial_potential_ratio

        # Initialize weights
        self.conv.weight.data.normal_(mean=w_mean, std=w_std)
        if bias:
            self.conv.bias.data.normal_(mean=w_mean, std=w_std)

        # Internal constants
        self._C1 = initial_potential_ratio * threshold_voltage
        self._C2 = 1 / self.threshold_voltage
        self._C3 = (1 - initial_potential_ratio) / window_t

    def init_if(self, batch_size: int) -> Tensor:
        return torch.full(
            (batch_size, self.conv.out_channels, *self.out_features),
            fill_value=self._C1,
            device=self.conv.weight.device, dtype=self.conv.weight.dtype
        )

    def forward(
            self,
            spikes: Tensor,
            mem: Tensor,
            hat_spikes: Optional[Tensor] = None
    ) -> (Tensor, Tensor, Optional[Tensor]):
        steps, batch_size, in_channels, in_feature_width, in_feature_height = spikes.shape

        out_channels = self.conv.out_channels
        device, dtype = spikes.device, spikes.dtype
        ss = torch.empty(
            (steps, batch_size, out_channels, *self.out_features),
            device=device, dtype=dtype
        )
        spikes = spikes.view(-1, in_channels, in_feature_width, in_feature_height)
        delta_us: Tensor = self.conv(spikes)
        _, out_channels, out_feature_width, out_feature_height = delta_us.shape
        delta_us = delta_us.view(steps, batch_size, out_channels, out_feature_width, out_feature_height)

        if self.training:
            with torch.no_grad():
                sum_u = torch.zeros(
                    (batch_size, out_channels, in_feature_width, in_feature_height),
                    device=device, dtype=dtype
                )
                for t in range(steps):
                    # Update membrane potential
                    mem += delta_us[t]
                    # Generate spikes
                    s = mem.gt(self.threshold_voltage).to(dtype)
                    ss[t] = s
                    # Reset membrane potential
                    mem.clamp_(0, self.threshold_voltage)
                    sum_u += mem
                    mem -= mem * s

                hat_s = ss.mean(dim=0)
                hat_u = sum_u / steps

            delta_u = self.conv(hat_spikes)  # .clamp_max(self.threshold_voltage)

            # Gradient calculation
            hat_u: Tensor = ForwardFirstBackwardSecond.apply(
                hat_u, delta_u
            )
            hat_s = ForwardFirstBackwardSecond.apply(
                hat_s,
                (hat_u * self._C2).clamp(self._C3, 1.5)
                # (hat_u * self._C2).clamp_min(self._C3)
                # hat_u * self._C2
            )
        else:
            for t in range(steps):
                # Update membrane potential
                mem += delta_us[t]
                # Generate spikes
                s = mem.gt(self.threshold_voltage).to(dtype)
                ss[t] = s
                # Reset membrane potential
                mem.relu_()
                mem -= mem * s
            hat_s = None

        return ss, mem, hat_s


class SNNConv2dWithBatchNorm(nn.Module):
    def __init__(
            self,
            out_features,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # TODO: refine this type
            device=None,
            dtype=None,
            window_t: int = WINDOW_T,
            threshold_voltage: Union[int, float] = THRESHOLD_VOLTAGE,
            initial_potential_ratio: Union[int, float] = INITIAL_POTENTIAL_RATIO,
            w_mean: Union[int, float] = WEIGHT_MEAN,
            w_std: Union[int, float] = WEIGHT_STD
    ):
        super().__init__()
        self.out_features = out_features
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)

        self.bn = SNNBatchNorm2d(out_channels, device=device, dtype=dtype)

        self.window_t = window_t
        self.threshold_voltage = threshold_voltage
        self.initial_potential_ratio = initial_potential_ratio

        # Initialize weights
        self.conv.weight.data.normal_(mean=w_mean, std=w_std)
        if bias:
            self.conv.bias.data.normal_(mean=w_mean, std=w_std)

        # Internal constants
        self._C1 = initial_potential_ratio * threshold_voltage
        self._C2 = 1 / self.threshold_voltage
        self._C3 = (1 - initial_potential_ratio) / window_t

    def init_if(self, batch_size: int) -> Tensor:
        return torch.full(
            (batch_size, self.conv.out_channels, *self.out_features),
            fill_value=self._C1,
            device=self.conv.weight.device, dtype=self.conv.weight.dtype
        )

    def forward(
            self,
            spikes: Tensor,
            mem: Tensor,
            hat_spikes: Optional[Tensor] = None
    ) -> (Tensor, Tensor, Optional[Tensor]):
        steps, batch_size, in_channels, in_feature_width, in_feature_height = spikes.shape

        out_channels = self.conv.out_channels
        device, dtype = spikes.device, spikes.dtype
        ss = torch.empty(
            (steps, batch_size, out_channels, *self.out_features),
            device=device, dtype=dtype
        )
        spikes = spikes.view(-1, in_channels, in_feature_width, in_feature_height)
        delta_us: Tensor = self.conv(spikes)
        _, out_channels, out_feature_width, out_feature_height = delta_us.shape
        delta_us = self.bn(delta_us.view(steps, batch_size, out_channels, out_feature_width, out_feature_height))

        if self.training:
            with torch.no_grad():
                sum_u = torch.zeros(
                    (batch_size, out_channels, in_feature_width, in_feature_height),
                    device=device, dtype=dtype
                )
                for t in range(steps):
                    # Update membrane potential
                    mem += delta_us[t]
                    # Generate spikes
                    s = mem.gt(self.threshold_voltage).to(dtype)
                    ss[t] = s
                    # Reset membrane potential
                    mem.clamp_(0, self.threshold_voltage)
                    sum_u += mem
                    mem -= mem * s

                hat_s = ss.mean(dim=0)
                hat_u = sum_u / steps
            delta_u = self.bn(self.conv(hat_spikes))  # .clamp_max(self.threshold_voltage)

            # Gradient calculation
            hat_u: Tensor = ForwardFirstBackwardSecond.apply(
                hat_u, delta_u
            )
            hat_s = ForwardFirstBackwardSecond.apply(
                hat_s,
                (hat_u * self._C2).clamp(self._C3, 1.5)
                #(hat_u * self._C2).clamp_min(self._C3)
                # hat_u * self._C2
            )
        else:
            for t in range(steps):
                # Update membrane potential
                mem += delta_us[t]
                # Generate spikes
                s = mem.gt(self.threshold_voltage).to(dtype)
                ss[t] = s
                # Reset membrane potential
                mem.relu_()
                mem -= mem * s
            hat_s = None

        return ss, mem, hat_s


class SNN(nn.Module):
    def __init__(
            self,
            shapes: Sequence[int],  # Shape array: [input_features, hidden1_features, ..., output_features]
            bias: bool = True,
            window_t: int = WINDOW_T, average_tau=AVERAGE_TAU,
            threshold_voltage: Union[int, float] = THRESHOLD_VOLTAGE,
            initial_potential_ratio: Union[int, float] = INITIAL_POTENTIAL_RATIO,
            w_mean: Union[Sequence[Union[None, int, float]], int, float] = WEIGHT_MEAN,
            w_std: Union[Sequence[Union[None, int, float]], int, float] = WEIGHT_STD,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.shapes = shapes
        self.window_t = window_t
        self.average_tau = average_tau
        num_layers = len(shapes) - 1  # Number of layers = shape array length - 1 (multi-layer structure)
        self.num_layers = num_layers
        # Validate shape array (must contain at least input and output features)
        if self.num_layers < 1:
            raise ValueError(f"num_layers < 1,num_layers:{len(shapes)}")

        if isinstance(w_mean, Sequence):
            w_mean = [WEIGHT_MEAN if wi is None else wi for wi in w_mean]
        else:
            w_mean = WEIGHT_MEAN if w_mean is None else w_mean
            w_mean = [w_mean] * num_layers

        if isinstance(w_std, Sequence):
            w_std = [WEIGHT_STD if wi is None else wi for wi in w_std]
        else:
            w_std = WEIGHT_STD if w_std is None else w_std
            w_std = [w_std] * num_layers

        # Build layer structure (input → hidden → output, matching paper Fig.1 multi-layer architecture)
        self.layers: Optional[nn.ModuleList, List[SNNLinear]] = nn.ModuleList()
        for i, (w_mean, w_std) in enumerate(zip(w_mean, w_std)):
            self.layers.append(
                SNNLinear(
                    shapes[i], shapes[i + 1],
                    bias, device, dtype,
                    window_t, threshold_voltage,
                    initial_potential_ratio,
                    w_mean, w_std
                )
            )

        self.mems: Optional[List[Tensor]] = None

    def reset(self):
        self.mems = None

    def forward(
            self,
            spikes: Tensor,  # Input spike sequence, shape: (time_steps, batch, input_features),
            mems: List[Tensor],
            hat_spk: Optional[Tensor] = None  # Input spike sequence average, shape: (batch, input_features)
    ) -> (Tensor, Optional[Tensor]):

        for i, layer in enumerate(self.layers):
            spikes, mem, hat_spk = layer(
                spikes,
                mems[i],
                hat_spk
            )
            mems[i] = mem
        return spikes, hat_spk

    def step(
            self,
            spikes: Tensor,  # Input spike sequence, shape: (time_steps, batch, input_features),
            hat_spk: Optional[Tensor] = None  # Input spike sequence average, shape: (batch, input_features)
    ) -> Optional[Tensor]:
        steps, batch_size, in_features = spikes.shape
        if self.mems is None:
            self.mems = [layer.init_if(batch_size) for layer in self.layers]

        return self(spikes, self.mems, hat_spk)


if __name__ == "__main__":
    pass
