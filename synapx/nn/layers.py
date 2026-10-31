
import math

import torch
import synapx
from synapx import nn, Tensor
from synapx.nn import functional as F, init


class Linear(nn.Module):
    
    def __init__(self, in_features:int, out_features:int, bias=True):
        """ 
        Applies a linear transformation to the incoming data: y = x @ w.T + b. 
        
        This layer is also known as Dense layer.
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        
        Args:
            in_features (int): The number of features in the input tensor.
            out_features (int): The number of features in the output tensor.
            weight_init_method (str): The method to use for initializing the weights.
                Options are 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform'.
                Defaults to 'he_normal'.
            bias: Whether to use a bias or not. Defaults to True.
        
        Returns:
            A tensor of shape (batch_size, output_size)
        
        Variables:
            - weight (synapx.Tensor) - the learnable weights of the module of shape
                (out_features, in_features). The values are initialized from `U(-sqrt(k), sqrt(k))`
                where `k = 1/in_features`.
            - bias (synapx.Tensor) - the learnable bias of the module of shape (out_features).
                If `bias=True`, the values are initialized from `U(-sqrt(k), sqrt(k))` where
                `k = 1/in_features`.
        
        Notes:
            - The input tensor is expected to have a shape of (batch_size, input_size).
            - The output tensor is expected to have a shape of (batch_size, output_size).
        
        Example:
            >>> layer = nn.Linear(input_size=3, output_size=4)
            >>> x = synapx.ones((2, 3))
            >>> y = layer(x)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
      
        self.weight = nn.Parameter(synapx.empty((out_features, in_features), dtype=torch.float32, requires_grad=True))
        if bias: 
            self.bias = nn.Parameter(synapx.empty((out_features,), dtype=torch.float32, requires_grad=True))
        else:
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        """ 
        Resets the parameters of the layer.
        """
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        std = 1. / math.sqrt(float(fan_in)) if fan_in > 0 else 0
        init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            init.uniform_(self.bias, -std, std)
        
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.in_features, f"Expected input size '{self.in_features}' but received '{x.shape[1]}'"
        return F.linear(x, self.weight, self.bias)
    
    
class Flatten(nn.Module):
    
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        """ 
        Flattens a tensor over the specified start and end dimensions
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.flatten.html
            
        Args:
            start_dim (int): The dimension to start flattening.
            end_dim (int): The dimension to end flattening.
        
        Returns:
            A tensor of shape (batch_size, *output_shape)
        
        Example:
            >>> layer = nn.Flatten()
            >>> x = synapgrad.ones((2, 3, 4, 5))
            >>> y = layer(x)
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)
    

class Dropout(nn.Module):

    def __init__(self, p=0.5) -> None:
        """
        Randomly zeroes some of the elements of the input tensor with probability p
        using samples from a Bernoulli distribution. The values are also scaled by 1/(1-p)
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
            
        Args:
            p (float): The probability of an element to be zeroed.
        
        Returns:
            A tensor of the same shape as the input tensor.
        
        Example:
            >>> layer = nn.Dropout(p=0.5)
            >>> x = synapgrad.ones((2, 3))
            >>> y = layer(x)
        """
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)


class MaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        """
        Applies a 2D max pooling over an input of shape (N, C, H, W).

        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)


class Conv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, bias=True) -> None:
        """
        Applies a 2D convolution over an input of shape (N, C, H, W).

        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Variables:
            - weight (synapx.Tensor) - the learnable weights, of shape
                (out_channels, in_channels, kernel_size[0], kernel_size[1]).
            - bias (synapx.Tensor) - the learnable bias, of shape (out_channels).
        """
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if padding == 'same':
            if stride != 1 and stride != (1, 1):
                raise ValueError("padding='same' is not supported for strided convolutions")
            if dilation != 1 or kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                raise ValueError("padding='same' needs an odd kernel_size and dilation=1")
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif padding == 'valid':
            padding = 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(synapx.empty((out_channels, in_channels, *kernel_size), dtype=torch.float32, requires_grad=True))
        if bias:
            self.bias = nn.Parameter(synapx.empty((out_channels,), dtype=torch.float32, requires_grad=True))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(float(fan_in)) if fan_in > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class BatchNorm2d(nn.Module):

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True) -> None:
        """
        Applies batch normalization over an input of shape (N, C, H, W).

        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.running_mean = synapx.zeros((num_features,), dtype=torch.float32) if track_running_stats else None
        self.running_var = synapx.ones((num_features,), dtype=torch.float32) if track_running_stats else None

        if affine:
            self.weight = nn.Parameter(synapx.ones((num_features,), dtype=torch.float32, requires_grad=True))
            self.bias = nn.Parameter(synapx.zeros((num_features,), dtype=torch.float32, requires_grad=True))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"Expected a 4D input but received {x.dim()}D"
        training = self.training or self.running_mean is None
        return F.batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var, training, self.momentum, self.eps)