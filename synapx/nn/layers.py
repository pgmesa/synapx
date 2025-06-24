
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