
from synapx import Tensor, nn
from synapx.nn import functional as F


class ReLU(nn.Module):
    """
    ReLU activation function. 
    
    The ReLU activation function is defined as:
    f(x) = max(0, x)
    """
    
    def forward(self, x:Tensor) -> Tensor:
        return F.relu(x)
    