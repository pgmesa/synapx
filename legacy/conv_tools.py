import ctypes
from pathlib import Path

import numpy as np
from synapgrad.conv_tools import get_conv2d_output_size

current_dir = Path(__file__).parent.absolute()
c_lib = ctypes.CDLL(str(current_dir/"conv_tools.so"))


def dilate_kernel(kernel_size:tuple, dilation:tuple):
    # Crea una matriz vacía para el kernel dilatado
    dilated_kernel = np.zeros((kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1), kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)))

    # Llena el kernel dilatado con unos en las posiciones correspondientes
    dilated_kernel[::dilation[0], ::dilation[1]] = 1

    return dilated_kernel


def im2col(a:np.ndarray, kernel_size, dilation=1, stride=1, padding=0, pad_value=0, as_unfold=False):
    kernel_size = np.broadcast_to(kernel_size, 2)
    dilation = np.broadcast_to(dilation, 2)
    padding = np.broadcast_to(padding, 2)
    stride = np.broadcast_to(stride, 2)
    
    lH, lW = get_conv2d_output_size(a.shape, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)
    
    N, C, H, W = a.shape
    
    out = np.zeros(lH*lW*N*C*kernel_size[0]*kernel_size[1]).astype(a.dtype)
    # Convert NumPy array to C pointer
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    c_array = a.ctypes.data_as(c_float_p)
    c_out = out.ctypes.data_as(c_float_p)

    # Llamar a la función sum_array en C
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    c_lib.im2col(c_array,
                ctypes.c_int(N), ctypes.c_int(C), ctypes.c_int(H), ctypes.c_int(W),
                ctypes.c_int(kH), ctypes.c_int(kW), ctypes.c_int(sH), ctypes.c_int(sW), ctypes.c_int(pH), ctypes.c_int(pW), ctypes.c_int(dH), ctypes.c_int(dW),
                c_out)

    out = out.reshape(C * kH * kW, N * lH * lW)

    return out
    
    
def col2im():
    ...
    
    
def extract_windows():
    ...
    
    
def place_windows():
    ...