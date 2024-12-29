# SynapX

**SynapX** is a Python Deep Learning library that implements its core functionality in C++ with Python bindings, designed to outperform my previous project, [SynapGrad](https://github.com/pgmesa/synapgrad). SynapGrad was built entirely in Python with NumPy as its tensor backend, while SynapX takes advantage of C++ for critical operations, offering higher performance and scalability.  

This project combines the raw computational power of C++ with Python's ease of use, leveraging **libtorch** as the backend for CPU tensor operations, and aims to support GPU acceleration in the near future.

## Backend Exploration  

The main branch uses libtorch as the tensor backend, selected for its efficiency and robust support for operations like broadcasting and batched matrix multiplications. For insights into the backend evaluation process and comparisons with alternatives like Xtensor, check out the [backend exploration branch](https://github.com/pgmesa/synapx/tree/xtensor-openblas).  

## Project Status  

SynapX is still under development...