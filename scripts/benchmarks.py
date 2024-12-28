import sys
import time
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

sys.path.append(str(Path(__file__).parent))


import torch
import synapx
import synapgrad
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt


@dataclass
class TestConfig:
    """Base configuration class that tests can extend"""
    num_runs: int = 100
    dtype: type = np.float32
    device: str = 'cpu'
    

class BenchmarkResult:
    def __init__(self, name: str, description: str, warmup=2):
        self.name = name
        self.warmup = warmup
        self.description = description
        self.forward_times: list[float] = []
        self.backward_times: list[float] = []
        
    @property
    def avg_forward_time(self) -> float:
        return np.mean(self.forward_times[self.warmup:]) if self.forward_times else 0
    
    @property
    def std_forward_time(self) -> float:
        return np.std(self.forward_times[self.warmup:]) if self.forward_times else 0
    
    @property
    def avg_backward_time(self) -> float:
        return np.mean(self.backward_times[self.warmup:]) if self.backward_times else 0
    
    @property
    def std_backward_time(self) -> float:
        return np.std(self.backward_times[self.warmup:]) if self.backward_times else 0

class BenchmarkTest(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the test"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the test does"""
        pass
    
    @abstractmethod
    def get_config(self) -> TestConfig:
        """Get configuration for this test"""
        pass
    
    @abstractmethod
    def run_numpy(self) -> Optional[BenchmarkResult]:
        """Run test with NumPy backend"""
        pass
    
    @abstractmethod
    def run_torch(self) -> Optional[BenchmarkResult]:
        """Run test with PyTorch backend"""
        pass
    
    @abstractmethod
    def run_synapx(self) -> Optional[BenchmarkResult]:
        """Run test with Synapx backend"""
        pass
    
    @abstractmethod
    def run_synapgrad(self) -> Optional[BenchmarkResult]:
        """Run test with Synapgrad backend"""
        pass

@dataclass
class MatmulConfig(TestConfig):
    input_size: int = (2, 2, 512, 512)

class MatmulTest(BenchmarkTest):
    def __init__(self, config:MatmulConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "Matmul"
    
    @property
    def description(self) -> str:
        return "Basic matrix multiplication test (A @ B)"
    
    def get_config(self) -> MatmulConfig:
        return self._config
    
    def _generate_data(self):
        shape_a = (*self._config.input_size,)
        shape_b = (*self._config.input_size[:-2], 
                  self._config.input_size[-1], 
                  self._config.input_size[-2])
        return (
            np.random.randn(*shape_a).astype(self._config.dtype),
            np.random.randn(*shape_b).astype(self._config.dtype)
        )
    
    def run_numpy(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (NumPy)", self.description)
        a, b = self._generate_data()
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            _ = np.matmul(a, b)
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
        return result
    
    def run_torch(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (PyTorch)", self.description)
        a, b = self._generate_data()
        
        torch_a = torch.from_numpy(a).requires_grad_(True)
        torch_b = torch.from_numpy(b).requires_grad_(True)
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            out = torch.matmul(torch_a, torch_b)
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
            start = time.perf_counter()
            grad = torch.ones_like(out)
            out.backward(grad)
            result.backward_times.append((time.perf_counter() - start) * 1000)
            
            torch_a.grad.zero_()
            torch_b.grad.zero_()
            
        return result
    
    def run_synapx(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (Synapx)", self.description)
        a, b = self._generate_data()
        
        synapx_a = synapx.from_numpy(a)
        synapx_b = synapx.from_numpy(b)
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            _ = synapx_a.matmul(synapx_b)
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
        return result
    
    def run_synapgrad(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (Synapgrad)", self.description)
        a, b = self._generate_data()
        
        synapgrad_a = synapgrad.tensor(a, requires_grad=True)
        synapgrad_b = synapgrad.tensor(b, requires_grad=True)
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            out = synapgrad_a @ synapgrad_b
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
            start = time.perf_counter()
            grad = synapgrad.ones_like(out)
            out.backward(grad)
            result.backward_times.append((time.perf_counter() - start) * 1000)
            
            synapgrad_a.zero_()
            synapgrad_b.zero_()
            
        return result

@dataclass
class MLPConfig(TestConfig):
    batch_size: int = 64
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    hidden_sizes: tuple[int, ...] = (200, 100)
    num_classes: int = 10
    dropout_p: float = 0.2
        
class MLPTest(BenchmarkTest):
    def __init__(self, config:MLPConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "MLP"
    
    @property
    def description(self) -> str:
        return "MLP architecture with multiple layers and operations"
    
    def get_config(self) -> MLPConfig:
        return self._config
    
    def _generate_data(self):
        return np.random.randn(
            self._config.batch_size, 
            self._config.input_channels,
            self._config.input_height, 
            self._config.input_width
        ).astype(self._config.dtype)
    
    def run_numpy(self) -> Optional[BenchmarkResult]:
        return None  # NumPy doesn't support automatic differentiation
    
    def run_torch(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (PyTorch)", self.description)
        x = self._generate_data()
        x_torch = torch.from_numpy(x).requires_grad_(True)
        
        input_size = self._config.input_channels * self._config.input_height * self._config.input_width
        
        # Define weights
        w1 = torch.randn(self._config.hidden_sizes[0], input_size, requires_grad=True)
        w2 = torch.randn(self._config.hidden_sizes[1], self._config.hidden_sizes[0], requires_grad=True)
        w3 = torch.randn(self._config.num_classes, self._config.hidden_sizes[1], requires_grad=True)
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            
            h = x_torch.reshape(self._config.batch_size, -1)
            h = h @ w1.t()
            h = F.relu(h)
            h = h @ w2.t()
            h = F.relu(h)
            h = h @ w3.t()
            out = F.log_softmax(h, dim=1)
            
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
            start = time.perf_counter()
            out.sum().backward()
            result.backward_times.append((time.perf_counter() - start) * 1000)
            
            x_torch.grad.zero_()
            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()
            
        return result
    
    def run_synapx(self) -> Optional[BenchmarkResult]:
        return None  # Synapx doesn't support automatic differentiation
    
    def run_synapgrad(self) -> BenchmarkResult:
        result = BenchmarkResult(self.name + " (Synapgrad)", self.description)
        x = self._generate_data()
        x_sg = synapgrad.tensor(x, requires_grad=True)
        
        input_size = self._config.input_channels * self._config.input_height * self._config.input_width
        
        # Define weights
        w1 = synapgrad.randn(self._config.hidden_sizes[0], input_size, requires_grad=True)
        w2 = synapgrad.randn(self._config.hidden_sizes[1], self._config.hidden_sizes[0], requires_grad=True)
        w3 = synapgrad.randn(self._config.num_classes, self._config.hidden_sizes[1], requires_grad=True)
        
        for _ in range(self._config.num_runs):
            start = time.perf_counter()
            
            h = x_sg.reshape((self._config.batch_size, -1))
            h = h @ w1.transpose(-1, -2)
            h = synapgrad.relu(h)
            h = h @ w2.transpose(-1, -2)
            h = synapgrad.relu(h)
            h = h @ w3.transpose(-1, -2)
            out = synapgrad.log_softmax(h, dim=1)
            
            result.forward_times.append((time.perf_counter() - start) * 1000)
            
            start = time.perf_counter()
            out.sum().backward()
            result.backward_times.append((time.perf_counter() - start) * 1000)
            
            x_sg.zero_()
            w1.zero_()
            w2.zero_()
            w3.zero_()
            
        return result

class BenchmarkSuite:
    def __init__(self):
        self.tests: List[BenchmarkTest] = []
        self.results: Dict[str, List[BenchmarkResult]] = {}
        
    def add_test(self, test: BenchmarkTest):
        """Add a test to the suite"""
        self.tests.append(test)
        
    def run(self):
        """Run all tests in the suite"""
        for test in self.tests:
            print(f"\nRunning {test.name}...")
            print(f"Configuration: {test.get_config()}")
            results = []
            
            # Run for each backend if implemented
            for backend, run_func in [
                ("NumPy", test.run_numpy),
                ("PyTorch", test.run_torch),
                ("Synapx", test.run_synapx),
                ("Synapgrad", test.run_synapgrad)
            ]:
                try:
                    result = run_func()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error running {backend} backend: {e}")
            
            self.results[test.name] = results
    
    def print_results(self):
        """Print results for all tests"""
        for test_name, results in self.results.items():
            print(f"\n{test_name} Results:")
            print("-" * 80)
            print(f"{'Backend':20} | {'Forward Avg':12} | {'Forward Std':12} | "
                  f"{'Backward Avg':12} | {'Backward Std':12}")
            print("-" * 80)
            
            for result in results:
                print(
                    f"{result.name:20} | "
                    f"{result.avg_forward_time:10.2f}ms | "
                    f"{result.std_forward_time:10.2f}ms | "
                    f"{result.avg_backward_time:10.2f}ms | "
                    f"{result.std_backward_time:10.2f}ms"
                )
    
    def plot_results(self):
        """Create plots for the benchmark results"""
        n_tests = len(self.results)
        fig, axes = plt.subplots(n_tests, 2, figsize=(15, 5 * n_tests))
        if n_tests == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (test_name, results) in enumerate(self.results.items()):
            # Forward pass plot
            self._create_subplot(
                axes[idx, 0],
                results,
                "Forward Pass Times",
                lambda x: x.avg_forward_time,
                lambda x: x.std_forward_time
            )
            
            # Backward pass plot
            self._create_subplot(
                axes[idx, 1],
                results,
                "Backward Pass Times",
                lambda x: x.avg_backward_time,
                lambda x: x.std_backward_time
            )
            
            axes[idx, 0].set_title(f"{test_name} - Forward Pass")
            axes[idx, 1].set_title(f"{test_name} - Backward Pass")
        
        plt.tight_layout()
        return fig
    
    def _create_subplot(self, ax, results, title, avg_func, std_func):
        """Helper function to create a subplot for the results"""
        backends = [r.name.split()[-1].strip('()') for r in results]
        avgs = [avg_func(r) for r in results]
        stds = [std_func(r) for r in results]
        
        # Create bars with error bars
        bars = ax.bar(backends, avgs, yerr=stds, capsize=5)
        ax.set_ylabel('Time (ms)')
        ax.set_xticks(range(len(backends)))
        ax.set_xticklabels(backends, rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms',
                   ha='center', va='bottom')
        
        # Customize appearance
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_title(title)
        

if __name__ == "__main__":
    # Create the benchmark suite
    suite = BenchmarkSuite()
    
    # Add matrix multiplication test
    config = MatmulConfig(
        input_size=(2, 2, 512, 512),
        num_runs=50,
        dtype=np.float32,
        device='cpu'
    )
    matmul_test = MatmulTest(config)
    suite.add_test(matmul_test)
    
    # Add MLP test
    config = MLPConfig(
        batch_size=64,
        input_channels=1,
        input_height=28,
        input_width=28,
        hidden_sizes=(200, 100),
        num_classes=10,
        dropout_p=0.2,
        num_runs=50,
        dtype=np.float32,
        device='cpu'
    )   
    mlp_test = MLPTest(config)
    suite.add_test(mlp_test)
    
    # Run all benchmarks
    suite.run()
    
    # Print results
    suite.print_results()
    
    # Plot and save results
    fig = suite.plot_results()
    # plt.savefig('benchmark_results.png', bbox_inches='tight', dpi=300)
    plt.show()