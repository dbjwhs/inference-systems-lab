#!/usr/bin/env python3
"""
Inference Benchmarker - ML performance analysis and benchmarking framework.

This tool provides comprehensive ML inference performance analysis including:
- Latency measurement with percentile analysis (p50, p95, p99)
- Throughput testing with batch size optimization
- Multi-model comparison and ranking
- GPU profiling integration (CUDA kernel timing, memory analysis)
- Performance regression detection
- Resource utilization monitoring
- Statistical analysis and visualization

Usage:
    # Benchmark single model latency
    python3 tools/benchmark_inference.py latency model.onnx \
        --iterations 1000 --warmup 100 --input-shape "[1,3,224,224]"
    
    # Benchmark throughput with batch scaling
    python3 tools/benchmark_inference.py throughput model.onnx \
        --batch-sizes "1,4,8,16,32" --duration 30 --target-latency 100
    
    # Compare multiple models
    python3 tools/benchmark_inference.py compare model1.onnx model2.onnx model3.onnx \
        --metrics latency,throughput,memory --output comparison.json
    
    # Profile GPU performance
    python3 tools/benchmark_inference.py profile model.onnx \
        --device cuda --profile-kernels --profile-memory
    
    # Regression testing
    python3 tools/benchmark_inference.py regression model.onnx \
        --baseline baseline.json --threshold 5.0 --save-results current.json
    
    # Resource monitoring
    python3 tools/benchmark_inference.py monitor model.onnx \
        --duration 60 --track-cpu --track-memory --track-gpu
"""

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import subprocess

# Try to import psutil (optional dependency)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Resource monitoring will be limited.", file=sys.stderr)

# Suppress warnings during import
warnings.filterwarnings("ignore")

# Try to import ML frameworks (optional dependencies)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some operations will be limited.", file=sys.stderr)

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Warning: ONNX Runtime not available. ONNX benchmarking will be disabled.", file=sys.stderr)

# Try to import GPU monitoring libraries
try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

# Try to import plotting library
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DeviceType(Enum):
    """Supported device types for inference."""
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    DIRECTML = "directml"
    TENSORRT = "tensorrt"


class MetricType(Enum):
    """Types of performance metrics to collect."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    GPU_UTILIZATION = "gpu_utilization"
    CPU_UTILIZATION = "cpu_utilization"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    iterations: int = 1000
    warmup_iterations: int = 100
    batch_sizes: List[int] = None
    duration: int = 30  # seconds
    target_latency: float = 100  # milliseconds
    device: DeviceType = DeviceType.CPU
    precision: str = "fp32"
    input_shapes: Optional[List[List[int]]] = None
    profile_kernels: bool = False
    profile_memory: bool = False
    track_cpu: bool = True
    track_memory: bool = True
    track_gpu: bool = False
    save_raw_data: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    p95: float
    p99: float
    p999: float
    raw_measurements: Optional[List[float]] = None


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""
    samples_per_second: float
    batch_size: int
    optimal_batch_size: int
    latency_ms: float
    efficiency: float  # samples/second per GPU/CPU core


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent: float
    memory_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    model_name: str
    device: str
    timestamp: str
    config: BenchmarkConfig
    latency: Optional[LatencyMetrics] = None
    throughput: List[ThroughputMetrics] = None
    resources: Optional[ResourceMetrics] = None
    metadata: Optional[Dict[str, Any]] = None


class GPUProfiler:
    """GPU profiling utilities."""
    
    def __init__(self):
        self.nvidia_available = NVIDIA_ML_AVAILABLE
        if self.nvidia_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                self.nvidia_available = False
                self.device_count = 0
        else:
            self.device_count = 0
    
    def get_gpu_utilization(self, device_id: int = 0) -> Tuple[float, float]:
        """Get GPU utilization and memory usage.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Tuple of (utilization_percent, memory_used_mb)
        """
        if not self.nvidia_available or device_id >= self.device_count:
            return 0.0, 0.0
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return float(util.gpu), float(mem_info.used) / (1024 * 1024)
        except Exception:
            return 0.0, 0.0
    
    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        """Get GPU device information.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Device information dictionary
        """
        if not self.nvidia_available or device_id >= self.device_count:
            return {}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "name": name,
                "total_memory_mb": mem_info.total / (1024 * 1024),
                "driver_version": pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            }
        except Exception:
            return {}


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, track_cpu: bool = True, track_memory: bool = True, track_gpu: bool = False):
        self.track_cpu = track_cpu
        self.track_memory = track_memory
        self.track_gpu = track_gpu
        self.gpu_profiler = GPUProfiler() if track_gpu else None
        
        self.measurements = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.monitoring = True
        self.measurements = []
        
        def monitor_loop():
            while self.monitoring:
                measurement = {}
                
                if self.track_cpu and PSUTIL_AVAILABLE:
                    measurement['cpu_percent'] = psutil.cpu_percent(interval=None)
                
                if self.track_memory and PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    measurement['memory_mb'] = memory.used / (1024 * 1024)
                    measurement['memory_percent'] = memory.percent
                
                if self.track_gpu and self.gpu_profiler:
                    gpu_util, gpu_mem = self.gpu_profiler.get_gpu_utilization()
                    measurement['gpu_utilization'] = gpu_util
                    measurement['gpu_memory_mb'] = gpu_mem
                
                measurement['timestamp'] = time.time()
                self.measurements.append(measurement)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return aggregate metrics.
        
        Returns:
            Aggregated resource metrics
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.measurements:
            return ResourceMetrics(cpu_percent=0.0, memory_mb=0.0)
        
        # Calculate averages
        cpu_values = [m.get('cpu_percent', 0) for m in self.measurements]
        memory_values = [m.get('memory_mb', 0) for m in self.measurements]
        gpu_util_values = [m.get('gpu_utilization', 0) for m in self.measurements if 'gpu_utilization' in m]
        gpu_mem_values = [m.get('gpu_memory_mb', 0) for m in self.measurements if 'gpu_memory_mb' in m]
        
        return ResourceMetrics(
            cpu_percent=statistics.mean(cpu_values) if cpu_values else 0.0,
            memory_mb=statistics.mean(memory_values) if memory_values else 0.0,
            gpu_utilization=statistics.mean(gpu_util_values) if gpu_util_values else None,
            gpu_memory_mb=statistics.mean(gpu_mem_values) if gpu_mem_values else None,
            peak_memory_mb=max(memory_values) if memory_values else None
        )


class ModelBenchmarker:
    """Main class for ML model benchmarking."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmarker.
        
        Args:
            config: Benchmarking configuration
        """
        self.config = config
        self.gpu_profiler = GPUProfiler()
    
    def create_inference_session(self, model_path: Path) -> Any:
        """Create inference session for the model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Inference session object
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime is required for benchmarking")
        
        # Configure providers based on device
        providers = []
        if self.config.device == DeviceType.CUDA:
            providers.append('CUDAExecutionProvider')
        elif self.config.device == DeviceType.TENSORRT:
            providers.append('TensorrtExecutionProvider')
        elif self.config.device == DeviceType.DIRECTML:
            providers.append('DmlExecutionProvider')
        elif self.config.device == DeviceType.OPENCL:
            providers.append('OpenVINOExecutionProvider')
        
        providers.append('CPUExecutionProvider')  # Fallback
        
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            if self.config.verbose:
                print(f"Created session with providers: {session.get_providers()}")
            return session
        except Exception as e:
            print(f"Failed to create inference session: {e}", file=sys.stderr)
            raise
    
    def create_test_input(self, session: Any) -> Dict[str, Any]:
        """Create test input for the model.
        
        Args:
            session: Inference session
            
        Returns:
            Dictionary of input tensors
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for creating test inputs")
        
        inputs = {}
        for input_meta in session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            
            # Handle dynamic dimensions
            actual_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None or dim < 0:
                    # Dynamic dimension, use batch size from config
                    actual_shape.append(self.config.batch_sizes[0] if self.config.batch_sizes else 1)
                else:
                    actual_shape.append(dim)
            
            # Override with config if provided
            if self.config.input_shapes and len(self.config.input_shapes) > 0:
                actual_shape = self.config.input_shapes[0]
            
            # Generate random input data
            if input_meta.type == 'tensor(float)':
                inputs[name] = np.random.randn(*actual_shape).astype(np.float32)
            elif input_meta.type == 'tensor(double)':
                inputs[name] = np.random.randn(*actual_shape).astype(np.float64)
            elif input_meta.type == 'tensor(int64)':
                inputs[name] = np.random.randint(0, 1000, actual_shape, dtype=np.int64)
            else:
                # Default to float32
                inputs[name] = np.random.randn(*actual_shape).astype(np.float32)
        
        return inputs
    
    def benchmark_latency(self, model_path: Path) -> LatencyMetrics:
        """Benchmark model inference latency.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Latency metrics
        """
        session = self.create_inference_session(model_path)
        test_input = self.create_test_input(session)
        
        # Warmup
        if self.config.verbose:
            print(f"Warming up with {self.config.warmup_iterations} iterations...")
        
        for _ in range(self.config.warmup_iterations):
            session.run(None, test_input)
        
        # Benchmark
        if self.config.verbose:
            print(f"Benchmarking latency with {self.config.iterations} iterations...")
        
        latencies = []
        for i in range(self.config.iterations):
            start_time = time.perf_counter()
            session.run(None, test_input)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if self.config.verbose and (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.config.iterations} iterations")
        
        # Calculate statistics
        latencies.sort()
        return LatencyMetrics(
            mean=statistics.mean(latencies),
            median=statistics.median(latencies),
            std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            min=min(latencies),
            max=max(latencies),
            p95=latencies[int(0.95 * len(latencies))],
            p99=latencies[int(0.99 * len(latencies))],
            p999=latencies[int(0.999 * len(latencies))],
            raw_measurements=latencies if self.config.save_raw_data else None
        )
    
    def benchmark_throughput(self, model_path: Path) -> List[ThroughputMetrics]:
        """Benchmark model throughput across different batch sizes.
        
        Args:
            model_path: Path to model file
            
        Returns:
            List of throughput metrics for each batch size
        """
        session = self.create_inference_session(model_path)
        results = []
        
        for batch_size in self.config.batch_sizes:
            if self.config.verbose:
                print(f"Benchmarking throughput with batch size {batch_size}...")
            
            # Create batch input
            single_input = self.create_test_input(session)
            batch_input = {}
            
            for name, tensor in single_input.items():
                # Expand batch dimension
                batch_shape = [batch_size] + list(tensor.shape[1:])
                batch_input[name] = np.random.randn(*batch_shape).astype(tensor.dtype)
            
            # Warmup
            for _ in range(min(10, self.config.warmup_iterations // 10)):
                session.run(None, batch_input)
            
            # Benchmark throughput
            start_time = time.perf_counter()
            iterations = 0
            total_samples = 0
            
            while time.perf_counter() - start_time < self.config.duration:
                iter_start = time.perf_counter()
                session.run(None, batch_input)
                iter_time = time.perf_counter() - iter_start
                
                iterations += 1
                total_samples += batch_size
                
                # Check if we're meeting target latency
                latency_ms = iter_time * 1000
                if latency_ms > self.config.target_latency:
                    if self.config.verbose:
                        print(f"  Target latency exceeded: {latency_ms:.2f}ms > {self.config.target_latency}ms")
                    break
            
            elapsed_time = time.perf_counter() - start_time
            samples_per_second = total_samples / elapsed_time
            avg_latency_ms = (elapsed_time / iterations) * 1000
            
            # Calculate efficiency (samples/second per core)
            if PSUTIL_AVAILABLE:
                cpu_cores = psutil.cpu_count(logical=False)
                efficiency = samples_per_second / cpu_cores if cpu_cores else samples_per_second
            else:
                efficiency = samples_per_second
            
            results.append(ThroughputMetrics(
                samples_per_second=samples_per_second,
                batch_size=batch_size,
                optimal_batch_size=batch_size,  # Will be updated later
                latency_ms=avg_latency_ms,
                efficiency=efficiency
            ))
            
            if self.config.verbose:
                print(f"  Batch {batch_size}: {samples_per_second:.1f} samples/sec, {avg_latency_ms:.2f}ms latency")
        
        # Find optimal batch size (highest samples/sec within latency target)
        valid_results = [r for r in results if r.latency_ms <= self.config.target_latency]
        if valid_results:
            optimal = max(valid_results, key=lambda x: x.samples_per_second)
            for result in results:
                result.optimal_batch_size = optimal.batch_size
        
        return results
    
    def benchmark_with_monitoring(self, model_path: Path) -> BenchmarkResults:
        """Run comprehensive benchmark with resource monitoring.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Complete benchmark results
        """
        if self.config.verbose:
            print(f"Starting comprehensive benchmark for {model_path}")
        
        # Start resource monitoring
        monitor = ResourceMonitor(
            track_cpu=self.config.track_cpu,
            track_memory=self.config.track_memory,
            track_gpu=self.config.track_gpu
        )
        monitor.start_monitoring()
        
        try:
            # Run latency benchmark
            latency_metrics = self.benchmark_latency(model_path)
            
            # Run throughput benchmark
            throughput_metrics = self.benchmark_throughput(model_path)
            
        finally:
            # Stop monitoring
            resource_metrics = monitor.stop_monitoring()
        
        # Collect metadata
        metadata = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        
        if PSUTIL_AVAILABLE:
            metadata.update({
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            })
        
        if self.config.track_gpu and self.gpu_profiler:
            gpu_info = self.gpu_profiler.get_device_info()
            if gpu_info:
                metadata["gpu_info"] = gpu_info
        
        return BenchmarkResults(
            model_name=model_path.name,
            device=self.config.device.value,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config,
            latency=latency_metrics,
            throughput=throughput_metrics,
            resources=resource_metrics,
            metadata=metadata
        )
    
    def compare_models(self, model_paths: List[Path]) -> Dict[str, BenchmarkResults]:
        """Compare performance across multiple models.
        
        Args:
            model_paths: List of model file paths
            
        Returns:
            Dictionary mapping model names to benchmark results
        """
        results = {}
        
        for model_path in model_paths:
            if self.config.verbose:
                print(f"\nBenchmarking {model_path.name}...")
            
            try:
                result = self.benchmark_with_monitoring(model_path)
                results[model_path.name] = result
            except Exception as e:
                print(f"Failed to benchmark {model_path.name}: {e}", file=sys.stderr)
                continue
        
        return results
    
    def save_results(self, results: Union[BenchmarkResults, Dict[str, BenchmarkResults]], 
                    output_path: Path):
        """Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results to save
            output_path: Output file path
        """
        if isinstance(results, BenchmarkResults):
            data = asdict(results)
        else:
            data = {name: asdict(result) for name, result in results.items()}
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.config.verbose:
            print(f"Results saved to {output_path}")
    
    def load_results(self, input_path: Path) -> Dict[str, Any]:
        """Load benchmark results from JSON file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded results data
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def detect_regression(self, current_results: BenchmarkResults, 
                         baseline_path: Path, threshold: float = 5.0) -> Dict[str, Any]:
        """Detect performance regression compared to baseline.
        
        Args:
            current_results: Current benchmark results
            baseline_path: Path to baseline results JSON
            threshold: Regression threshold percentage
            
        Returns:
            Regression analysis results
        """
        baseline_data = self.load_results(baseline_path)
        
        # Handle both single model and multi-model baselines
        if "latency" in baseline_data:
            # Single model baseline
            baseline_latency = baseline_data["latency"]["mean"]
        else:
            # Multi-model baseline, find matching model
            model_name = current_results.model_name
            if model_name not in baseline_data:
                raise ValueError(f"Model {model_name} not found in baseline")
            baseline_latency = baseline_data[model_name]["latency"]["mean"]
        
        current_latency = current_results.latency.mean
        regression_percent = ((current_latency - baseline_latency) / baseline_latency) * 100
        
        is_regression = regression_percent > threshold
        
        return {
            "is_regression": is_regression,
            "regression_percent": regression_percent,
            "current_latency_ms": current_latency,
            "baseline_latency_ms": baseline_latency,
            "threshold_percent": threshold,
            "status": "REGRESSION" if is_regression else "PASS"
        }


def parse_batch_sizes(batch_sizes_str: str) -> List[int]:
    """Parse batch sizes string to list of integers.
    
    Args:
        batch_sizes_str: Comma-separated batch sizes (e.g., "1,4,8,16")
        
    Returns:
        List of batch sizes
    """
    return [int(size.strip()) for size in batch_sizes_str.split(',')]


def parse_input_shapes(shapes_str: str) -> List[List[int]]:
    """Parse input shapes string to list of shape lists.
    
    Args:
        shapes_str: Shape string like "[1,3,224,224]" or "1,3,224,224"
        
    Returns:
        List of shape lists
    """
    # Remove brackets and spaces
    shapes_str = shapes_str.strip().strip('[]').replace(' ', '')
    # Split by comma and convert to int
    shape = [int(dim) for dim in shapes_str.split(',')]
    return [shape]


def main():
    """Main entry point for the inference benchmarker CLI."""
    parser = argparse.ArgumentParser(
        description="Inference Benchmarker - ML performance analysis framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Benchmark commands")
    
    # Latency benchmark command
    latency_parser = subparsers.add_parser("latency", help="Benchmark model latency")
    latency_parser.add_argument("model", type=Path, help="Path to model file")
    latency_parser.add_argument("--iterations", type=int, default=1000, 
                               help="Number of inference iterations")
    latency_parser.add_argument("--warmup", type=int, default=100, 
                               help="Number of warmup iterations")
    latency_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                               default="cpu", help="Inference device")
    latency_parser.add_argument("--input-shape", type=str, 
                               help="Input tensor shape (e.g., '[1,3,224,224]')")
    latency_parser.add_argument("--output", type=Path, help="Output JSON file")
    latency_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Throughput benchmark command
    throughput_parser = subparsers.add_parser("throughput", help="Benchmark model throughput")
    throughput_parser.add_argument("model", type=Path, help="Path to model file")
    throughput_parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                                  help="Comma-separated batch sizes")
    throughput_parser.add_argument("--duration", type=int, default=30, 
                                  help="Benchmark duration in seconds")
    throughput_parser.add_argument("--target-latency", type=float, default=100,
                                  help="Target latency in milliseconds")
    throughput_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                                  default="cpu", help="Inference device")
    throughput_parser.add_argument("--input-shape", type=str, 
                                  help="Input tensor shape (e.g., '[1,3,224,224]')")
    throughput_parser.add_argument("--output", type=Path, help="Output JSON file")
    throughput_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("models", nargs="+", type=Path, help="Paths to model files")
    compare_parser.add_argument("--metrics", type=str, default="latency,throughput",
                               help="Metrics to compare (latency,throughput,memory)")
    compare_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                               default="cpu", help="Inference device")
    compare_parser.add_argument("--iterations", type=int, default=500, 
                               help="Number of iterations for each model")
    compare_parser.add_argument("--output", type=Path, help="Output JSON file")
    compare_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile model performance")
    profile_parser.add_argument("model", type=Path, help="Path to model file")
    profile_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                               default="cpu", help="Inference device")
    profile_parser.add_argument("--profile-kernels", action="store_true", 
                               help="Profile GPU kernels")
    profile_parser.add_argument("--profile-memory", action="store_true", 
                               help="Profile memory usage")
    profile_parser.add_argument("--duration", type=int, default=30, 
                               help="Profiling duration in seconds")
    profile_parser.add_argument("--output", type=Path, help="Output JSON file")
    profile_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Regression testing command
    regression_parser = subparsers.add_parser("regression", help="Detect performance regression")
    regression_parser.add_argument("model", type=Path, help="Path to model file")
    regression_parser.add_argument("--baseline", type=Path, required=True,
                                  help="Baseline results JSON file")
    regression_parser.add_argument("--threshold", type=float, default=5.0,
                                  help="Regression threshold percentage")
    regression_parser.add_argument("--save-results", type=Path, 
                                  help="Save current results to file")
    regression_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                                  default="cpu", help="Inference device")
    regression_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor resource usage")
    monitor_parser.add_argument("model", type=Path, help="Path to model file")
    monitor_parser.add_argument("--duration", type=int, default=60, 
                               help="Monitoring duration in seconds")
    monitor_parser.add_argument("--track-cpu", action="store_true", default=True,
                               help="Track CPU utilization")
    monitor_parser.add_argument("--track-memory", action="store_true", default=True,
                               help="Track memory usage")
    monitor_parser.add_argument("--track-gpu", action="store_true", 
                               help="Track GPU utilization")
    monitor_parser.add_argument("--device", choices=["cpu", "cuda", "opencl", "directml"], 
                               default="cpu", help="Inference device")
    monitor_parser.add_argument("--output", type=Path, help="Output JSON file")
    monitor_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Create benchmark configuration
        config = BenchmarkConfig()
        
        if hasattr(args, 'verbose'):
            config.verbose = args.verbose
        
        if hasattr(args, 'device'):
            config.device = DeviceType(args.device)
        
        if hasattr(args, 'iterations'):
            config.iterations = args.iterations
        
        if hasattr(args, 'warmup'):
            config.warmup_iterations = args.warmup
        
        if hasattr(args, 'batch_sizes'):
            config.batch_sizes = parse_batch_sizes(args.batch_sizes)
        
        if hasattr(args, 'duration'):
            config.duration = args.duration
        
        if hasattr(args, 'target_latency'):
            config.target_latency = args.target_latency
        
        if hasattr(args, 'input_shape') and args.input_shape:
            config.input_shapes = parse_input_shapes(args.input_shape)
        
        if hasattr(args, 'profile_kernels'):
            config.profile_kernels = args.profile_kernels
        
        if hasattr(args, 'profile_memory'):
            config.profile_memory = args.profile_memory
        
        if hasattr(args, 'track_cpu'):
            config.track_cpu = args.track_cpu
        
        if hasattr(args, 'track_memory'):
            config.track_memory = args.track_memory
        
        if hasattr(args, 'track_gpu'):
            config.track_gpu = args.track_gpu
        
        # Create benchmarker
        benchmarker = ModelBenchmarker(config)
        
        # Execute command
        if args.command == "latency":
            results = benchmarker.benchmark_latency(args.model)
            
            print(f"🕐 Latency Results for {args.model.name}:")
            print(f"   Mean: {results.mean:.2f}ms")
            print(f"   Median: {results.median:.2f}ms")
            print(f"   Std Dev: {results.std:.2f}ms")
            print(f"   Min/Max: {results.min:.2f}ms / {results.max:.2f}ms")
            print(f"   P95: {results.p95:.2f}ms")
            print(f"   P99: {results.p99:.2f}ms")
            print(f"   P99.9: {results.p999:.2f}ms")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(results), f, indent=2)
                print(f"✅ Results saved to {args.output}")
        
        elif args.command == "throughput":
            results = benchmarker.benchmark_throughput(args.model)
            
            print(f"🚀 Throughput Results for {args.model.name}:")
            for result in results:
                print(f"   Batch {result.batch_size}: {result.samples_per_second:.1f} samples/sec "
                      f"({result.latency_ms:.2f}ms latency)")
            
            # Find optimal batch size
            valid_results = [r for r in results if r.latency_ms <= config.target_latency]
            if valid_results:
                optimal = max(valid_results, key=lambda x: x.samples_per_second)
                print(f"🎯 Optimal batch size: {optimal.batch_size} "
                      f"({optimal.samples_per_second:.1f} samples/sec)")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                print(f"✅ Results saved to {args.output}")
        
        elif args.command == "compare":
            results = benchmarker.compare_models(args.models)
            
            print(f"📊 Model Comparison Results:")
            print(f"{'Model':<20} {'Mean Latency (ms)':<18} {'Peak Throughput':<15}")
            print("-" * 55)
            
            for model_name, result in results.items():
                latency = result.latency.mean if result.latency else 0
                throughput = max(t.samples_per_second for t in result.throughput) if result.throughput else 0
                print(f"{model_name:<20} {latency:<18.2f} {throughput:<15.1f}")
            
            if args.output:
                benchmarker.save_results(results, args.output)
                print(f"✅ Results saved to {args.output}")
        
        elif args.command == "profile":
            config.profile_kernels = args.profile_kernels
            config.profile_memory = args.profile_memory
            config.track_gpu = True  # Enable GPU tracking for profiling
            
            results = benchmarker.benchmark_with_monitoring(args.model)
            
            print(f"🔍 Profile Results for {args.model.name}:")
            print(f"   Latency: {results.latency.mean:.2f}ms ± {results.latency.std:.2f}ms")
            print(f"   CPU Usage: {results.resources.cpu_percent:.1f}%")
            print(f"   Memory: {results.resources.memory_mb:.1f}MB")
            if results.resources.gpu_utilization is not None:
                print(f"   GPU Usage: {results.resources.gpu_utilization:.1f}%")
                print(f"   GPU Memory: {results.resources.gpu_memory_mb:.1f}MB")
            
            if args.output:
                benchmarker.save_results(results, args.output)
                print(f"✅ Results saved to {args.output}")
        
        elif args.command == "regression":
            results = benchmarker.benchmark_with_monitoring(args.model)
            regression_analysis = benchmarker.detect_regression(
                results, args.baseline, args.threshold
            )
            
            print(f"📈 Regression Analysis for {args.model.name}:")
            print(f"   Current Latency: {regression_analysis['current_latency_ms']:.2f}ms")
            print(f"   Baseline Latency: {regression_analysis['baseline_latency_ms']:.2f}ms")
            print(f"   Change: {regression_analysis['regression_percent']:+.1f}%")
            print(f"   Status: {regression_analysis['status']}")
            
            if regression_analysis['is_regression']:
                print(f"❌ REGRESSION DETECTED: Exceeds {args.threshold}% threshold")
                return_code = 1
            else:
                print(f"✅ No regression detected (within {args.threshold}% threshold)")
                return_code = 0
            
            if args.save_results:
                benchmarker.save_results(results, args.save_results)
                print(f"✅ Current results saved to {args.save_results}")
            
            return return_code
        
        elif args.command == "monitor":
            results = benchmarker.benchmark_with_monitoring(args.model)
            
            print(f"📊 Resource Monitoring for {args.model.name}:")
            print(f"   Avg CPU Usage: {results.resources.cpu_percent:.1f}%")
            print(f"   Avg Memory: {results.resources.memory_mb:.1f}MB")
            print(f"   Peak Memory: {results.resources.peak_memory_mb:.1f}MB")
            if results.resources.gpu_utilization is not None:
                print(f"   Avg GPU Usage: {results.resources.gpu_utilization:.1f}%")
                print(f"   Avg GPU Memory: {results.resources.gpu_memory_mb:.1f}MB")
            
            if args.output:
                benchmarker.save_results(results, args.output)
                print(f"✅ Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())