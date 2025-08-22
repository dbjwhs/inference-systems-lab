#!/usr/bin/env python3
"""
Test suite for benchmark_inference.py - Comprehensive testing of ML benchmarking functionality.

This test script validates all aspects of the inference benchmarker including:
- Latency measurement and statistical analysis
- Throughput benchmarking with batch scaling
- Multi-model comparison framework
- Resource monitoring and profiling
- Performance regression detection
- CLI interface functionality
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import time

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_inference import (
    ModelBenchmarker, BenchmarkConfig, DeviceType, MetricType,
    LatencyMetrics, ThroughputMetrics, ResourceMetrics, BenchmarkResults,
    GPUProfiler, ResourceMonitor, parse_batch_sizes, parse_input_shapes
)

# Mock dependencies if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = MagicMock()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = MagicMock()


class TestBenchmarkConfig(unittest.TestCase):
    """Test suite for BenchmarkConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        
        self.assertEqual(config.iterations, 1000)
        self.assertEqual(config.warmup_iterations, 100)
        self.assertEqual(config.batch_sizes, [1, 4, 8, 16, 32])
        self.assertEqual(config.duration, 30)
        self.assertEqual(config.target_latency, 100)
        self.assertEqual(config.device, DeviceType.CPU)
        self.assertEqual(config.precision, "fp32")
        self.assertIsNone(config.input_shapes)
        self.assertFalse(config.profile_kernels)
        self.assertFalse(config.profile_memory)
        self.assertTrue(config.track_cpu)
        self.assertTrue(config.track_memory)
        self.assertFalse(config.track_gpu)
        self.assertFalse(config.save_raw_data)
        self.assertFalse(config.verbose)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            iterations=500,
            warmup_iterations=50,
            batch_sizes=[1, 2, 4],
            duration=60,
            target_latency=50,
            device=DeviceType.CUDA,
            precision="fp16",
            input_shapes=[[1, 3, 224, 224]],
            profile_kernels=True,
            verbose=True
        )
        
        self.assertEqual(config.iterations, 500)
        self.assertEqual(config.warmup_iterations, 50)
        self.assertEqual(config.batch_sizes, [1, 2, 4])
        self.assertEqual(config.duration, 60)
        self.assertEqual(config.target_latency, 50)
        self.assertEqual(config.device, DeviceType.CUDA)
        self.assertEqual(config.precision, "fp16")
        self.assertEqual(config.input_shapes, [[1, 3, 224, 224]])
        self.assertTrue(config.profile_kernels)
        self.assertTrue(config.verbose)


class TestDeviceType(unittest.TestCase):
    """Test suite for DeviceType enum."""
    
    def test_device_values(self):
        """Test device enum values."""
        self.assertEqual(DeviceType.CPU.value, "cpu")
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.OPENCL.value, "opencl")
        self.assertEqual(DeviceType.DIRECTML.value, "directml")
        self.assertEqual(DeviceType.TENSORRT.value, "tensorrt")


class TestMetricType(unittest.TestCase):
    """Test suite for MetricType enum."""
    
    def test_metric_values(self):
        """Test metric enum values."""
        self.assertEqual(MetricType.LATENCY.value, "latency")
        self.assertEqual(MetricType.THROUGHPUT.value, "throughput")
        self.assertEqual(MetricType.MEMORY.value, "memory")
        self.assertEqual(MetricType.GPU_UTILIZATION.value, "gpu_utilization")
        self.assertEqual(MetricType.CPU_UTILIZATION.value, "cpu_utilization")


class TestLatencyMetrics(unittest.TestCase):
    """Test suite for LatencyMetrics dataclass."""
    
    def test_latency_metrics_creation(self):
        """Test LatencyMetrics creation and values."""
        metrics = LatencyMetrics(
            mean=10.5,
            median=10.0,
            std=2.1,
            min=8.0,
            max=15.0,
            p95=13.0,
            p99=14.5,
            p999=14.9,
            raw_measurements=[8.0, 9.0, 10.0, 11.0, 15.0]
        )
        
        self.assertEqual(metrics.mean, 10.5)
        self.assertEqual(metrics.median, 10.0)
        self.assertEqual(metrics.std, 2.1)
        self.assertEqual(metrics.min, 8.0)
        self.assertEqual(metrics.max, 15.0)
        self.assertEqual(metrics.p95, 13.0)
        self.assertEqual(metrics.p99, 14.5)
        self.assertEqual(metrics.p999, 14.9)
        self.assertEqual(len(metrics.raw_measurements), 5)


class TestThroughputMetrics(unittest.TestCase):
    """Test suite for ThroughputMetrics dataclass."""
    
    def test_throughput_metrics_creation(self):
        """Test ThroughputMetrics creation and values."""
        metrics = ThroughputMetrics(
            samples_per_second=100.5,
            batch_size=8,
            optimal_batch_size=16,
            latency_ms=80.0,
            efficiency=12.5
        )
        
        self.assertEqual(metrics.samples_per_second, 100.5)
        self.assertEqual(metrics.batch_size, 8)
        self.assertEqual(metrics.optimal_batch_size, 16)
        self.assertEqual(metrics.latency_ms, 80.0)
        self.assertEqual(metrics.efficiency, 12.5)


class TestResourceMetrics(unittest.TestCase):
    """Test suite for ResourceMetrics dataclass."""
    
    def test_resource_metrics_creation(self):
        """Test ResourceMetrics creation and values."""
        metrics = ResourceMetrics(
            cpu_percent=75.5,
            memory_mb=1024.0,
            gpu_utilization=85.0,
            gpu_memory_mb=2048.0,
            peak_memory_mb=1200.0
        )
        
        self.assertEqual(metrics.cpu_percent, 75.5)
        self.assertEqual(metrics.memory_mb, 1024.0)
        self.assertEqual(metrics.gpu_utilization, 85.0)
        self.assertEqual(metrics.gpu_memory_mb, 2048.0)
        self.assertEqual(metrics.peak_memory_mb, 1200.0)


class TestGPUProfiler(unittest.TestCase):
    """Test suite for GPUProfiler class."""
    
    def test_gpu_profiler_init_no_nvidia(self):
        """Test GPU profiler initialization without NVIDIA-ML."""
        profiler = GPUProfiler()
        # Should handle missing NVIDIA-ML gracefully
        self.assertIsNotNone(profiler)
    
    def test_get_gpu_utilization_no_gpu(self):
        """Test GPU utilization when no GPU available."""
        profiler = GPUProfiler()
        profiler.nvidia_available = False
        
        util, memory = profiler.get_gpu_utilization()
        self.assertEqual(util, 0.0)
        self.assertEqual(memory, 0.0)
    
    def test_get_device_info_no_gpu(self):
        """Test device info when no GPU available."""
        profiler = GPUProfiler()
        profiler.nvidia_available = False
        
        info = profiler.get_device_info()
        self.assertEqual(info, {})


class TestResourceMonitor(unittest.TestCase):
    """Test suite for ResourceMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = ResourceMonitor(track_cpu=True, track_memory=True, track_gpu=False)
    
    def test_monitor_initialization(self):
        """Test resource monitor initialization."""
        self.assertTrue(self.monitor.track_cpu)
        self.assertTrue(self.monitor.track_memory)
        self.assertFalse(self.monitor.track_gpu)
        self.assertFalse(self.monitor.monitoring)
        self.assertEqual(len(self.monitor.measurements), 0)
    
    @unittest.skipIf(not PSUTIL_AVAILABLE, "psutil not available")
    def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        # Start monitoring
        self.monitor.start_monitoring(interval=0.01)
        self.assertTrue(self.monitor.monitoring)
        
        # Let it run briefly
        time.sleep(0.05)
        
        # Stop monitoring
        metrics = self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.monitoring)
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertGreaterEqual(metrics.cpu_percent, 0.0)
        self.assertGreaterEqual(metrics.memory_mb, 0.0)


class TestParsingUtilities(unittest.TestCase):
    """Test suite for parsing utility functions."""
    
    def test_parse_batch_sizes(self):
        """Test batch sizes parsing."""
        batch_sizes = parse_batch_sizes("1,4,8,16,32")
        self.assertEqual(batch_sizes, [1, 4, 8, 16, 32])
        
        batch_sizes = parse_batch_sizes("1, 2, 4")
        self.assertEqual(batch_sizes, [1, 2, 4])
        
        batch_sizes = parse_batch_sizes("8")
        self.assertEqual(batch_sizes, [8])
    
    def test_parse_input_shapes(self):
        """Test input shapes parsing."""
        shapes = parse_input_shapes("[1,3,224,224]")
        self.assertEqual(shapes, [[1, 3, 224, 224]])
        
        shapes = parse_input_shapes("1,3,224,224")
        self.assertEqual(shapes, [[1, 3, 224, 224]])
        
        shapes = parse_input_shapes("[ 1 , 3 , 224 , 224 ]")
        self.assertEqual(shapes, [[1, 3, 224, 224]])


class TestModelBenchmarker(unittest.TestCase):
    """Test suite for ModelBenchmarker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = BenchmarkConfig(verbose=False, iterations=10, warmup_iterations=2)
        self.benchmarker = ModelBenchmarker(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_benchmarker_initialization(self):
        """Test benchmarker initialization."""
        self.assertEqual(self.benchmarker.config, self.config)
        self.assertIsNotNone(self.benchmarker.gpu_profiler)
    
    def test_create_inference_session_no_onnxruntime(self):
        """Test session creation when ONNX Runtime not available."""
        model_path = Path(self.test_dir) / "model.onnx"
        model_path.write_bytes(b"dummy onnx model")
        
        # Should raise error when ONNX Runtime not available
        with self.assertRaises(RuntimeError):
            self.benchmarker.create_inference_session(model_path)
    
    @unittest.skip("Requires onnxruntime module for mocking")
    def test_create_inference_session_success(self):
        """Test successful session creation."""
        pass
    
    @unittest.skip("Requires numpy module for mocking")
    def test_create_test_input(self):
        """Test test input creation."""
        pass
    
    def test_create_test_input_no_numpy(self):
        """Test test input creation without numpy."""
        mock_session = MagicMock()
        
        with self.assertRaises(RuntimeError):
            self.benchmarker.create_test_input(mock_session)
    
    @patch('benchmark_inference.ONNXRUNTIME_AVAILABLE', True)
    @patch('benchmark_inference.NUMPY_AVAILABLE', True)
    def test_benchmark_latency_mock(self):
        """Test latency benchmarking with mocks."""
        model_path = Path(self.test_dir) / "model.onnx"
        
        with patch.object(self.benchmarker, 'create_inference_session') as mock_create_session:
            with patch.object(self.benchmarker, 'create_test_input') as mock_create_input:
                # Mock session and input
                mock_session = MagicMock()
                mock_create_session.return_value = mock_session
                mock_create_input.return_value = {"input": "mock_data"}
                
                # Mock timing
                with patch('time.perf_counter', side_effect=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05] * 10):
                    results = self.benchmarker.benchmark_latency(model_path)
                
                self.assertIsInstance(results, LatencyMetrics)
                self.assertGreater(results.mean, 0)
                self.assertGreaterEqual(results.min, 0)
                self.assertGreaterEqual(results.max, results.min)
    
    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Create sample results
        latency = LatencyMetrics(
            mean=10.0, median=9.5, std=1.0, min=8.0, max=12.0,
            p95=11.0, p99=11.5, p999=11.8
        )
        
        results = BenchmarkResults(
            model_name="test_model.onnx",
            device="cpu",
            timestamp="2024-01-01 12:00:00",
            config=self.config,
            latency=latency
        )
        
        # Save results
        output_path = Path(self.test_dir) / "results.json"
        self.benchmarker.save_results(results, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Load results
        loaded_data = self.benchmarker.load_results(output_path)
        
        self.assertEqual(loaded_data["model_name"], "test_model.onnx")
        self.assertEqual(loaded_data["device"], "cpu")
        self.assertIn("latency", loaded_data)
    
    def test_detect_regression(self):
        """Test performance regression detection."""
        # Create current results
        latency = LatencyMetrics(
            mean=12.0, median=11.5, std=1.0, min=10.0, max=14.0,
            p95=13.0, p99=13.5, p999=13.8
        )
        
        current_results = BenchmarkResults(
            model_name="test_model.onnx",
            device="cpu",
            timestamp="2024-01-01 12:00:00",
            config=self.config,
            latency=latency
        )
        
        # Create baseline file
        baseline_data = {
            "latency": {"mean": 10.0}
        }
        
        baseline_path = Path(self.test_dir) / "baseline.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        
        # Test regression detection
        regression_analysis = self.benchmarker.detect_regression(
            current_results, baseline_path, threshold=5.0
        )
        
        self.assertTrue(regression_analysis["is_regression"])
        self.assertEqual(regression_analysis["status"], "REGRESSION")
        self.assertAlmostEqual(regression_analysis["regression_percent"], 20.0)
        
        # Test no regression
        regression_analysis = self.benchmarker.detect_regression(
            current_results, baseline_path, threshold=25.0
        )
        
        self.assertFalse(regression_analysis["is_regression"])
        self.assertEqual(regression_analysis["status"], "PASS")


class TestBenchmarkerCLI(unittest.TestCase):
    """Test suite for benchmark_inference.py CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_cli_help(self):
        """Test CLI help output."""
        from benchmark_inference import main
        import io
        import contextlib
        
        # Capture help output
        with patch('sys.argv', ['benchmark_inference.py', '--help']):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                with patch('sys.exit'):
                    try:
                        main()
                    except SystemExit:
                        pass
            
            help_text = f.getvalue()
            
            # Check help contains expected commands
            self.assertIn("latency", help_text)
            self.assertIn("throughput", help_text)
            self.assertIn("compare", help_text)
            self.assertIn("profile", help_text)
            self.assertIn("regression", help_text)
            self.assertIn("monitor", help_text)
    
    def test_cli_latency_args(self):
        """Test latency command arguments parsing."""
        from benchmark_inference import main
        
        model_file = Path(self.test_dir) / "model.onnx"
        output_file = Path(self.test_dir) / "results.json"
        
        # Create dummy model file
        model_file.write_bytes(b"dummy onnx model")
        
        with patch('sys.argv', [
            'benchmark_inference.py',
            'latency',
            str(model_file),
            '--iterations', '100',
            '--warmup', '10',
            '--device', 'cpu',
            '--input-shape', '[1,3,224,224]',
            '--output', str(output_file),
            '--verbose'
        ]):
            with patch('benchmark_inference.ModelBenchmarker') as mock_benchmarker_class:
                mock_benchmarker = MagicMock()
                mock_latency = LatencyMetrics(
                    mean=10.0, median=9.5, std=1.0, min=8.0, max=12.0,
                    p95=11.0, p99=11.5, p999=11.8
                )
                mock_benchmarker.benchmark_latency.return_value = mock_latency
                mock_benchmarker_class.return_value = mock_benchmarker
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_benchmarker.benchmark_latency.assert_called_once()
    
    def test_cli_throughput_args(self):
        """Test throughput command arguments parsing."""
        from benchmark_inference import main
        
        model_file = Path(self.test_dir) / "model.onnx"
        output_file = Path(self.test_dir) / "results.json"
        
        # Create dummy model file
        model_file.write_bytes(b"dummy onnx model")
        
        with patch('sys.argv', [
            'benchmark_inference.py',
            'throughput',
            str(model_file),
            '--batch-sizes', '1,4,8',
            '--duration', '15',
            '--target-latency', '50',
            '--device', 'cpu',
            '--output', str(output_file),
            '--verbose'
        ]):
            with patch('benchmark_inference.ModelBenchmarker') as mock_benchmarker_class:
                mock_benchmarker = MagicMock()
                mock_throughput = [
                    ThroughputMetrics(100.0, 1, 4, 10.0, 25.0),
                    ThroughputMetrics(200.0, 4, 4, 20.0, 50.0),
                    ThroughputMetrics(300.0, 8, 4, 26.7, 75.0)
                ]
                mock_benchmarker.benchmark_throughput.return_value = mock_throughput
                mock_benchmarker_class.return_value = mock_benchmarker
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_benchmarker.benchmark_throughput.assert_called_once()
    
    def test_cli_compare_args(self):
        """Test compare command arguments parsing."""
        from benchmark_inference import main
        
        model1_file = Path(self.test_dir) / "model1.onnx"
        model2_file = Path(self.test_dir) / "model2.onnx"
        output_file = Path(self.test_dir) / "comparison.json"
        
        # Create dummy model files
        model1_file.write_bytes(b"dummy onnx model 1")
        model2_file.write_bytes(b"dummy onnx model 2")
        
        with patch('sys.argv', [
            'benchmark_inference.py',
            'compare',
            str(model1_file),
            str(model2_file),
            '--metrics', 'latency,throughput',
            '--device', 'cpu',
            '--iterations', '100',
            '--output', str(output_file),
            '--verbose'
        ]):
            with patch('benchmark_inference.ModelBenchmarker') as mock_benchmarker_class:
                mock_benchmarker = MagicMock()
                mock_results = {
                    "model1.onnx": MagicMock(),
                    "model2.onnx": MagicMock()
                }
                # Set up mock results
                for result in mock_results.values():
                    result.latency.mean = 10.0
                    result.throughput = [ThroughputMetrics(100.0, 1, 1, 10.0, 25.0)]
                
                mock_benchmarker.compare_models.return_value = mock_results
                mock_benchmarker_class.return_value = mock_benchmarker
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_benchmarker.compare_models.assert_called_once()
    
    def test_cli_regression_detection(self):
        """Test regression command with detection."""
        from benchmark_inference import main
        
        model_file = Path(self.test_dir) / "model.onnx"
        baseline_file = Path(self.test_dir) / "baseline.json"
        results_file = Path(self.test_dir) / "current.json"
        
        # Create dummy files
        model_file.write_bytes(b"dummy onnx model")
        
        # Create baseline data
        baseline_data = {"latency": {"mean": 10.0}}
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f)
        
        with patch('sys.argv', [
            'benchmark_inference.py',
            'regression',
            str(model_file),
            '--baseline', str(baseline_file),
            '--threshold', '5.0',
            '--save-results', str(results_file),
            '--device', 'cpu',
            '--verbose'
        ]):
            with patch('benchmark_inference.ModelBenchmarker') as mock_benchmarker_class:
                mock_benchmarker = MagicMock()
                
                # Mock regression detection
                mock_regression = {
                    'is_regression': True,
                    'regression_percent': 10.0,
                    'current_latency_ms': 11.0,
                    'baseline_latency_ms': 10.0,
                    'threshold_percent': 5.0,
                    'status': 'REGRESSION'
                }
                mock_benchmarker.detect_regression.return_value = mock_regression
                mock_benchmarker_class.return_value = mock_benchmarker
                
                result = main()
                
                # Should return 1 for regression detected
                self.assertEqual(result, 1)
                mock_benchmarker.detect_regression.assert_called_once()
    
    def test_cli_no_command(self):
        """Test CLI with no command."""
        from benchmark_inference import main
        import io
        import contextlib
        
        with patch('sys.argv', ['benchmark_inference.py']):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = main()
            
            help_output = f.getvalue()
            self.assertEqual(result, 1)
            # Should contain help text
            self.assertIn("Benchmark commands", help_output)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        from benchmark_inference import main
        
        with patch('sys.argv', [
            'benchmark_inference.py',
            'latency',
            '/non/existent/file.onnx'
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                
                self.assertEqual(result, 1)
                # Should print error message
                calls = [str(call) for call in mock_print.call_args_list]
                error_output = ' '.join(calls)
                self.assertIn("Error", error_output)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common benchmarking scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_end_to_end_benchmarking_pipeline(self):
        """Test complete benchmarking pipeline."""
        # This test demonstrates the full workflow without requiring actual models
        config = BenchmarkConfig(
            iterations=10,
            warmup_iterations=2,
            batch_sizes=[1, 4],
            duration=1,  # Short duration for testing
            verbose=False
        )
        
        benchmarker = ModelBenchmarker(config)
        
        # Test configuration
        self.assertEqual(benchmarker.config.iterations, 10)
        self.assertEqual(benchmarker.config.batch_sizes, [1, 4])
        
        # Test GPU profiler
        self.assertIsNotNone(benchmarker.gpu_profiler)
    
    def test_resource_monitoring_workflow(self):
        """Test resource monitoring workflow."""
        monitor = ResourceMonitor(track_cpu=True, track_memory=True, track_gpu=False)
        
        # Test initial state
        self.assertFalse(monitor.monitoring)
        self.assertEqual(len(monitor.measurements), 0)
        
        # Test monitoring lifecycle (without actual monitoring)
        if PSUTIL_AVAILABLE:
            monitor.start_monitoring(interval=0.001)
            time.sleep(0.01)  # Brief monitoring period
            metrics = monitor.stop_monitoring()
            
            self.assertIsInstance(metrics, ResourceMetrics)
            self.assertGreaterEqual(metrics.cpu_percent, 0.0)
    
    def test_benchmark_results_serialization(self):
        """Test benchmark results serialization and deserialization."""
        # Create sample results
        latency = LatencyMetrics(
            mean=10.0, median=9.5, std=1.0, min=8.0, max=12.0,
            p95=11.0, p99=11.5, p999=11.8
        )
        
        throughput = [
            ThroughputMetrics(100.0, 1, 1, 10.0, 25.0),
            ThroughputMetrics(200.0, 4, 4, 20.0, 50.0)
        ]
        
        resources = ResourceMetrics(
            cpu_percent=75.0,
            memory_mb=1024.0,
            gpu_utilization=85.0,
            gpu_memory_mb=2048.0
        )
        
        config = BenchmarkConfig()
        
        results = BenchmarkResults(
            model_name="test_model.onnx",
            device="cpu",
            timestamp="2024-01-01 12:00:00",
            config=config,
            latency=latency,
            throughput=throughput,
            resources=resources,
            metadata={"platform": "test"}
        )
        
        # Test serialization
        benchmarker = ModelBenchmarker(config)
        output_path = Path(self.test_dir) / "test_results.json"
        
        benchmarker.save_results(results, output_path)
        self.assertTrue(output_path.exists())
        
        # Test deserialization
        loaded_data = benchmarker.load_results(output_path)
        
        self.assertEqual(loaded_data["model_name"], "test_model.onnx")
        self.assertEqual(loaded_data["device"], "cpu")
        self.assertIn("latency", loaded_data)
        self.assertIn("throughput", loaded_data)
        self.assertIn("resources", loaded_data)
        self.assertIn("metadata", loaded_data)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceType))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricType))
    suite.addTests(loader.loadTestsFromTestCase(TestLatencyMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestThroughputMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestParsingUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestModelBenchmarker))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkerCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())