#!/usr/bin/env python3
"""
Test suite for Python bindings of the Inference Systems Laboratory

This module provides comprehensive testing for the Python-C++ integration,
including tensor operations, inference engines, and error handling.
"""

import unittest
import sys
import os
from typing import Any, List

# Add build directory to path for importing the compiled module
# This assumes the typical build structure: build/engines/src/python_bindings/
build_paths = [
    os.path.join(os.path.dirname(__file__), "..", "..", "build"),
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "engines", "src", "python_bindings"),
    os.path.join(os.path.dirname(__file__), "..", "..", "cmake-build-debug"),
    os.path.join(os.path.dirname(__file__), "..", "..", "cmake-build-release"),
]

for path in build_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

# Try to import numpy for tensor tests (graceful fallback if not available)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available, skipping tensor conversion tests")

# Try to import the compiled inference_lab module
try:
    import inference_lab
    HAS_MODULE = True
except ImportError as e:
    HAS_MODULE = False
    print(f"Warning: inference_lab module not available: {e}")
    print("Build the module first with: mkdir build && cd build && cmake .. -DBUILD_PYTHON_BINDINGS=ON && make")


class TestInferenceLabModule(unittest.TestCase):
    """Test basic module functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_MODULE:
            self.skipTest("inference_lab module not available")
    
    def test_module_import(self):
        """Test that the module can be imported successfully"""
        self.assertTrue(hasattr(inference_lab, '__version__'))
        self.assertTrue(hasattr(inference_lab, '__author__'))
    
    def test_build_info(self):
        """Test build information is accessible"""
        build_info = inference_lab.get_build_info()
        self.assertIsInstance(build_info, dict)
        self.assertIn('version', build_info)
        self.assertIn('build_type', build_info)
    
    def test_submodules_exist(self):
        """Test that expected submodules are available"""
        expected_modules = ['result', 'tensor', 'inference']
        for module_name in expected_modules:
            self.assertTrue(hasattr(inference_lab, module_name),
                          f"Missing submodule: {module_name}")


class TestResultTypes(unittest.TestCase):
    """Test Result<T, E> bindings"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_MODULE:
            self.skipTest("inference_lab module not available")
    
    def test_string_result_creation(self):
        """Test creating string results"""
        # Test success result
        ok_result = inference_lab.result.ok("success value")
        self.assertTrue(ok_result.is_ok())
        self.assertFalse(ok_result.is_err())
        self.assertEqual(ok_result.unwrap(), "success value")
        
        # Test error result
        err_result = inference_lab.result.err("error message")
        self.assertFalse(err_result.is_ok())
        self.assertTrue(err_result.is_err())
        self.assertEqual(err_result.unwrap_err(), "error message")
    
    def test_result_unwrap_error(self):
        """Test that unwrap() raises exception for error results"""
        err_result = inference_lab.result.err("test error")
        with self.assertRaises(ValueError):
            err_result.unwrap()
    
    def test_result_unwrap_or(self):
        """Test unwrap_or functionality"""
        ok_result = inference_lab.result.ok("success")
        err_result = inference_lab.result.err("error")
        
        self.assertEqual(ok_result.unwrap_or("default"), "success")
        self.assertEqual(err_result.unwrap_or("default"), "default")
    
    def test_result_collect(self):
        """Test collecting multiple results"""
        # All success case
        results = [
            inference_lab.result.ok("value1"),
            inference_lab.result.ok("value2"),
            inference_lab.result.ok("value3")
        ]
        collected = inference_lab.result.collect(results)
        self.assertTrue(collected.is_ok())
        values = collected.unwrap()
        self.assertEqual(values, ["value1", "value2", "value3"])
        
        # Contains error case
        results_with_error = [
            inference_lab.result.ok("value1"),
            inference_lab.result.err("error"),
            inference_lab.result.ok("value3")
        ]
        collected_error = inference_lab.result.collect(results_with_error)
        self.assertTrue(collected_error.is_err())
        self.assertEqual(collected_error.unwrap_err(), "error")
    
    def test_safe_result_wrapper(self):
        """Test SafeResult wrapper functionality"""
        # Test safe execution with success
        def success_func():
            return "success"
        
        safe_result = inference_lab.result.safe_execute(success_func)
        self.assertTrue(safe_result.is_success())
        result_value = safe_result.get()
        self.assertEqual(result_value, "success")
        
        # Test safe execution with error
        def error_func():
            raise ValueError("test error")
        
        safe_error_result = inference_lab.result.safe_execute(error_func)
        self.assertFalse(safe_error_result.is_success())
        self.assertIn("test error", safe_error_result.get_error_message())
        
        with self.assertRaises(RuntimeError):
            safe_error_result.get()


@unittest.skipUnless(HAS_NUMPY, "NumPy not available")
class TestTensorOperations(unittest.TestCase):
    """Test tensor data exchange between Python and C++"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_MODULE:
            self.skipTest("inference_lab module not available")
    
    def test_tensor_creation(self):
        """Test basic tensor creation"""
        # Test creating tensors with different shapes
        tensor_1d = inference_lab.tensor.Tensor([10])
        self.assertEqual(tensor_1d.shape(), [10])
        self.assertEqual(tensor_1d.size(), 10)
        
        tensor_2d = inference_lab.tensor.Tensor([3, 4])
        self.assertEqual(tensor_2d.shape(), [3, 4])
        self.assertEqual(tensor_2d.size(), 12)
    
    def test_tensor_fill_operations(self):
        """Test tensor fill operations"""
        tensor = inference_lab.tensor.Tensor([2, 3])
        
        # Fill with specific value
        tensor.fill(42.0)
        numpy_array = tensor.to_numpy()
        np.testing.assert_array_equal(numpy_array, np.full((2, 3), 42.0))
    
    def test_factory_functions(self):
        """Test zeros and ones factory functions"""
        # Test zeros
        zeros_tensor = inference_lab.tensor.zeros([2, 3])
        zeros_array = zeros_tensor.to_numpy()
        np.testing.assert_array_equal(zeros_array, np.zeros((2, 3)))
        
        # Test ones
        ones_tensor = inference_lab.tensor.ones([2, 3])
        ones_array = ones_tensor.to_numpy()
        np.testing.assert_array_equal(ones_array, np.ones((2, 3)))
    
    def test_numpy_roundtrip(self):
        """Test NumPy to C++ tensor and back conversion"""
        # Test with different data types
        test_arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[1, 2], [3, 4]], dtype=np.int64),
        ]
        
        for original_array in test_arrays:
            with self.subTest(dtype=original_array.dtype):
                result_array = inference_lab.tensor.test_numpy_roundtrip(original_array)
                
                # Should be float32 after conversion
                self.assertEqual(result_array.dtype, np.float32)
                
                # Values should be preserved (within float32 precision)
                np.testing.assert_allclose(
                    result_array, 
                    original_array.astype(np.float32), 
                    rtol=1e-6
                )
    
    def test_dtype_support(self):
        """Test supported data type information"""
        supported_types = inference_lab.tensor.test_dtype_support()
        self.assertIsInstance(supported_types, dict)
        
        # Check expected supported types
        self.assertTrue(supported_types.get('float32'))
        self.assertTrue(supported_types.get('float64'))
        self.assertTrue(supported_types.get('int32'))
        self.assertTrue(supported_types.get('int64'))
        
        # Check expected unsupported types
        self.assertFalse(supported_types.get('int8'))
        self.assertFalse(supported_types.get('uint8'))
    
    def test_invalid_tensor_operations(self):
        """Test error handling for invalid tensor operations"""
        # Test empty array conversion
        with self.assertRaises(RuntimeError):
            empty_array = np.array([])
            inference_lab.tensor.test_numpy_roundtrip(empty_array)
        
        # Test 0-dimensional array conversion
        with self.assertRaises(RuntimeError):
            scalar_array = np.array(42.0)
            inference_lab.tensor.test_numpy_roundtrip(scalar_array)
    
    def test_benchmark_conversion(self):
        """Test tensor conversion benchmarking"""
        test_array = np.random.random((100, 100)).astype(np.float32)
        
        # Benchmark with small iteration count
        avg_time = inference_lab.tensor.benchmark_conversion(test_array, 10)
        self.assertIsInstance(avg_time, float)
        self.assertGreater(avg_time, 0)  # Should take some time
        
        # Test with invalid iterations
        with self.assertRaises(ValueError):
            inference_lab.tensor.benchmark_conversion(test_array, 0)


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine bindings"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_MODULE:
            self.skipTest("inference_lab module not available")
    
    def test_inference_error_enum(self):
        """Test InferenceError enum values"""
        # Test that enum values exist and are accessible
        error_types = [
            'MODEL_LOAD_FAILED',
            'UNSUPPORTED_MODEL_FORMAT',
            'MODEL_VERSION_MISMATCH',
            'INFERENCE_FAILED',
            'INVALID_INPUT',
            'TIMEOUT',
            'OUT_OF_MEMORY',
            'GPU_NOT_AVAILABLE',
            'BACKEND_NOT_AVAILABLE',
            'CONFIGURATION_ERROR',
            'SERIALIZATION_ERROR',
            'NETWORK_ERROR',
            'UNKNOWN_ERROR'
        ]
        
        for error_type in error_types:
            self.assertTrue(hasattr(inference_lab.inference.InferenceError, error_type))
    
    def test_inference_backend_enum(self):
        """Test InferenceBackend enum values"""
        backend_types = [
            'RULE_BASED',
            'TENSORRT_GPU',
            'ONNX_RUNTIME',
            'MOCK',
            'HYBRID_NEURAL_SYMBOLIC'
        ]
        
        for backend_type in backend_types:
            self.assertTrue(hasattr(inference_lab.inference.InferenceBackend, backend_type))
    
    def test_model_config(self):
        """Test ModelConfig struct"""
        config = inference_lab.inference.ModelConfig()
        
        # Test default values and attribute access
        self.assertEqual(config.max_batch_size, 1)
        self.assertEqual(config.enable_optimization, False)
        self.assertEqual(config.enable_profiling, False)
        self.assertEqual(config.gpu_device_id, -1)
        
        # Test setting values
        config.model_path = "/path/to/model"
        config.max_batch_size = 8
        config.enable_optimization = True
        
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.max_batch_size, 8)
        self.assertEqual(config.enable_optimization, True)
    
    def test_inference_request_response(self):
        """Test InferenceRequest and InferenceResponse structs"""
        # Test InferenceRequest
        request = inference_lab.inference.InferenceRequest()
        request.input_names = ["input1", "input2"]
        request.batch_size = 4
        
        self.assertEqual(request.input_names, ["input1", "input2"])
        self.assertEqual(request.batch_size, 4)
        
        # Test InferenceResponse
        response = inference_lab.inference.InferenceResponse()
        response.output_names = ["output1"]
        response.inference_time_ms = 15.5
        response.memory_used_bytes = 1024
        
        self.assertEqual(response.output_names, ["output1"])
        self.assertEqual(response.inference_time_ms, 15.5)
        self.assertEqual(response.memory_used_bytes, 1024)
    
    def test_available_backends(self):
        """Test getting available backends"""
        backends = inference_lab.inference.get_available_backends()
        self.assertIsInstance(backends, list)
        self.assertIn("cpu", backends)
        self.assertIn("mock", backends)
    
    def test_placeholder_engine(self):
        """Test placeholder inference engine functionality"""
        # Create engine
        engine = inference_lab.inference.Engine("test_engine")
        self.assertEqual(engine.name(), "test_engine")
        self.assertFalse(engine.is_initialized())
        
        # Initialize engine
        result = engine.initialize("/fake/path")
        self.assertTrue(result)
        self.assertTrue(engine.is_initialized())
        
        # Run inference
        output = engine.run_inference("test input")
        self.assertIn("test input", output)
        
        # Shutdown
        engine.shutdown()
        self.assertFalse(engine.is_initialized())
        
        # Test error on uninitialized engine
        with self.assertRaises(RuntimeError):
            engine.run_inference("test input")


class TestExceptionTranslation(unittest.TestCase):
    """Test exception translation between C++ and Python"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not HAS_MODULE:
            self.skipTest("inference_lab module not available")
    
    def test_exception_translation(self):
        """Test intelligent exception translation"""
        test_cases = [
            ("invalid input", ValueError),
            ("type mismatch", TypeError),
            ("not found", KeyError),
            ("index out of range", IndexError),
            ("tensor operation failed", ValueError),
            ("model load failed", RuntimeError),
            ("inference failed", RuntimeError),
            ("unknown error", RuntimeError),
        ]
        
        for error_msg, expected_exception in test_cases:
            with self.subTest(error_msg=error_msg):
                with self.assertRaises(expected_exception):
                    inference_lab.result.translate_exception(error_msg)


def run_tests():
    """Run all tests with appropriate test discovery"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestInferenceLabModule,
        TestResultTypes,
        TestTensorOperations,
        TestInferenceEngine,
        TestExceptionTranslation,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("Inference Systems Laboratory - Python Bindings Test Suite")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy available: {HAS_NUMPY}")
    print(f"Module available: {HAS_MODULE}")
    print("=" * 70)
    
    success = run_tests()
    
    print("=" * 70)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
