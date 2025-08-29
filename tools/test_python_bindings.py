#!/usr/bin/env python3
"""
Test script for Python bindings integration.
Tests the basic functionality of the inference_lab Python module.
"""

import sys
import numpy as np

def test_module_import():
    """Test basic module import"""
    try:
        import inference_lab
        print("✅ Module import: SUCCESS")
        print(f"   Version: {inference_lab.__version__}")
        print(f"   Build info: {inference_lab.get_build_info()}")
        return True
    except ImportError as e:
        print(f"❌ Module import: FAILED - {e}")
        return False

def test_result_types():
    """Test Result<T,E> bindings"""
    try:
        import inference_lab.result as result
        
        # Test success result
        ok_result = result.ok("success")
        assert ok_result.is_ok()
        assert ok_result.unwrap() == "success"
        
        # Test error result
        err_result = result.err("error")
        assert err_result.is_err()
        assert err_result.unwrap_err() == "error"
        
        print("✅ Result types: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Result types: FAILED - {e}")
        return False

def test_tensor_operations():
    """Test tensor and NumPy integration"""
    try:
        import inference_lab.tensor as tensor
        
        # Create tensor
        t = tensor.Tensor([2, 3])
        assert t.shape() == [2, 3]
        assert t.size() == 6
        
        # Test NumPy conversion
        np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        cpp_tensor = tensor.Tensor.from_numpy(np_array)
        converted_back = cpp_tensor.to_numpy()
        
        # Test roundtrip
        roundtrip_result = tensor.test_numpy_roundtrip(np_array)
        
        print("✅ Tensor operations: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Tensor operations: FAILED - {e}")
        return False

def test_inference_engine():
    """Test inference engine bindings"""
    try:
        import inference_lab.inference as inference
        
        # Create config
        config = inference.Config()
        config.model_path = "test_model.bin"
        config.backend = "mock"
        config.batch_size = 4
        
        # Create engine
        engine = inference.create_engine(config)
        assert engine.is_initialized()
        
        # Run inference
        result = engine.run_inference("test input")
        assert "test input" in result
        
        print("✅ Inference engine: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Inference engine: FAILED - {e}")
        return False

def test_logging_system():
    """Test logging system integration"""
    try:
        import inference_lab.logging as logging
        
        # Create logger
        logger = logging.Logger("test_component")
        
        # Test log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        print("✅ Logging system: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Logging system: FAILED - {e}")
        return False

def test_performance():
    """Test performance characteristics"""
    try:
        import inference_lab.tensor as tensor
        import inference_lab.inference as inference
        import time
        
        # Test tensor conversion performance
        large_array = np.random.random((1000, 1000)).astype(np.float32)
        conversion_time = tensor.benchmark_conversion(large_array, 100)
        print(f"   Tensor conversion: {conversion_time:.2f} μs/op")
        
        # Test inference performance
        config = inference.Config()
        config.model_path = "test"
        config.backend = "mock"
        engine = inference.create_engine(config)
        
        inference_time = inference.benchmark_inference(engine, "test", 1000)
        print(f"   Inference: {inference_time:.2f} μs/op")
        
        print("✅ Performance tests: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Performance tests: FAILED - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Python Bindings Integration")
    print("=" * 50)
    
    tests = [
        test_module_import,
        test_result_types,
        test_tensor_operations,
        test_inference_engine,
        test_logging_system,
        test_performance,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Python binding tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
