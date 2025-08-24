#!/usr/bin/env python3
"""
Integration tests for Python-C++ toolchain integration

This module tests the integration between Python ML tools and the build system,
ensuring that the Python bindings work correctly with existing Python tooling.
"""

import unittest
import sys
import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for importing Python tools
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import our Python ML tools
try:
    from tools.model_manager import ModelManager
    HAS_MODEL_MANAGER = True
except ImportError:
    HAS_MODEL_MANAGER = False

try:
    from tools.convert_model import ModelConverter
    HAS_CONVERTER = True
except ImportError:
    HAS_CONVERTER = False


class TestBuildSystem(unittest.TestCase):
    """Test build system integration with Python bindings"""
    
    def test_cmake_configuration(self):
        """Test that CMake can configure with Python bindings enabled"""
        # Check if CMakeLists.txt exists and contains Python bindings option
        cmake_file = PROJECT_ROOT / "CMakeLists.txt"
        self.assertTrue(cmake_file.exists(), "Main CMakeLists.txt should exist")
        
        with open(cmake_file, 'r') as f:
            content = f.read()
            self.assertIn("BUILD_PYTHON_BINDINGS", content,
                         "CMakeLists.txt should contain BUILD_PYTHON_BINDINGS option")
    
    def test_pybind11_cmake_module(self):
        """Test that Pybind11.cmake module exists"""
        pybind_cmake = PROJECT_ROOT / "cmake" / "Pybind11.cmake"
        self.assertTrue(pybind_cmake.exists(), "Pybind11.cmake module should exist")
    
    def test_python_bindings_directory(self):
        """Test that Python bindings directory structure is correct"""
        bindings_dir = PROJECT_ROOT / "engines" / "src" / "python_bindings"
        self.assertTrue(bindings_dir.exists(), "Python bindings directory should exist")
        
        expected_files = [
            "main.cpp",
            "tensor_bindings.cpp", 
            "inference_bindings.cpp",
            "result_bindings.cpp",
            "logging_bindings.cpp",
            "CMakeLists.txt"
        ]
        
        for filename in expected_files:
            file_path = bindings_dir / filename
            self.assertTrue(file_path.exists(), f"Should have {filename}")
    
    def test_setup_py_exists(self):
        """Test that setup.py exists for pip installation"""
        setup_file = PROJECT_ROOT / "setup.py"
        self.assertTrue(setup_file.exists(), "setup.py should exist")
        
        with open(setup_file, 'r') as f:
            content = f.read()
            self.assertIn("inference_lab", content)
            self.assertIn("CMakeExtension", content)


class TestPythonToolsIntegration(unittest.TestCase):
    """Test integration with existing Python ML tools"""
    
    @unittest.skipUnless(HAS_MODEL_MANAGER, "ModelManager not available")
    def test_model_manager_integration(self):
        """Test ModelManager can work with inference lab"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock model registry
            registry_path = Path(temp_dir) / "model_registry.db"
            
            # Test that ModelManager can be instantiated
            manager = ModelManager(str(registry_path))
            self.assertIsInstance(manager, ModelManager)
            
            # Test basic operations don't crash
            models = manager.list_models()
            self.assertIsInstance(models, list)
    
    def test_requirements_file(self):
        """Test that requirements-dev.txt includes necessary dependencies"""
        req_file = PROJECT_ROOT / "requirements-dev.txt"
        self.assertTrue(req_file.exists(), "requirements-dev.txt should exist")
        
        with open(req_file, 'r') as f:
            content = f.read()
            self.assertIn("numpy", content)
    
    def test_python_path_structure(self):
        """Test that Python module structure is correct"""
        tools_dir = PROJECT_ROOT / "tools"
        self.assertTrue(tools_dir.exists(), "tools directory should exist")
        
        expected_tools = [
            "model_manager.py",
            "convert_model.py", 
            "benchmark_inference.py",
            "validate_model.py"
        ]
        
        for tool_name in expected_tools:
            tool_path = tools_dir / tool_name
            self.assertTrue(tool_path.exists(), f"Should have {tool_name}")


class TestDocumentation(unittest.TestCase):
    """Test documentation and examples"""
    
    def test_binding_documentation(self):
        """Test that Python binding documentation exists"""
        docs_file = PROJECT_ROOT / "docs" / "PYTHON_CPP_BINDING.md"
        self.assertTrue(docs_file.exists(), "Python binding documentation should exist")
        
        with open(docs_file, 'r') as f:
            content = f.read()
            self.assertIn("tensor", content.lower())
            self.assertIn("inference", content.lower())
            self.assertIn("result", content.lower())
    
    def test_readme_mentions_python(self):
        """Test that main README mentions Python bindings"""
        readme_file = PROJECT_ROOT / "README.md"
        self.assertTrue(readme_file.exists(), "README.md should exist")
        
        with open(readme_file, 'r') as f:
            content = f.read()
            # Should mention Python somewhere
            self.assertTrue("python" in content.lower() or "bindings" in content.lower(),
                           "README should mention Python or bindings")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling integration across languages"""
    
    def test_result_hpp_exists(self):
        """Test that Result<T,E> implementation exists"""
        result_file = PROJECT_ROOT / "common" / "src" / "result.hpp"
        self.assertTrue(result_file.exists(), "Result<T,E> implementation should exist")
        
        with open(result_file, 'r') as f:
            content = f.read()
            self.assertIn("Result", content)
            self.assertIn("Ok", content)
            self.assertIn("Err", content)
    
    def test_exception_patterns(self):
        """Test that exception handling patterns are consistent"""
        bindings_files = [
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "result_bindings.cpp",
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "tensor_bindings.cpp",
        ]
        
        for bindings_file in bindings_files:
            self.assertTrue(bindings_file.exists(), f"Should have {bindings_file.name}")
            
            with open(bindings_file, 'r') as f:
                content = f.read()
                # Should have exception handling
                self.assertTrue("exception" in content.lower() or "error" in content.lower(),
                               f"{bindings_file.name} should handle exceptions")


class TestCompilationTest(unittest.TestCase):
    """Test compilation without actually building (syntax checks)"""
    
    def test_cpp_syntax(self):
        """Test that C++ files have valid syntax (basic checks)"""
        cpp_files = [
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "main.cpp",
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "tensor_bindings.cpp",
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "inference_bindings.cpp",
            PROJECT_ROOT / "engines" / "src" / "python_bindings" / "result_bindings.cpp",
        ]
        
        for cpp_file in cpp_files:
            self.assertTrue(cpp_file.exists(), f"Should have {cpp_file.name}")
            
            with open(cpp_file, 'r') as f:
                content = f.read()
                # Basic syntax checks
                self.assertIn("#include", content, f"{cpp_file.name} should have includes")
                self.assertIn("pybind11", content, f"{cpp_file.name} should use pybind11")
                
                # Check for balanced braces (very basic)
                open_braces = content.count('{')
                close_braces = content.count('}')
                self.assertEqual(open_braces, close_braces, 
                               f"{cpp_file.name} should have balanced braces")
    
    def test_python_syntax(self):
        """Test that Python test file has valid syntax"""
        test_file = PROJECT_ROOT / "engines" / "tests" / "test_python_bindings.py"
        self.assertTrue(test_file.exists(), "Python test file should exist")
        
        # Try to compile the Python file
        try:
            with open(test_file, 'r') as f:
                compile(f.read(), str(test_file), 'exec')
        except SyntaxError as e:
            self.fail(f"Python test file has syntax error: {e}")


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBuildSystem,
        TestPythonToolsIntegration,
        TestDocumentation,
        TestErrorHandlingIntegration,
        TestCompilationTest,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 70)
    print("Inference Systems Laboratory - Integration Test Suite")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model Manager available: {HAS_MODEL_MANAGER}")
    print(f"Converter available: {HAS_CONVERTER}")
    print("=" * 70)
    
    success = run_integration_tests()
    
    print("=" * 70)
    if success:
        print("✅ All integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some integration tests failed!")
        sys.exit(1)
