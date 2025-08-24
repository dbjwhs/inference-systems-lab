#!/usr/bin/env python3
"""
Integration tests for ModelRegistry cross-language compatibility.

This module tests both the Python ModelRegistryClient and the C++ ModelRegistry 
bindings to ensure they provide compatible interfaces and can work with the 
same SQLite database.
"""

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build" / "engines" / "src" / "python_bindings"))

try:
    from model_registry_client import (
        ModelRegistryClient,
        ModelStatus,
        ModelType,
        InferenceBackend,
    )
    PYTHON_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Python ModelRegistryClient not available: {e}")
    PYTHON_CLIENT_AVAILABLE = False

try:
    import inference_lab
    CPP_BINDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: C++ bindings not available: {e}")
    CPP_BINDINGS_AVAILABLE = False


class TestModelRegistryIntegration(unittest.TestCase):
    """Test integration between Python and C++ ModelRegistry implementations."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_registry.db")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not PYTHON_CLIENT_AVAILABLE, "Python client not available")
    def test_python_client_basic_operations(self):
        """Test basic operations with Python ModelRegistryClient."""
        client = ModelRegistryClient(self.db_path)
        
        # Test model registration
        model_id = client.register_model(
            model_name="test_model",
            model_version="1.0.0",
            model_type=ModelType.ONNX,
            backend=InferenceBackend.RULE_BASED,
            model_path="/path/to/model.onnx",
            description="Test model for integration tests",
            author="Test Suite"
        )
        
        self.assertIsInstance(model_id, int)
        self.assertGreater(model_id, 0)
        
        # Test model retrieval
        model_info = client.get_model(model_id)
        self.assertEqual(model_info['model_name'], "test_model")
        self.assertEqual(model_info['model_version'], "1.0.0")
        self.assertEqual(model_info['model_type'], "onnx")
        self.assertEqual(model_info['description'], "Test model for integration tests")
        
        # Test model listing
        models = client.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['model_id'], model_id)
        
        # Test status update
        client.update_model_status(model_id, ModelStatus.PRODUCTION)
        updated_model = client.get_model(model_id)
        self.assertEqual(updated_model['status'], 'production')

    @unittest.skipIf(not CPP_BINDINGS_AVAILABLE, "C++ bindings not available")
    def test_cpp_bindings_basic_operations(self):
        """Test basic operations with C++ ModelRegistry bindings."""
        registry = inference_lab.ModelRegistry.create(self.db_path)
        
        # Create ModelInfo structure
        model_info = inference_lab.ModelInfo()
        model_info.model_name = "cpp_test_model"
        model_info.model_version = "2.0.0"
        model_info.model_type = inference_lab.ModelType.TENSORRT
        model_info.backend = inference_lab.InferenceBackend.TENSORRT_GPU
        model_info.model_path = "/path/to/model.trt"
        model_info.status = inference_lab.ModelStatus.DEVELOPMENT
        
        # Test model registration
        model_id = registry.register_model(model_info)
        self.assertIsInstance(model_id, int)
        self.assertGreater(model_id, 0)
        
        # Test model retrieval by ID
        retrieved_model = registry.get_model(model_id)
        self.assertEqual(retrieved_model.model_name, "cpp_test_model")
        self.assertEqual(retrieved_model.model_version, "2.0.0")
        self.assertEqual(retrieved_model.model_type, inference_lab.ModelType.TENSORRT)
        
        # Test model retrieval by name and version
        retrieved_model2 = registry.get_model("cpp_test_model", "2.0.0")
        self.assertEqual(retrieved_model2.model_id, model_id)
        
        # Test model listing
        models = registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].model_id, model_id)
        
        # Test status update
        registry.update_model_status(model_id, inference_lab.ModelStatus.STAGING)
        updated_model = registry.get_model(model_id)
        self.assertEqual(updated_model.status, inference_lab.ModelStatus.STAGING)

    @unittest.skipIf(not (PYTHON_CLIENT_AVAILABLE and CPP_BINDINGS_AVAILABLE), 
                     "Both Python and C++ implementations not available")
    def test_cross_language_compatibility(self):
        """Test that Python and C++ implementations can work with same database."""
        # Create model with Python client
        python_client = ModelRegistryClient(self.db_path)
        python_model_id = python_client.register_model(
            model_name="cross_lang_model",
            model_version="1.0.0",
            model_type=ModelType.PYTORCH,
            backend=InferenceBackend.ONNX_RUNTIME,
            model_path="/path/to/model.pt"
        )
        
        # Read model with C++ registry
        cpp_registry = inference_lab.ModelRegistry.create(self.db_path)
        cpp_model = cpp_registry.get_model(python_model_id)
        
        # Verify compatibility
        self.assertEqual(cpp_model.model_name, "cross_lang_model")
        self.assertEqual(cpp_model.model_version, "1.0.0")
        self.assertEqual(cpp_model.model_type, inference_lab.ModelType.PYTORCH)
        self.assertEqual(cpp_model.backend, inference_lab.InferenceBackend.ONNX_RUNTIME)
        
        # Create model with C++ registry
        cpp_model_info = inference_lab.ModelInfo()
        cpp_model_info.model_name = "cpp_created_model"
        cpp_model_info.model_version = "1.0.0"
        cpp_model_info.model_type = inference_lab.ModelType.RULE_BASED
        cpp_model_info.backend = inference_lab.InferenceBackend.RULE_BASED
        cpp_model_info.model_path = "/path/to/rules.json"
        cpp_model_info.status = inference_lab.ModelStatus.DEVELOPMENT
        
        cpp_model_id = cpp_registry.register_model(cpp_model_info)
        
        # Read model with Python client
        python_model = python_client.get_model(cpp_model_id)
        
        # Verify compatibility
        self.assertEqual(python_model['model_name'], "cpp_created_model")
        self.assertEqual(python_model['model_version'], "1.0.0")
        self.assertEqual(python_model['model_type'], "rule_based")
        self.assertEqual(python_model['backend'], "RULE_BASED")
        
        # Verify both implementations see same models
        python_models = python_client.list_models()
        cpp_models = cpp_registry.list_models()
        
        self.assertEqual(len(python_models), 2)
        self.assertEqual(len(cpp_models), 2)

    @unittest.skipIf(not CPP_BINDINGS_AVAILABLE, "C++ bindings not available")
    def test_cpp_performance_metrics(self):
        """Test performance metrics structures with C++ bindings."""
        registry = inference_lab.ModelRegistry.create(self.db_path)
        
        # Test that we can create performance metrics structures
        # (Recording not yet implemented in C++)
        metrics = inference_lab.PerformanceMetrics()
        metrics.model_id = 1
        metrics.inference_time_ms = 15.5
        metrics.memory_usage_mb = 128.0
        metrics.batch_size = 1
        
        # Verify structure fields are accessible
        self.assertEqual(metrics.model_id, 1)
        self.assertAlmostEqual(metrics.inference_time_ms, 15.5)
        self.assertAlmostEqual(metrics.memory_usage_mb, 128.0)
        self.assertEqual(metrics.batch_size, 1)

    @unittest.skipIf(not CPP_BINDINGS_AVAILABLE, "C++ bindings not available")
    def test_cpp_validation_results(self):
        """Test validation result structures with C++ bindings."""
        registry = inference_lab.ModelRegistry.create(self.db_path)
        
        # Test that we can create validation result structures
        # (Recording not yet implemented in C++)
        validation = inference_lab.ValidationResult()
        validation.model_id = 1
        validation.validation_type = "accuracy"
        validation.validation_status = "passed"
        
        # Verify structure fields are accessible
        self.assertEqual(validation.model_id, 1)
        self.assertEqual(validation.validation_type, "accuracy")
        self.assertEqual(validation.validation_status, "passed")

    @unittest.skipIf(not CPP_BINDINGS_AVAILABLE, "C++ bindings not available")
    def test_cpp_error_handling(self):
        """Test error handling in C++ bindings."""
        registry = inference_lab.ModelRegistry.create(self.db_path)
        
        # Test getting non-existent model
        with self.assertRaises(ValueError) as context:
            registry.get_model(999999)
        self.assertIn("Failed to get model", str(context.exception))
        
        # Test getting non-existent model by name/version
        with self.assertRaises(ValueError) as context:
            registry.get_model("non_existent", "1.0.0")
        self.assertIn("Failed to get model", str(context.exception))

    @unittest.skipIf(not CPP_BINDINGS_AVAILABLE, "C++ bindings not available")
    def test_cpp_utility_methods(self):
        """Test basic utility methods in C++ bindings."""
        registry = inference_lab.ModelRegistry.create(self.db_path)
        
        # Test connection status
        self.assertTrue(registry.is_connected())
        
        # Test database path
        self.assertEqual(registry.get_db_path(), self.db_path)
        
        # Create some test models to verify basic listing works
        for i in range(3):
            model_info = inference_lab.ModelInfo()
            model_info.model_name = f"utility_test_model_{i}"
            model_info.model_version = "1.0.0"
            model_info.model_type = inference_lab.ModelType.ONNX
            model_info.backend = inference_lab.InferenceBackend.ONNX_RUNTIME
            model_info.model_path = f"/path/to/model_{i}.onnx"
            model_info.status = inference_lab.ModelStatus.PRODUCTION if i == 0 else inference_lab.ModelStatus.DEVELOPMENT
            
            registry.register_model(model_info)
        
        # Test basic listing (filtering methods not yet implemented)
        all_models = registry.list_models()
        self.assertEqual(len(all_models), 3)

    @unittest.skipIf(not PYTHON_CLIENT_AVAILABLE, "Python client not available")
    def test_python_client_filtering(self):
        """Test model filtering capabilities in Python client."""
        client = ModelRegistryClient(self.db_path)
        
        # Create models with different statuses and types
        models_data = [
            ("model_a", ModelType.ONNX, ModelStatus.DEVELOPMENT),
            ("model_b", ModelType.TENSORRT, ModelStatus.PRODUCTION),
            ("model_c", ModelType.ONNX, ModelStatus.PRODUCTION),
            ("model_d", ModelType.PYTORCH, ModelStatus.STAGING)
        ]
        
        for name, model_type, status in models_data:
            model_id = client.register_model(
                model_name=name,
                model_version="1.0.0",
                model_type=model_type,
                backend=InferenceBackend.RULE_BASED,
                model_path=f"/path/to/{name}.model"
            )
            client.update_model_status(model_id, status)
        
        # Test filtering by status
        production_models = client.list_models(status=ModelStatus.PRODUCTION)
        self.assertEqual(len(production_models), 2)
        production_names = [m['model_name'] for m in production_models]
        self.assertIn("model_b", production_names)
        self.assertIn("model_c", production_names)
        
        # Test filtering by type
        onnx_models = client.list_models(model_type=ModelType.ONNX)
        self.assertEqual(len(onnx_models), 2)
        onnx_names = [m['model_name'] for m in onnx_models]
        self.assertIn("model_a", onnx_names)
        self.assertIn("model_c", onnx_names)


class TestModelRegistrySchema(unittest.TestCase):
    """Test database schema and data integrity."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "schema_test.db")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not PYTHON_CLIENT_AVAILABLE, "Python client not available")
    def test_database_schema_initialization(self):
        """Test that database schema is properly initialized."""
        import sqlite3
        
        # Create database through registry
        client = ModelRegistryClient(self.db_path)
        
        # Check that database file exists
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that required tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'models',
            'deployment_history', 
            'performance_metrics',
            'validation_results',
            'model_dependencies'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found in database")
        
        conn.close()

    @unittest.skipIf(not PYTHON_CLIENT_AVAILABLE, "Python client not available")
    def test_database_constraints(self):
        """Test database constraints and data validation."""
        client = ModelRegistryClient(self.db_path)
        
        # Test unique constraint on (model_name, model_version)
        client.register_model(
            model_name="unique_test",
            model_version="1.0.0",
            model_type=ModelType.ONNX,
            backend=InferenceBackend.RULE_BASED,
            model_path="/path/to/model1.onnx"
        )
        
        # This should raise an error due to unique constraint
        with self.assertRaises(Exception):
            client.register_model(
                model_name="unique_test",
                model_version="1.0.0",  # Same name and version
                model_type=ModelType.PYTORCH,
                backend=InferenceBackend.RULE_BASED,
                model_path="/path/to/model2.pt"
            )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
