#!/usr/bin/env python3
"""
Test suite for model_manager.py - Comprehensive testing of model registry functionality.

This test script validates all aspects of the model manager including:
- Model registration and versioning
- Lifecycle management (dev/staging/production)
- Model validation and metadata extraction
- Version comparison and rollback
- Registry export/import
- Error handling and edge cases
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import (
    ModelRegistry, ModelMetadata, ModelStage, ModelFramework
)


class TestModelRegistry(unittest.TestCase):
    """Test suite for ModelRegistry class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test registry
        self.test_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.test_dir) / "test_models.db"
        self.registry = ModelRegistry(self.registry_path)
        
        # Create test model files
        self.test_models_dir = Path(self.test_dir) / "test_models"
        self.test_models_dir.mkdir(exist_ok=True)
        
        # Create dummy ONNX model file
        self.onnx_model = self.test_models_dir / "resnet50.onnx"
        self.onnx_model.write_bytes(b"ONNX_MODEL_DATA_PLACEHOLDER")
        
        # Create dummy TensorRT engine file
        self.trt_model = self.test_models_dir / "resnet50.engine"
        self.trt_model.write_bytes(b"TENSORRT_ENGINE_DATA_PLACEHOLDER")
        
        # Create dummy PyTorch model file
        self.pytorch_model = self.test_models_dir / "model.pt"
        self.pytorch_model.write_bytes(b"PYTORCH_MODEL_DATA_PLACEHOLDER")
    
    def tearDown(self):
        """Clean up test environment after each test."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_registry_initialization(self):
        """Test registry database initialization."""
        # Check database file exists
        self.assertTrue(self.registry_path.exists())
        
        # Check models directory exists
        models_dir = self.registry_path.parent / "models"
        self.assertTrue(models_dir.exists())
        
        # Check database schema
        with sqlite3.connect(self.registry_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor]
            self.assertIn("models", tables)
            
            # Check indexes
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor]
            self.assertIn("idx_model_name", indexes)
            self.assertIn("idx_model_stage", indexes)
    
    def test_model_registration(self):
        """Test basic model registration."""
        metadata = self.registry.register(
            self.onnx_model,
            name="resnet50",
            version="1.0.0",
            stage="dev",
            description="Test ResNet50 model",
            author="Test Author",
            tags=["computer-vision", "classification"]
        )
        
        # Verify metadata
        self.assertEqual(metadata.name, "resnet50")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.framework, "onnx")
        self.assertEqual(metadata.stage, "dev")
        self.assertEqual(metadata.description, "Test ResNet50 model")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.tags, "computer-vision,classification")
        self.assertIsNotNone(metadata.file_hash)
        self.assertGreater(metadata.file_size, 0)
        
        # Verify file was copied to registry
        dest_path = Path(metadata.file_path)
        self.assertTrue(dest_path.exists())
        self.assertEqual(dest_path.read_bytes(), b"ONNX_MODEL_DATA_PLACEHOLDER")
    
    def test_auto_version_increment(self):
        """Test automatic version incrementing."""
        # Register first version
        v1 = self.registry.register(self.onnx_model, name="test_model")
        self.assertEqual(v1.version, "1.0.0")
        
        # Register second version (auto-increment)
        v2 = self.registry.register(self.onnx_model, name="test_model")
        self.assertEqual(v2.version, "1.0.1")
        
        # Register third version (auto-increment)
        v3 = self.registry.register(self.onnx_model, name="test_model")
        self.assertEqual(v3.version, "1.0.2")
    
    def test_duplicate_version_error(self):
        """Test error when registering duplicate version."""
        # Register first time
        self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0"
        )
        
        # Try to register same version again
        with self.assertRaises(ValueError) as context:
            self.registry.register(
                self.onnx_model,
                name="test_model",
                version="1.0.0"
            )
        self.assertIn("already exists", str(context.exception))
    
    def test_framework_detection(self):
        """Test ML framework detection from file extensions."""
        # Test ONNX
        framework = self.registry._detect_framework(self.onnx_model)
        self.assertEqual(framework, ModelFramework.ONNX)
        
        # Test TensorRT
        framework = self.registry._detect_framework(self.trt_model)
        self.assertEqual(framework, ModelFramework.TENSORRT)
        
        # Test PyTorch
        framework = self.registry._detect_framework(self.pytorch_model)
        self.assertEqual(framework, ModelFramework.PYTORCH)
        
        # Test unknown
        unknown_file = self.test_models_dir / "unknown.xyz"
        unknown_file.write_text("data")
        framework = self.registry._detect_framework(unknown_file)
        self.assertEqual(framework, ModelFramework.UNKNOWN)
    
    def test_semantic_version_validation(self):
        """Test semantic version format validation."""
        # Valid versions
        self.assertTrue(self.registry._validate_version("1.0.0"))
        self.assertTrue(self.registry._validate_version("2.1.3"))
        self.assertTrue(self.registry._validate_version("10.20.30"))
        
        # Invalid versions
        self.assertFalse(self.registry._validate_version("1.0"))
        self.assertFalse(self.registry._validate_version("1.0.0.0"))
        self.assertFalse(self.registry._validate_version("v1.0.0"))
        self.assertFalse(self.registry._validate_version("1.a.0"))
    
    def test_get_model(self):
        """Test retrieving models from registry."""
        # Register multiple versions
        v1 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0"
        )
        v2 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.1.0"
        )
        
        # Get specific version
        model = self.registry.get_model("test_model", "1.0.0")
        self.assertIsNotNone(model)
        self.assertEqual(model.version, "1.0.0")
        
        # Get latest version (no version specified)
        latest = self.registry.get_model("test_model")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.version, "1.1.0")
        
        # Get non-existent model
        none_model = self.registry.get_model("non_existent")
        self.assertIsNone(none_model)
    
    def test_list_models(self):
        """Test listing models with filters."""
        # Register multiple models
        self.registry.register(
            self.onnx_model,
            name="resnet50",
            version="1.0.0",
            stage="dev",
            tags=["vision", "classification"]
        )
        self.registry.register(
            self.onnx_model,
            name="resnet50",
            version="1.1.0",
            stage="staging",
            tags=["vision", "classification"]
        )
        self.registry.register(
            self.pytorch_model,
            name="bert",
            version="2.0.0",
            stage="production",
            tags=["nlp", "transformer"]
        )
        
        # List all models
        all_models = self.registry.list_models()
        self.assertEqual(len(all_models), 3)
        
        # Filter by name
        resnet_models = self.registry.list_models(name_filter="resnet")
        self.assertEqual(len(resnet_models), 2)
        
        # Filter by stage
        prod_models = self.registry.list_models(stage_filter="production")
        self.assertEqual(len(prod_models), 1)
        self.assertEqual(prod_models[0].name, "bert")
        
        # Filter by tag
        vision_models = self.registry.list_models(tag_filter="vision")
        self.assertEqual(len(vision_models), 2)
    
    def test_model_promotion(self):
        """Test promoting models between lifecycle stages."""
        # Register model
        model = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0",
            stage="dev"
        )
        
        # Promote to staging
        promoted = self.registry.promote("test_model", "1.0.0", "staging")
        self.assertEqual(promoted.stage, "staging")
        self.assertIsNotNone(promoted.promoted_at)
        
        # Promote to production
        promoted = self.registry.promote("test_model", "1.0.0", "production")
        self.assertEqual(promoted.stage, "production")
        
        # Verify in database
        model = self.registry.get_model("test_model", "1.0.0")
        self.assertEqual(model.stage, "production")
    
    def test_model_rollback(self):
        """Test rolling back to previous model version."""
        # Register multiple versions
        v1 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0",
            stage="production"
        )
        v2 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.1.0",
            stage="production"
        )
        
        # Rollback to v1.0.0
        rolled_back = self.registry.rollback("test_model", "1.0.0")
        self.assertEqual(rolled_back.version, "1.0.0")
        self.assertEqual(rolled_back.stage, "production")
        
        # Check that v1.1.0 was archived
        v2_updated = self.registry.get_model("test_model", "1.1.0")
        self.assertEqual(v2_updated.stage, "archived")
    
    def test_model_validation(self):
        """Test model file validation."""
        # Valid model
        results = self.registry.validate(self.onnx_model)
        self.assertTrue(results["valid"])
        self.assertEqual(results["info"]["framework"], "onnx")
        self.assertIn("file_hash", results["info"])
        self.assertGreater(results["info"]["file_size"], 0)
        
        # Non-existent file
        results = self.registry.validate(Path("/non/existent/file.onnx"))
        self.assertFalse(results["valid"])
        self.assertIn("File not found", results["errors"][0])
        
        # Empty file
        empty_file = self.test_models_dir / "empty.onnx"
        empty_file.write_text("")
        results = self.registry.validate(empty_file)
        self.assertFalse(results["valid"])
        self.assertIn("File is empty", results["errors"][0])
    
    def test_model_comparison(self):
        """Test comparing two model versions."""
        # Register two versions with different metadata
        v1 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0",
            stage="production"
        )
        
        # Modify file for different hash
        self.onnx_model.write_bytes(b"MODIFIED_ONNX_DATA")
        v2 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.1.0",
            stage="dev"
        )
        
        # Compare versions
        comparison = self.registry.compare("test_model", "1.0.0", "1.1.0")
        
        self.assertEqual(comparison["name"], "test_model")
        self.assertIn("1.0.0", comparison["versions"])
        self.assertIn("1.1.0", comparison["versions"])
        
        # Check differences
        self.assertIn("stage", comparison["differences"])
        self.assertIn("file_hash", comparison["differences"])
        self.assertEqual(
            comparison["differences"]["stage"]["1.0.0"],
            "production"
        )
        self.assertEqual(
            comparison["differences"]["stage"]["1.1.0"],
            "dev"
        )
    
    def test_update_metrics(self):
        """Test updating model metrics."""
        # Register model
        model = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0"
        )
        
        # Update metrics
        metrics = {
            "accuracy": 0.95,
            "latency_ms": 12.5,
            "throughput_rps": 80
        }
        self.registry.update_metrics("test_model", "1.0.0", metrics)
        
        # Verify metrics were saved
        model = self.registry.get_model("test_model", "1.0.0")
        self.assertIsNotNone(model.metrics)
        saved_metrics = json.loads(model.metrics)
        self.assertEqual(saved_metrics["accuracy"], 0.95)
        self.assertEqual(saved_metrics["latency_ms"], 12.5)
    
    def test_export_registry(self):
        """Test exporting registry to JSON."""
        # Register multiple models
        self.registry.register(
            self.onnx_model,
            name="resnet50",
            version="1.0.0",
            stage="production"
        )
        self.registry.register(
            self.onnx_model,
            name="resnet50",
            version="1.1.0",
            stage="dev"
        )
        self.registry.register(
            self.pytorch_model,
            name="bert",
            version="2.0.0",
            stage="staging"
        )
        
        # Export to dictionary
        export_data = self.registry.export_registry()
        
        self.assertEqual(export_data["version"], "1.0.0")
        self.assertEqual(export_data["model_count"], 3)
        self.assertIn("resnet50", export_data["models"])
        self.assertIn("bert", export_data["models"])
        self.assertEqual(len(export_data["models"]["resnet50"]), 2)
        self.assertEqual(len(export_data["models"]["bert"]), 1)
        
        # Export to file
        export_file = Path(self.test_dir) / "export.json"
        self.registry.export_registry(export_file)
        self.assertTrue(export_file.exists())
        
        with open(export_file) as f:
            file_data = json.load(f)
        self.assertEqual(file_data["model_count"], 3)
    
    def test_parent_version_tracking(self):
        """Test model lineage tracking with parent versions."""
        # Register base version
        v1 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0"
        )
        
        # Register derived version
        v2 = self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.1.0",
            parent_version="1.0.0"
        )
        
        self.assertEqual(v2.parent_version, "1.0.0")
        
        # Verify in database
        model = self.registry.get_model("test_model", "1.1.0")
        self.assertEqual(model.parent_version, "1.0.0")
    
    def test_file_hash_computation(self):
        """Test SHA256 hash computation for files."""
        # Create test file with known content
        test_file = self.test_models_dir / "test.bin"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Compute hash
        computed_hash = self.registry._compute_file_hash(test_file)
        
        # Expected SHA256 hash of "Hello, World!"
        expected_hash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        self.assertEqual(computed_hash, expected_hash)
    
    def test_invalid_stage_promotion(self):
        """Test error handling for invalid stage promotion."""
        # Register model
        self.registry.register(
            self.onnx_model,
            name="test_model",
            version="1.0.0",
            stage="dev"
        )
        
        # Try to promote to invalid stage
        with self.assertRaises(ValueError) as context:
            self.registry.promote("test_model", "1.0.0", "invalid_stage")
        self.assertIn("Invalid stage", str(context.exception))
    
    def test_model_not_found_errors(self):
        """Test error handling for non-existent models."""
        # Try to promote non-existent model
        with self.assertRaises(ValueError) as context:
            self.registry.promote("non_existent", "1.0.0", "production")
        self.assertIn("not found", str(context.exception))
        
        # Try to rollback to non-existent version
        with self.assertRaises(ValueError) as context:
            self.registry.rollback("non_existent", "1.0.0")
        self.assertIn("not found", str(context.exception))
        
        # Try to compare non-existent models
        with self.assertRaises(ValueError) as context:
            self.registry.compare("non_existent", "1.0.0", "1.1.0")
        self.assertIn("not found", str(context.exception))


class TestModelManagerCLI(unittest.TestCase):
    """Test suite for model_manager.py CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.model_file = Path(self.test_dir) / "model.onnx"
        self.model_file.write_bytes(b"TEST_MODEL_DATA")
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
        # Clean up default registry if created
        default_registry = Path.home() / ".inference-lab"
        if default_registry.exists():
            shutil.rmtree(default_registry)
    
    def test_cli_help(self):
        """Test CLI help output."""
        from model_manager import main
        import io
        import contextlib
        
        # Capture help output using stdout redirection
        with patch('sys.argv', ['model_manager.py', '--help']):
            # Capture stdout
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                with patch('sys.exit'):
                    try:
                        main()
                    except SystemExit:
                        pass
            
            help_text = f.getvalue()
            
            # Check help contains expected text
            # The argparse help should contain the description
            self.assertIn("Version control and lifecycle management", help_text)
            self.assertIn("register", help_text)
            self.assertIn("list", help_text)
    
    def test_cli_register(self):
        """Test CLI model registration."""
        from model_manager import main
        
        registry_path = Path(self.test_dir) / "cli_test.db"
        
        with patch('sys.argv', [
            'model_manager.py',
            '--registry', str(registry_path),
            'register',
            str(self.model_file),
            '--name', 'cli_model',
            '--version', '1.0.0',
            '--stage', 'dev',
            '--description', 'CLI test model',
            '--author', 'Test Author'
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                self.assertEqual(result, 0)
                
                # Check success message
                calls = [str(call) for call in mock_print.call_args_list]
                output = ' '.join(calls)
                self.assertIn("Registered model", output)
                self.assertIn("cli_model", output)
                self.assertIn("v1.0.0", output)
    
    def test_cli_list_empty(self):
        """Test CLI list with empty registry."""
        from model_manager import main
        
        registry_path = Path(self.test_dir) / "empty.db"
        
        with patch('sys.argv', [
            'model_manager.py',
            '--registry', str(registry_path),
            'list'
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                self.assertEqual(result, 0)
                
                # Check empty message
                calls = [str(call) for call in mock_print.call_args_list]
                output = ' '.join(calls)
                self.assertIn("No models found", output)
    
    def test_cli_validate(self):
        """Test CLI model validation."""
        from model_manager import main
        
        with patch('sys.argv', [
            'model_manager.py',
            'validate',
            str(self.model_file)
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                self.assertEqual(result, 0)
                
                # Check validation output
                calls = [str(call) for call in mock_print.call_args_list]
                output = ' '.join(calls)
                self.assertIn("validation passed", output)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        from model_manager import main
        
        # Test with non-existent file
        with patch('sys.argv', [
            'model_manager.py',
            'register',
            '/non/existent/file.onnx'
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                self.assertEqual(result, 1)
                
                # Check error message
                calls = [str(call) for call in mock_print.call_args_list]
                output = ' '.join(calls)
                self.assertIn("Error", output)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestModelManagerCLI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())