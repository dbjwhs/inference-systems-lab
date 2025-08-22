#!/usr/bin/env python3
"""
Test suite for validate_model.py - Comprehensive testing of model validation functionality.
This test script validates all aspects of the model validator including:
- File existence and integrity checking
- Model structure validation (ONNX, PyTorch)
- Numerical accuracy verification
- Determinism testing
- Edge case handling
- Cross-platform consistency checks
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Try to import numpy (optional dependency)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some tests will be skipped.", file=sys.stderr)

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from validate_model import (
    ModelValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationResult,
    ValidationReport,
    ValidationMetrics,
    main
)


class TestValidationConfig(unittest.TestCase):
    """Test ValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        self.assertEqual(config.level, ValidationLevel.STANDARD)
        self.assertEqual(config.tolerance, 1e-5)
        self.assertEqual(config.relative_tolerance, 1e-3)
        self.assertEqual(config.max_samples, 1000)
        self.assertEqual(config.random_seed, 42)
        self.assertTrue(config.check_determinism)
        self.assertTrue(config.check_numerical_stability)
        self.assertTrue(config.check_edge_cases)
        self.assertFalse(config.check_performance)
        self.assertFalse(config.save_outputs)
        self.assertIsNone(config.output_dir)
        self.assertFalse(config.verbose)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            level=ValidationLevel.STRICT,
            tolerance=1e-6,
            relative_tolerance=1e-4,
            max_samples=500,
            random_seed=123,
            check_determinism=False,
            verbose=True
        )
        
        self.assertEqual(config.level, ValidationLevel.STRICT)
        self.assertEqual(config.tolerance, 1e-6)
        self.assertEqual(config.relative_tolerance, 1e-4)
        self.assertEqual(config.max_samples, 500)
        self.assertEqual(config.random_seed, 123)
        self.assertFalse(config.check_determinism)
        self.assertTrue(config.verbose)


class TestValidationLevel(unittest.TestCase):
    """Test ValidationLevel enum."""
    
    def test_level_values(self):
        """Test validation level enum values."""
        self.assertEqual(ValidationLevel.BASIC.value, "basic")
        self.assertEqual(ValidationLevel.STANDARD.value, "standard")
        self.assertEqual(ValidationLevel.STRICT.value, "strict")
        self.assertEqual(ValidationLevel.EXHAUSTIVE.value, "exhaustive")


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult enum."""
    
    def test_result_values(self):
        """Test validation result enum values."""
        self.assertEqual(ValidationResult.PASS.value, "pass")
        self.assertEqual(ValidationResult.FAIL.value, "fail")
        self.assertEqual(ValidationResult.WARNING.value, "warning")
        self.assertEqual(ValidationResult.SKIP.value, "skip")
        self.assertEqual(ValidationResult.ERROR.value, "error")


class TestValidationMetrics(unittest.TestCase):
    """Test ValidationMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = ValidationMetrics(
            max_absolute_error=1e-5,
            mean_absolute_error=1e-6,
            max_relative_error=1e-4,
            mean_relative_error=1e-5,
            correlation=0.99,
            pass_rate=0.95,
            execution_time_ms=100.0
        )
        
        self.assertEqual(metrics.max_absolute_error, 1e-5)
        self.assertEqual(metrics.mean_absolute_error, 1e-6)
        self.assertEqual(metrics.max_relative_error, 1e-4)
        self.assertEqual(metrics.mean_relative_error, 1e-5)
        self.assertEqual(metrics.correlation, 0.99)
        self.assertEqual(metrics.pass_rate, 0.95)
        self.assertEqual(metrics.execution_time_ms, 100.0)


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport dataclass."""
    
    def test_report_creation(self):
        """Test validation report creation."""
        model_path = Path("test_model.onnx")
        report = ValidationReport(
            model_path=model_path,
            validation_level=ValidationLevel.STANDARD,
            timestamp="2024-01-01 12:00:00",
            platform_info={"platform": "test"},
            overall_result=ValidationResult.PASS
        )
        
        self.assertEqual(report.model_path, model_path)
        self.assertEqual(report.validation_level, ValidationLevel.STANDARD)
        self.assertEqual(report.timestamp, "2024-01-01 12:00:00")
        self.assertEqual(report.platform_info, {"platform": "test"})
        self.assertEqual(report.overall_result, ValidationResult.PASS)
        self.assertEqual(len(report.test_results), 0)
        self.assertIsNone(report.metrics)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(len(report.warnings), 0)


class TestModelValidator(unittest.TestCase):
    """Test ModelValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = ValidationConfig(verbose=True)
        self.validator = ModelValidator(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.config, self.config)
        self.assertEqual(len(self.validator.reference_outputs), 0)
    
    def test_get_platform_info(self):
        """Test platform info gathering."""
        platform_info = self.validator._get_platform_info()
        
        self.assertIn("platform", platform_info)
        self.assertIn("python_version", platform_info)
        self.assertIn("numpy_available", platform_info)
        self.assertIn("pytorch_available", platform_info)
        self.assertIn("onnxruntime_available", platform_info)
        self.assertIn("tensorrt_available", platform_info)
    
    def test_check_file_exists_missing(self):
        """Test file existence check with missing file."""
        missing_file = Path(self.test_dir) / "missing_model.onnx"
        report = ValidationReport(
            model_path=missing_file,
            validation_level=ValidationLevel.BASIC,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        self.validator._check_file_exists(missing_file, report)
        
        self.assertIn("file_exists", report.test_results)
        result, message = report.test_results["file_exists"]
        self.assertEqual(result, ValidationResult.FAIL)
        self.assertIn("does not exist", message)
    
    def test_check_file_exists_success(self):
        """Test file existence check with existing file."""
        test_file = Path(self.test_dir) / "test_model.onnx"
        test_file.write_bytes(b"dummy model content")
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.BASIC,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        self.validator._check_file_exists(test_file, report)
        
        self.assertIn("file_exists", report.test_results)
        result, message = report.test_results["file_exists"]
        self.assertEqual(result, ValidationResult.PASS)
        self.assertIn("exists and is readable", message)
    
    def test_check_file_integrity(self):
        """Test file integrity check."""
        test_file = Path(self.test_dir) / "test_model.onnx"
        test_content = b"dummy model content"
        test_file.write_bytes(test_content)
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.BASIC,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        self.validator._check_file_integrity(test_file, report)
        
        self.assertIn("file_integrity", report.test_results)
        result, message = report.test_results["file_integrity"]
        self.assertEqual(result, ValidationResult.PASS)
        self.assertIn("integrity check passed", message)
    
    def test_check_file_integrity_empty(self):
        """Test file integrity check with empty file."""
        test_file = Path(self.test_dir) / "empty_model.onnx"
        test_file.write_bytes(b"")
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.BASIC,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        self.validator._check_file_integrity(test_file, report)
        
        self.assertIn("file_integrity", report.test_results)
        result, message = report.test_results["file_integrity"]
        self.assertEqual(result, ValidationResult.FAIL)
        self.assertIn("empty", message)
    
    def test_validate_model_structure_unsupported(self):
        """Test model structure validation with unsupported format."""
        test_file = Path(self.test_dir) / "test_model.unknown"
        test_file.write_bytes(b"dummy content")
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        self.validator._validate_model_structure(test_file, report)
        
        self.assertIn("model_structure", report.test_results)
        result, message = report.test_results["model_structure"]
        self.assertEqual(result, ValidationResult.SKIP)
        self.assertIn("No validator available", message)
    
    def test_validate_onnx_structure_not_available(self):
        """Test ONNX structure validation without ONNX Runtime."""
        test_file = Path(self.test_dir) / "test_model.onnx"
        test_file.write_bytes(b"dummy onnx content")
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        with patch('validate_model.ONNXRUNTIME_AVAILABLE', False):
            self.validator._validate_onnx_structure(test_file, report)
        
        self.assertIn("onnx_structure", report.test_results)
        result, message = report.test_results["onnx_structure"]
        self.assertEqual(result, ValidationResult.SKIP)
        self.assertIn("ONNX Runtime not available", message)
    
    def test_validate_pytorch_structure_not_available(self):
        """Test PyTorch structure validation without PyTorch."""
        test_file = Path(self.test_dir) / "test_model.pt"
        test_file.write_bytes(b"dummy pytorch content")
        
        report = ValidationReport(
            model_path=test_file,
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        with patch('validate_model.PYTORCH_AVAILABLE', False):
            self.validator._validate_pytorch_structure(test_file, report)
        
        self.assertIn("pytorch_structure", report.test_results)
        result, message = report.test_results["pytorch_structure"]
        self.assertEqual(result, ValidationResult.SKIP)
        self.assertIn("PyTorch not available", message)
    
    def test_get_model_input_shape_default(self):
        """Test getting model input shape with default fallback."""
        test_file = Path(self.test_dir) / "test_model.unknown"
        test_file.write_bytes(b"dummy content")
        
        shape = self.validator._get_model_input_shape(test_file)
        self.assertEqual(shape, [1, 3, 224, 224])
    
    def test_run_inference_unsupported(self):
        """Test inference with unsupported model format."""
        test_file = Path(self.test_dir) / "test_model.unknown"
        test_file.write_bytes(b"dummy content")
        
        inputs = {"input": "test_input"}
        outputs = self.validator._run_inference(test_file, inputs)
        
        self.assertIn("output", outputs)
        self.assertEqual(outputs["output"], "dummy_output")
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_compare_outputs(self):
        """Test output comparison with numerical data."""
        outputs1 = {"output": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"output": np.array([1.001, 2.001, 3.001])}
        
        metrics = self.validator._compare_outputs(outputs1, outputs2)
        
        self.assertGreater(metrics.max_absolute_error, 0)
        self.assertGreater(metrics.mean_absolute_error, 0)
        self.assertLess(metrics.max_absolute_error, 0.002)
    
    def test_compare_outputs_no_numpy(self):
        """Test output comparison without NumPy."""
        outputs1 = {"output": [1.0, 2.0, 3.0]}
        outputs2 = {"output": [1.001, 2.001, 3.001]}
        
        with patch('validate_model.NUMPY_AVAILABLE', False):
            metrics = self.validator._compare_outputs(outputs1, outputs2)
        
        self.assertEqual(metrics.max_absolute_error, 0.0)
        self.assertEqual(metrics.mean_absolute_error, 0.0)
        self.assertEqual(metrics.correlation, 1.0)
    
    def test_outputs_equal_identical(self):
        """Test output equality with identical outputs."""
        outputs1 = {"output": "test_output"}
        outputs2 = {"output": "test_output"}
        
        self.assertTrue(self.validator._outputs_equal(outputs1, outputs2))
    
    def test_outputs_equal_different(self):
        """Test output equality with different outputs."""
        outputs1 = {"output": "test_output1"}
        outputs2 = {"output": "test_output2"}
        
        self.assertFalse(self.validator._outputs_equal(outputs1, outputs2))
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_is_valid_output_valid(self):
        """Test valid output detection."""
        output = {"result": np.array([1.0, 2.0, 3.0])}
        self.assertTrue(self.validator._is_valid_output(output))
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_is_valid_output_nan(self):
        """Test invalid output detection with NaN."""
        output = {"result": np.array([1.0, np.nan, 3.0])}
        self.assertFalse(self.validator._is_valid_output(output))
    
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy not available")
    def test_is_valid_output_inf(self):
        """Test invalid output detection with infinity."""
        output = {"result": np.array([1.0, np.inf, 3.0])}
        self.assertFalse(self.validator._is_valid_output(output))
    
    def test_calculate_overall_result_pass(self):
        """Test overall result calculation with all passes."""
        report = ValidationReport(
            model_path=Path("test.onnx"),
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        report.test_results = {
            "test1": (ValidationResult.PASS, "Test 1 passed"),
            "test2": (ValidationResult.PASS, "Test 2 passed")
        }
        
        self.validator._calculate_overall_result(report)
        self.assertEqual(report.overall_result, ValidationResult.PASS)
    
    def test_calculate_overall_result_fail(self):
        """Test overall result calculation with failures."""
        report = ValidationReport(
            model_path=Path("test.onnx"),
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        report.test_results = {
            "test1": (ValidationResult.PASS, "Test 1 passed"),
            "test2": (ValidationResult.FAIL, "Test 2 failed")
        }
        
        self.validator._calculate_overall_result(report)
        self.assertEqual(report.overall_result, ValidationResult.FAIL)
    
    def test_calculate_overall_result_error(self):
        """Test overall result calculation with errors."""
        report = ValidationReport(
            model_path=Path("test.onnx"),
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        report.test_results = {
            "test1": (ValidationResult.PASS, "Test 1 passed"),
            "test2": (ValidationResult.ERROR, "Test 2 error")
        }
        
        self.validator._calculate_overall_result(report)
        self.assertEqual(report.overall_result, ValidationResult.ERROR)
    
    def test_calculate_overall_result_warning(self):
        """Test overall result calculation with warnings."""
        report = ValidationReport(
            model_path=Path("test.onnx"),
            validation_level=ValidationLevel.STANDARD,
            timestamp="test",
            platform_info={},
            overall_result=ValidationResult.PASS
        )
        
        report.test_results = {
            "test1": (ValidationResult.PASS, "Test 1 passed"),
            "test2": (ValidationResult.WARNING, "Test 2 warning")
        }
        
        self.validator._calculate_overall_result(report)
        self.assertEqual(report.overall_result, ValidationResult.WARNING)
    
    def test_save_report(self):
        """Test saving validation report to file."""
        report = ValidationReport(
            model_path=Path("test.onnx"),
            validation_level=ValidationLevel.STANDARD,
            timestamp="2024-01-01 12:00:00",
            platform_info={"platform": "test"},
            overall_result=ValidationResult.PASS
        )
        
        report.test_results = {
            "test1": (ValidationResult.PASS, "Test passed")
        }
        
        report.metrics = ValidationMetrics(
            max_absolute_error=1e-5,
            mean_absolute_error=1e-6,
            max_relative_error=1e-4,
            mean_relative_error=1e-5,
            correlation=0.99,
            pass_rate=1.0,
            execution_time_ms=100.0
        )
        
        output_file = Path(self.test_dir) / "report.json"
        self.validator.save_report(report, output_file)
        
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["model_path"], "test.onnx")
        self.assertEqual(saved_data["validation_level"], "standard")
        self.assertEqual(saved_data["overall_result"], "pass")
        self.assertIn("test1", saved_data["test_results"])
    
    def test_validate_model_basic(self):
        """Test basic model validation."""
        test_file = Path(self.test_dir) / "test_model.onnx"
        test_file.write_bytes(b"dummy onnx content")
        
        config = ValidationConfig(level=ValidationLevel.BASIC)
        validator = ModelValidator(config)
        
        report = validator.validate_model(test_file)
        
        self.assertEqual(report.model_path, test_file)
        self.assertEqual(report.validation_level, ValidationLevel.BASIC)
        self.assertIn("file_exists", report.test_results)
        self.assertIn("file_integrity", report.test_results)


class TestCLI(unittest.TestCase):
    """Test command-line interface."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help output."""
        from validate_model import main
        import io
        import contextlib
        
        # Capture help output using stdout redirection
        with patch('sys.argv', ['validate_model.py', '--help']):
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
            self.assertIn("Model Validator", help_text)
            self.assertIn("validate", help_text)
            self.assertIn("batch-validate", help_text)
    
    def test_cli_no_command(self):
        """Test CLI with no command."""
        from validate_model import main
        import io
        import contextlib
        
        with patch('sys.argv', ['validate_model.py']):
            # Capture stdout to see if help is printed
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = main()
            
            help_output = f.getvalue()
            self.assertEqual(result, 1)
            # Should contain help text
            self.assertIn("Validation commands", help_output)
    
    def test_cli_validate_missing_file(self):
        """Test CLI validate with missing file."""
        from validate_model import main
        
        missing_file = Path(self.test_dir) / "missing.onnx"
        
        with patch('sys.argv', ['validate_model.py', 'validate', str(missing_file)]):
            with patch('builtins.print'):  # Suppress output
                result = main()
        
        self.assertEqual(result, 1)  # Should fail
    
    def test_cli_validate_existing_file(self):
        """Test CLI validate with existing file."""
        from validate_model import main
        
        test_file = Path(self.test_dir) / "test.onnx"
        test_file.write_bytes(b"dummy onnx content")
        
        with patch('sys.argv', ['validate_model.py', 'validate', str(test_file), '--level', 'basic']):
            with patch('builtins.print'):  # Suppress output
                result = main()
        
        self.assertEqual(result, 0)  # Should pass
    
    def test_cli_batch_validate(self):
        """Test CLI batch validation."""
        from validate_model import main
        
        # Create test models
        models_dir = Path(self.test_dir) / "models"
        models_dir.mkdir()
        
        (models_dir / "model1.onnx").write_bytes(b"dummy content 1")
        (models_dir / "model2.onnx").write_bytes(b"dummy content 2")
        
        output_dir = Path(self.test_dir) / "reports"
        
        with patch('sys.argv', [
            'validate_model.py',
            'batch-validate',
            str(models_dir),
            '--pattern', '*.onnx',
            '--level', 'basic',
            '--output', str(output_dir)
        ]):
            with patch('builtins.print'):  # Suppress output
                result = main()
        
        self.assertEqual(result, 0)
        self.assertTrue(output_dir.exists())
        
        # Check that report files were created
        report_files = list(output_dir.glob("*_validation_report.json"))
        self.assertEqual(len(report_files), 2)
    
    def test_cli_compare_models(self):
        """Test CLI model comparison."""
        from validate_model import main
        
        model1 = Path(self.test_dir) / "model1.onnx"
        model2 = Path(self.test_dir) / "model2.onnx"
        
        model1.write_bytes(b"dummy content 1")
        model2.write_bytes(b"dummy content 2")
        
        with patch('sys.argv', [
            'validate_model.py',
            'compare',
            str(model1),
            str(model2),
            '--tolerance', '1e-5'
        ]):
            with patch('builtins.print'):  # Suppress output
                result = main()
        
        # Result depends on comparison, but should not crash
        self.assertIn(result, [0, 1])
    
    def test_cli_generate_data(self):
        """Test CLI test data generation."""
        from validate_model import main
        
        test_model = Path(self.test_dir) / "model.onnx"
        test_model.write_bytes(b"dummy onnx content")
        
        output_file = Path(self.test_dir) / "test_data.pkl"
        
        with patch('sys.argv', [
            'validate_model.py',
            'generate-data',
            str(test_model),
            '--samples', '10',
            '--output', str(output_file)
        ]):
            with patch('builtins.print'):  # Suppress output
                result = main()
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        
        # Verify the generated data
        with open(output_file, 'rb') as f:
            test_data = pickle.load(f)
        
        self.assertEqual(len(test_data), 10)
        self.assertIsInstance(test_data[0], tuple)
        self.assertEqual(len(test_data[0]), 2)  # (inputs, outputs)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        # Create test model
        test_model = Path(self.test_dir) / "test_model.onnx"
        test_model.write_bytes(b"dummy onnx model content")
        
        # Configure validator
        config = ValidationConfig(
            level=ValidationLevel.STANDARD,
            tolerance=1e-5,
            verbose=False
        )
        
        validator = ModelValidator(config)
        
        # Run validation
        report = validator.validate_model(test_model)
        
        # Verify report structure
        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.model_path, test_model)
        self.assertEqual(report.validation_level, ValidationLevel.STANDARD)
        self.assertIsNotNone(report.timestamp)
        self.assertIsInstance(report.platform_info, dict)
        self.assertIn(report.overall_result, [ValidationResult.PASS, ValidationResult.WARNING, ValidationResult.FAIL])
        
        # Check that basic tests were run
        self.assertIn("file_exists", report.test_results)
        self.assertIn("file_integrity", report.test_results)
        
        # Save and load report
        report_file = Path(self.test_dir) / "validation_report.json"
        validator.save_report(report, report_file)
        
        self.assertTrue(report_file.exists())
        
        with open(report_file, 'r') as f:
            saved_report = json.load(f)
        
        self.assertEqual(saved_report["model_path"], str(test_model))
        self.assertEqual(saved_report["validation_level"], "standard")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)