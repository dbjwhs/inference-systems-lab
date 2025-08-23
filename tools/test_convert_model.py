#!/usr/bin/env python3
"""
Test suite for convert_model.py - Comprehensive testing of model conversion functionality.

This test script validates all aspects of the model converter including:
- PyTorch to ONNX conversion
- ONNX optimization and graph simplification
- Precision conversion (FP32/FP16)
- Model validation and comparison
- Error handling and edge cases
- CLI interface functionality
"""

import json
import os
import shutil
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
    # Create mock numpy
    np = MagicMock()
    print("Warning: NumPy not available. Some tests will be skipped.", file=sys.stderr)

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from convert_model import (
    ModelConverter, ConversionConfig, Precision, parse_shape
)

# Mock PyTorch if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch module
    torch = MagicMock()
    nn = MagicMock()

# Mock ONNX if not available
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = MagicMock()


class SimpleModel(nn.Module):
    """Simple PyTorch model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.linear(x))


class TestConversionConfig(unittest.TestCase):
    """Test suite for ConversionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()
        
        self.assertIsNone(config.input_shapes)
        self.assertIsNone(config.input_names)
        self.assertIsNone(config.output_names)
        self.assertIsNone(config.dynamic_axes)
        self.assertEqual(config.opset_version, 13)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.max_batch_size, 1)
        self.assertEqual(config.precision, Precision.FP32)
        self.assertTrue(config.optimize)
        self.assertFalse(config.simplify)
        self.assertTrue(config.fold_constants)
        self.assertFalse(config.verbose)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            input_shapes=[[1, 3, 224, 224]],
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            batch_size=4,
            precision=Precision.FP16,
            verbose=True
        )
        
        self.assertEqual(config.input_shapes, [[1, 3, 224, 224]])
        self.assertEqual(config.input_names, ["input"])
        self.assertEqual(config.output_names, ["output"])
        self.assertEqual(config.opset_version, 14)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.precision, Precision.FP16)
        self.assertTrue(config.verbose)


class TestPrecisionEnum(unittest.TestCase):
    """Test suite for Precision enum."""
    
    def test_precision_values(self):
        """Test precision enum values."""
        self.assertEqual(Precision.FP32.value, "fp32")
        self.assertEqual(Precision.FP16.value, "fp16")
        self.assertEqual(Precision.INT8.value, "int8")
        self.assertEqual(Precision.BF16.value, "bf16")
    
    def test_precision_creation(self):
        """Test precision creation from string."""
        self.assertEqual(Precision("fp32"), Precision.FP32)
        self.assertEqual(Precision("fp16"), Precision.FP16)
        self.assertEqual(Precision("int8"), Precision.INT8)


class TestParseShape(unittest.TestCase):
    """Test suite for shape parsing utility."""
    
    def test_parse_shape_with_brackets(self):
        """Test parsing shape with brackets."""
        shape = parse_shape("[1,3,224,224]")
        self.assertEqual(shape, [1, 3, 224, 224])
    
    def test_parse_shape_without_brackets(self):
        """Test parsing shape without brackets."""
        shape = parse_shape("1,3,224,224")
        self.assertEqual(shape, [1, 3, 224, 224])
    
    def test_parse_shape_with_spaces(self):
        """Test parsing shape with spaces."""
        shape = parse_shape("[ 1 , 3 , 224 , 224 ]")
        self.assertEqual(shape, [1, 3, 224, 224])
    
    def test_parse_shape_single_dimension(self):
        """Test parsing single dimension shape."""
        shape = parse_shape("10")
        self.assertEqual(shape, [10])
    
    def test_parse_shape_empty(self):
        """Test parsing empty shape."""
        with self.assertRaises(ValueError):
            parse_shape("")


class TestModelConverter(unittest.TestCase):
    """Test suite for ModelConverter class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config = ConversionConfig(verbose=False)  # Disable verbose for tests
        self.converter = ModelConverter(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        # Test with config
        converter = ModelConverter(self.config)
        self.assertEqual(converter.config, self.config)
        
        # Test without config (default)
        converter = ModelConverter()
        self.assertIsInstance(converter.config, ConversionConfig)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_pytorch_to_onnx_simple(self):
        """Test simple PyTorch to ONNX conversion."""
        # Create simple model
        model = SimpleModel()
        
        # Save PyTorch model
        pytorch_path = Path(self.test_dir) / "model.pt"
        torch.save(model.state_dict(), pytorch_path)
        
        # Set up conversion config
        self.config.input_shapes = [[1, 10]]
        self.config.input_names = ["input"]
        self.config.output_names = ["output"]
        
        # Convert to ONNX
        onnx_path = Path(self.test_dir) / "model.onnx"
        
        with patch('torch.onnx.export') as mock_export:
            with patch('onnx.load') as mock_load:
                with patch('onnx.checker.check_model') as mock_check:
                    mock_load.return_value = MagicMock()
                    
                    success = self.converter.pytorch_to_onnx(
                        pytorch_path, onnx_path, SimpleModel
                    )
                    
                    self.assertTrue(success)
                    mock_export.assert_called_once()
                    mock_check.assert_called_once()
    
    def test_pytorch_to_onnx_missing_requirements(self):
        """Test PyTorch to ONNX conversion with missing requirements."""
        pytorch_path = Path(self.test_dir) / "model.pt"
        onnx_path = Path(self.test_dir) / "model.onnx"
        
        # Test missing PyTorch installation
        with self.assertRaises(RuntimeError):
            self.converter.pytorch_to_onnx(pytorch_path, onnx_path)
    
    def test_onnx_to_tensorrt_not_available(self):
        """Test ONNX to TensorRT conversion when TensorRT not available."""
        onnx_path = Path(self.test_dir) / "model.onnx"
        engine_path = Path(self.test_dir) / "model.engine"
        
        # This should fail gracefully since TensorRT is likely not installed
        success = self.converter.onnx_to_tensorrt(onnx_path, engine_path)
        self.assertFalse(success)
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not available")
    def test_optimize_onnx(self):
        """Test ONNX model optimization."""
        # Create dummy ONNX model file
        onnx_input = Path(self.test_dir) / "input.onnx"
        onnx_output = Path(self.test_dir) / "output.onnx"
        
        # Create minimal ONNX model for testing
        with patch('onnx.load') as mock_load:
            with patch('onnx.save') as mock_save:
                with patch('onnx.checker.check_model') as mock_check:
                    with patch('onnx.shape_inference.infer_shapes') as mock_infer:
                        mock_model = MagicMock()
                        mock_load.return_value = mock_model
                        mock_infer.return_value = mock_model
                        
                        # Create a dummy file
                        onnx_input.write_text("dummy onnx content")
                        
                        success = self.converter.optimize_onnx(onnx_input, onnx_output)
                        
                        self.assertTrue(success)
                        mock_load.assert_called_once()
                        mock_save.assert_called_once()
                        mock_check.assert_called_once()
    
    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not available")
    def test_convert_precision_fp16(self):
        """Test precision conversion to FP16."""
        input_path = Path(self.test_dir) / "input.onnx"
        output_path = Path(self.test_dir) / "output.onnx"
        
        with patch('onnx.load') as mock_load:
            with patch('onnx.save') as mock_save:
                with patch('onnx.checker.check_model') as mock_check:
                    # Create mock model with graph structure
                    mock_model = MagicMock()
                    mock_model.graph.initializer = []
                    mock_model.graph.value_info = []
                    mock_model.graph.input = []
                    mock_model.graph.output = []
                    
                    mock_load.return_value = mock_model
                    
                    # Create dummy file
                    input_path.write_text("dummy onnx content")
                    
                    success = self.converter.convert_precision(
                        input_path, output_path, Precision.FP16
                    )
                    
                    self.assertTrue(success)
                    mock_load.assert_called_once()
                    mock_save.assert_called_once()
                    mock_check.assert_called_once()
    
    def test_validate_conversion_no_onnxruntime(self):
        """Test validation when ONNX Runtime not available."""
        model1_path = Path(self.test_dir) / "model1.onnx"
        model2_path = Path(self.test_dir) / "model2.onnx"
        
        # This should warn and return False since ONNX Runtime likely not available
        success = self.converter.validate_conversion(model1_path, model2_path)
        self.assertFalse(success)


class TestModelConverterCLI(unittest.TestCase):
    """Test suite for convert_model.py CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_cli_help(self):
        """Test CLI help output."""
        from convert_model import main
        import io
        import contextlib
        
        # Capture help output
        with patch('sys.argv', ['convert_model.py', '--help']):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                with patch('sys.exit'):
                    try:
                        main()
                    except SystemExit:
                        pass
            
            help_text = f.getvalue()
            
            # Check help contains expected commands
            self.assertIn("pytorch-to-onnx", help_text)
            self.assertIn("onnx-to-tensorrt", help_text)
            self.assertIn("optimize-onnx", help_text)
            self.assertIn("convert-precision", help_text)
            self.assertIn("validate", help_text)
    
    def test_cli_pytorch_to_onnx_args(self):
        """Test PyTorch to ONNX CLI arguments parsing."""
        from convert_model import main
        
        input_file = Path(self.test_dir) / "model.pt"
        output_file = Path(self.test_dir) / "model.onnx"
        
        # Create dummy input file
        input_file.write_bytes(b"dummy pytorch model")
        
        with patch('sys.argv', [
            'convert_model.py',
            'pytorch-to-onnx',
            str(input_file),
            str(output_file),
            '--input-shapes', '[1,3,224,224]',
            '--input-names', 'input',
            '--output-names', 'output',
            '--opset', '14',
            '--dynamic-batch',
            '--simplify',
            '--verbose'
        ]):
            with patch('convert_model.ModelConverter') as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter.pytorch_to_onnx.return_value = True
                mock_converter_class.return_value = mock_converter
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_converter.pytorch_to_onnx.assert_called_once()
    
    def test_cli_onnx_to_tensorrt_args(self):
        """Test ONNX to TensorRT CLI arguments parsing."""
        from convert_model import main
        
        input_file = Path(self.test_dir) / "model.onnx"
        output_file = Path(self.test_dir) / "model.engine"
        
        # Create dummy input file
        input_file.write_bytes(b"dummy onnx model")
        
        with patch('sys.argv', [
            'convert_model.py',
            'onnx-to-tensorrt',
            str(input_file),
            str(output_file),
            '--precision', 'fp16',
            '--batch-size', '4',
            '--max-batch-size', '32',
            '--verbose'
        ]):
            with patch('convert_model.ModelConverter') as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter.onnx_to_tensorrt.return_value = True
                mock_converter_class.return_value = mock_converter
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_converter.onnx_to_tensorrt.assert_called_once()
    
    def test_cli_optimize_onnx_args(self):
        """Test ONNX optimization CLI arguments parsing."""
        from convert_model import main
        
        input_file = Path(self.test_dir) / "input.onnx"
        output_file = Path(self.test_dir) / "output.onnx"
        
        # Create dummy input file
        input_file.write_bytes(b"dummy onnx model")
        
        with patch('sys.argv', [
            'convert_model.py',
            'optimize-onnx',
            str(input_file),
            str(output_file),
            '--simplify',
            '--fold-constants',
            '--verbose'
        ]):
            with patch('convert_model.ModelConverter') as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter.optimize_onnx.return_value = True
                mock_converter_class.return_value = mock_converter
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_converter.optimize_onnx.assert_called_once()
    
    def test_cli_convert_precision_args(self):
        """Test precision conversion CLI arguments parsing."""
        from convert_model import main
        
        input_file = Path(self.test_dir) / "input.onnx"
        output_file = Path(self.test_dir) / "output.onnx"
        
        # Create dummy input file
        input_file.write_bytes(b"dummy onnx model")
        
        with patch('sys.argv', [
            'convert_model.py',
            'convert-precision',
            str(input_file),
            str(output_file),
            '--from', 'fp32',
            '--to', 'fp16',
            '--verbose'
        ]):
            with patch('convert_model.ModelConverter') as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter.convert_precision.return_value = True
                mock_converter_class.return_value = mock_converter
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_converter.convert_precision.assert_called_once()
    
    def test_cli_validate_args(self):
        """Test validation CLI arguments parsing."""
        from convert_model import main
        
        model1_file = Path(self.test_dir) / "model1.onnx"
        model2_file = Path(self.test_dir) / "model2.onnx"
        test_input_file = Path(self.test_dir) / "test_input.npy"
        
        # Create dummy files
        model1_file.write_bytes(b"dummy onnx model 1")
        model2_file.write_bytes(b"dummy onnx model 2")
        
        if NUMPY_AVAILABLE:
            np.save(test_input_file, np.random.randn(1, 3, 224, 224))
        else:
            # Create dummy numpy file
            test_input_file.write_bytes(b"dummy numpy data")
        
        with patch('sys.argv', [
            'convert_model.py',
            'validate',
            str(model1_file),
            str(model2_file),
            '--tolerance', '1e-4',
            '--test-input', str(test_input_file),
            '--verbose'
        ]):
            with patch('convert_model.ModelConverter') as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter.validate_conversion.return_value = True
                mock_converter_class.return_value = mock_converter
                
                result = main()
                
                self.assertEqual(result, 0)
                mock_converter.validate_conversion.assert_called_once()
    
    def test_cli_no_command(self):
        """Test CLI with no command."""
        from convert_model import main
        import io
        import contextlib
        
        with patch('sys.argv', ['convert_model.py']):
            # Capture stdout to see if help is printed
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = main()
            
            help_output = f.getvalue()
            self.assertEqual(result, 1)
            # Should contain help text
            self.assertIn("Conversion commands", help_output)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        from convert_model import main
        
        with patch('sys.argv', [
            'convert_model.py',
            'pytorch-to-onnx',
            '/non/existent/file.pt',
            '/tmp/output.onnx'
        ]):
            with patch('builtins.print') as mock_print:
                result = main()
                
                self.assertEqual(result, 1)
                # Should print error message
                calls = [str(call) for call in mock_print.call_args_list]
                error_output = ' '.join(calls)
                self.assertIn("Error", error_output)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common conversion scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_full_pytorch_to_onnx_pipeline(self):
        """Test complete PyTorch to ONNX conversion pipeline."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        # Create model
        model = SimpleModel()
        
        # Save model
        pytorch_path = Path(self.test_dir) / "model.pt"
        torch.save(model.state_dict(), pytorch_path)
        
        # Configure converter
        config = ConversionConfig(
            input_shapes=[[1, 10]],
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            simplify=True,
            verbose=False
        )
        
        converter = ModelConverter(config)
        
        # Mock ONNX operations for testing
        with patch('torch.onnx.export') as mock_export:
            with patch('onnx.load') as mock_load:
                with patch('onnx.checker.check_model') as mock_check:
                    mock_load.return_value = MagicMock()
                    
                    onnx_path = Path(self.test_dir) / "model.onnx"
                    success = converter.pytorch_to_onnx(
                        pytorch_path, onnx_path, SimpleModel
                    )
                    
                    self.assertTrue(success)
    
    def test_model_optimization_workflow(self):
        """Test model optimization workflow."""
        if not ONNX_AVAILABLE:
            self.skipTest("ONNX not available")
        
        # Create dummy ONNX files
        input_path = Path(self.test_dir) / "original.onnx"
        optimized_path = Path(self.test_dir) / "optimized.onnx"
        fp16_path = Path(self.test_dir) / "fp16.onnx"
        
        input_path.write_text("dummy onnx model")
        
        config = ConversionConfig(
            simplify=True,
            fold_constants=True,
            verbose=False
        )
        
        converter = ModelConverter(config)
        
        # Mock ONNX operations
        with patch('onnx.load') as mock_load:
            with patch('onnx.save') as mock_save:
                with patch('onnx.checker.check_model') as mock_check:
                    with patch('onnx.shape_inference.infer_shapes') as mock_infer:
                        mock_model = MagicMock()
                        mock_model.graph.initializer = []
                        mock_model.graph.value_info = []
                        mock_model.graph.input = []
                        mock_model.graph.output = []
                        
                        mock_load.return_value = mock_model
                        mock_infer.return_value = mock_model
                        
                        # Test optimization
                        success1 = converter.optimize_onnx(input_path, optimized_path)
                        self.assertTrue(success1)
                        
                        # Test precision conversion
                        success2 = converter.convert_precision(
                            optimized_path, fp16_path, Precision.FP16
                        )
                        self.assertTrue(success2)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConversionConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPrecisionEnum))
    suite.addTests(loader.loadTestsFromTestCase(TestParseShape))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConverterCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
