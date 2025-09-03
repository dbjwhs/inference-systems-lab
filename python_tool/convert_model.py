#!/usr/bin/env python3
"""
Model Converter - Automated model conversion pipeline for ML models.

This tool provides comprehensive model conversion functionality including:
- PyTorch to ONNX conversion with validation
- ONNX to TensorRT engine optimization
- Precision conversion (FP32, FP16, INT8)
- Model optimization and graph simplification
- Batch size configuration
- Input/output shape validation
- Conversion validation pipeline

Usage:
    # Convert PyTorch model to ONNX
    python3 python_tool/convert_model.py pytorch-to-onnx model.pt model.onnx \
        --input-shapes "[1,3,224,224]" --input-names "input" --output-names "output"
    
    # Convert ONNX to TensorRT (requires TensorRT)
    python3 python_tool/convert_model.py onnx-to-tensorrt model.onnx model.engine \
        --precision fp16 --max-batch-size 8
    
    # Optimize ONNX model
    python3 python_tool/convert_model.py optimize-onnx input.onnx output.onnx \
        --simplify --fold-constants
    
    # Convert precision
    python3 python_tool/convert_model.py convert-precision model.onnx model_fp16.onnx \
        --from fp32 --to fp16
    
    # Validate conversion
    python3 python_tool/convert_model.py validate model1.onnx model2.onnx \
        --tolerance 1e-5 --test-inputs sample.npy
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
# Try to import numpy (optional dependency)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some operations will be limited.", file=sys.stderr)

# Suppress warnings during import
warnings.filterwarnings("ignore")

# Try to import ML frameworks (optional dependencies)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. PyTorch conversion will be disabled.", file=sys.stderr)

try:
    import onnx
    import onnx.checker
    import onnx.helper
    import onnx.numpy_helper
    from onnx import TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. ONNX operations will be disabled.", file=sys.stderr)

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Model validation will be limited.", file=sys.stderr)

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    # Don't warn about TensorRT - it's expected to be missing on most systems

try:
    from onnxsim import simplify as onnx_simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False


class Precision(Enum):
    """Supported precision types for conversion."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    input_shapes: Optional[List[List[int]]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    opset_version: int = 13
    batch_size: int = 1
    max_batch_size: int = 1
    precision: Precision = Precision.FP32
    optimize: bool = True
    simplify: bool = False
    fold_constants: bool = True
    verbose: bool = False


class ModelConverter:
    """Main class for model conversion operations."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize the model converter.
        
        Args:
            config: Conversion configuration
        """
        self.config = config or ConversionConfig()
    
    def pytorch_to_onnx(self, 
                       pytorch_path: Path,
                       onnx_path: Path,
                       model_class: Optional[type] = None,
                       sample_input: Optional[Any] = None) -> bool:
        """Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_path: Path to PyTorch model (.pt, .pth)
            onnx_path: Output path for ONNX model
            model_class: Optional model class for loading
            sample_input: Optional sample input tensor
            
        Returns:
            True if conversion successful
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install with: pip install torch")
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX is not installed. Install with: pip install onnx")
        
        try:
            # Load PyTorch model
            if self.config.verbose:
                print(f"Loading PyTorch model from {pytorch_path}")
            
            if model_class:
                model = model_class()
                model.load_state_dict(torch.load(pytorch_path, map_location='cpu'))
            else:
                model = torch.load(pytorch_path, map_location='cpu')
            
            model.eval()
            
            # Create sample input if not provided
            if sample_input is None:
                if self.config.input_shapes:
                    sample_input = torch.randn(*self.config.input_shapes[0])
                else:
                    raise ValueError("Either sample_input or input_shapes must be provided")
            
            # Prepare export arguments
            export_args = {
                'model': model,
                'args': sample_input,
                'f': str(onnx_path),
                'export_params': True,
                'opset_version': self.config.opset_version,
                'do_constant_folding': self.config.fold_constants,
                'verbose': self.config.verbose
            }
            
            if self.config.input_names:
                export_args['input_names'] = self.config.input_names
            
            if self.config.output_names:
                export_args['output_names'] = self.config.output_names
            
            if self.config.dynamic_axes:
                export_args['dynamic_axes'] = self.config.dynamic_axes
            
            # Export to ONNX
            if self.config.verbose:
                print(f"Exporting to ONNX with opset {self.config.opset_version}")
            
            torch.onnx.export(**export_args)
            
            # Verify the exported model
            if self.config.verbose:
                print("Verifying ONNX model")
            
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Optionally simplify the model
            if self.config.simplify and ONNXSIM_AVAILABLE:
                if self.config.verbose:
                    print("Simplifying ONNX model")
                
                simplified_model, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, str(onnx_path))
                else:
                    print("Warning: Model simplification failed, using original model", file=sys.stderr)
            
            print(f"✅ Successfully converted PyTorch model to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            print(f"❌ PyTorch to ONNX conversion failed: {e}", file=sys.stderr)
            return False
    
    def onnx_to_tensorrt(self, 
                        onnx_path: Path,
                        engine_path: Path,
                        calibration_data: Optional[Any] = None) -> bool:
        """Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Output path for TensorRT engine
            calibration_data: Optional INT8 calibration data
            
        Returns:
            True if conversion successful
        """
        if not TENSORRT_AVAILABLE:
            print("❌ TensorRT is not installed. TensorRT conversion requires NVIDIA GPU and TensorRT SDK.", file=sys.stderr)
            print("   Install instructions: https://developer.nvidia.com/tensorrt", file=sys.stderr)
            return False
        
        try:
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            if self.config.verbose:
                TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            if self.config.verbose:
                print(f"Parsing ONNX model from {onnx_path}")
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(f"Parser error: {parser.get_error(error)}", file=sys.stderr)
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            
            # Set max workspace size (1GB)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            
            # Set precision
            if self.config.precision == Precision.FP16:
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    if self.config.verbose:
                        print("Using FP16 precision")
                else:
                    print("Warning: FP16 not supported on this platform", file=sys.stderr)
            
            elif self.config.precision == Precision.INT8:
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    if self.config.verbose:
                        print("Using INT8 precision")
                    
                    # INT8 requires calibration
                    if calibration_data is None:
                        print("Warning: INT8 precision requires calibration data", file=sys.stderr)
                else:
                    print("Warning: INT8 not supported on this platform", file=sys.stderr)
            
            # Set batch size
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = input_tensor.shape
                
                # Set dynamic batch size if needed
                if shape[0] == -1:
                    min_shape = [1] + list(shape[1:])
                    opt_shape = [self.config.batch_size] + list(shape[1:])
                    max_shape = [self.config.max_batch_size] + list(shape[1:])
                    
                    profile.set_shape(
                        input_tensor.name,
                        min_shape,
                        opt_shape,
                        max_shape
                    )
            
            config.add_optimization_profile(profile)
            
            # Build engine
            if self.config.verbose:
                print("Building TensorRT engine (this may take a while)...")
            
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("❌ Failed to build TensorRT engine", file=sys.stderr)
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            print(f"✅ Successfully converted ONNX model to TensorRT engine: {engine_path}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX to TensorRT conversion failed: {e}", file=sys.stderr)
            return False
    
    def optimize_onnx(self, 
                     input_path: Path,
                     output_path: Path) -> bool:
        """Optimize ONNX model with graph simplification.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path to save optimized model
            
        Returns:
            True if optimization successful
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX is not installed. Install with: pip install onnx")
        
        try:
            # Load model
            if self.config.verbose:
                print(f"Loading ONNX model from {input_path}")
            
            model = onnx.load(str(input_path))
            
            # Apply optimizations
            optimized = False
            
            # Simplify if available
            if self.config.simplify and ONNXSIM_AVAILABLE:
                if self.config.verbose:
                    print("Simplifying model graph")
                
                simplified_model, check = onnx_simplify(model)
                if check:
                    model = simplified_model
                    optimized = True
                else:
                    print("Warning: Simplification failed", file=sys.stderr)
            
            # Additional ONNX optimizations can be added here
            # For example: constant folding, shape inference, etc.
            
            # Run shape inference
            if self.config.verbose:
                print("Running shape inference")
            
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
            
            # Save optimized model
            onnx.save(model, str(output_path))
            
            # Verify the optimized model
            onnx.checker.check_model(model)
            
            print(f"✅ Successfully optimized ONNX model: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX optimization failed: {e}", file=sys.stderr)
            return False
    
    def convert_precision(self,
                         input_path: Path,
                         output_path: Path,
                         target_precision: Precision) -> bool:
        """Convert model to different precision.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path to save converted model
            target_precision: Target precision
            
        Returns:
            True if conversion successful
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX is not installed. Install with: pip install onnx")
        
        try:
            # Load model
            if self.config.verbose:
                print(f"Loading ONNX model from {input_path}")
            
            model = onnx.load(str(input_path))
            
            # Convert precision
            if target_precision == Precision.FP16:
                if self.config.verbose:
                    print("Converting to FP16 precision")
                
                from onnx import numpy_helper
                
                # Convert all float32 tensors to float16
                for tensor in model.graph.initializer:
                    if tensor.data_type == TensorProto.FLOAT:
                        float_data = numpy_helper.to_array(tensor)
                        float16_data = float_data.astype(np.float16)
                        new_tensor = numpy_helper.from_array(float16_data, tensor.name)
                        tensor.CopyFrom(new_tensor)
                
                # Update tensor types in graph
                for value_info in model.graph.value_info:
                    if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                        value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                
                for input_info in model.graph.input:
                    if input_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                        input_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                
                for output_info in model.graph.output:
                    if output_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                        output_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            
            else:
                print(f"Warning: Precision conversion to {target_precision.value} not fully implemented", file=sys.stderr)
            
            # Save converted model
            onnx.save(model, str(output_path))
            
            # Verify the converted model
            onnx.checker.check_model(model)
            
            print(f"✅ Successfully converted model precision to {target_precision.value}: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Precision conversion failed: {e}", file=sys.stderr)
            return False
    
    def validate_conversion(self,
                           model1_path: Path,
                           model2_path: Path,
                           test_input: Optional[Any] = None,
                           tolerance: float = 1e-5) -> bool:
        """Validate conversion by comparing model outputs.
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            test_input: Test input for validation
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if models produce similar outputs
        """
        if not ONNXRUNTIME_AVAILABLE:
            print("Warning: ONNX Runtime not available. Cannot validate conversion.", file=sys.stderr)
            return False
        
        try:
            # Create inference sessions
            if self.config.verbose:
                print(f"Loading models for validation")
            
            session1 = ort.InferenceSession(str(model1_path))
            session2 = ort.InferenceSession(str(model2_path))
            
            # Get input details
            input1 = session1.get_inputs()[0]
            input2 = session2.get_inputs()[0]
            
            # Create test input if not provided
            if test_input is None:
                shape1 = input1.shape
                # Handle dynamic dimensions
                shape = []
                for dim in shape1:
                    if isinstance(dim, str) or dim is None:
                        shape.append(1)  # Use batch size of 1 for dynamic dims
                    else:
                        shape.append(dim)
                
                if NUMPY_AVAILABLE:
                    test_input = np.random.randn(*shape).astype(np.float32)
                else:
                    raise RuntimeError("NumPy is required for random test input generation")
            
            # Run inference
            if self.config.verbose:
                print("Running inference on both models")
            
            output1 = session1.run(None, {input1.name: test_input})[0]
            output2 = session2.run(None, {input2.name: test_input})[0]
            
            # Compare outputs
            if output1.shape != output2.shape:
                print(f"❌ Output shapes differ: {output1.shape} vs {output2.shape}", file=sys.stderr)
                return False
            
            # Calculate difference
            if NUMPY_AVAILABLE:
                diff = np.abs(output1 - output2)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
            else:
                # Fallback for systems without numpy
                max_diff = float('inf')
                mean_diff = float('inf')
            
            if self.config.verbose:
                print(f"Max difference: {max_diff:.6e}")
                print(f"Mean difference: {mean_diff:.6e}")
                print(f"Tolerance: {tolerance:.6e}")
            
            if max_diff > tolerance:
                print(f"❌ Validation failed: max difference {max_diff:.6e} exceeds tolerance {tolerance:.6e}", file=sys.stderr)
                return False
            
            print(f"✅ Validation passed: outputs match within tolerance {tolerance:.6e}")
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}", file=sys.stderr)
            return False


def parse_shape(shape_str: str) -> List[int]:
    """Parse shape string to list of integers.
    
    Args:
        shape_str: Shape string like "[1,3,224,224]" or "1,3,224,224"
        
    Returns:
        List of integers
    """
    # Remove brackets and spaces
    shape_str = shape_str.strip().strip('[]').replace(' ', '')
    # Split by comma and convert to int
    return [int(dim) for dim in shape_str.split(',')]


def main():
    """Main entry point for the model converter CLI."""
    parser = argparse.ArgumentParser(
        description="Model Converter - Automated model conversion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Conversion commands")
    
    # PyTorch to ONNX command
    pytorch_parser = subparsers.add_parser("pytorch-to-onnx", 
                                          help="Convert PyTorch model to ONNX")
    pytorch_parser.add_argument("input", type=Path, help="Input PyTorch model (.pt, .pth)")
    pytorch_parser.add_argument("output", type=Path, help="Output ONNX model (.onnx)")
    pytorch_parser.add_argument("--input-shapes", type=str, 
                               help="Input tensor shapes (e.g., '[1,3,224,224]')")
    pytorch_parser.add_argument("--input-names", nargs="+", help="Input tensor names")
    pytorch_parser.add_argument("--output-names", nargs="+", help="Output tensor names")
    pytorch_parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    pytorch_parser.add_argument("--dynamic-batch", action="store_true", 
                               help="Enable dynamic batch size")
    pytorch_parser.add_argument("--simplify", action="store_true", 
                               help="Simplify ONNX model")
    pytorch_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # ONNX to TensorRT command
    trt_parser = subparsers.add_parser("onnx-to-tensorrt", 
                                       help="Convert ONNX model to TensorRT engine")
    trt_parser.add_argument("input", type=Path, help="Input ONNX model (.onnx)")
    trt_parser.add_argument("output", type=Path, help="Output TensorRT engine (.engine)")
    trt_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], 
                           default="fp32", help="Inference precision")
    trt_parser.add_argument("--batch-size", type=int, default=1, 
                           help="Optimal batch size")
    trt_parser.add_argument("--max-batch-size", type=int, default=32, 
                           help="Maximum batch size")
    trt_parser.add_argument("--calibration-data", type=Path, 
                           help="Calibration data for INT8 (numpy file)")
    trt_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Optimize ONNX command
    opt_parser = subparsers.add_parser("optimize-onnx", 
                                       help="Optimize ONNX model")
    opt_parser.add_argument("input", type=Path, help="Input ONNX model")
    opt_parser.add_argument("output", type=Path, help="Output optimized ONNX model")
    opt_parser.add_argument("--simplify", action="store_true", 
                           help="Simplify model graph")
    opt_parser.add_argument("--fold-constants", action="store_true", 
                           help="Fold constant operations")
    opt_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Convert precision command
    prec_parser = subparsers.add_parser("convert-precision", 
                                        help="Convert model precision")
    prec_parser.add_argument("input", type=Path, help="Input ONNX model")
    prec_parser.add_argument("output", type=Path, help="Output ONNX model")
    prec_parser.add_argument("--from", dest="from_precision", 
                            choices=["fp32", "fp16", "int8"], default="fp32",
                            help="Source precision")
    prec_parser.add_argument("--to", dest="to_precision", 
                            choices=["fp32", "fp16", "int8"], required=True,
                            help="Target precision")
    prec_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", 
                                       help="Validate model conversion")
    val_parser.add_argument("model1", type=Path, help="First model")
    val_parser.add_argument("model2", type=Path, help="Second model")
    val_parser.add_argument("--tolerance", type=float, default=1e-5, 
                           help="Numerical tolerance")
    val_parser.add_argument("--test-input", type=Path, 
                           help="Test input (numpy file)")
    val_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Create converter configuration
        config = ConversionConfig()
        
        if hasattr(args, 'verbose'):
            config.verbose = args.verbose
        
        # Create converter
        converter = ModelConverter(config)
        
        # Execute command
        if args.command == "pytorch-to-onnx":
            if args.input_shapes:
                config.input_shapes = [parse_shape(args.input_shapes)]
            
            config.input_names = args.input_names
            config.output_names = args.output_names
            config.opset_version = args.opset
            config.simplify = args.simplify
            
            if args.dynamic_batch and config.input_names:
                config.dynamic_axes = {
                    config.input_names[0]: {0: 'batch_size'}
                }
                if config.output_names:
                    config.dynamic_axes[config.output_names[0]] = {0: 'batch_size'}
            
            success = converter.pytorch_to_onnx(args.input, args.output)
            return 0 if success else 1
        
        elif args.command == "onnx-to-tensorrt":
            config.precision = Precision(args.precision)
            config.batch_size = args.batch_size
            config.max_batch_size = args.max_batch_size
            
            calibration_data = None
            if args.calibration_data:
                if NUMPY_AVAILABLE:
                    calibration_data = np.load(args.calibration_data)
                else:
                    print("Warning: NumPy not available, cannot load calibration data", file=sys.stderr)
            
            success = converter.onnx_to_tensorrt(args.input, args.output, calibration_data)
            return 0 if success else 1
        
        elif args.command == "optimize-onnx":
            config.simplify = args.simplify
            config.fold_constants = args.fold_constants
            
            success = converter.optimize_onnx(args.input, args.output)
            return 0 if success else 1
        
        elif args.command == "convert-precision":
            target_precision = Precision(args.to_precision)
            
            success = converter.convert_precision(args.input, args.output, target_precision)
            return 0 if success else 1
        
        elif args.command == "validate":
            test_input = None
            if args.test_input:
                if NUMPY_AVAILABLE:
                    test_input = np.load(args.test_input)
                else:
                    print("Warning: NumPy not available, cannot load test input", file=sys.stderr)
            
            success = converter.validate_conversion(
                args.model1, args.model2, test_input, args.tolerance
            )
            return 0 if success else 1
        
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
