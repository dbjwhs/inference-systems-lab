#!/usr/bin/env python3
"""
Test script to verify all ML dependencies are working correctly in the Nix environment.
Run with: nix develop -c python3 test_ml_dependencies.py
"""

import sys

def test_dependency(name, import_statement, version_check=None):
    """Test if a dependency can be imported and optionally check its version."""
    try:
        exec(import_statement)
        if version_check:
            version = eval(version_check)
            print(f"✅ {name}: {version}")
        else:
            print(f"✅ {name}: Available")
        return True
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {name}: Import succeeded but version check failed - {e}")
        return True

def main():
    print("🧪 Testing ML Dependencies in Nix Environment")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test core Python ML libraries
    tests = [
        ("NumPy", "import numpy", "f'v{numpy.__version__}'"),
        ("ONNX", "import onnx", "f'v{onnx.__version__}'"),
        ("OpenCV (Python)", "import cv2", "f'v{cv2.__version__}'"),
        ("PyTorch", "import torch", "f'v{torch.__version__}'"),
    ]
    
    for name, import_stmt, version_check in tests:
        total_tests += 1
        if test_dependency(name, import_stmt, version_check):
            success_count += 1
    
    print("\n" + "=" * 50)
    
    # PyTorch specific tests
    if success_count == total_tests:
        print("\n🚀 PyTorch Device Information:")
        try:
            import torch
            print(f"   CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   CUDA device count: {torch.cuda.device_count()}")
                print(f"   Current device: {torch.cuda.current_device()}")
                print(f"   Device name: {torch.cuda.get_device_name(0)}")
            else:
                print("   Running on CPU (expected on macOS)")
            
            # Test basic tensor operations
            x = torch.randn(3, 3)
            y = torch.randn(3, 3)
            z = torch.matmul(x, y)
            print(f"   Tensor operations: ✅ Working (result shape: {z.shape})")
            
        except Exception as e:
            print(f"   PyTorch device test failed: {e}")
    
    print(f"\n📊 Results: {success_count}/{total_tests} dependencies working")
    
    if success_count == total_tests:
        print("🎉 All ML dependencies are working correctly!")
        return 0
    else:
        print("⚠️  Some dependencies are missing or broken")
        return 1

if __name__ == "__main__":
    sys.exit(main())