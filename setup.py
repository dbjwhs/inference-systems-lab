"""Setup script for inference_lab Python package"""

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building with CMake instead of setuptools"""
    
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension using CMake"""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        
        # CMake build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTING=OFF",  # Skip tests for pip install
        ]
        
        # Pile on the rest of the arguments
        build_args = ["--config", cfg]
        
        # Set CMAKE_BUILD_PARALLEL_LEVEL to use multiple cores
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Use half the available cores
            import multiprocessing as mp
            build_args += [f"-j{mp.cpu_count() // 2}"]
        
        # Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )


# Read version from CMakeLists.txt
def get_version():
    cmake_file = Path(__file__).parent / "CMakeLists.txt"
    with open(cmake_file, "r") as f:
        content = f.read()
        version_match = re.search(r'project\([^)]*VERSION\s+([0-9.]+)', content)
        if version_match:
            return version_match.group(1)
    return "0.1.0"  # Default version


# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "Inference Systems Laboratory - Python Bindings"


setup(
    name="inference_lab",
    version=get_version(),
    author="Inference Systems Laboratory",
    author_email="",
    description="Python bindings for Inference Systems Laboratory C++ libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbjwhs/inference-systems-lab",
    ext_modules=[CMakeExtension("inference_lab")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",  # For tensor data exchange
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
        ],
        "ml": [
            "torch>=1.9.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
)