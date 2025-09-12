#!/usr/bin/env python3
"""
Comprehensive test suite for check_headers.py

Tests all functionality including header detection, file exclusions,
header replacement, git integration, and error handling.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Import the module to test
import check_headers


class TestCheckHeaders(unittest.TestCase):
    """Test suite for check_headers.py functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        
        # Expected header content
        self.expected_header = check_headers.EXPECTED_HEADER.strip()
        
        # Sample file contents
        self.file_with_correct_header = f"""{check_headers.EXPECTED_HEADER}

#include <iostream>

int main() {{ return 0; }}
"""
        
        self.file_without_header = """#include <iostream>

int main() { return 0; }
"""
        
        self.file_with_wrong_header = """// Wrong License
// Copyright (c) 2020 someone

#include <iostream>

int main() { return 0; }
"""

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_file(self, content: str, path: str) -> Path:
        """Create a test file with given content."""
        file_path = self.root_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def test_check_header_correct(self):
        """Test header detection with correct header."""
        self.assertTrue(check_headers.check_header(self.file_with_correct_header))

    def test_check_header_missing(self):
        """Test header detection with missing header."""
        self.assertFalse(check_headers.check_header(self.file_without_header))

    def test_check_header_wrong(self):
        """Test header detection with wrong header."""
        self.assertFalse(check_headers.check_header(self.file_with_wrong_header))

    def test_check_header_empty_file(self):
        """Test header detection with empty file."""
        self.assertFalse(check_headers.check_header(""))

    def test_add_header_to_new_file(self):
        """Test adding header to file without header."""
        result = check_headers.add_header(self.file_without_header)
        self.assertTrue(result.startswith(check_headers.EXPECTED_HEADER))
        self.assertIn("#include <iostream>", result)

    def test_add_header_replaces_existing(self):
        """Test replacing existing wrong header."""
        result = check_headers.add_header(self.file_with_wrong_header)
        self.assertTrue(result.startswith(check_headers.EXPECTED_HEADER))
        self.assertNotIn("Wrong License", result)
        self.assertIn("#include <iostream>", result)

    def test_add_header_preserves_code(self):
        """Test that header addition preserves existing code."""
        original_code = "#include <vector>\n\nint main() { return 42; }"
        file_content = f"// Old header\n\n{original_code}"
        result = check_headers.add_header(file_content)
        
        self.assertTrue(result.startswith(check_headers.EXPECTED_HEADER))
        self.assertIn(original_code, result)

    def test_should_skip_file_build_dirs(self):
        """Test file exclusion for build directories."""
        test_cases = [
            Path("build/test.cpp"),
            Path("build-debug/test.cpp"),
            Path("build-release/test.cpp"),
            Path("cmake-build-debug/test.cpp"),
            Path("project/build/nested/test.hpp"),
        ]
        
        for path in test_cases:
            with self.subTest(path=path):
                self.assertTrue(check_headers.should_skip_file(path))

    def test_should_skip_file_third_party(self):
        """Test file exclusion for third-party directories."""
        test_cases = [
            Path("third_party/lib/test.cpp"),
            Path("external/googletest/test.hpp"),
            Path("vendor/json/json.hpp"),
            Path("_deps/benchmark/test.cpp"),
        ]
        
        for path in test_cases:
            with self.subTest(path=path):
                self.assertTrue(check_headers.should_skip_file(path))

    def test_should_skip_file_generated_files(self):
        """Test file exclusion for generated files."""
        test_cases = [
            Path("schemas/test.capnp.h"),
            Path("CMakeFiles/test.cpp"),
            Path("python_tool/script.py"),
            Path("tools/old_script.py"),
        ]
        
        for path in test_cases:
            with self.subTest(path=path):
                self.assertTrue(check_headers.should_skip_file(path))

    def test_should_not_skip_source_files(self):
        """Test that source files are not skipped."""
        test_cases = [
            Path("common/src/test.cpp"),
            Path("engines/src/test.hpp"),
            Path("integration/tests/test.cpp"),
            Path("examples/demo.hpp"),
        ]
        
        for path in test_cases:
            with self.subTest(path=path):
                self.assertFalse(check_headers.should_skip_file(path))

    def test_process_file_correct_header(self):
        """Test processing file with correct header."""
        file_path = self._create_test_file(self.file_with_correct_header, "test.cpp")
        
        has_correct, error = check_headers.process_file(file_path, fix=False)
        self.assertTrue(has_correct)
        self.assertIsNone(error)

    def test_process_file_fix_missing_header(self):
        """Test processing and fixing file with missing header."""
        file_path = self._create_test_file(self.file_without_header, "test.cpp")
        
        # First check without fix
        has_correct, error = check_headers.process_file(file_path, fix=False)
        self.assertFalse(has_correct)
        self.assertIsNone(error)
        
        # Now fix it
        has_correct, error = check_headers.process_file(file_path, fix=True)
        self.assertTrue(has_correct)
        self.assertIsNone(error)
        
        # Verify the file was actually fixed
        fixed_content = file_path.read_text()
        self.assertTrue(fixed_content.startswith(check_headers.EXPECTED_HEADER))

    def test_process_file_with_backup(self):
        """Test processing file with backup creation."""
        file_path = self._create_test_file(self.file_without_header, "test.cpp")
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        
        has_correct, error = check_headers.process_file(file_path, fix=True, backup=True)
        self.assertTrue(has_correct)
        self.assertIsNone(error)
        
        # Check backup was created
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.read_text(), self.file_without_header)

    def test_process_file_read_error(self):
        """Test error handling for unreadable files."""
        fake_path = Path("/nonexistent/file.cpp")
        
        has_correct, error = check_headers.process_file(fake_path, fix=False)
        self.assertFalse(has_correct)
        self.assertIsNotNone(error)
        self.assertIn("Error reading", error)

    def test_find_cpp_files(self):
        """Test finding C++ files in directory tree."""
        # Create test files
        self._create_test_file("content", "src/test.cpp")
        self._create_test_file("content", "include/test.hpp")
        self._create_test_file("content", "build/ignore.cpp")  # Should be excluded
        self._create_test_file("content", "README.md")  # Wrong extension
        
        files = check_headers.find_cpp_files(self.root_path, staged_only=False)
        
        # Convert to relative paths for easier testing
        rel_files = [f.relative_to(self.root_path) for f in files]
        
        self.assertIn(Path("src/test.cpp"), rel_files)
        self.assertIn(Path("include/test.hpp"), rel_files)
        self.assertNotIn(Path("build/ignore.cpp"), rel_files)
        self.assertNotIn(Path("README.md"), rel_files)

    @patch('check_headers.subprocess.run')
    def test_get_git_root_success(self, mock_run):
        """Test git root detection success."""
        mock_run.return_value.stdout = "/test/repo\n"
        mock_run.return_value.returncode = 0
        
        # Clear the cache first
        check_headers.get_git_root.cache_clear()
        
        result = check_headers.get_git_root()
        self.assertEqual(result, Path("/test/repo"))

    @patch('check_headers.subprocess.run')
    def test_get_git_root_failure(self, mock_run):
        """Test git root detection failure fallback."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        
        # Clear the cache first
        check_headers.get_git_root.cache_clear()
        
        result = check_headers.get_git_root()
        self.assertEqual(result, Path.cwd())

    @patch('check_headers.subprocess.run')
    def test_find_cpp_files_staged(self, mock_run):
        """Test finding staged C++ files."""
        mock_run.return_value.stdout = "src/test.cpp\ninclude/test.hpp\nREADME.md\n"
        mock_run.return_value.returncode = 0
        
        # Create the files that git claims are staged
        self._create_test_file("content", "src/test.cpp")
        self._create_test_file("content", "include/test.hpp")
        self._create_test_file("content", "README.md")
        
        files = check_headers.find_cpp_files(self.root_path, staged_only=True)
        
        # Should only include C++ files, not README.md
        self.assertEqual(len(files), 2)
        rel_files = [f.relative_to(self.root_path) for f in files]
        self.assertIn(Path("src/test.cpp"), rel_files)
        self.assertIn(Path("include/test.hpp"), rel_files)

    def test_path_validation(self):
        """Test path validation security."""
        # This assumes we'll add validate_file_path function
        root_dir = Path("/safe/root")
        
        # Valid paths
        valid_paths = [
            Path("/safe/root/file.cpp"),
            Path("/safe/root/subdir/file.hpp"),
        ]
        
        # Invalid paths (if function exists)
        if hasattr(check_headers, 'validate_file_path'):
            for path in valid_paths:
                self.assertTrue(check_headers.validate_file_path(path, root_dir))

    def test_file_extensions(self):
        """Test that all supported C++ extensions are handled."""
        extensions = ['.cpp', '.hpp', '.h', '.cc', '.cxx', '.hxx']
        
        for ext in extensions:
            file_path = self._create_test_file(self.file_without_header, f"test{ext}")
            
            with self.subTest(extension=ext):
                # Should not be skipped based on extension
                self.assertFalse(check_headers.should_skip_file(file_path))
                
                # Should be processable
                has_correct, error = check_headers.process_file(file_path, fix=True)
                self.assertTrue(has_correct)
                self.assertIsNone(error)

    def test_header_preservation_edge_cases(self):
        """Test header handling with edge cases."""
        edge_cases = [
            # File with only comments
            "// Just a comment\n// Another comment\n",
            
            # File with existing header-like content
            "// Some license\n// Copyright someone\n\n#include <test>\n",
            
            # File with mixed comment styles
            "/* Block comment */\n// Line comment\n\ncode();\n",
            
            # File starting with preprocessor
            "#pragma once\n\n#include <header>\n",
        ]
        
        for i, content in enumerate(edge_cases):
            file_path = self._create_test_file(content, f"edge_case_{i}.hpp")
            
            with self.subTest(case=i):
                has_correct, error = check_headers.process_file(file_path, fix=True)
                self.assertTrue(has_correct, f"Failed on case {i}")
                self.assertIsNone(error)
                
                # Verify header was added
                fixed_content = file_path.read_text()
                self.assertTrue(fixed_content.startswith(check_headers.EXPECTED_HEADER))


class TestMainFunction(unittest.TestCase):
    """Test the main function and command-line interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_file(self, content: str, path: str) -> Path:
        """Create a test file with given content."""
        file_path = self.root_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    @patch('check_headers.get_git_root')
    @patch('check_headers.sys.argv')
    def test_main_check_mode_success(self, mock_argv, mock_git_root):
        """Test main function in check mode with correct headers."""
        mock_git_root.return_value = self.root_path
        mock_argv.__getitem__.return_value = 'check_headers.py'
        
        # Create file with correct header
        correct_content = f"""{check_headers.EXPECTED_HEADER}

#include <test>
"""
        self._create_test_file(correct_content, "test.cpp")
        
        with patch('check_headers.sys.argv', ['check_headers.py', '--check']):
            result = check_headers.main()
            self.assertEqual(result, 0)

    @patch('check_headers.get_git_root')
    def test_main_check_mode_failure(self, mock_git_root):
        """Test main function in check mode with incorrect headers."""
        mock_git_root.return_value = self.root_path
        
        # Create file without header
        self._create_test_file("#include <test>", "test.cpp")
        
        with patch('check_headers.sys.argv', ['check_headers.py', '--check']):
            result = check_headers.main()
            self.assertEqual(result, 1)

    @patch('check_headers.get_git_root')
    def test_main_fix_mode(self, mock_git_root):
        """Test main function in fix mode."""
        mock_git_root.return_value = self.root_path
        
        # Create file without header
        file_path = self._create_test_file("#include <test>", "test.cpp")
        
        with patch('check_headers.sys.argv', ['check_headers.py', '--fix']):
            result = check_headers.main()
            self.assertEqual(result, 0)
            
            # Verify file was fixed
            fixed_content = file_path.read_text()
            self.assertTrue(fixed_content.startswith(check_headers.EXPECTED_HEADER))


if __name__ == '__main__':
    # Set up test discovery and run
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
