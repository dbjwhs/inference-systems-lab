#!/usr/bin/env python3
"""
Test script for Configuration Management System

This script demonstrates the configuration system without requiring PyYAML,
using a JSON configuration file instead.
"""

import json
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import ConfigurationLoader, ConfigurationError


class JSONConfigurationLoader(ConfigurationLoader):
    """Configuration loader that supports JSON files for testing."""
    
    def load(self, config_path=None):
        """Load configuration from JSON file instead of YAML."""
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            # Load JSON configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                raise ConfigurationError("Configuration file must contain a JSON object")
            
            print(f"✅ Loaded configuration from {self.config_path}")
            
            # Apply environment-specific overrides
            config = self._apply_environment_overrides(config)
            
            # Resolve environment variables
            config = self.env_resolver.resolve(config)
            
            # Validate configuration
            validation_errors = self.validator.validate(config, self.validate_strict)
            if validation_errors:
                print(f"⚠️  Configuration validation issues: {len(validation_errors)} errors")
                for error in validation_errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(validation_errors) > 3:
                    print(f"    ... and {len(validation_errors) - 3} more errors")
            else:
                print("✅ Configuration validation passed")
            
            self._config = config
            self._is_loaded = True
            
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Failed to parse JSON configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")


def test_configuration_system():
    """Test the configuration management system."""
    print("🧪 Testing Configuration Management System")
    print("=" * 50)
    
    # Find the test config file
    config_path = Path(__file__).parent.parent.parent / "config" / "test_config.json"
    schema_path = Path(__file__).parent.parent.parent / "config" / "schema" / "inference_config.json"
    
    if not config_path.exists():
        print(f"❌ Test config file not found: {config_path}")
        return False
    
    try:
        # Test 1: Basic Configuration Loading
        print("\n📋 Test 1: Basic Configuration Loading")
        config_loader = JSONConfigurationLoader(
            config_path=config_path,
            schema_path=schema_path if schema_path.exists() else None,
            environment="development"
        )
        
        config = config_loader.load()
        print(f"✅ Configuration loaded successfully")
        print(f"   Schema version: {config.get('schema_version', 'N/A')}")
        print(f"   Application name: {config_loader.get('application.name', 'N/A')}")
        print(f"   Environment: {config_loader.environment_name}")
        
        # Test 2: Type-safe Access Methods
        print("\n🔍 Test 2: Type-safe Configuration Access")
        
        debug_mode = config_loader.get_bool('application.debug_mode')
        print(f"   Debug mode (bool): {debug_mode}")
        
        max_threads = config_loader.get_int('application.settings.max_threads')
        print(f"   Max threads (int): {max_threads}")
        
        log_level = config_loader.get('logging.level', 'INFO')
        print(f"   Log level (string): {log_level}")
        
        console_config = config_loader.get_dict('logging.outputs.console')
        print(f"   Console logging enabled: {console_config.get('enabled', False)}")
        
        # Test 3: Feature Flag Checking
        print("\n🚩 Test 3: Feature Flag Management")
        
        experimental_features = config_loader.get_dict('features.experimental')
        for feature, enabled in experimental_features.items():
            status = "🟢 ENABLED" if enabled else "🔴 DISABLED"
            print(f"   experimental.{feature}: {status}")
        
        beta_features = config_loader.get_dict('features.beta')
        for feature, enabled in beta_features.items():
            status = "🟢 ENABLED" if enabled else "🔴 DISABLED"
            print(f"   beta.{feature}: {status}")
        
        # Test 4: Environment Variable Resolution
        print("\n🌍 Test 4: Environment Variable Resolution")
        
        # Set a test environment variable
        os.environ['TEST_DB_PATH'] = '/tmp/test_registry.db'
        
        # Create a test config with environment variable
        test_config = {
            "database_path": "${TEST_DB_PATH}",
            "database_with_default": "${MISSING_VAR:default_value}",
            "nested": {
                "config": "${TEST_DB_PATH}/nested"
            }
        }
        
        resolved_config = config_loader.env_resolver.resolve(test_config)
        print(f"   Original: {test_config['database_path']}")
        print(f"   Resolved: {resolved_config['database_path']}")
        print(f"   With default: {resolved_config['database_with_default']}")
        print(f"   Nested: {resolved_config['nested']['config']}")
        
        # Test 5: Configuration Validation
        print("\n✅ Test 5: Configuration Validation")
        
        # Test with valid config
        validation_errors = config_loader.validate()
        if not validation_errors:
            print("   ✅ Valid configuration passed validation")
        else:
            print(f"   ⚠️  Configuration has {len(validation_errors)} validation issues")
        
        # Test with invalid config
        invalid_config = {
            "application": {
                "name": "test",
                "version": "1.0.0",
                "environment": "invalid_environment"  # Should be development/staging/production
            }
        }
        
        validation_errors = config_loader.validate(invalid_config)
        if validation_errors:
            print(f"   ✅ Invalid configuration correctly rejected ({len(validation_errors)} errors)")
        else:
            print("   ⚠️  Invalid configuration was not rejected")
        
        # Test 6: Configuration Properties
        print("\n📊 Test 6: Configuration Properties")
        
        print(f"   Environment name: {config_loader.environment_name}")
        print(f"   Is development: {config_loader.is_development}")
        print(f"   Is staging: {config_loader.is_staging}")
        print(f"   Is production: {config_loader.is_production}")
        print(f"   Debug mode: {config_loader.debug_mode}")
        
        # Test 7: Key Existence and Listing
        print("\n🔑 Test 7: Key Management")
        
        print(f"   Has 'application.name': {config_loader.has('application.name')}")
        print(f"   Has 'non.existent.key': {config_loader.has('non.existent.key')}")
        
        root_keys = config_loader.keys()
        print(f"   Root keys: {', '.join(root_keys[:5])}{'...' if len(root_keys) > 5 else ''}")
        
        app_keys = config_loader.keys('application')
        print(f"   Application keys: {', '.join(app_keys)}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_configuration_system()
    sys.exit(0 if success else 1)
