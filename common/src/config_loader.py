#!/usr/bin/env python3
"""
Configuration Management System for Inference Systems Laboratory

This module provides a comprehensive configuration management system that supports:
- YAML configuration file loading and validation
- Environment variable substitution
- Environment-specific overrides (development/staging/production)  
- JSON Schema validation
- Type-safe configuration access
- Configuration merging and inheritance
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

# Set up module logger
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigurationFileNotFoundError(ConfigurationError):
    """Exception raised when configuration file cannot be found."""
    pass


class EnvironmentVariableResolver:
    """Handles environment variable substitution in configuration values."""
    
    # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
    ENV_VAR_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}')
    
    @classmethod
    def resolve(cls, value: Any) -> Any:
        """
        Recursively resolve environment variables in configuration values.
        
        Args:
            value: Configuration value (can be string, dict, list, etc.)
            
        Returns:
            Configuration value with environment variables resolved
        """
        if isinstance(value, str):
            return cls._resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item) for item in value]
        else:
            return value
    
    @classmethod
    def _resolve_string(cls, value: str) -> str:
        """
        Resolve environment variables in a string value.
        
        Supports formats:
        - ${VAR_NAME} - Required environment variable
        - ${VAR_NAME:default} - Environment variable with default value
        
        Args:
            value: String potentially containing environment variable references
            
        Returns:
            String with environment variables resolved
            
        Raises:
            ConfigurationError: If required environment variable is not set
        """
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ConfigurationError(
                    f"Required environment variable '{var_name}' is not set"
                )
        
        return cls.ENV_VAR_PATTERN.sub(replace_env_var, value)


class ConfigurationValidator:
    """Validates configuration against JSON Schema."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize configuration validator.
        
        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        self.schema = None
        self.validator = None
        
        if JSONSCHEMA_AVAILABLE and schema_path:
            self._load_schema(schema_path)
    
    def _load_schema(self, schema_path: Path) -> None:
        """Load JSON schema from file."""
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
            self.validator = jsonschema.Draft7Validator(self.schema)
            logger.debug(f"Loaded configuration schema from {schema_path}")
        except Exception as e:
            logger.warning(f"Failed to load schema from {schema_path}: {e}")
            self.schema = None
            self.validator = None
    
    def validate(self, config: Dict[str, Any], strict: bool = True) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            strict: If True, raise exception on validation errors
            
        Returns:
            List of validation error messages (empty if valid)
            
        Raises:
            ConfigurationValidationError: If strict=True and validation fails
        """
        errors = []
        
        if not JSONSCHEMA_AVAILABLE:
            if strict:
                logger.warning("jsonschema not available - skipping validation")
            return errors
        
        if not self.validator:
            if strict:
                logger.warning("No schema loaded - skipping validation")
            return errors
        
        try:
            # Collect all validation errors
            for error in self.validator.iter_errors(config):
                error_msg = f"Validation error at {'.'.join(str(p) for p in error.absolute_path)}: {error.message}"
                errors.append(error_msg)
                logger.debug(error_msg)
            
            if errors and strict:
                raise ConfigurationValidationError(
                    f"Configuration validation failed:\n" + "\n".join(errors)
                )
            
        except jsonschema.exceptions.SchemaError as e:
            error_msg = f"Schema validation error: {e}"
            errors.append(error_msg)
            if strict:
                raise ConfigurationValidationError(error_msg)
        
        return errors


class ConfigurationLoader:
    """
    Main configuration loader class that handles loading, validation, and management
    of YAML configuration files with environment-specific overrides.
    """
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 schema_path: Optional[Union[str, Path]] = None,
                 environment: Optional[str] = None,
                 validate_strict: bool = True):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to main configuration file
            schema_path: Path to JSON schema file for validation
            environment: Target environment (development/staging/production)
            validate_strict: Whether to raise exceptions on validation errors
        """
        self.config_path = Path(config_path) if config_path else None
        self.schema_path = Path(schema_path) if schema_path else None
        self.environment = environment or os.getenv('INFERENCE_LAB_ENV', 'development')
        self.validate_strict = validate_strict
        
        # Initialize components
        self.env_resolver = EnvironmentVariableResolver()
        self.validator = ConfigurationValidator(self.schema_path)
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._is_loaded = False
        
        # Auto-discover paths if not provided
        if not self.config_path:
            self.config_path = self._discover_config_path()
        if not self.schema_path:
            self.schema_path = self._discover_schema_path()
            if self.schema_path and not self.validator.schema:
                self.validator = ConfigurationValidator(self.schema_path)
    
    def _discover_config_path(self) -> Optional[Path]:
        """Auto-discover configuration file path."""
        possible_paths = [
            Path("config/inference_lab_config.yaml"),
            Path("inference_lab_config.yaml"),
            Path("config.yaml"),
            Path("../config/inference_lab_config.yaml"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.debug(f"Auto-discovered config path: {path}")
                return path
        
        logger.warning("No configuration file found in standard locations")
        return None
    
    def _discover_schema_path(self) -> Optional[Path]:
        """Auto-discover schema file path."""
        possible_paths = [
            Path("config/schema/inference_config.json"),
            Path("schema/inference_config.json"),
            Path("inference_config.json"),
            Path("../config/schema/inference_config.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.debug(f"Auto-discovered schema path: {path}")
                return path
        
        logger.debug("No schema file found - validation will be skipped")
        return None
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment overrides.
        
        Args:
            config_path: Override config path for this load
            
        Returns:
            Loaded and processed configuration dictionary
            
        Raises:
            ConfigurationFileNotFoundError: If config file cannot be found
            ConfigurationError: If config file cannot be parsed
            ConfigurationValidationError: If validation fails (in strict mode)
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise ConfigurationFileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        if not YAML_AVAILABLE:
            raise ConfigurationError(
                "PyYAML is required for configuration loading. "
                "Install with: pip install PyYAML"
            )
        
        try:
            # Load base configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ConfigurationError("Configuration file must contain a YAML object")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
            # Apply environment-specific overrides
            config = self._apply_environment_overrides(config)
            
            # Resolve environment variables
            config = self.env_resolver.resolve(config)
            
            # Validate configuration
            validation_errors = self.validator.validate(config, self.validate_strict)
            if validation_errors:
                logger.warning(f"Configuration validation issues: {len(validation_errors)} errors")
                for error in validation_errors:
                    logger.warning(f"  - {error}")
            else:
                logger.debug("Configuration validation passed")
            
            self._config = config
            self._is_loaded = True
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment-specific configuration overrides.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        if 'environments' not in config:
            return config
        
        env_overrides = config.get('environments', {}).get(self.environment, {})
        if not env_overrides:
            logger.debug(f"No environment overrides found for '{self.environment}'")
            return config
        
        logger.info(f"Applying environment overrides for '{self.environment}'")
        
        # Deep merge environment overrides
        merged_config = self._deep_merge(config.copy(), env_overrides)
        
        # Remove the environments section from final config
        merged_config.pop('environments', None)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot-notation path.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self._is_loaded:
            self.load()
        
        current = self._config
        
        for key in key_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_bool(self, key_path: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(key_path, default)
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        else:
            return bool(value)
    
    def get_int(self, key_path: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(key_path, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def get_float(self, key_path: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(key_path, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def get_list(self, key_path: str, default: Optional[List[Any]] = None) -> List[Any]:
        """Get configuration value as list."""
        value = self.get(key_path, default or [])
        return value if isinstance(value, list) else [value]
    
    def get_dict(self, key_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration value as dictionary."""
        value = self.get(key_path, default or {})
        return value if isinstance(value, dict) else {}
    
    def has(self, key_path: str) -> bool:
        """Check if configuration key exists."""
        if not self._is_loaded:
            self.load()
        
        current = self._config
        
        for key in key_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        return True
    
    def keys(self, key_path: str = '') -> List[str]:
        """
        Get list of keys at the specified path.
        
        Args:
            key_path: Path to get keys for (empty string for root keys)
            
        Returns:
            List of keys at the specified path
        """
        if key_path:
            value = self.get(key_path, {})
        else:
            if not self._is_loaded:
                self.load()
            value = self._config
        
        if isinstance(value, dict):
            return list(value.keys())
        else:
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return complete configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        if not self._is_loaded:
            self.load()
        
        return self._config.copy()
    
    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.
        
        Returns:
            Reloaded configuration dictionary
        """
        self._is_loaded = False
        return self.load()
    
    def validate(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate (uses loaded config if None)
            
        Returns:
            List of validation error messages
        """
        if config is None:
            if not self._is_loaded:
                self.load()
            config = self._config
        
        return self.validator.validate(config, strict=False)
    
    @property
    def environment_name(self) -> str:
        """Get current environment name."""
        return self.environment
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == 'staging'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get_bool('application.debug_mode', False)


# Global configuration instance for easy access
_global_config: Optional[ConfigurationLoader] = None


def get_config() -> ConfigurationLoader:
    """
    Get global configuration instance.
    
    Returns:
        Global configuration loader instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigurationLoader()
    
    return _global_config


def init_config(config_path: Optional[Union[str, Path]] = None,
                schema_path: Optional[Union[str, Path]] = None,
                environment: Optional[str] = None,
                validate_strict: bool = True) -> ConfigurationLoader:
    """
    Initialize global configuration.
    
    Args:
        config_path: Path to configuration file
        schema_path: Path to schema file
        environment: Target environment
        validate_strict: Whether to validate strictly
        
    Returns:
        Initialized configuration loader
    """
    global _global_config
    
    _global_config = ConfigurationLoader(
        config_path=config_path,
        schema_path=schema_path,
        environment=environment,
        validate_strict=validate_strict
    )
    
    return _global_config


# Convenience functions for common operations
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    loader = ConfigurationLoader(config_path)
    return loader.load()


def get_setting(key_path: str, default: Any = None) -> Any:
    """Get configuration setting using global config."""
    return get_config().get(key_path, default)


def is_feature_enabled(feature_path: str) -> bool:
    """Check if a feature flag is enabled."""
    return get_config().get_bool(f'features.{feature_path}', False)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config/inference_lab_config.yaml'
    
    try:
        # Initialize configuration
        config = init_config(config_file)
        
        # Load and display configuration
        config.load()
        
        print(f"Configuration loaded successfully!")
        print(f"Environment: {config.environment_name}")
        print(f"Debug mode: {config.debug_mode}")
        print(f"Application name: {config.get('application.name')}")
        print(f"Logging level: {config.get('logging.level')}")
        print(f"Registry database path: {config.get('registry.database.path')}")
        
        # Check feature flags
        print("\nFeature flags:")
        for feature_type in ['experimental', 'beta', 'legacy']:
            features = config.get_dict(f'features.{feature_type}')
            for feature, enabled in features.items():
                print(f"  {feature_type}.{feature}: {enabled}")
        
        # Validate configuration
        errors = config.validate()
        if errors:
            print(f"\nValidation errors: {len(errors)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nConfiguration is valid!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
