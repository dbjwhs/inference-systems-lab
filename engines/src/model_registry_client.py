#!/usr/bin/env python3
"""
Model Registry Client for Inference Systems Laboratory

Provides Python interface to the SQLite model registry, enabling model
registration, versioning, lifecycle management, and metadata tracking.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Supported model types"""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    RULE_BASED = "rule_based"
    OTHER = "other"


class InferenceBackend(Enum):
    """Inference backend types matching C++ enum"""
    RULE_BASED = "RULE_BASED"
    TENSORRT_GPU = "TENSORRT_GPU"
    ONNX_RUNTIME = "ONNX_RUNTIME"
    HYBRID_NEURAL_SYMBOLIC = "HYBRID_NEURAL_SYMBOLIC"


class ModelRegistryClient:
    """
    Client for interacting with the model registry database.
    
    This class provides a high-level interface for model management operations
    including registration, versioning, deployment tracking, and performance monitoring.
    """
    
    def __init__(self, db_path: str = "model_registry.db"):
        """
        Initialize the model registry client.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with schema if it doesn't exist"""
        # Check if schema file exists
        schema_file = Path(__file__).parent / "model_registry_schema.sql"
        
        if not Path(self.db_path).exists() and schema_file.exists():
            logger.info(f"Creating new model registry database at {self.db_path}")
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def register_model(self,
                       model_name: str,
                       model_version: str,
                       model_type: ModelType,
                       backend: InferenceBackend,
                       model_path: str,
                       **kwargs) -> int:
        """
        Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            model_version: Semantic version (e.g., "1.0.0")
            model_type: Type of model (ONNX, TensorRT, etc.)
            backend: Inference backend to use
            model_path: Path to the model file
            **kwargs: Additional optional fields (description, author, tags, etc.)
        
        Returns:
            model_id: The ID of the registered model
        
        Raises:
            sqlite3.IntegrityError: If model with same name and version exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare optional fields
            fields = {
                'model_name': model_name,
                'model_version': model_version,
                'model_type': model_type.value,
                'backend': backend.value,
                'model_path': model_path,
                'status': kwargs.get('status', ModelStatus.DEVELOPMENT.value)
            }
            
            # Add optional fields if provided
            optional_fields = [
                'config_path', 'metadata_path', 'input_shape', 'output_shape',
                'input_dtype', 'output_dtype', 'model_size_bytes', 
                'estimated_memory_mb', 'description', 'author', 'tags',
                'custom_metadata'
            ]
            
            for field in optional_fields:
                if field in kwargs:
                    value = kwargs[field]
                    # Convert lists/dicts to JSON strings
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)
                    fields[field] = value
            
            # Build INSERT query
            columns = ', '.join(fields.keys())
            placeholders = ', '.join(['?' for _ in fields])
            query = f"INSERT INTO models ({columns}) VALUES ({placeholders})"
            
            try:
                cursor.execute(query, list(fields.values()))
                conn.commit()
                model_id = cursor.lastrowid
                logger.info(f"Registered model: {model_name} v{model_version} (ID: {model_id})")
                return model_id
            except sqlite3.IntegrityError as e:
                logger.error(f"Model {model_name} v{model_version} already exists")
                raise
    
    def get_model(self, model_id: Optional[int] = None,
                  model_name: Optional[str] = None,
                  model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve model information from the registry.
        
        Args:
            model_id: Model ID (if provided, takes precedence)
            model_name: Model name (used with model_version)
            model_version: Model version (used with model_name)
        
        Returns:
            Dictionary with model information or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if model_id:
                query = "SELECT * FROM models WHERE model_id = ?"
                cursor.execute(query, (model_id,))
            elif model_name and model_version:
                query = "SELECT * FROM models WHERE model_name = ? AND model_version = ?"
                cursor.execute(query, (model_name, model_version))
            else:
                logger.error("Must provide either model_id or (model_name, model_version)")
                return None
            
            row = cursor.fetchone()
            if row:
                # Convert Row to dictionary and parse JSON fields
                model = dict(row)
                for field in ['input_shape', 'output_shape', 'tags', 'custom_metadata']:
                    if model.get(field):
                        try:
                            model[field] = json.loads(model[field])
                        except json.JSONDecodeError:
                            pass
                return model
            return None
    
    def list_models(self, 
                    status: Optional[ModelStatus] = None,
                    model_type: Optional[ModelType] = None,
                    backend: Optional[InferenceBackend] = None) -> List[Dict[str, Any]]:
        """
        List models with optional filtering.
        
        Args:
            status: Filter by model status
            model_type: Filter by model type
            backend: Filter by inference backend
        
        Returns:
            List of model dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            
            if backend:
                query += " AND backend = ?"
                params.append(backend.value)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            models = []
            for row in cursor.fetchall():
                model = dict(row)
                # Parse JSON fields
                for field in ['input_shape', 'output_shape', 'tags', 'custom_metadata']:
                    if model.get(field):
                        try:
                            model[field] = json.loads(model[field])
                        except json.JSONDecodeError:
                            pass
                models.append(model)
            
            return models
    
    def update_model_status(self, model_id: int, new_status: ModelStatus) -> bool:
        """
        Update the status of a model (lifecycle management).
        
        Args:
            model_id: ID of the model to update
            new_status: New status for the model
        
        Returns:
            True if successful, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Special handling for deployment to production
            if new_status == ModelStatus.PRODUCTION:
                cursor.execute(
                    "UPDATE models SET status = ?, deployed_at = CURRENT_TIMESTAMP WHERE model_id = ?",
                    (new_status.value, model_id)
                )
            elif new_status == ModelStatus.DEPRECATED:
                cursor.execute(
                    "UPDATE models SET status = ?, deprecated_at = CURRENT_TIMESTAMP WHERE model_id = ?",
                    (new_status.value, model_id)
                )
            else:
                cursor.execute(
                    "UPDATE models SET status = ? WHERE model_id = ?",
                    (new_status.value, model_id)
                )
            
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"Updated model {model_id} status to {new_status.value}")
            else:
                logger.warning(f"No model found with ID {model_id}")
            
            return success
    
    def record_deployment(self, model_id: int,
                         deployment_environment: str,
                         deployed_by: str,
                         deployment_notes: Optional[str] = None,
                         previous_model_id: Optional[int] = None) -> int:
        """
        Record a model deployment event.
        
        Args:
            model_id: ID of the model being deployed
            deployment_environment: Target environment (development/staging/production)
            deployed_by: Person or system deploying the model
            deployment_notes: Optional deployment notes
            previous_model_id: ID of model being replaced (for rollback tracking)
        
        Returns:
            deployment_id: ID of the deployment record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO deployment_history 
                (model_id, deployment_status, deployment_environment, 
                 deployed_by, deployment_notes, previous_model_id)
                VALUES (?, 'deployed', ?, ?, ?, ?)
            """, (model_id, deployment_environment, deployed_by, 
                  deployment_notes, previous_model_id))
            
            conn.commit()
            deployment_id = cursor.lastrowid
            
            logger.info(f"Recorded deployment {deployment_id} for model {model_id}")
            return deployment_id
    
    def record_performance_metrics(self, model_id: int,
                                  inference_time_ms: float,
                                  **kwargs) -> int:
        """
        Record performance metrics for a model.
        
        Args:
            model_id: ID of the model
            inference_time_ms: Inference time in milliseconds
            **kwargs: Additional metrics (memory_usage_mb, throughput_qps, etc.)
        
        Returns:
            metric_id: ID of the metric record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            fields = {
                'model_id': model_id,
                'inference_time_ms': inference_time_ms
            }
            
            # Add optional metrics
            optional_fields = [
                'preprocessing_time_ms', 'postprocessing_time_ms',
                'memory_usage_mb', 'gpu_memory_mb', 
                'cpu_utilization_percent', 'gpu_utilization_percent',
                'batch_size', 'throughput_qps', 'hardware_info',
                'inference_backend', 'optimization_level'
            ]
            
            for field in optional_fields:
                if field in kwargs:
                    value = kwargs[field]
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    fields[field] = value
            
            columns = ', '.join(fields.keys())
            placeholders = ', '.join(['?' for _ in fields])
            query = f"INSERT INTO performance_metrics ({columns}) VALUES ({placeholders})"
            
            cursor.execute(query, list(fields.values()))
            conn.commit()
            
            return cursor.lastrowid
    
    def record_validation_result(self, model_id: int,
                                validation_type: str,
                                validation_status: str,
                                **kwargs) -> int:
        """
        Record validation results for a model.
        
        Args:
            model_id: ID of the model
            validation_type: Type of validation performed
            validation_status: Result status (passed/failed/warning)
            **kwargs: Additional validation metrics and details
        
        Returns:
            validation_id: ID of the validation record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            fields = {
                'model_id': model_id,
                'validation_type': validation_type,
                'validation_status': validation_status
            }
            
            # Add optional fields
            optional_fields = [
                'validation_dataset', 'accuracy_score', 'precision_score',
                'recall_score', 'f1_score', 'custom_metrics',
                'error_count', 'error_details', 'validation_config',
                'validator_version'
            ]
            
            for field in optional_fields:
                if field in kwargs:
                    value = kwargs[field]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    fields[field] = value
            
            columns = ', '.join(fields.keys())
            placeholders = ', '.join(['?' for _ in fields])
            query = f"INSERT INTO validation_results ({columns}) VALUES ({placeholders})"
            
            cursor.execute(query, list(fields.values()))
            conn.commit()
            
            logger.info(f"Recorded {validation_status} validation for model {model_id}")
            return cursor.lastrowid
    
    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a model by name.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Latest version string or None if model not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(model_version) FROM models WHERE model_name = ?",
                (model_name,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
    
    def get_production_models(self) -> List[Dict[str, Any]]:
        """
        Get all models currently in production.
        
        Returns:
            List of production model dictionaries
        """
        return self.list_models(status=ModelStatus.PRODUCTION)
    
    def get_model_performance_summary(self, model_id: int) -> Dict[str, Any]:
        """
        Get performance summary for a model.
        
        Args:
            model_id: ID of the model
        
        Returns:
            Dictionary with performance statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_performance_summary WHERE model_id = ?
            """, (model_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def rollback_deployment(self, deployment_id: int, rollback_reason: str) -> bool:
        """
        Record a deployment rollback.
        
        Args:
            deployment_id: ID of the deployment to rollback
            rollback_reason: Reason for the rollback
        
        Returns:
            True if successful, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get the deployment info
            cursor.execute(
                "SELECT model_id, previous_model_id FROM deployment_history WHERE deployment_id = ?",
                (deployment_id,)
            )
            deployment = cursor.fetchone()
            
            if not deployment:
                logger.error(f"Deployment {deployment_id} not found")
                return False
            
            # Update deployment status
            cursor.execute("""
                UPDATE deployment_history 
                SET deployment_status = 'rolled_back', rollback_reason = ?
                WHERE deployment_id = ?
            """, (rollback_reason, deployment_id))
            
            # If there was a previous model, restore it to production
            if deployment['previous_model_id']:
                cursor.execute(
                    "UPDATE models SET status = 'production' WHERE model_id = ?",
                    (deployment['previous_model_id'],)
                )
                cursor.execute(
                    "UPDATE models SET status = 'development' WHERE model_id = ?",
                    (deployment['model_id'],)
                )
            
            conn.commit()
            logger.info(f"Rolled back deployment {deployment_id}")
            return True
    
    def cleanup_old_metrics(self, days_to_keep: int = 30) -> int:
        """
        Clean up old performance metrics.
        
        Args:
            days_to_keep: Number of days of metrics to retain
        
        Returns:
            Number of records deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM performance_metrics 
                WHERE recorded_at < datetime('now', '-' || ? || ' days')
            """, (days_to_keep,))
            
            conn.commit()
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old metric records")
            
            return deleted_count


# Example usage and testing
if __name__ == "__main__":
    # Create a test registry
    registry = ModelRegistryClient("test_registry.db")
    
    # Register a model
    model_id = registry.register_model(
        model_name="resnet50",
        model_version="1.0.0",
        model_type=ModelType.ONNX,
        backend=InferenceBackend.ONNX_RUNTIME,
        model_path="/models/resnet50_v1.onnx",
        description="ResNet50 image classification model",
        author="ML Team",
        tags=["computer_vision", "classification"],
        input_shape=[1, 3, 224, 224],
        output_shape=[1, 1000]
    )
    
    print(f"Registered model with ID: {model_id}")
    
    # Retrieve the model
    model = registry.get_model(model_id=model_id)
    print(f"Retrieved model: {model['model_name']} v{model['model_version']}")
    
    # Update status
    registry.update_model_status(model_id, ModelStatus.PRODUCTION)
    
    # Record performance metrics
    registry.record_performance_metrics(
        model_id=model_id,
        inference_time_ms=15.3,
        memory_usage_mb=512,
        throughput_qps=65.4
    )
    
    # List production models
    prod_models = registry.get_production_models()
    print(f"Production models: {len(prod_models)}")
