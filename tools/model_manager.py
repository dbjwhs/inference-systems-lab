#!/usr/bin/env python3
"""
Model Manager - Version control and lifecycle management for ML models.

This tool provides comprehensive model registry functionality including:
- Model registration with semantic versioning
- Version listing and rollback capabilities
- Model validation and metadata extraction
- Integration with schema evolution system
- Lifecycle management (dev/staging/production)

Usage:
    # Register a new model
    python3 tools/model_manager.py register model.onnx --version 1.2.0 --stage dev
    
    # List all registered models
    python3 tools/model_manager.py list
    
    # Get model details
    python3 tools/model_manager.py info resnet50 --version 1.2.0
    
    # Promote model to production
    python3 tools/model_manager.py promote resnet50 --version 1.2.0 --to production
    
    # Rollback to previous version
    python3 tools/model_manager.py rollback resnet50 --to 1.1.0
    
    # Validate model file
    python3 tools/model_manager.py validate model.onnx
    
    # Compare two versions
    python3 tools/model_manager.py compare resnet50 --from 1.1.0 --to 1.2.0
    
    # Export registry to JSON
    python3 tools/model_manager.py export --output registry.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import tempfile

# Try to import ONNX for model validation (optional dependency)
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Model validation will be limited.", file=sys.stderr)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelFramework(Enum):
    """Supported ML frameworks."""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Model metadata structure."""
    name: str
    version: str
    framework: str
    stage: str
    file_path: str
    file_hash: str
    file_size: int
    input_shapes: Optional[str] = None  # JSON string
    output_shapes: Optional[str] = None  # JSON string
    data_types: Optional[str] = None  # JSON string
    metrics: Optional[str] = None  # JSON string
    description: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[str] = None  # Comma-separated
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    promoted_at: Optional[str] = None
    parent_version: Optional[str] = None


class ModelRegistry:
    """Model registry for version control and lifecycle management."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to registry database (default: ~/.inference-lab/models.db)
        """
        if registry_path is None:
            registry_dir = Path.home() / ".inference-lab"
            registry_dir.mkdir(parents=True, exist_ok=True)
            registry_path = registry_dir / "models.db"
        
        self.registry_path = registry_path
        self.models_dir = registry_path.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    input_shapes TEXT,
                    output_shapes TEXT,
                    data_types TEXT,
                    metrics TEXT,
                    description TEXT,
                    author TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    promoted_at TEXT,
                    parent_version TEXT,
                    UNIQUE(name, version)
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON models(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_stage ON models(stage)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_created ON models(created_at)")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex string of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _detect_framework(self, file_path: Path) -> ModelFramework:
        """Detect ML framework from file extension and content.
        
        Args:
            file_path: Path to model file
            
        Returns:
            Detected framework
        """
        suffix = file_path.suffix.lower()
        
        if suffix == ".onnx":
            return ModelFramework.ONNX
        elif suffix in [".engine", ".plan"]:
            return ModelFramework.TENSORRT
        elif suffix in [".pb", ".h5", ".keras"]:
            return ModelFramework.TENSORFLOW
        elif suffix in [".pt", ".pth", ".pkl"]:
            return ModelFramework.PYTORCH
        else:
            return ModelFramework.UNKNOWN
    
    def _extract_onnx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from ONNX model.
        
        Args:
            file_path: Path to ONNX model
            
        Returns:
            Dictionary with input_shapes, output_shapes, data_types
        """
        if not ONNX_AVAILABLE:
            return {}
        
        try:
            model = onnx.load(str(file_path))
            
            # Extract input information
            input_shapes = {}
            input_types = {}
            for input in model.graph.input:
                name = input.name
                shape = []
                for dim in input.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)  # Dynamic dimension
                input_shapes[name] = shape
                input_types[name] = input.type.tensor_type.elem_type
            
            # Extract output information
            output_shapes = {}
            output_types = {}
            for output in model.graph.output:
                name = output.name
                shape = []
                for dim in output.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)  # Dynamic dimension
                output_shapes[name] = shape
                output_types[name] = output.type.tensor_type.elem_type
            
            return {
                "input_shapes": json.dumps(input_shapes),
                "output_shapes": json.dumps(output_shapes),
                "data_types": json.dumps({"inputs": input_types, "outputs": output_types})
            }
        except Exception as e:
            print(f"Warning: Failed to extract ONNX metadata: {e}", file=sys.stderr)
            return {}
    
    def _validate_version(self, version: str) -> bool:
        """Validate semantic version format.
        
        Args:
            version: Version string (e.g., "1.2.0")
            
        Returns:
            True if valid semantic version
        """
        parts = version.split(".")
        if len(parts) != 3:
            return False
        
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False
    
    def register(self, model_path: Path, name: Optional[str] = None,
                 version: Optional[str] = None, stage: str = "dev",
                 description: Optional[str] = None, author: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 parent_version: Optional[str] = None) -> ModelMetadata:
        """Register a new model in the registry.
        
        Args:
            model_path: Path to model file
            name: Model name (default: filename without extension)
            version: Semantic version (default: auto-increment)
            stage: Lifecycle stage (dev/staging/production)
            description: Model description
            author: Model author
            tags: List of tags
            parent_version: Parent version for lineage tracking
            
        Returns:
            Registered model metadata
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine model name
        if name is None:
            name = model_path.stem
        
        # Validate or auto-generate version
        if version is None:
            version = self._get_next_version(name)
        elif not self._validate_version(version):
            raise ValueError(f"Invalid semantic version: {version}")
        
        # Check if version already exists
        existing = self.get_model(name, version)
        if existing:
            raise ValueError(f"Model {name} version {version} already exists")
        
        # Detect framework
        framework = self._detect_framework(model_path)
        
        # Copy model to registry storage
        model_dir = self.models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        dest_path = model_dir / model_path.name
        shutil.copy2(model_path, dest_path)
        
        # Compute file hash and size
        file_hash = self._compute_file_hash(dest_path)
        file_size = dest_path.stat().st_size
        
        # Extract model-specific metadata
        metadata_dict = {}
        if framework == ModelFramework.ONNX:
            metadata_dict.update(self._extract_onnx_metadata(dest_path))
        
        # Create metadata object
        now = datetime.now().isoformat()
        metadata = ModelMetadata(
            name=name,
            version=version,
            framework=framework.value,
            stage=stage,
            file_path=str(dest_path),
            file_hash=file_hash,
            file_size=file_size,
            input_shapes=metadata_dict.get("input_shapes"),
            output_shapes=metadata_dict.get("output_shapes"),
            data_types=metadata_dict.get("data_types"),
            description=description,
            author=author,
            tags=",".join(tags) if tags else None,
            created_at=now,
            updated_at=now,
            parent_version=parent_version
        )
        
        # Store in database
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                INSERT INTO models (
                    name, version, framework, stage, file_path, file_hash,
                    file_size, input_shapes, output_shapes, data_types,
                    description, author, tags, created_at, updated_at,
                    parent_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name, metadata.version, metadata.framework,
                metadata.stage, metadata.file_path, metadata.file_hash,
                metadata.file_size, metadata.input_shapes, metadata.output_shapes,
                metadata.data_types, metadata.description, metadata.author,
                metadata.tags, metadata.created_at, metadata.updated_at,
                metadata.parent_version
            ))
        
        return metadata
    
    def _get_next_version(self, name: str) -> str:
        """Get next auto-incremented version for a model.
        
        Args:
            name: Model name
            
        Returns:
            Next semantic version string
        """
        with sqlite3.connect(self.registry_path) as conn:
            cursor = conn.execute("""
                SELECT version FROM models
                WHERE name = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (name,))
            row = cursor.fetchone()
            
            if row is None:
                return "1.0.0"
            
            # Parse current version and increment patch
            parts = row[0].split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get model metadata from registry.
        
        Args:
            name: Model name
            version: Model version (default: latest)
            
        Returns:
            Model metadata or None if not found
        """
        with sqlite3.connect(self.registry_path) as conn:
            if version:
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE name = ? AND version = ?
                """, (name, version))
            else:
                cursor = conn.execute("""
                    SELECT * FROM models
                    WHERE name = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (name,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            # Convert row to ModelMetadata
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            data.pop("id", None)  # Remove internal ID
            return ModelMetadata(**data)
    
    def list_models(self, name_filter: Optional[str] = None,
                   stage_filter: Optional[str] = None,
                   tag_filter: Optional[str] = None) -> List[ModelMetadata]:
        """List all models in registry with optional filters.
        
        Args:
            name_filter: Filter by model name pattern
            stage_filter: Filter by lifecycle stage
            tag_filter: Filter by tag
            
        Returns:
            List of model metadata
        """
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if name_filter:
            query += " AND name LIKE ?"
            params.append(f"%{name_filter}%")
        
        if stage_filter:
            query += " AND stage = ?"
            params.append(stage_filter)
        
        if tag_filter:
            query += " AND tags LIKE ?"
            params.append(f"%{tag_filter}%")
        
        query += " ORDER BY name, created_at DESC"
        
        with sqlite3.connect(self.registry_path) as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            models = []
            for row in cursor:
                data = dict(zip(columns, row))
                data.pop("id", None)
                models.append(ModelMetadata(**data))
            
            return models
    
    def promote(self, name: str, version: str, to_stage: str) -> ModelMetadata:
        """Promote model to a different lifecycle stage.
        
        Args:
            name: Model name
            version: Model version
            to_stage: Target stage
            
        Returns:
            Updated model metadata
        """
        model = self.get_model(name, version)
        if not model:
            raise ValueError(f"Model {name} version {version} not found")
        
        # Validate stage transition
        valid_stages = [s.value for s in ModelStage]
        if to_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {to_stage}. Must be one of {valid_stages}")
        
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                UPDATE models
                SET stage = ?, updated_at = ?, promoted_at = ?
                WHERE name = ? AND version = ?
            """, (to_stage, now, now, name, version))
        
        model.stage = to_stage
        model.updated_at = now
        model.promoted_at = now
        return model
    
    def rollback(self, name: str, to_version: str) -> ModelMetadata:
        """Rollback to a previous model version by promoting it.
        
        Args:
            name: Model name
            to_version: Target version to rollback to
            
        Returns:
            Promoted model metadata
        """
        model = self.get_model(name, to_version)
        if not model:
            raise ValueError(f"Model {name} version {to_version} not found")
        
        # Find all production versions and archive them
        with sqlite3.connect(self.registry_path) as conn:
            cursor = conn.execute("""
                SELECT version FROM models
                WHERE name = ? AND stage = ?
            """, (name, ModelStage.PRODUCTION.value))
            
            production_versions = [row[0] for row in cursor]
            
        # Archive all current production versions except the target
        for version in production_versions:
            if version != to_version:
                self.promote(name, version, ModelStage.ARCHIVED.value)
        
        # Promote target version to production
        return self.promote(name, to_version, ModelStage.PRODUCTION.value)
    
    def _get_production_model(self, name: str) -> Optional[ModelMetadata]:
        """Get current production model for a given name.
        
        Args:
            name: Model name
            
        Returns:
            Production model metadata or None
        """
        with sqlite3.connect(self.registry_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM models
                WHERE name = ? AND stage = ?
                ORDER BY promoted_at DESC
                LIMIT 1
            """, (name, ModelStage.PRODUCTION.value))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            data.pop("id", None)
            return ModelMetadata(**data)
    
    def validate(self, model_path: Path) -> Dict[str, Any]:
        """Validate a model file before registration.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Validation results dictionary
        """
        model_path = Path(model_path)
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check file exists
        if not model_path.exists():
            results["valid"] = False
            results["errors"].append(f"File not found: {model_path}")
            return results
        
        # Check file size
        file_size = model_path.stat().st_size
        results["info"]["file_size"] = file_size
        if file_size == 0:
            results["valid"] = False
            results["errors"].append("File is empty")
            return results
        
        # Detect framework
        framework = self._detect_framework(model_path)
        results["info"]["framework"] = framework.value
        
        if framework == ModelFramework.UNKNOWN:
            results["warnings"].append("Unknown model framework")
        
        # Framework-specific validation
        if framework == ModelFramework.ONNX and ONNX_AVAILABLE:
            try:
                model = onnx.load(str(model_path))
                onnx.checker.check_model(model)
                results["info"]["onnx_valid"] = True
                
                # Extract metadata
                metadata = self._extract_onnx_metadata(model_path)
                if metadata:
                    results["info"].update(metadata)
            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"ONNX validation failed: {e}")
        
        # Compute hash for integrity
        results["info"]["file_hash"] = self._compute_file_hash(model_path)
        
        return results
    
    def compare(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        if not model1:
            raise ValueError(f"Model {name} version {version1} not found")
        if not model2:
            raise ValueError(f"Model {name} version {version2} not found")
        
        comparison = {
            "name": name,
            "versions": [version1, version2],
            "differences": {},
            "metrics_comparison": {}
        }
        
        # Compare basic metadata
        for field in ["framework", "file_size", "file_hash", "stage"]:
            val1 = getattr(model1, field)
            val2 = getattr(model2, field)
            if val1 != val2:
                comparison["differences"][field] = {
                    version1: val1,
                    version2: val2
                }
        
        # Compare shapes if available
        if model1.input_shapes and model2.input_shapes:
            shapes1 = json.loads(model1.input_shapes)
            shapes2 = json.loads(model2.input_shapes)
            if shapes1 != shapes2:
                comparison["differences"]["input_shapes"] = {
                    version1: shapes1,
                    version2: shapes2
                }
        
        # Compare metrics if available
        if model1.metrics and model2.metrics:
            metrics1 = json.loads(model1.metrics)
            metrics2 = json.loads(model2.metrics)
            comparison["metrics_comparison"] = {
                "version1": metrics1,
                "version2": metrics2
            }
        
        return comparison
    
    def export_registry(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export registry to JSON format.
        
        Args:
            output_path: Path to save JSON (optional)
            
        Returns:
            Registry data as dictionary
        """
        models = self.list_models()
        
        registry_data = {
            "version": "1.0.0",
            "exported_at": datetime.now().isoformat(),
            "model_count": len(models),
            "models": []
        }
        
        # Group models by name
        models_by_name = {}
        for model in models:
            if model.name not in models_by_name:
                models_by_name[model.name] = []
            models_by_name[model.name].append(asdict(model))
        
        registry_data["models"] = models_by_name
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(registry_data, f, indent=2)
        
        return registry_data
    
    def update_metrics(self, name: str, version: str, metrics: Dict[str, Any]):
        """Update model metrics.
        
        Args:
            name: Model name
            version: Model version
            metrics: Metrics dictionary
        """
        model = self.get_model(name, version)
        if not model:
            raise ValueError(f"Model {name} version {version} not found")
        
        metrics_json = json.dumps(metrics)
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                UPDATE models
                SET metrics = ?, updated_at = ?
                WHERE name = ? AND version = ?
            """, (metrics_json, now, name, version))


def main():
    """Main entry point for the model manager CLI."""
    parser = argparse.ArgumentParser(
        description="Model Manager - Version control and lifecycle management for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--registry",
        type=Path,
        help="Path to registry database (default: ~/.inference-lab/models.db)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("model_path", type=Path, help="Path to model file")
    register_parser.add_argument("--name", help="Model name (default: filename)")
    register_parser.add_argument("--version", help="Semantic version (default: auto-increment)")
    register_parser.add_argument("--stage", default="dev",
                                choices=["dev", "staging", "production"],
                                help="Lifecycle stage")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--author", help="Model author")
    register_parser.add_argument("--tags", nargs="+", help="Model tags")
    register_parser.add_argument("--parent-version", help="Parent version for lineage")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument("--name", help="Filter by name pattern")
    list_parser.add_argument("--stage", choices=["dev", "staging", "production", "archived"],
                           help="Filter by stage")
    list_parser.add_argument("--tag", help="Filter by tag")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get model details")
    info_parser.add_argument("name", help="Model name")
    info_parser.add_argument("--version", help="Model version (default: latest)")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to different stage")
    promote_parser.add_argument("name", help="Model name")
    promote_parser.add_argument("--version", required=True, help="Model version")
    promote_parser.add_argument("--to", required=True,
                              choices=["dev", "staging", "production", "archived"],
                              help="Target stage")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument("name", help="Model name")
    rollback_parser.add_argument("--to", required=True, help="Target version")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model file")
    validate_parser.add_argument("model_path", type=Path, help="Path to model file")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument("name", help="Model name")
    compare_parser.add_argument("--from", dest="version1", required=True, help="First version")
    compare_parser.add_argument("--to", dest="version2", required=True, help="Second version")
    compare_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export registry to JSON")
    export_parser.add_argument("--output", type=Path, help="Output file path")
    
    # Update metrics command
    metrics_parser = subparsers.add_parser("update-metrics", help="Update model metrics")
    metrics_parser.add_argument("name", help="Model name")
    metrics_parser.add_argument("--version", required=True, help="Model version")
    metrics_parser.add_argument("--metrics", required=True, help="JSON string of metrics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize registry
    registry = ModelRegistry(args.registry)
    
    try:
        if args.command == "register":
            metadata = registry.register(
                args.model_path,
                name=args.name,
                version=args.version,
                stage=args.stage,
                description=args.description,
                author=args.author,
                tags=args.tags,
                parent_version=args.parent_version
            )
            print(f"✅ Registered model: {metadata.name} v{metadata.version}")
            print(f"   Framework: {metadata.framework}")
            print(f"   Stage: {metadata.stage}")
            print(f"   Path: {metadata.file_path}")
            print(f"   Hash: {metadata.file_hash[:16]}...")
        
        elif args.command == "list":
            models = registry.list_models(
                name_filter=args.name,
                stage_filter=args.stage,
                tag_filter=args.tag
            )
            
            if args.json:
                print(json.dumps([asdict(m) for m in models], indent=2))
            else:
                if not models:
                    print("No models found")
                else:
                    # Group by name
                    by_name = {}
                    for model in models:
                        if model.name not in by_name:
                            by_name[model.name] = []
                        by_name[model.name].append(model)
                    
                    for name, versions in by_name.items():
                        print(f"\n📦 {name}")
                        for model in versions:
                            stage_emoji = {
                                "dev": "🔧",
                                "staging": "🔄",
                                "production": "✅",
                                "archived": "📦"
                            }.get(model.stage, "❓")
                            print(f"   {stage_emoji} v{model.version} ({model.stage}) - {model.created_at[:10]}")
        
        elif args.command == "info":
            model = registry.get_model(args.name, args.version)
            if not model:
                print(f"Model {args.name} not found", file=sys.stderr)
                return 1
            
            if args.json:
                print(json.dumps(asdict(model), indent=2))
            else:
                print(f"📦 Model: {model.name} v{model.version}")
                print(f"   Framework: {model.framework}")
                print(f"   Stage: {model.stage}")
                print(f"   File: {model.file_path}")
                print(f"   Size: {model.file_size:,} bytes")
                print(f"   Hash: {model.file_hash[:32]}...")
                print(f"   Created: {model.created_at}")
                print(f"   Updated: {model.updated_at}")
                
                if model.description:
                    print(f"   Description: {model.description}")
                if model.author:
                    print(f"   Author: {model.author}")
                if model.tags:
                    print(f"   Tags: {model.tags}")
                if model.input_shapes:
                    print(f"   Input shapes: {model.input_shapes}")
                if model.output_shapes:
                    print(f"   Output shapes: {model.output_shapes}")
        
        elif args.command == "promote":
            model = registry.promote(args.name, args.version, args.to)
            print(f"✅ Promoted {model.name} v{model.version} to {model.stage}")
        
        elif args.command == "rollback":
            model = registry.rollback(args.name, args.to)
            print(f"✅ Rolled back {model.name} to v{model.version} (now in {model.stage})")
        
        elif args.command == "validate":
            results = registry.validate(args.model_path)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                if results["valid"]:
                    print(f"✅ Model validation passed")
                else:
                    print(f"❌ Model validation failed")
                
                if results["errors"]:
                    print("\nErrors:")
                    for error in results["errors"]:
                        print(f"   ❌ {error}")
                
                if results["warnings"]:
                    print("\nWarnings:")
                    for warning in results["warnings"]:
                        print(f"   ⚠️  {warning}")
                
                if results["info"]:
                    print("\nInfo:")
                    for key, value in results["info"].items():
                        print(f"   {key}: {value}")
        
        elif args.command == "compare":
            comparison = registry.compare(args.name, args.version1, args.version2)
            
            if args.json:
                print(json.dumps(comparison, indent=2))
            else:
                print(f"📊 Comparing {args.name} v{args.version1} vs v{args.version2}")
                
                if comparison["differences"]:
                    print("\nDifferences:")
                    for field, values in comparison["differences"].items():
                        print(f"   {field}:")
                        for version, value in values.items():
                            print(f"      v{version}: {value}")
                else:
                    print("\nNo differences in metadata")
                
                if comparison["metrics_comparison"]:
                    print("\nMetrics:")
                    for version, metrics in comparison["metrics_comparison"].items():
                        print(f"   {version}: {metrics}")
        
        elif args.command == "export":
            data = registry.export_registry(args.output)
            if args.output:
                print(f"✅ Exported registry to {args.output}")
            else:
                print(json.dumps(data, indent=2))
        
        elif args.command == "update-metrics":
            try:
                metrics = json.loads(args.metrics)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON for metrics: {e}", file=sys.stderr)
                return 1
            
            registry.update_metrics(args.name, args.version, metrics)
            print(f"✅ Updated metrics for {args.name} v{args.version}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
