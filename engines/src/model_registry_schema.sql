-- Model Registry Schema for Inference Systems Laboratory
-- This SQLite schema provides a unified model registry accessible from both Python and C++
-- Supports versioning, lifecycle management, and metadata tracking for ML models

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Model metadata table
-- Stores core information about each registered model
CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,  -- Semantic versioning: major.minor.patch
    model_type TEXT NOT NULL,      -- 'onnx', 'tensorrt', 'pytorch', 'rule_based'
    backend TEXT NOT NULL,         -- 'RULE_BASED', 'TENSORRT_GPU', 'ONNX_RUNTIME', 'HYBRID_NEURAL_SYMBOLIC'
    
    -- File paths and storage
    model_path TEXT NOT NULL,      -- Path to model file
    config_path TEXT,               -- Optional path to configuration file
    metadata_path TEXT,             -- Optional path to metadata file
    
    -- Model characteristics
    input_shape TEXT,               -- JSON array of input dimensions
    output_shape TEXT,              -- JSON array of output dimensions
    input_dtype TEXT,               -- Data type of inputs (float32, int32, etc.)
    output_dtype TEXT,              -- Data type of outputs
    
    -- Performance metrics
    model_size_bytes INTEGER,       -- Size of model file in bytes
    estimated_memory_mb INTEGER,    -- Estimated memory usage in MB
    avg_inference_time_ms REAL,     -- Average inference time in milliseconds
    
    -- Lifecycle management
    status TEXT NOT NULL DEFAULT 'development',  -- 'development', 'staging', 'production', 'deprecated', 'archived'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    deprecated_at TIMESTAMP,
    
    -- Metadata
    description TEXT,
    author TEXT,
    tags TEXT,                      -- JSON array of tags
    custom_metadata TEXT,           -- JSON object for arbitrary metadata
    
    -- Constraints
    UNIQUE(model_name, model_version),
    CHECK(status IN ('development', 'staging', 'production', 'deprecated', 'archived')),
    CHECK(model_type IN ('onnx', 'tensorrt', 'pytorch', 'rule_based', 'other'))
);

-- Model deployment history
-- Tracks deployment lifecycle and rollback capability
CREATE TABLE IF NOT EXISTS deployment_history (
    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    
    -- Deployment details
    deployment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployment_status TEXT NOT NULL,  -- 'deployed', 'rolled_back', 'failed'
    previous_model_id INTEGER,        -- For tracking rollbacks
    deployment_environment TEXT,      -- 'development', 'staging', 'production'
    
    -- Performance at deployment
    initial_latency_ms REAL,
    initial_throughput_qps REAL,
    
    -- Deployment metadata
    deployed_by TEXT,
    deployment_notes TEXT,
    rollback_reason TEXT,
    
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
    FOREIGN KEY (previous_model_id) REFERENCES models(model_id) ON DELETE SET NULL
);

-- Model performance metrics
-- Tracks performance over time for monitoring and optimization
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    
    -- Timing information
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance metrics
    inference_time_ms REAL NOT NULL,
    preprocessing_time_ms REAL,
    postprocessing_time_ms REAL,
    
    -- Resource usage
    memory_usage_mb REAL,
    gpu_memory_mb REAL,
    cpu_utilization_percent REAL,
    gpu_utilization_percent REAL,
    
    -- Throughput metrics
    batch_size INTEGER DEFAULT 1,
    throughput_qps REAL,
    
    -- Context
    hardware_info TEXT,           -- JSON with CPU/GPU details
    inference_backend TEXT,
    optimization_level TEXT,      -- 'none', 'fp16', 'int8', etc.
    
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Model validation results
-- Stores validation and testing results for quality assurance
CREATE TABLE IF NOT EXISTS validation_results (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    
    -- Validation metadata
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validation_type TEXT NOT NULL,    -- 'accuracy', 'performance', 'correctness', 'regression'
    validation_dataset TEXT,
    
    -- Results
    validation_status TEXT NOT NULL,  -- 'passed', 'failed', 'warning'
    accuracy_score REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    custom_metrics TEXT,              -- JSON object for additional metrics
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    error_details TEXT,               -- JSON array of error descriptions
    
    -- Validation configuration
    validation_config TEXT,           -- JSON object with validation parameters
    validator_version TEXT,
    
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Model dependencies
-- Tracks dependencies between models for ensemble and pipeline scenarios
CREATE TABLE IF NOT EXISTS model_dependencies (
    dependency_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    depends_on_model_id INTEGER NOT NULL,
    
    dependency_type TEXT NOT NULL,    -- 'preprocessing', 'ensemble_member', 'feature_extractor', 'postprocessing'
    is_required BOOLEAN DEFAULT TRUE,
    load_order INTEGER,               -- For determining initialization sequence
    
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_model_id) REFERENCES models(model_id) ON DELETE RESTRICT,
    UNIQUE(model_id, depends_on_model_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at);
CREATE INDEX IF NOT EXISTS idx_deployment_model ON deployment_history(model_id);
CREATE INDEX IF NOT EXISTS idx_deployment_timestamp ON deployment_history(deployment_timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON performance_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON performance_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_validation_model ON validation_results(model_id);

-- Triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_model_timestamp 
AFTER UPDATE ON models
BEGIN
    UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE model_id = NEW.model_id;
END;

-- Views for common queries
-- Active production models
CREATE VIEW IF NOT EXISTS active_production_models AS
SELECT 
    m.*,
    MAX(d.deployment_timestamp) as last_deployed,
    COUNT(DISTINCT v.validation_id) as validation_count,
    AVG(p.inference_time_ms) as avg_latency_ms
FROM models m
LEFT JOIN deployment_history d ON m.model_id = d.model_id
LEFT JOIN validation_results v ON m.model_id = v.model_id
LEFT JOIN performance_metrics p ON m.model_id = p.model_id
WHERE m.status = 'production'
GROUP BY m.model_id;

-- Model performance summary
CREATE VIEW IF NOT EXISTS model_performance_summary AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_version,
    m.status,
    COUNT(p.metric_id) as metric_count,
    AVG(p.inference_time_ms) as avg_inference_ms,
    MIN(p.inference_time_ms) as min_inference_ms,
    MAX(p.inference_time_ms) as max_inference_ms,
    AVG(p.memory_usage_mb) as avg_memory_mb,
    AVG(p.throughput_qps) as avg_throughput_qps
FROM models m
LEFT JOIN performance_metrics p ON m.model_id = p.model_id
GROUP BY m.model_id;

-- Latest model versions
CREATE VIEW IF NOT EXISTS latest_model_versions AS
SELECT 
    model_name,
    MAX(model_version) as latest_version,
    COUNT(*) as version_count,
    GROUP_CONCAT(status) as all_statuses
FROM models
GROUP BY model_name;