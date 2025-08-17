"""
EVOLVE-MEM: Configuration

Central configuration for all system parameters.
"""
import os
from typing import Dict, Any
import logging

class EVOLVEMConfig:
    """Configuration class for EVOLVE-MEM system."""
    
    # LLM Configuration
    LLM_BACKEND = os.getenv('LLM_BACKEND', 'gemini')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-1.5-flash')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'thenlper/gte-small')
    EMBEDDING_DIMENSION = 384  # GTE-Small dimension
    
    # Memory System Configuration
    RETRIEVAL_MODE = os.getenv('RETRIEVAL_MODE', 'hybrid')  # 'embedding', 'hybrid', 'context'
    ENABLE_EVOLUTION = os.getenv('ENABLE_EVOLUTION', 'true').lower() == 'true'
    ENABLE_CLUSTERING = os.getenv('ENABLE_CLUSTERING', 'true').lower() == 'true'
    ENABLE_SELF_IMPROVEMENT = os.getenv('ENABLE_SELF_IMPROVEMENT', 'true').lower() == 'true'
    
    # Hierarchical Configuration
    DEFAULT_LEVEL1_CLUSTERS = int(os.getenv('DEFAULT_LEVEL1_CLUSTERS', '3'))
    DEFAULT_LEVEL2_CLUSTERS = int(os.getenv('DEFAULT_LEVEL2_CLUSTERS', '1'))
    MIN_NOTES_FOR_CLUSTERING = int(os.getenv('MIN_NOTES_FOR_CLUSTERING', '3'))
    
    # Self-Improvement Configuration
    ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', '0.8'))
    SPEED_THRESHOLD = float(os.getenv('SPEED_THRESHOLD', '0.15'))
    REORGANIZATION_COOLDOWN = int(os.getenv('REORGANIZATION_COOLDOWN', '10'))
    
    # Evaluation Configuration
    DEFAULT_DATASET_RATIO = float(os.getenv('DEFAULT_DATASET_RATIO', '0.1'))
    DEFAULT_N_RUNS = int(os.getenv('DEFAULT_N_RUNS', '3'))
    DATASET_PATH = os.getenv('DATASET_PATH', '../AgenticMemory/data/locomo10.json')
    
    # Performance Configuration
    MAX_SUMMARY_LENGTH = int(os.getenv('MAX_SUMMARY_LENGTH', '1024'))
    MAX_TOKENS_SUMMARY = int(os.getenv('MAX_TOKENS_SUMMARY', '60'))
    MAX_TOKENS_ABSTRACTION = int(os.getenv('MAX_TOKENS_ABSTRACTION', '40'))
    SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))
    
    # Storage Configuration
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'evolve_mem_dynamic')
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        errors = []
        
        # Check required API keys
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required for Gemini backend")
        
        # Check numeric ranges
        if cls.DEFAULT_LEVEL1_CLUSTERS < 1:
            errors.append("DEFAULT_LEVEL1_CLUSTERS must be >= 1")
        if cls.DEFAULT_LEVEL2_CLUSTERS < 1:
            errors.append("DEFAULT_LEVEL2_CLUSTERS must be >= 1")
        if cls.ACCURACY_THRESHOLD < 0 or cls.ACCURACY_THRESHOLD > 1:
            errors.append("ACCURACY_THRESHOLD must be between 0 and 1")
        if cls.SPEED_THRESHOLD < 0:
            errors.append("SPEED_THRESHOLD must be >= 0")
        
        if errors:
            logging.error("Configuration validation failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def get_system_config(cls) -> Dict[str, Any]:
        """Get configuration for system initialization."""
        return {
            'retrieval_mode': cls.RETRIEVAL_MODE,
            'enable_evolution': cls.ENABLE_EVOLUTION,
            'enable_clustering': cls.ENABLE_CLUSTERING,
            'enable_self_improvement': cls.ENABLE_SELF_IMPROVEMENT
        }
    
    @classmethod
    def get_hierarchical_config(cls) -> Dict[str, Any]:
        """Get configuration for hierarchical manager."""
        return {
            'n_clusters_level1': cls.DEFAULT_LEVEL1_CLUSTERS,
            'n_clusters_level2': cls.DEFAULT_LEVEL2_CLUSTERS,
            'min_notes_for_clustering': cls.MIN_NOTES_FOR_CLUSTERING
        }
    
    @classmethod
    def get_evaluation_config(cls) -> Dict[str, Any]:
        """Get configuration for evaluation."""
        return {
            'dataset_path': cls.DATASET_PATH,
            'ratio': cls.DEFAULT_DATASET_RATIO,
            'n_runs': cls.DEFAULT_N_RUNS
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("EVOLVE-MEM Configuration:")
        print("=" * 40)
        print(f"LLM Backend: {cls.LLM_BACKEND}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Retrieval Mode: {cls.RETRIEVAL_MODE}")
        print(f"Enable Evolution: {cls.ENABLE_EVOLUTION}")
        print(f"Enable Clustering: {cls.ENABLE_CLUSTERING}")
        print(f"Enable Self-Improvement: {cls.ENABLE_SELF_IMPROVEMENT}")
        print(f"Level 1 Clusters: {cls.DEFAULT_LEVEL1_CLUSTERS}")
        print(f"Level 2 Clusters: {cls.DEFAULT_LEVEL2_CLUSTERS}")
        print(f"Accuracy Threshold: {cls.ACCURACY_THRESHOLD}")
        print(f"Speed Threshold: {cls.SPEED_THRESHOLD}")
        print("=" * 40) 
