"""
EVOLVE-MEM: Core Memory System

Implements the three-tier architecture:
- Tier 1: DynamicMemoryNetwork
- Tier 2: HierarchicalMemoryManager (with 3-level hierarchy)
- Tier 3: SelfImprovementEngine

Simplified version focused on core functionality.
"""
from evolve_mem_tiers.dynamic_memory import DynamicMemoryNetwork
from evolve_mem_tiers.hierarchical_manager import HierarchicalMemoryManager
from evolve_mem_tiers.self_improvement import SelfImprovementEngine
from core.llm_backend import get_llm_backend, BaseLLM
from typing import Dict, List, Optional, Any
import logging

class EvolveMemSystem:
    """
    The main orchestrator for EVOLVE-MEM, integrating all tiers.
    Simplified version focused on core functionality.
    """
    def __init__(self, 
                 llm_backend: Optional[BaseLLM] = None, 
                 retrieval_mode: str = 'hybrid',
                 enable_evolution: bool = True, 
                 enable_clustering: bool = True, 
                 enable_self_improvement: bool = True):
        """
        Initialize the EVOLVE-MEM system.
        
        Args:
            llm_backend: LLM backend for summarization and abstraction
            retrieval_mode: 'embedding', 'hybrid', or 'context'
            enable_evolution: Enable memory evolution
            enable_clustering: Enable hierarchical clustering
            enable_self_improvement: Enable self-improvement engine
        """
        logging.info("Initializing EVOLVE-MEM system...")
        
        # Initialize Tier 1: Dynamic Memory Network
        self.dynamic_memory = DynamicMemoryNetwork()
        
        # Initialize LLM backend
        self.llm_backend = llm_backend or get_llm_backend("gemini")
        
        # Configuration
        self.retrieval_mode = retrieval_mode
        self.enable_evolution = enable_evolution
        self.enable_clustering = enable_clustering
        self.enable_self_improvement = enable_self_improvement
        
        # Initialize Tier 2: Hierarchical Memory Manager
        self.hierarchical_manager = HierarchicalMemoryManager(
            self.dynamic_memory, 
            llm=self.llm_backend, 
            retrieval_mode=self.retrieval_mode
        )
        
        # Initialize Tier 3: Self-Improvement Engine
        if self.enable_self_improvement:
            self.self_improvement = SelfImprovementEngine(self)
        else:
            self.self_improvement = None
        
        # System statistics
        self.total_experiences = 0
        self.total_queries = 0
        
        logging.info(f"EVOLVE-MEM system initialized successfully")
        logging.info(f"Configuration: evolution={enable_evolution}, clustering={enable_clustering}, self_improvement={enable_self_improvement}")

    def add_experience(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add a new raw experience to the system (Tier 1).
        
        Args:
            text: The experience text to add
            metadata: Optional metadata for the experience
            
        Returns:
            Dict: The created memory note
        """
        try:
            if not text or not text.strip():
                raise ValueError("Experience text cannot be empty")
            
            logging.info(f"Adding experience: {text[:50]}...")
            
            # Add to Tier 1
            note = self.dynamic_memory.add_note(text, metadata)
            self.total_experiences += 1
            
            # Update hierarchy if evolution is enabled
            if self.enable_evolution and self.enable_clustering:
                success = self.hierarchical_manager.update_hierarchy(note)
                if success:
                    logging.info(f"Hierarchy updated for experience {note['id']}")
                else:
                    logging.warning(f"Hierarchy update failed for experience {note['id']}")
            
            logging.info(f"Experience added successfully (total: {self.total_experiences})")
            return note
            
        except Exception as e:
            logging.error(f"Failed to add experience: {e}")
            raise

    def query(self, query_text: str, category: str = None) -> Dict[str, Any]:
        """
        Query the memory system using multi-level retrieval.
        
        Args:
            query_text: The query string
            category: The reasoning category (optional)
            
        Returns:
            Dict: Query results and metadata
        """
        try:
            if not query_text or not query_text.strip():
                return {'error': 'Empty query', 'level': -1}
            
            logging.info(f"Processing query: {query_text[:50]}...")
            
            # Query the hierarchical manager with category and self-improvement
            result = self.hierarchical_manager.retrieve(query_text, category=category, self_improvement_engine=self.self_improvement)
            self.total_queries += 1
            
            # Add system metadata
            result['system_metadata'] = {
                'total_experiences': self.total_experiences,
                'total_queries': self.total_queries,
                'retrieval_mode': self.retrieval_mode,
                'enable_evolution': self.enable_evolution,
                'enable_clustering': self.enable_clustering,
                'enable_self_improvement': self.enable_self_improvement
            }
            
            logging.info(f"Query processed successfully (level: {result.get('level', -1)})")
            return result
            
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return {
                'error': str(e),
                'level': -1,
                'system_metadata': {
                    'total_experiences': self.total_experiences,
                    'total_queries': self.total_queries
                }
            }

    def run_self_improvement(self) -> Optional[Dict[str, Any]]:
        """
        Trigger the self-improvement engine (Tier 3).
        
        Returns:
            Dict: Improvement results or None if disabled
        """
        if not self.self_improvement:
            logging.info("Self-improvement engine is disabled")
            return None
        
        try:
            logging.info("Running self-improvement cycle...")
            result = self.self_improvement.run()
            logging.info(f"Self-improvement cycle completed: {result.get('action', 'unknown')}")
            return result
            
        except Exception as e:
            logging.error(f"Self-improvement failed: {e}")
            return {'error': str(e), 'action': 'failed'}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict: System statistics from all tiers
        """
        try:
            # Get stats from each tier
            tier1_stats = self.dynamic_memory.get_stats()
            tier2_stats = self.hierarchical_manager.get_stats()
            tier3_stats = self.self_improvement.get_stats() if self.self_improvement else {}
            
            # Combine all stats
            system_stats = {
                'system': {
                    'total_experiences': self.total_experiences,
                    'total_queries': self.total_queries,
                    'retrieval_mode': self.retrieval_mode,
                    'enable_evolution': self.enable_evolution,
                    'enable_clustering': self.enable_clustering,
                    'enable_self_improvement': self.enable_self_improvement
                },
                'tier1_dynamic_memory': tier1_stats,
                'tier2_hierarchical_manager': tier2_stats,
                'tier3_self_improvement': tier3_stats
            }
            
            return system_stats
            
        except Exception as e:
            logging.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}

    def clear(self):
        """Clear all memories and reset the system (for testing)."""
        try:
            logging.info("Clearing all memories...")
            self.dynamic_memory.clear()
            self.hierarchical_manager.level1_clusters.clear()
            self.hierarchical_manager.level2_clusters.clear()
            self.hierarchical_manager.retrieval_stats = {
                'level0_count': 0,
                'level1_count': 0,
                'level2_count': 0,
                'failed_count': 0
            }
            self.total_experiences = 0
            self.total_queries = 0
            logging.info("System cleared successfully")
            
        except Exception as e:
            logging.error(f"Failed to clear system: {e}")

    def add_batch_experiences(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Add multiple experiences in batch.
        
        Args:
            texts: List of experience texts
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List[Dict]: List of created memory notes
        """
        try:
            if not texts:
                return []
            
            logging.info(f"Adding batch of {len(texts)} experiences...")
            
            notes = []
            for i, text in enumerate(texts):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                try:
                    note = self.add_experience(text, metadata)
                    notes.append(note)
                except Exception as e:
                    logging.warning(f"Failed to add experience {i}: {e}")
                    continue
            
            logging.info(f"Batch addition completed: {len(notes)}/{len(texts)} experiences added")
            return notes
            
        except Exception as e:
            logging.error(f"Batch addition failed: {e}")
            return [] 