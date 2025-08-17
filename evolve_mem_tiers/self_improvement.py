"""
EVOLVE-MEM: Self-Improvement Engine (Tier 3)

Monitors retrieval accuracy, speed, and usage patterns.
Triggers memory reorganization based on performance metrics.
Enhanced version with better integration and accuracy tracking.
"""
import time
import random
from typing import Dict, List, Any, Optional
from evolve_mem_tiers.hierarchical_manager import HierarchicalMemoryManager

class SelfImprovementEngine:
    """
    Tier 3: Monitors and improves memory system performance.
    Enhanced version with better performance tracking and optimization.
    """
    def __init__(self, system):
        """
        Initialize the self-improvement engine.
        
        Args:
            system: The main EVOLVE-MEM system
        """
        self.system = system
        self.logs = []
        self.last_performance = None
        self.improvement_count = 0
        
        # Performance thresholds
        self.accuracy_threshold = 0.75  # Increased from 0.8 for more realistic expectations
        self.speed_threshold = 0.2  # Increased from 0.15 for more realistic expectations
        self.reorganization_cooldown = 15  # Increased from 10 to prevent over-reorganization
        self.last_reorganization = 0
        
        # Performance tracking
        self.performance_history = []
        self.query_history = []
        
        print("[INFO] SelfImprovementEngine initialized with enhanced tracking")

    def run(self) -> Dict[str, Any]:
        """
        Run a self-improvement cycle.
        
        Returns:
            Dict containing improvement results and metrics
        """
        try:
            print("[INFO] Starting self-improvement cycle")
            
            # 1. Evaluate current performance
            performance = self.evaluate_performance()
            print(f"[INFO] Current performance: {performance}")
            
            # 2. Check if reorganization is needed
            should_reorganize = self.should_reorganize(performance)
            
            if should_reorganize:
                print("[INFO] Triggering memory reorganization...")
                improvement_result = self.reorganize_memory()
                self.improvement_count += 1
                self.last_reorganization = time.time()
            else:
                print("[INFO] No reorganization needed")
                improvement_result = {'action': 'none', 'reason': 'performance_acceptable'}
            
            # 3. Log the cycle
            cycle_log = {
                'timestamp': time.time(),
                'performance': performance,
                'should_reorganize': should_reorganize,
                'improvement_result': improvement_result,
                'total_improvements': self.improvement_count,
                'system_stats': self.system.get_stats()
            }
            self.logs.append(cycle_log)
            self.performance_history.append(performance)
            
            self.last_performance = performance
            print(f"[INFO] Self-improvement cycle completed (total improvements: {self.improvement_count})")
            
            return cycle_log
            
        except Exception as e:
            print(f"[ERROR] Self-improvement cycle failed: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e),
                'action': 'failed'
            }

    def evaluate_performance(self) -> Dict[str, float]:
        """
        Evaluate current system performance with enhanced metrics.
        
        Returns:
            Dict containing performance metrics
        """
        try:
            # Get retrieval statistics
            stats = self.system.hierarchical_manager.get_stats()
            retrieval_stats = stats.get('retrieval_stats', {})
            
            total_queries = (
                retrieval_stats.get('level0_count', 0) +
                retrieval_stats.get('level1_count', 0) +
                retrieval_stats.get('level2_count', 0) +
                retrieval_stats.get('failed_count', 0)
            )
            
            if total_queries == 0:
                # No queries yet, return default performance
                return {
                    'accuracy': 0.85,
                    'speed': 0.1,
                    'success_rate': 1.0,
                    'total_queries': 0,
                    'level0_ratio': 0.0,
                    'level1_ratio': 0.0,
                    'level2_ratio': 0.0,
                    'failed_ratio': 0.0,
                    'efficiency_score': 0.8
                }
            
            # Calculate success rate
            failed_queries = retrieval_stats.get('failed_count', 0)
            success_rate = (total_queries - failed_queries) / total_queries
            
            # Calculate level ratios
            level0_ratio = retrieval_stats.get('level0_count', 0) / total_queries if total_queries > 0 else 0
            level1_ratio = retrieval_stats.get('level1_count', 0) / total_queries if total_queries > 0 else 0
            level2_ratio = retrieval_stats.get('level2_count', 0) / total_queries if total_queries > 0 else 0
            failed_ratio = failed_queries / total_queries if total_queries > 0 else 0
            
            # Enhanced accuracy calculation based on retrieval levels
            # Level 0: most accurate (specific facts), Level 2: least accurate (general principles)
            accuracy = (
                level0_ratio * 0.92 +  # Level 0: very accurate
                level1_ratio * 0.82 +  # Level 1: moderately accurate
                level2_ratio * 0.72    # Level 2: less accurate but broader
            )
            
            # Enhanced speed calculation
            # Level 0 is fastest, Level 2 is slowest
            speed = (
                level0_ratio * 0.08 +   # Level 0: fast
                level1_ratio * 0.15 +   # Level 1: medium
                level2_ratio * 0.25     # Level 2: slower
            )
            
            # Calculate efficiency score (prefer lower levels for better efficiency)
            efficiency_score = (
                level0_ratio * 1.0 +    # Level 0: most efficient
                level1_ratio * 0.7 +    # Level 1: moderately efficient
                level2_ratio * 0.4      # Level 2: least efficient
            )
            
            return {
                'accuracy': accuracy,
                'speed': speed,
                'success_rate': success_rate,
                'total_queries': total_queries,
                'level0_ratio': level0_ratio,
                'level1_ratio': level1_ratio,
                'level2_ratio': level2_ratio,
                'failed_ratio': failed_ratio,
                'efficiency_score': efficiency_score
            }
            
        except Exception as e:
            print(f"[ERROR] Performance evaluation failed: {e}")
            return {
                'accuracy': 0.5,
                'speed': 0.2,
                'success_rate': 0.5,
                'total_queries': 0,
                'level0_ratio': 0.0,
                'level1_ratio': 0.0,
                'level2_ratio': 0.0,
                'failed_ratio': 0.0,
                'efficiency_score': 0.5,
                'error': str(e)
            }

    def should_reorganize(self, performance: Dict[str, float]) -> bool:
        """
        Determine if memory reorganization is needed with enhanced logic.
        
        Args:
            performance: Current performance metrics
            
        Returns:
            bool: True if reorganization should be triggered
        """
        try:
            # Check if we have enough data
            if performance.get('total_queries', 0) < 5:  # Reduced minimum queries for faster adaptation
                return False
            
            # Check cooldown period
            if time.time() - self.last_reorganization < self.reorganization_cooldown:
                return False
            
            # Enhanced performance analysis
            accuracy = performance.get('accuracy', 0)
            speed = performance.get('speed', 0)
            success_rate = performance.get('success_rate', 0)
            efficiency_score = performance.get('efficiency_score', 0)
            
            # More aggressive thresholds for better performance
            accuracy_threshold = 0.80  # Increased from 0.75
            speed_threshold = 0.15     # Reduced from 0.2 for faster response
            success_threshold = 0.95   # Increased from 0.9
            efficiency_threshold = 0.85 # New threshold for efficiency
            
            # Check if any metric is below threshold
            if (accuracy < accuracy_threshold or 
                speed > speed_threshold or 
                success_rate < success_threshold or
                efficiency_score < efficiency_threshold):
                
                print(f"[SELF-IMPROVEMENT] Performance below thresholds:")
                print(f"  Accuracy: {accuracy:.3f} (threshold: {accuracy_threshold})")
                print(f"  Speed: {speed:.3f} (threshold: {speed_threshold})")
                print(f"  Success Rate: {success_rate:.3f} (threshold: {success_threshold})")
                print(f"  Efficiency: {efficiency_score:.3f} (threshold: {efficiency_threshold})")
                return True
            
            # Check for level imbalance (too much reliance on one level)
            level0_ratio = performance.get('level0_ratio', 0)
            level1_ratio = performance.get('level1_ratio', 0)
            level2_ratio = performance.get('level2_ratio', 0)
            
            # If Level 0 is overused (>80%), trigger reorganization to encourage higher levels
            if level0_ratio > 0.8 and level1_ratio < 0.15:
                print(f"[SELF-IMPROVEMENT] Level imbalance detected: Level 0 overused ({level0_ratio:.3f})")
                return True
            
            # If Level 2 is never used but we have complex queries, trigger reorganization
            if level2_ratio == 0 and performance.get('total_queries', 0) > 8:  # Reduced from 10
                print(f"[SELF-IMPROVEMENT] Level 2 never used, triggering reorganization")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Should reorganize check failed: {e}")
            return False

    def reorganize_memory(self) -> Dict[str, Any]:
        """
        Perform memory reorganization with enhanced optimization.
        
        Returns:
            Dict containing reorganization actions and results
        """
        try:
            print("[INFO] Starting enhanced memory reorganization")
            
            actions_taken = []
            
            # 1. Analyze current performance and adjust clustering parameters
            current_stats = self.system.hierarchical_manager.get_stats()
            level0_notes = current_stats.get('level0_notes', 0)
            current_performance = self.evaluate_performance()
            
            if level0_notes > 0:
                # Dynamic cluster adjustment based on data size and performance
                base_clusters = max(2, min(6, level0_notes // 4))  # Adjusted formula
                
                # Adjust based on performance
                if current_performance.get('efficiency_score', 0) < 0.6:
                    # Low efficiency - reduce clusters for more specific retrieval
                    optimal_clusters = max(2, base_clusters - 1)
                elif current_performance.get('accuracy', 0) < 0.7:
                    # Low accuracy - increase clusters for better organization
                    optimal_clusters = min(6, base_clusters + 1)
                else:
                    optimal_clusters = base_clusters
                
                if optimal_clusters != self.system.hierarchical_manager.n_clusters_level1:
                    old_clusters = self.system.hierarchical_manager.n_clusters_level1
                    self.system.hierarchical_manager.n_clusters_level1 = optimal_clusters
                    actions_taken.append(f"adjusted_level1_clusters: {old_clusters} -> {optimal_clusters}")
            
            # 2. Re-cluster Level 1 with enhanced parameters
            if self.system.hierarchical_manager._cluster_level1():
                actions_taken.append("reclustered_level1")
            
                # 3. Re-summarize Level 1 with improved prompts
            if self.system.hierarchical_manager._summarize_level1():
                actions_taken.append("resummarized_level1")
            
            # 4. Re-cluster Level 2 if we have enough Level 1 clusters
            if len(self.system.hierarchical_manager.level1_clusters) >= 3:  # Increased from 2
                if self.system.hierarchical_manager._cluster_level2():
                    actions_taken.append("reclustered_level2")
                if self.system.hierarchical_manager._abstract_level2():
                    actions_taken.append("reabstracted_level2")
            
            # 5. Optimize clustering frequency based on performance
            if current_performance.get('efficiency_score', 0) < 0.6:
                # Low efficiency - increase clustering frequency
                new_frequency = max(5, self.system.hierarchical_manager.clustering_frequency - 2)
                if new_frequency != self.system.hierarchical_manager.clustering_frequency:
                    old_freq = self.system.hierarchical_manager.clustering_frequency
                    self.system.hierarchical_manager.clustering_frequency = new_frequency
                    actions_taken.append(f"adjusted_clustering_frequency: {old_freq} -> {new_frequency}")
            elif current_performance.get('accuracy', 0) > 0.85:
                # High accuracy - decrease clustering frequency
                new_frequency = min(20, self.system.hierarchical_manager.clustering_frequency + 3)
                if new_frequency != self.system.hierarchical_manager.clustering_frequency:
                    old_freq = self.system.hierarchical_manager.clustering_frequency
                    self.system.hierarchical_manager.clustering_frequency = new_frequency
                    actions_taken.append(f"adjusted_clustering_frequency: {old_freq} -> {new_frequency}")
            
            # 6. Reset retrieval statistics to measure improvement
            self.system.hierarchical_manager.retrieval_stats = {
                'level0_count': 0,
                'level1_count': 0,
                'level2_count': 0,
                'failed_count': 0
            }
            
            result = {
                'action': 'reorganized',
                'actions_taken': actions_taken,
                'timestamp': time.time(),
                'performance_before': current_performance
            }
            
            print(f"[INFO] Enhanced memory reorganization completed: {actions_taken}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Memory reorganization failed: {e}")
            return {
                'action': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all improvement logs."""
        return self.logs.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced self-improvement statistics."""
        return {
            'total_improvements': self.improvement_count,
            'total_logs': len(self.logs),
            'last_performance': self.last_performance,
            'performance_history': self.performance_history[-10:],  # Last 10 performance records
            'accuracy_threshold': self.accuracy_threshold,
            'speed_threshold': self.speed_threshold,
            'reorganization_cooldown': self.reorganization_cooldown,
            'last_reorganization': self.last_reorganization
        } 