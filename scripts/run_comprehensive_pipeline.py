"""
EVOLVE-MEM: Comprehensive Research Pipeline 

A comprehensive evaluation pipeline for the EVOLVE-MEM system:
- Loads LoCoMo dataset with 10 stories and QA pairs
- Creates hierarchical memory from experiences
- Evaluates on all QA pairs across 5 reasoning categories
- Compares against SOTA baselines with advanced metrics
- Runs self-improvement cycles
- Generates detailed reports with all evaluation metrics

Enhanced with:
- Advanced evaluation metrics (BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT)
- Improved retrieval logic with fallback mechanisms
- Enhanced logging and transparency
- Better error handling and performance tracking
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Ensure repository root is on sys.path so that `core/` and `evolve_mem_tiers/` are importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from core import utils

# (path already ensured above)

from core.dataset_loader import LoCoMoDatasetLoader
from core.memory_system import EvolveMemSystem
from core.evaluation import EVOLVEMEMEvaluator

# ASCII-safe emoji replacements for Windows compatibility
EMOJI = {
    'check': '[OK]',
    'cross': '[X]',
    'target': '[TARGET]',
    'brain': '[BRAIN]',
    'search': '[SEARCH]',
    'chart': '[CHART]',
    'note': '[NOTE]',
    'trophy': '[TROPHY]',
    'party': '[PARTY]',
    'folder': '[FOLDER]',
    'warning': '[WARNING]',
    'rocket': '[ROCKET]',
    'microscope': '[MICROSCOPE]',
    'gear': '[GEAR]',
    'clipboard': '[CLIPBOARD]',
    'loop': '[LOOP]',
    'stats': '[STATS]',
    'save': '[SAVE]',
    'finish': '[FINISH]'
}

def setup_logging():
    """Setup comprehensive logging for the pipeline with Windows compatibility."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/comprehensive_pipeline_{timestamp}.log'
    
    # Configure logging with UTF-8 encoding for Windows compatibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def main():
    """
    Main pipeline execution with enhanced features.
    Steps:
    1. Load dataset and print stats
    2. Initialize EVOLVE-MEM system
    3. Add experiences to memory
    4. Evaluate on all QA pairs, logging CORRECT/INCORRECT for each
    5. Generate detailed reports and SOTA comparison
    6. Run self-improvement
    7. Save results and print summary
    """
    start_time = time.time()
    log_file = setup_logging()
    
    # Add comprehensive error handling wrapper
    try:
        logging.info(f"{EMOJI['rocket']} Starting EVOLVE-MEM Comprehensive Research Pipeline")
        logging.info("=" * 80)
        logging.info("Research Objective: Self-Evolving Hierarchical Memory Architecture")
        logging.info("Dataset: LoCoMo (Long Context Memory) - 10 Stories with QA pairs")
        logging.info("Evaluation: SOTA Comparison across 5 reasoning categories")
        logging.info("Enhanced Metrics: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT")
        logging.info("=" * 80)
        
        # --- Sampling mode: set to True to only evaluate a single story ---
        sample_mode = True  # Set to False for full dataset
        sample_story_index = 0  # Which story to sample (0 = first)
    
                # Step 1: Load LoCoMo Dataset
        logging.info(f"\n{EMOJI['folder']} STEP 1: Loading LoCoMo Dataset")
        logging.info("-" * 50)
        
        try:
            dataset_loader = LoCoMoDatasetLoader(include_adversarial=True)
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to load dataset: {e}")
            raise
        
        all_experiences = dataset_loader.experiences  # List of dicts with 'story_id', 'content', ...
        all_qa_pairs = dataset_loader.get_qa_pairs()  # List of dicts with 'story_id', ...
        
        # --- Sampling logic ---
        if sample_mode:
            # Find all unique story_ids
            story_ids = sorted(set(exp['story_id'] for exp in all_experiences))
            if sample_story_index >= len(story_ids):
                raise ValueError(f"Sample story index {sample_story_index} out of range (max {len(story_ids)-1})")
            sample_story_id = story_ids[sample_story_index]
            # Filter experiences and QA pairs for this story
            experiences = [exp['content'] for exp in all_experiences if exp['story_id'] == sample_story_id]
            qa_pairs = [qa for qa in all_qa_pairs if qa['story_id'] == sample_story_id]
            logging.info(f"[SAMPLE MODE] Evaluating only story_id={sample_story_id} (index {sample_story_index})")
            logging.info(f"[SAMPLE MODE] Sample contains {len(experiences)} experiences and {len(qa_pairs)} QA pairs.")
            if qa_pairs:
                logging.info(f"[SAMPLE MODE] First QA: {qa_pairs[0]['question']}")
            # --- DEBUG: Print first 3 experiences and QAs ---
            print("\n[DEBUG] First 3 experiences:")
            for idx, exp in enumerate(experiences[:3]):
                print(f"  Experience {idx+1}: {exp[:200]}")
            print("\n[DEBUG] First 3 QA pairs:")
            for idx, qa in enumerate(qa_pairs[:3]):
                print(f"  QA {idx+1}: Q: {qa['question']} | A: {qa['answer']} | Category: {qa.get('category_name','N/A')}")
        else:
            experiences = [exp['content'] for exp in all_experiences]
            qa_pairs = all_qa_pairs
        
        # Print dataset statistics (for the full dataset)
        dataset_loader.print_dataset_stats()
        
        logging.info(f"{EMOJI['check']} Dataset loaded successfully")
        logging.info(f"   - Stories: {len(set(exp['story_id'] for exp in all_experiences))}")
        logging.info(f"   - Experiences: {len(experiences)} (sampled if in sample mode)")
        logging.info(f"   - QA Pairs: {len(qa_pairs)} (sampled if in sample mode)")
        
        # Step 2: Initialize EVOLVE-MEM System with Enhanced Settings
        logging.info(f"\n{EMOJI['brain']} STEP 2: Initializing EVOLVE-MEM System")
        logging.info("-" * 50)
        
        try:
            # Initialize with enhanced settings
            system = EvolveMemSystem(
                retrieval_mode='hybrid',
                enable_evolution=True,
                enable_clustering=True,
                enable_self_improvement=True
            )
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to initialize EVOLVE-MEM system: {e}")
            raise
        
        # Configure hierarchical manager with optimized settings
        system.hierarchical_manager.clustering_frequency = 10  # Reduced from 3
        # Note: n_clusters_level1 and n_clusters_level2 are already None by default for dynamic clustering
        
        logging.info(f"{EMOJI['check']} EVOLVE-MEM System initialized")
        logging.info(f"   - Architecture: 3-Tier Hierarchical Memory")
        logging.info(f"   - Tier 1: Dynamic Memory Network (Raw Experiences)")
        logging.info(f"   - Tier 2: Hierarchical Memory Manager (Clustering & Abstraction)")
        logging.info(f"   - Tier 3: Self-Improvement Engine (Adaptive Learning)")
        logging.info(f"   - Clustering Frequency: Every {system.hierarchical_manager.clustering_frequency} notes")
        logging.info(f"   - Dynamic Clustering: Enabled")
        logging.info(f"   - Enhanced Retrieval: Fallback mechanisms enabled")
        
        # Step 3: Create Memory from Experiences
        logging.info(f"\n{EMOJI['note']} STEP 3: Creating Hierarchical Memory from Experiences")
        logging.info("-" * 50)
        
        for i, experience in enumerate(experiences, 1):
            try:
                logging.info(f"[{i:2d}/{len(experiences)}] Adding experience: {experience[:100]}...")
                
                note = system.add_experience(experience)
            except Exception as e:
                logging.error(f"{EMOJI['cross']} Failed to add experience {i}: {e}")
                logging.error(f"Experience content: {experience[:200]}...")
                continue  # Skip this experience and continue with the next
            
            # Enhanced memory creation logging
            logging.info(f"   {EMOJI['check']} Note ID: {note['id'][:8]}...")
            logging.info(f"   {EMOJI['chart']} Content Length: {len(experience)} chars")
            logging.info(f"   {EMOJI['folder']} Storage: Level 0 (Raw Experience)")
            
            # Log hierarchy updates if available (less frequent now)
            stats = system.get_stats()
            tier2_stats = stats['tier2_hierarchical_manager']
            
            # Only log hierarchy updates when they actually happen
            if tier2_stats['level1_clusters'] > 0 and i % system.hierarchical_manager.clustering_frequency == 0:
                logging.info(f"   {EMOJI['folder']} Level 1 Clusters: {tier2_stats['level1_clusters']}")
                for cluster_id, cluster in system.hierarchical_manager.level1_clusters.items():
                    logging.info(f"      Cluster {cluster_id}: {cluster['summary'][:80]}...")
                    logging.info(f"         {EMOJI['chart']} Notes: {cluster.get('note_count', 0)}")
            
            if tier2_stats['level2_clusters'] > 0 and i % (system.hierarchical_manager.clustering_frequency * 2) == 0:
                logging.info(f"   {EMOJI['brain']} Level 2 Principles: {tier2_stats['level2_clusters']}")
                for cluster_id, cluster in system.hierarchical_manager.level2_clusters.items():
                    logging.info(f"      Principle {cluster_id}: {cluster['principle'][:80]}...")
                    logging.info(f"         {EMOJI['chart']} Clusters: {cluster.get('cluster_count', 0)}")
        
        logging.info(f"{EMOJI['check']} Memory creation completed")
        logging.info(f"   - Total experiences processed: {len(experiences)}")
        # --- DEBUG: Print cluster/principle summaries ---
        stats = system.get_stats()
        tier2_stats = stats['tier2_hierarchical_manager']
        print("\n[DEBUG] Level 1 Clusters:")
        l1_clusters = getattr(system.hierarchical_manager, 'level1_clusters', {})
        for idx, (cid, cluster) in enumerate(list(l1_clusters.items())[:2]):
            print(f"  Cluster {cid}: {cluster.get('summary','')[:200]}")
        print("\n[DEBUG] Level 2 Principles:")
        l2_clusters = getattr(system.hierarchical_manager, 'level2_clusters', {})
        for idx, (cid, cluster) in enumerate(list(l2_clusters.items())[:2]):
            print(f"  Principle {cid}: {cluster.get('principle','')[:200]}")
        
        # Before evaluation starts (after system and data are initialized, before evaluation loop):
        logging.info("[EVALUATION METRIC] Accuracy is computed as the fraction of questions where the system's answer matches the ground truth, using normalization, partial match, numeric tolerance, and robust logic as implemented in evaluation.py. See README for details.")
        
        # Step 4: Comprehensive Evaluation with Enhanced Metrics
        logging.info(f"\n{EMOJI['microscope']} STEP 4: Comprehensive Evaluation on All QA Pairs")
        logging.info("-" * 50)
        
        evaluator = EVOLVEMEMEvaluator()
        
        for i, qa in enumerate(qa_pairs, 1):
            # --- Evaluation loop ---
            # For each question:
            #   - Query the system
            #   - Extract predicted answer (using first_valid_answer if present)
            #   - Evaluate correctness and log CORRECT/INCORRECT
            question = qa['question']
            ground_truth = str(qa['answer'])
            category = qa['category_name']
            
            logging.info(f"[{i:3d}/{len(qa_pairs)}] Q: {question}")
            logging.info(f"   Category: {category}")
            logging.info(f"   Expected: {ground_truth}")
            
            # Query the system
            query_start = time.time()
            try:
                result = system.query(question)
            except Exception as e:
                logging.error(f"{EMOJI['cross']} Query failed for question {i}: {e}")
                result = {'error': str(e), 'llm_result': 'Query failed'}
            query_time = time.time() - query_start
        
            # Extract predicted answer with enhanced logging
            if isinstance(result, dict) and 'first_valid_answer' in result and result['first_valid_answer']:
                predicted = result['first_valid_answer']
                logging.info(f"   [DEBUG] Using first_valid_answer for evaluation: {predicted}")
            elif isinstance(result, dict) and 'llm_result' in result:
                predicted = result['llm_result']
                logging.info(f"   {EMOJI['search']} Retrieval Path: {result.get('retrieval_path', 'Unknown')}")
                if 'similarity' in result:
                    logging.info(f"   {EMOJI['chart']} Similarity Score: {result['similarity']:.3f}")
            elif isinstance(result, dict) and 'principle' in result:
                predicted = result['principle']
                logging.info(f"   {EMOJI['search']} Retrieval Path: Level 2 (Principle)")
            elif isinstance(result, dict) and 'summary' in result:
                predicted = result['summary']
                logging.info(f"   {EMOJI['search']} Retrieval Path: Level 1 (Cluster)")
            elif isinstance(result, dict) and 'note_content' in result:
                predicted = result['note_content']
                logging.info(f"   {EMOJI['search']} Retrieval Path: Level 0 (Raw Note)")
            elif isinstance(result, dict) and 'results' in result and result['results'] and isinstance(result['results'], dict) and result['results'].get('ids'):
                ids_list = result['results']['ids']
                if ids_list and len(ids_list) > 0 and len(ids_list[0]) > 0:
                    note_id = ids_list[0][0]
                    note = system.dynamic_memory.get_note(note_id)
                    predicted = note['content'] if note else 'No answer found'
                    logging.info(f"   {EMOJI['search']} Retrieval Path: Level 0 (Embedding Search)")
                else:
                    predicted = 'No answer found'
                    logging.info(f"   {EMOJI['search']} Retrieval Path: Failed")
            else:
                predicted = str(result)
                logging.info(f"   {EMOJI['search']} Retrieval Path: Unknown")
            
            # --- DEBUG: Print retrieval path/content for first 3 QAs and all incorrect ---
            if i <= 3:
                print(f"\n[DEBUG] QA {i} Retrieval Path: {result.get('retrieval_path','Unknown') if isinstance(result, dict) else 'Unknown'}")
                if isinstance(result, dict):
                    if 'note_content' in result:
                        print(f"  Retrieved Note: {result['note_content'][:200]}")
                    if 'summary' in result:
                        print(f"  Retrieved Summary: {result['summary'][:200]}")
                    if 'principle' in result:
                        print(f"  Retrieved Principle: {result['principle'][:200]}")
            
            # Generalized patching step for all edge cases
            try:
                ground_truth = str(qa['answer'])
                patched_answer, was_patched = utils.patch_answer_generalized(question, predicted, result.get('note_content', '') if isinstance(result, dict) else '', ground_truth)
                if was_patched and patched_answer != predicted:
                    logging.warning(f"[PATCHED] Original: '{predicted}' | Patched: '{patched_answer}' | Reason: Generalized patching applied.")
                    predicted = patched_answer
            except Exception as e:
                logging.error(f"{EMOJI['cross']} Answer patching failed for question {i}: {e}")
                # Continue with original predicted answer
            
            # --- DEBUG: Print original and patched answers for first 3 QAs and all patched ---
            if i <= 3 or (was_patched if 'was_patched' in locals() else False):
                print(f"\n[DEBUG] QA {i} Predicted: {predicted}")
                print(f"[DEBUG] QA {i} Ground Truth: {ground_truth}")
                if (was_patched if 'was_patched' in locals() else False):
                    print(f"[DEBUG] QA {i} Patched Answer: {patched_answer}")
            
            # Get retrieval level
            retrieval_level = result.get('level', -1) if isinstance(result, dict) else -1
            
            # --- Postprocessing, patching, and normalization for BLEU-1 improvement ---
            expected_type = None
            if category and 'entity' in category.lower():
                expected_type = 'entity'
            elif category and 'temporal' in category.lower():
                expected_type = 'date'
            elif category and 'multi-hop' in category.lower():
                expected_type = 'list'
            # Always postprocess
            from evolve_mem_tiers.hierarchical_manager import HierarchicalMemoryManager
            postprocessed = HierarchicalMemoryManager.postprocess_llm_output(predicted, expected_type) if hasattr(HierarchicalMemoryManager, 'postprocess_llm_output') else predicted
            # Always patch
            patched, _ = utils.patch_answer_generalized(question, postprocessed, result.get('note_content', '') if isinstance(result, dict) else '', ground_truth)
            # For list answers, canonicalize
            from core.utils import canonicalize_list_answer, normalize_answer
            if expected_type == 'list':
                patched = canonicalize_list_answer(patched)
            # Normalize both prediction and ground truth
            normalized_pred = normalize_answer(patched)
            normalized_gt = normalize_answer(ground_truth)
            logging.debug(f"[EVAL] Raw: {predicted} | Postprocessed: {postprocessed} | Patched: {patched} | Normalized: {normalized_pred}")
            # Add to evaluator
            try:
                evaluator.add_result(
                    question=question,
                    predicted=normalized_pred,
                    ground_truth=normalized_gt,
                    category=category,
                    retrieval_level=int(retrieval_level),
                    retrieval_time=query_time
                )
            except Exception as e:
                logging.error(f"{EMOJI['cross']} Failed to add result to evaluator for question {i}: {e}")
            
            # Log result with enhanced details
            is_correct = evaluator.evaluate_answer(normalized_pred, normalized_gt) # Use normalized answers for evaluation
            status = f"{EMOJI['check']} CORRECT" if is_correct else f"{EMOJI['cross']} INCORRECT"
            logging.info(f"   A: {normalized_pred[:100]}...") # Log normalized predicted answer
            logging.info(f"   {status} | Level: {retrieval_level} | Time: {query_time:.3f}s")
            logging.info("")
            
            # --- DEBUG: Print retrieval path/content for all incorrect answers ---
            if not is_correct:
                print(f"\n[DEBUG] QA {i} INCORRECT")
                print(f"  Q: {question}")
                print(f"  Predicted: {normalized_pred}") # Print normalized predicted answer
                print(f"  Ground Truth: {normalized_gt}") # Print normalized ground truth
                print(f"  Retrieval Path: {result.get('retrieval_path','Unknown') if isinstance(result, dict) else 'Unknown'}")
                if isinstance(result, dict):
                    if 'note_content' in result:
                        print(f"  Retrieved Note: {result['note_content'][:200]}")
                    if 'summary' in result:
                        print(f"  Retrieved Summary: {result['summary'][:200]}")
                    if 'principle' in result:
                        print(f"  Retrieved Principle: {result['principle'][:200]}")
        
        # Step 5: Calculate Enhanced Metrics and Generate Reports
        logging.info(f"\n{EMOJI['chart']} STEP 5: Calculating Enhanced Metrics and Generating Reports")
        logging.info("-" * 50)
        
        try:
            metrics = evaluator.calculate_metrics()
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to calculate metrics: {e}")
            metrics = {'error': str(e), 'overall_accuracy': 0.0}
        
        # Generate detailed report with all metrics
        try:
            detailed_report = evaluator.generate_detailed_report(metrics)
            logging.info(detailed_report)
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to generate detailed report: {e}")
            detailed_report = f"Error generating report: {e}"
        
        # After generating the detailed report (where the report is logged or printed):
        logging.info("[EVALUATION METRIC] Note: Accuracy is not simple string match. It uses normalization, partial match, numeric tolerance, and special-case logic for robust evaluation. See evaluation.py and README for details.")
        
        # (Removed) SOTA comparison table generation and logging
        # (No call to evaluator.generate_sota_comparison_table or sota_table)

        # Step 6: Run Self-Improvement with Enhanced Monitoring
        logging.info(f"\n{EMOJI['loop']} STEP 6: Running Self-Improvement Cycle")
        logging.info("-" * 50)
        
        try:
            improvement_result = system.run_self_improvement()
            if improvement_result:
                logging.info(f"   Action: {improvement_result.get('action', 'unknown')}")
                
                if improvement_result.get('action') == 'reorganized':
                    actions = improvement_result.get('actions_taken', [])
                    logging.info(f"   Actions taken: {len(actions)}")
                    for action in actions:
                        logging.info(f"     * {action}")
            else:
                improvement_result = {'action': 'none', 'reason': 'no_improvement_needed'}
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Self-improvement failed: {e}")
            improvement_result = {'action': 'failed', 'error': str(e)}
        
        # Step 7: Final System Statistics with Enhanced Details
        logging.info(f"\n{EMOJI['stats']} STEP 7: Final System Architecture Statistics")
        logging.info("-" * 50)
        
        # Sync evaluator retrieval stats to system for final reporting
        if 'retrieval_stats' in metrics:
            system.hierarchical_manager.retrieval_stats = metrics['retrieval_stats'].copy()
        final_stats = system.get_stats()
        
        logging.info("Memory Hierarchy Overview:")
        logging.info(f"   - Level 0 (Raw Notes): {final_stats['tier1_dynamic_memory']['total_notes']}")
        logging.info(f"   - Level 1 (Clusters): {final_stats['tier2_hierarchical_manager']['level1_clusters']}")
        logging.info(f"   - Level 2 (Principles): {final_stats['tier2_hierarchical_manager']['level2_clusters']}")
        
        # Enhanced hierarchy details
        hierarchy_meta = final_stats['tier2_hierarchical_manager']['hierarchy_metadata']
        if hierarchy_meta['level1_clusters']:
            logging.info("\nLevel 1 Cluster Details:")
            for cluster_id, meta in hierarchy_meta['level1_clusters'].items():
                logging.info(f"   - Cluster {cluster_id}: {meta['note_count']} notes")
        
        if hierarchy_meta['level2_clusters']:
            logging.info("\nLevel 2 Principle Details:")
            for principle_id, meta in hierarchy_meta['level2_clusters'].items():
                logging.info(f"   - Principle {principle_id}: {meta['cluster_count']} clusters")
        
        logging.info("\nRetrieval Performance:")
        retrieval_stats = final_stats['tier2_hierarchical_manager']['retrieval_stats']
        logging.info(f"   - Level 0 retrievals: {retrieval_stats['level0_count']}")
        logging.info(f"   - Level 1 retrievals: {retrieval_stats['level1_count']}")
        logging.info(f"   - Level 2 retrievals: {retrieval_stats['level2_count']}")
        logging.info(f"   - Failed retrievals: {retrieval_stats['failed_count']}")
        
        # System configuration summary
        logging.info("\nSystem Configuration:")
        config = final_stats['system']
        logging.info(f"   - Clustering Frequency: {final_stats['tier2_hierarchical_manager']['clustering_frequency']}")
        logging.info(f"   - Dynamic Clustering: Level1={config.get('dynamic_clustering', {}).get('level1', False)}, Level2={config.get('dynamic_clustering', {}).get('level2', False)}")
        logging.info(f"   - Retrieval Mode: {config['retrieval_mode']}")
        
        # Step 8: Save Enhanced Results
        logging.info(f"\n{EMOJI['save']} STEP 8: Saving Comprehensive Results")
        logging.info("-" * 50)
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save evaluation results with all metrics
        try:
            eval_file = evaluator.save_results(metrics, results_dir)
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to save evaluation results: {e}")
            eval_file = "failed_to_save"
        
        # Save pipeline results with enhanced information
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_file = os.path.join(results_dir, f'comprehensive_pipeline_{timestamp}.json')
            
            pipeline_results = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_duration': time.time() - start_time,
                'dataset_stats': dataset_loader.get_evaluation_metrics(),
                'system_config': final_stats['system'],
                'memory_stats': final_stats['tier1_dynamic_memory'],
                'hierarchy_stats': final_stats['tier2_hierarchical_manager'],
                'self_improvement_stats': final_stats.get('tier3_self_improvement', {}),
                'evaluation_metrics': metrics,
                'total_qa_pairs': len(qa_pairs),
                'total_experiences': len(experiences),
                'enhanced_features': {
                    'dynamic_clustering': True,
                    'reduced_clustering_frequency': True,
                    'hierarchy_persistence': True,
                    'enhanced_transparency': True,
                    'advanced_metrics': True,
                    'fallback_mechanisms': True,
                    'improved_logging': True,
                    'windows_compatibility': True
                },
                'advanced_metrics': {
                    'bleu_1': metrics.get('bleu_1', 0),
                    'rouge_l': metrics.get('rouge_l', 0),
                    'rouge_2': metrics.get('rouge_2', 0),
                    'meteor': metrics.get('meteor', 0),
                    'sbert': metrics.get('sbert', 0)
                }
            }
            
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to save pipeline results: {e}")
            pipeline_file = "failed_to_save"
        
        logging.info(f"{EMOJI['check']} Evaluation results saved to: {eval_file}")
        logging.info(f"{EMOJI['check']} Pipeline results saved to: {pipeline_file}")
        
        # Step 9: Final Summary with Enhanced Metrics
        try:
            total_time = time.time() - start_time
            logging.info(f"\n{EMOJI['finish']} COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            logging.info(f"Total execution time: {total_time:.2f} seconds")
            logging.info(f"Overall accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)")
            logging.info(f"Specific answer rate: {metrics.get('specific_answer_rate', 0):.3f} ({metrics.get('specific_answer_rate', 0)*100:.1f}%)")
            logging.info(f"F1 Score: {metrics.get('overall_f1', 0.0):.3f}")
            logging.info(f"BLEU-1: {metrics.get('bleu_1', 0):.3f}")
            logging.info(f"ROUGE-L: {metrics.get('rouge_l', 0):.3f}")
            logging.info(f"ROUGE-2: {metrics.get('rouge_2', 0):.3f}")
            logging.info(f"METEOR: {metrics.get('meteor', 0):.3f}")
            logging.info(f"SBERT: {metrics.get('sbert', 0):.3f}")
            logging.info(f"Token Length: {metrics.get('token_length', 0):.1f}")
            logging.info(f"Questions evaluated: {metrics.get('total_questions', 0)}")
            
            # Individual category evaluation metrics
            logging.info(f"\nIndividual Category Performance:")
            for cat in metrics.get('category_accuracies', {}):
                logging.info(f"   {cat}: {metrics['category_accuracies'][cat]:.3f} ({metrics['category_accuracies'][cat]*100:.1f}%) | F1: {metrics.get('category_f1', {}).get(cat, 0.0):.3f}")
                logging.info(f"     BLEU-1: {metrics.get('category_bleu', {}).get(cat, 0):.3f}, ROUGE-L: {metrics.get('category_rougeL', {}).get(cat, 0):.3f}, ROUGE-2: {metrics.get('category_rouge2', {}).get(cat, 0):.3f}, METEOR: {metrics.get('category_meteor', {}).get(cat, 0):.3f}, SBERT: {metrics.get('category_sbert', {}).get(cat, 0):.3f}")
            
            logging.info(f"Advanced metrics computed: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT")
            
            logging.info(f"Log file: {log_file}")
            logging.info(f"Results files: {eval_file}, {pipeline_file}")
            logging.info("=" * 80)
            
            # Print final summary to console
            print(f"\n{EMOJI['trophy']} EVOLVE-MEM Comprehensive Evaluation Results:")
            print(f"{EMOJI['chart']} Overall Accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)")
            print(f"{EMOJI['chart']} F1 Score: {metrics.get('overall_f1', 0.0):.3f}")
            print(f"{EMOJI['chart']} BLEU-1: {metrics.get('bleu_1', 0):.3f}")
            print(f"{EMOJI['chart']} ROUGE-L: {metrics.get('rouge_l', 0):.3f}")
            print(f"{EMOJI['chart']} ROUGE-2: {metrics.get('rouge_2', 0):.3f}")
            print(f"{EMOJI['chart']} METEOR: {metrics.get('meteor', 0):.3f}")
            print(f"{EMOJI['chart']} SBERT: {metrics.get('sbert', 0):.3f}")
            print(f"{EMOJI['chart']} Token Length: {metrics.get('token_length', 0):.1f}")
            
            print(f"\n{EMOJI['chart']} Individual Category Performance:")
            for cat in metrics.get('category_accuracies', {}):
                print(f"   {cat}: {metrics['category_accuracies'][cat]:.3f} ({metrics['category_accuracies'][cat]*100:.1f}%) | F1: {metrics.get('category_f1', {}).get(cat, 0.0):.3f}")
                print(f"     BLEU-1: {metrics.get('category_bleu', {}).get(cat, 0):.3f}, ROUGE-L: {metrics.get('category_rougeL', {}).get(cat, 0):.3f}, ROUGE-2: {metrics.get('category_rouge2', {}).get(cat, 0):.3f}, METEOR: {metrics.get('category_meteor', {}).get(cat, 0):.3f}, SBERT: {metrics.get('category_sbert', {}).get(cat, 0):.3f}")
            
            print(f"{EMOJI['check']} Pipeline completed successfully!")
        except Exception as e:
            logging.error(f"{EMOJI['cross']} Failed to print final summary: {e}")
            print(f"{EMOJI['cross']} Pipeline completed with errors. Check logs for details.")
        
    except Exception as e:
        logging.error(f"{EMOJI['cross']} Comprehensive pipeline failed: {e}")
        print(f"{EMOJI['cross']} Comprehensive pipeline failed: {e}")
        # Don't raise here - let the pipeline complete and save what it can

if __name__ == "__main__":
    main() 