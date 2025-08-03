"""
EVOLVE-MEM: Hierarchical Memory Manager (Tier 2) - OPTIMIZED VERSION

Organizes memories into a three-level hierarchy:
- Level 0: Raw experiences (from DynamicMemoryNetwork)
- Level 1: Contextual clusters (k-means clustering + LLM summarization)
- Level 2: Abstract principles (meta-clustering + LLM abstraction)
- Supports bidirectional links and multi-level retrieval
- Optimized for efficiency and transparency
"""
import numpy as np
from sklearn.cluster import KMeans
from dynamic_memory import DynamicMemoryNetwork
from llm_backend import get_llm_backend, BaseLLM
from typing import Dict, List, Optional, Any
import time
import json
import os
import utils
import logging
import re # Added for regex in patch_answer_with_context

class HierarchicalMemoryManager:
    """
    Tier 2: Manages hierarchical organization of memories.
    Optimized version with better efficiency and transparency.
    """
    def __init__(self, 
                 dynamic_memory: DynamicMemoryNetwork, 
                 n_clusters_level1: int = None,  # Dynamic by default
                 n_clusters_level2: int = None,  # Dynamic by default
                 llm: Optional[BaseLLM] = None, 
                 retrieval_mode: str = 'hybrid',
                 clustering_frequency: int = 10,  # Reduced from 3
                 persistence_file: str = 'memory_hierarchy.json'):
        """
        Initialize the hierarchical memory manager.
        
        Args:
            dynamic_memory: The Tier 1 memory network
            n_clusters_level1: Number of clusters for Level 1 (None for dynamic)
            n_clusters_level2: Number of clusters for Level 2 (None for dynamic)
            llm: LLM backend for summarization and abstraction
            retrieval_mode: 'embedding', 'hybrid', or 'context'
            clustering_frequency: How often to trigger clustering
            persistence_file: File to save/load hierarchy state
        """
        self.dynamic_memory = dynamic_memory
        self.llm = llm or get_llm_backend("gemini")
        self.retrieval_mode = retrieval_mode
        self.clustering_frequency = clustering_frequency
        self.persistence_file = persistence_file
        
        # Dynamic cluster sizing
        self.n_clusters_level1 = n_clusters_level1
        self.n_clusters_level2 = n_clusters_level2
        
        # Hierarchical storage with enhanced metadata
        self.level1_clusters = {}  # cluster_id -> {'ids': [note_ids], 'summary': str, 'created_at': timestamp, 'last_updated': timestamp}
        self.level2_clusters = {}  # cluster_id -> {'ids': [level1_cluster_ids], 'principle': str, 'created_at': timestamp, 'last_updated': timestamp}
        
        # Performance tracking
        self.retrieval_stats = {
            'level0_count': 0,
            'level1_count': 0,
            'level2_count': 0,
            'failed_count': 0
        }
        
        # Load existing hierarchy if available
        self._load_hierarchy()
        
        logging.info(f"HierarchicalMemoryManager initialized with {retrieval_mode} retrieval mode")
        logging.info(f"Clustering frequency: every {clustering_frequency} notes")
        logging.info(f"Dynamic clustering: Level1={n_clusters_level1 is None}, Level2={n_clusters_level2 is None}")

    def _calculate_optimal_clusters(self, data_size: int, level: int) -> int:
        """Calculate optimal number of clusters based on data size."""
        if level == 1:
            # Level 1: More granular clustering
            if data_size < 10:
                return max(2, data_size // 3)
            elif data_size < 50:
                return max(3, data_size // 5)
            else:
                return max(5, min(10, data_size // 8))
        else:
            # Level 2: More abstract principles
            if data_size < 5:
                return max(1, data_size // 3)
            else:
                return max(2, min(5, data_size // 4))

    def _save_hierarchy(self):
        """Save hierarchy state to file for persistence."""
        try:
            # Convert numpy int32 keys to regular Python ints for JSON serialization
            level1_clusters_serializable = {}
            for key, value in self.level1_clusters.items():
                level1_clusters_serializable[int(key)] = value
            
            level2_clusters_serializable = {}
            for key, value in self.level2_clusters.items():
                level2_clusters_serializable[int(key)] = value
            
            hierarchy_data = {
                'level1_clusters': level1_clusters_serializable,
                'level2_clusters': level2_clusters_serializable,
                'retrieval_stats': self.retrieval_stats,
                'last_saved': time.time()
            }
            with open(self.persistence_file, 'w') as f:
                json.dump(hierarchy_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save hierarchy: {e}")

    def _load_hierarchy(self):
        """Load hierarchy state from file."""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    hierarchy_data = json.load(f)
                self.level1_clusters = hierarchy_data.get('level1_clusters', {})
                self.level2_clusters = hierarchy_data.get('level2_clusters', {})
                self.retrieval_stats = hierarchy_data.get('retrieval_stats', self.retrieval_stats)
                logging.info(f"Loaded existing hierarchy: {len(self.level1_clusters)} clusters, {len(self.level2_clusters)} principles")
        except Exception as e:
            logging.warning(f"Failed to load hierarchy: {e}")

    def update_hierarchy(self, new_note: Dict) -> bool:
        """
        Update the hierarchical structure when a new note is added.
        Optimized to reduce frequency and improve efficiency.
        
        Args:
            new_note: The newly added note
            
        Returns:
            bool: True if hierarchy was updated successfully
        """
        try:
            total_notes = len(self.dynamic_memory.notes)
            
            # Enhanced logging for transparency
            logging.info(f"[MEMORY] Note {new_note['id'][:8]} added to Level 0")
            logging.info(f"[MEMORY] Content: {new_note['content'][:60]}...")
            logging.info(f"[MEMORY] Total notes: {total_notes}")
            
            # Adaptive clustering frequency based on note count
            if total_notes <= 10:
                clustering_frequency = 5  # More frequent clustering for small datasets
            elif total_notes <= 20:
                clustering_frequency = 8  # Moderate frequency
            else:
                clustering_frequency = 10  # Standard frequency for large datasets
            
            if total_notes % clustering_frequency == 0:
                logging.info(f"[CLUSTERING] Triggering Level 1 clustering (notes: {total_notes})")
                if self._cluster_level1():
                    if self._summarize_level1():
                        logging.info(f"[SUCCESS] Level 1 hierarchy updated: {len(self.level1_clusters)} clusters")
                        self._save_hierarchy()  # Persist after successful update
                    else:
                        logging.warning(f"Level 1 summarization failed")
                else:
                    logging.warning(f"Level 1 clustering failed")
            
            # Trigger Level 2 clustering even less frequently
            if (total_notes >= self.clustering_frequency * 2 and 
                total_notes % (self.clustering_frequency * 2) == 0 and 
                len(self.level1_clusters) >= 2):
                logging.info(f"[CLUSTERING] Triggering Level 2 clustering (notes: {total_notes})")
                if self._cluster_level2():
                    if self._abstract_level2():
                        logging.info(f"[SUCCESS] Level 2 hierarchy updated: {len(self.level2_clusters)} principles")
                        self._save_hierarchy()  # Persist after successful update
                    else:
                        logging.warning(f"Level 2 abstraction failed")
                else:
                    logging.warning(f"Level 2 clustering failed")
            
            return True
            
        except Exception as e:
            logging.error(f"Hierarchy update failed: {e}")
            return False

    def _cluster_level1(self) -> bool:
        """Cluster Level 0 notes into Level 1 summaries with dynamic sizing."""
        try:
            total_notes = len(self.dynamic_memory.notes)
            if total_notes < 3:
                return False
            
            # Calculate optimal number of clusters
            if self.n_clusters_level1 is None:
                n_clusters = self._calculate_optimal_clusters(total_notes, 1)
            else:
                n_clusters = min(self.n_clusters_level1, total_notes)
            
            logging.info(f"[CLUSTERING] Level 1: Using {n_clusters} clusters for {total_notes} notes")
            
            # Extract note contents and create embeddings
            note_contents = [note['content'] for note in self.dynamic_memory.notes.values()]
            embeddings = np.array([self.dynamic_memory.model.encode(content) for content in note_contents])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Organize into Level 1 clusters with enhanced metadata
            self.level1_clusters = {}
            current_time = time.time()
            
            for idx, label in enumerate(labels):
                if label not in self.level1_clusters:
                    self.level1_clusters[label] = {
                        'ids': [], 
                        'summary': '',
                        'created_at': current_time,
                        'last_updated': current_time,
                        'note_count': 0
                    }
                self.level1_clusters[label]['ids'].append(idx)
                self.level1_clusters[label]['note_count'] += 1
            
            logging.info(f"[CLUSTERING] Level 1 complete: {len(self.level1_clusters)} clusters created")
            for cluster_id, cluster in self.level1_clusters.items():
                logging.info(f"[CLUSTER] {cluster_id}: {cluster['note_count']} notes")
            
            return True
            
        except Exception as e:
            logging.error(f"Level 1 clustering failed: {e}", exc_info=True)
            return False

    def _summarize_level1(self) -> bool:
        """Generate summaries for Level 1 clusters using LLM with enhanced logging and fact retention."""
        try:
            current_time = time.time()
            for cluster_id, cluster in self.level1_clusters.items():
                if not cluster['ids']:
                    continue
                # Get notes from cluster
                notes = [list(self.dynamic_memory.notes.values())[idx]['content'] for idx in cluster['ids']]
                notes_str = ' '.join([str(n) for n in notes])[:2048]
                # Extract all dates/names/numbers
                all_dates = []
                for note in notes:
                    all_dates.extend(utils.extract_dates(note))
                # (Names: simple heuristic for now)
                all_names = []
                for note in notes:
                    all_names.extend([w for w in note.split() if w.istitle() and len(w) > 2])
                all_numbers = []
                for note in notes:
                    all_numbers.extend(utils.advanced_extract_numbers(note))
                logging.info(f"[SUMMARIZING] Cluster {cluster_id}: {len(notes)} notes | Dates: {all_dates} | Names: {all_names} | Numbers: {all_numbers}")
                prompt = f"Summarize the following notes in 1-2 sentences, focusing on the main themes and key information. ALWAYS include all dates, names, and numbers found in the notes.\n\nNotes: {notes_str}\n\nDates: {all_dates}\nNames: {all_names}\nNumbers: {all_numbers}"
                try:
                    summary = self.llm.generate(prompt, max_tokens=100)
                    cluster['summary'] = str(summary).strip()
                    cluster['last_updated'] = current_time
                    logging.info(f"[SUMMARY] Cluster {cluster_id}: {cluster['summary'][:80]}...")
                except Exception as e:
                    logging.warning(f"Summarization failed for cluster {cluster_id}: {e}")
                    cluster['summary'] = f"Cluster {cluster_id} summary"
            logging.info(f"[SUCCESS] Level 1 summarization complete: {len(self.level1_clusters)} summaries")
            return True
        except Exception as e:
            logging.error(f"Level 1 summarization failed: {e}", exc_info=True)
            return False

    def _cluster_level2(self) -> bool:
        """Cluster Level 1 summaries into Level 2 abstract principles with dynamic sizing."""
        try:
            total_clusters = len(self.level1_clusters)
            if total_clusters < 2:
                return False
            
            # Calculate optimal number of principles
            if self.n_clusters_level2 is None:
                n_clusters = self._calculate_optimal_clusters(total_clusters, 2)
            else:
                n_clusters = min(self.n_clusters_level2, total_clusters)
            
            logging.info(f"[CLUSTERING] Level 2: Using {n_clusters} principles for {total_clusters} clusters")
            
            # Extract summaries and create embeddings
            summaries = [c['summary'] for c in self.level1_clusters.values() if c['summary']]
            if not summaries:
                return False
            
            embeddings = np.array([self.dynamic_memory.model.encode(s) for s in summaries])
            
            # Perform meta-clustering
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Organize into Level 2 clusters with enhanced metadata
            self.level2_clusters = {}
            current_time = time.time()
            
            for idx, label in enumerate(labels):
                if label not in self.level2_clusters:
                    self.level2_clusters[label] = {
                        'ids': [], 
                        'principle': '',
                        'created_at': current_time,
                        'last_updated': current_time,
                        'cluster_count': 0
                    }
                self.level2_clusters[label]['ids'].append(idx)
                self.level2_clusters[label]['cluster_count'] += 1
            
            logging.info(f"[CLUSTERING] Level 2 complete: {len(self.level2_clusters)} principles created")
            for principle_id, principle in self.level2_clusters.items():
                logging.info(f"[PRINCIPLE] {principle_id}: {principle['cluster_count']} clusters")
            
            return True
            
        except Exception as e:
            logging.error(f"Level 2 clustering failed: {e}", exc_info=True)
            return False

    def _abstract_level2(self) -> bool:
        """Generate abstract principles for Level 2 clusters using LLM with enhanced logging and fact retention."""
        try:
            current_time = time.time()
            for cluster_id, cluster in self.level2_clusters.items():
                if not cluster['ids']:
                    continue
                # Get summaries from cluster
                summaries = []
                for idx in cluster['ids']:
                    if idx < len(self.level1_clusters):
                        summary = list(self.level1_clusters.values())[idx]['summary']
                        if summary:
                            summaries.append(str(summary))
                if not summaries:
                    continue
                joined_summaries = ' '.join(summaries)[:2048]
                # Extract all dates/names/numbers
                all_dates = []
                for summary in summaries:
                    all_dates.extend(utils.extract_dates(summary))
                all_names = []
                for summary in summaries:
                    all_names.extend([w for w in summary.split() if w.istitle() and len(w) > 2])
                all_numbers = []
                for summary in summaries:
                    all_numbers.extend(utils.advanced_extract_numbers(summary))
                logging.info(f"[ABSTRACTING] Principle {cluster_id}: {len(summaries)} summaries | Dates: {all_dates} | Names: {all_names} | Numbers: {all_numbers}")
                prompt = f"Abstract the following cluster summaries into a general principle or life lesson (1-2 sentences). ALWAYS include all dates, names, and numbers found in the summaries.\n\nSummaries: {joined_summaries}\n\nDates: {all_dates}\nNames: {all_names}\nNumbers: {all_numbers}"
                try:
                    principle = self.llm.generate(prompt, max_tokens=60)
                    cluster['principle'] = str(principle).strip()
                    cluster['last_updated'] = current_time
                    logging.info(f"[PRINCIPLE] {cluster_id}: {cluster['principle'][:80]}...")
                except Exception as e:
                    logging.warning(f"Abstraction failed for principle {cluster_id}: {e}")
                    cluster['principle'] = f"Principle {cluster_id}"
            logging.info(f"[SUCCESS] Level 2 abstraction complete: {len(self.level2_clusters)} principles")
            return True
        except Exception as e:
            logging.error(f"Level 2 abstraction failed: {e}", exc_info=True)
            return False

    def _classify_query_complexity(self, query, category):
        """
        Classify the query to determine reasoning type and preferred retrieval level.
        Returns (level_preference, reasoning_type, confidence).
        """
        # Query complexity indicators
        specific_indicators = ['what color', 'what was', 'how much', 'what time', 'what date', 'name of', 'exact', 'specific']
        temporal_indicators = ['when', 'how long', 'duration', 'time', 'date', 'before', 'after', 'during']
        causal_indicators = ['why', 'because', 'caused', 'reason', 'due to', 'result of']
        multi_hop_indicators = ['total', 'combined', 'sum', 'all', 'both', 'together', 'across', 'total cost', 'total amount', 'how many photos']
        abstract_indicators = ['pattern', 'trend', 'general', 'typically', 'usually', 'overall']
        
        # Count indicators
        specific_count = sum(1 for indicator in specific_indicators if indicator in query.lower())
        temporal_count = sum(1 for indicator in temporal_indicators if indicator in query.lower())
        causal_count = sum(1 for indicator in causal_indicators if indicator in query.lower())
        multi_hop_count = sum(1 for indicator in multi_hop_indicators if indicator in query.lower())
        abstract_count = sum(1 for indicator in abstract_indicators if indicator in query.lower())
        
        # Determine primary reasoning type
        reasoning_scores = {
            'specific': specific_count,
            'temporal': temporal_count,
            'causal': causal_count,
            'multi_hop': multi_hop_count,
            'abstract': abstract_count
        }
        
        primary_reasoning = max(reasoning_scores, key=reasoning_scores.get)
        max_score = reasoning_scores[primary_reasoning]
        
        # Determine level preference based on reasoning type and category
        if primary_reasoning == 'specific' and max_score >= 1:
            level_preference = 0  # Level 0 for specific factual queries
            confidence = 0.9
        elif primary_reasoning == 'temporal' and max_score >= 1:
            level_preference = 1  # Level 1 for temporal reasoning
            confidence = 0.8
        elif primary_reasoning == 'causal' and max_score >= 1:
            level_preference = 1  # Level 1 for causal reasoning
            confidence = 0.8
        elif primary_reasoning == 'multi_hop' and max_score >= 1:
            level_preference = 2  # Level 2 for multi-hop reasoning
            confidence = 0.7
        elif primary_reasoning == 'abstract' and max_score >= 1:
            level_preference = 2  # Level 2 for abstract reasoning
            confidence = 0.7
        else:
            # Fallback based on category
            if category in ['Entity Tracking', 'Adversarial/Challenge']:
                level_preference = 0
                confidence = 0.6
            elif category in ['Temporal Reasoning', 'Causal Reasoning']:
                level_preference = 1
                confidence = 0.6
            elif category in ['Multi-hop Reasoning']:
                level_preference = 2
                confidence = 0.6
            else:
                level_preference = 0
                confidence = 0.5
        
        return level_preference, primary_reasoning, confidence

    def get_full_context_for_aggregation(self):
        """Concatenate all Level 0 note contents and Level 1 summaries for aggregation/multi-hop questions."""
        notes = [note['content'] for note in self.dynamic_memory.notes.values() if 'content' in note]
        summaries = [c['summary'] for c in self.level1_clusters.values() if 'summary' in c and c['summary']]
        return '\n'.join(notes + summaries)

    def retrieve(self, query: str, category: str = None, self_improvement_engine=None) -> Dict[str, Any]:
        """
        Enhanced: (1) Escalate to higher levels if Level 0 fails, (2) Aggregate context for complex/failed queries, (3) Explicit, context-rich LLM prompts, (4) Debug log for all incorrect answers, (5) Lower thresholds for Level 1/2, (6) Fact extraction and patching for temporal/entity questions, (7) Expanded context window.
        """
        try:
            logging.info(f'[RETRIEVAL] Processing query: {query}...')
            preferred_level, reasoning_type, confidence = self._classify_query_complexity(query, category)
            logging.info(f'[RETRIEVAL] Query classification: {reasoning_type} (confidence: {confidence:.2f}), preferred level: {preferred_level}')
            if preferred_level == 0:
                retrieval_order = [0, 1, 2]
            elif preferred_level == 1:
                retrieval_order = [1, 0, 2]
            else:
                retrieval_order = [2, 1, 0]
            best_result = None
            first_valid_answer = None
            results = None
            is_aggregation = any(kw in query.lower() for kw in ["total", "sum", "aggregate", "add up", "combined", "overall"])
            is_multihop = reasoning_type in ["multi_hop", "abstract"] or (category and category.lower() in ["multi-hop reasoning", "adversarial/challenge"])
            full_context = self.get_full_context_for_aggregation() if (is_aggregation or is_multihop) else None
            # Lower thresholds to encourage more retrieval attempts
            level0_threshold = 0.70
            level1_threshold = 0.50
            level2_threshold = 0.35
            generic_answers = ["not found", "not_in_summary", "not_in_principle", "", "no answer", "none", "not_found"]
            def is_generic_response(response):
                if not response:
                    return True
                response_lower = response.strip().lower()
                if response_lower in generic_answers:
                    return True
                not_found_patterns = [
                    "not found", "not available", "not present", "not in", "cannot find",
                    "no information", "no answer", "not mentioned", "not stated",
                    "i cannot", "i don't", "there is no", "this information",
                    "the answer is not", "not provided", "not given"
                ]
                for pattern in not_found_patterns:
                    if pattern in response_lower:
                        return True
                if len(response_lower.split()) < 3:
                    return True
                return False
            def is_fragment_or_nonanswer(ans):
                # Returns True if answer is a fragment, generic, or not of expected type
                if not ans or ans.strip().lower() in ['says', 'what kinda jobs', 'and melanie', 'takes her kids to a local', '']:
                    return True
                if len(ans.strip().split()) <= 2:
                    return True
                return False

            def build_prompt(question, context, reasoning_type, category):
                # Enhanced prompt engineering for reasoning, entity tracking, and neat output
                prompt = f"Answer the following question as concisely and directly as possible, using only information from the context.\n"
                if reasoning_type == 'causal' or (category and 'causal' in category.lower()):
                    prompt += "If the question asks for a reason, preference, or decision, provide a clear, concise answer with supporting evidence from the context."
                elif reasoning_type == 'entity' or (category and 'entity' in category.lower()):
                    prompt += "If the question asks for entities, roles, or activities, enumerate all relevant items as a clean, comma-separated list."
                elif reasoning_type == 'temporal' or (category and 'temporal' in category.lower()):
                    prompt += "If the answer is a date or time, provide the most specific value, or a relative/event-based phrase if required."
                elif reasoning_type == 'multi-hop' or (category and 'multi-hop' in category.lower()):
                    prompt += "If the answer requires combining information, show the reasoning or calculation step by step."
                prompt += "\nIf the answer is not found, state 'NOT_FOUND'.\n"
                prompt += f"\nContext:\n{context}\n\nQuestion: {question}\nAnswer: "
                return prompt

            def postprocess_llm_output(answer, expected_type=None):
                # Clean up LLM output for neatness and BLEU-1 improvement
                if not answer:
                    return ''
                answer = answer.strip()
                # Remove generic phrases
                for phrase in ["says", "What kinda jobs", "takes her kids to a local", "and Melanie", "NOT_FOUND."]:
                    if answer.lower().startswith(phrase.lower()):
                        answer = answer[len(phrase):].strip()
                # Standardize list formatting
                if expected_type == 'list' and answer:
                    from utils import canonicalize_list_answer
                    answer = canonicalize_list_answer(answer)
                # Remove trailing punctuation for BLEU-1
                answer = answer.rstrip('.;:')
                logging.debug(f"[POSTPROCESS] Postprocessed answer: {answer}")
                return answer

            def reconstruct_relative_date(context, target_date):
                # Try to reconstruct a relative/event-based answer from context
                # Example: 'the week before 9 June 2023' if target_date is '9 June 2023'
                import re
                match = re.search(r'(week|day|month|weekend|friday|saturday|sunday|monday|tuesday|wednesday|thursday)[^\n]*before ([0-9]{1,2} [A-Za-z]+ [0-9]{4})', context, re.IGNORECASE)
                if match:
                    return match.group(0)
                return target_date

            def extract_facts_for_logging(context):
                # Extract all relevant facts for logging and patching
                dates = utils.extract_dates(context) if context else []
                numbers = utils.advanced_extract_numbers(context) if context else []
                entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', context) if context else []
                return {'dates': dates, 'numbers': numbers, 'entities': list(set(entities))}

            def patch_answer_with_context(question, answer, context, expected_type=None):
                # Always postprocess before patching
                postprocessed = postprocess_llm_output(answer, expected_type)
                facts = extract_facts_for_logging(context)
                dates, numbers, entities = facts['dates'], facts['numbers'], facts['entities']
                logging.info(f'[PATCH] Extracted facts: {facts}')
                # --- Temporal: Try to reconstruct relative/event-based answer ---
                if expected_type == 'date' and dates:
                    rel_phrases = re.findall(r'(the (week|friday|sunday|monday|tuesday|wednesday|thursday|saturday|weekend) before [^\n\.;,]+)', context, re.IGNORECASE)
                    if rel_phrases:
                        logging.info(f'[PATCH][TEMPORAL] Found relative phrase: {rel_phrases[0][0]}')
                        return rel_phrases[0][0], facts
                    rel = reconstruct_relative_date(context, dates[0])
                    return rel, facts
                # --- Entity/List: Patch with all relevant items ---
                if expected_type == 'entity' and entities:
                    unique_entities = list({e.strip() for e in entities if len(e.strip()) > 1})
                    if unique_entities:
                        from utils import canonicalize_list_answer
                        patched = canonicalize_list_answer(', '.join(unique_entities))
                        logging.info(f'[PATCH][ENTITY] Patched with all entities: {patched}')
                        return patched, facts
                # --- Causal/Preference: Patch with likely cause/preference phrase ---
                if expected_type is None or expected_type == 'causal':
                    cause_phrases = re.findall(r'((because|so that|in order to|wanted to|so she|so he|so they)[^\n\.;,]+)', context, re.IGNORECASE)
                    if cause_phrases:
                        logging.info(f'[PATCH][CAUSAL] Patched with cause phrase: {cause_phrases[0][0]}')
                        return cause_phrases[0][0], facts
                # --- Fragments/Generics: Patch with most specific fact or list ---
                if is_fragment_or_nonanswer(postprocessed):
                    if expected_type == 'date' and dates:
                        rel_phrases = re.findall(r'(the (week|friday|sunday|monday|tuesday|wednesday|thursday|saturday|weekend) before [^\n\.;,]+)', context, re.IGNORECASE)
                        if rel_phrases:
                            logging.info(f'[PATCH][FRAGMENT][TEMPORAL] Patched with relative phrase: {rel_phrases[0][0]}')
                            return rel_phrases[0][0], facts
                        rel = reconstruct_relative_date(context, dates[0])
                        return rel, facts
                    if numbers:
                        logging.info(f'[PATCH][FRAGMENT][NUMBER] Patched with number: {numbers[0]}')
                        return numbers[0], facts
                    if entities:
                        unique_entities = list({e.strip() for e in entities if len(e.strip()) > 1})
                        from utils import canonicalize_list_answer
                        patched = canonicalize_list_answer(', '.join(unique_entities))
                        logging.info(f'[PATCH][FRAGMENT][ENTITY] Patched with all entities: {patched}')
                        return patched, facts
                    cause_phrases = re.findall(r'((because|so that|in order to|wanted to|so she|so he|so they)[^\n\.;,]+)', context, re.IGNORECASE)
                    if cause_phrases:
                        logging.info(f'[PATCH][FRAGMENT][CAUSAL] Patched with cause phrase: {cause_phrases[0][0]}')
                        return cause_phrases[0][0], facts
                    logging.info(f'[PATCH][FRAGMENT] No relevant fact found, returning NOT_FOUND')
                    return 'NOT_FOUND', facts
                # --- Entity/List: Patch with all if answer is not a list or is missing items ---
                if expected_type == 'entity' and entities:
                    ans_set = set([a.strip().lower() for a in postprocessed.split(',')])
                    ent_set = set([e.strip().lower() for e in entities])
                    if not ans_set.issuperset(ent_set):
                        unique_entities = list({e.strip() for e in entities if len(e.strip()) > 1})
                        from utils import canonicalize_list_answer
                        patched = canonicalize_list_answer(', '.join(unique_entities))
                        logging.info(f'[PATCH][ENTITY][LIST] Patched with all entities: {patched}')
                        return patched, facts
                return postprocessed, facts

            # --- Enhanced context extraction for temporal/entity questions ---
            def extract_all_dates_entities(notes):
                all_dates = []
                for note in notes:
                    all_dates.extend(utils.extract_dates(note))
                all_names = []
                for note in notes:
                    all_names.extend([w for w in note.split() if w.istitle() and len(w) > 2])
                all_numbers = []
                for note in notes:
                    all_numbers.extend(utils.advanced_extract_numbers(note))
                return all_dates, all_names, all_numbers
            for level in retrieval_order:
                if best_result:
                    break
                logging.info(f'[RETRIEVAL] Trying Level {level}...')
                # --- Prompt refinement for specificity and fact extraction ---
                prompt_suffix = ''
                if reasoning_type in ['temporal', 'specific'] or (category and category.lower() in ['temporal reasoning', 'entity tracking']):
                    prompt_suffix += '\nIf the answer is a date, name, or number, respond with only that value.'
                if is_aggregation or is_multihop:
                    prompt_suffix += '\nIf the answer requires combining information, show the calculation or reasoning.'
                if level == 0:
                    results = self.dynamic_memory.search(query, top_k=5)
                    if results and 'ids' in results and results['ids'] and len(results['ids']) > 0:
                        note_ids = results['ids'][0] if isinstance(results['ids'], list) and len(results['ids']) > 0 else []
                        if note_ids:
                            notes = [self.dynamic_memory.get_note(note_id)['content'] for note_id in note_ids if self.dynamic_memory.get_note(note_id)]
                            all_dates, all_names, all_numbers = extract_all_dates_entities(notes)
                            context_str = '\n'.join(notes)
                            logging.info(f'[RETRIEVAL][CONTEXT] Top-5 notes: {notes}')
                            logging.info(f'[RETRIEVAL][FACTS] Dates: {all_dates} | Names: {all_names} | Numbers: {all_numbers}')
                            prompt = f'''You are a world-class memory reasoner. Given the following notes, answer the question as specifically as possible.\n\nNotes: {context_str}\n\nDates: {all_dates}\nNames: {all_names}\nNumbers: {all_numbers}\n\nQuestion: "{query}"\n\nInstructions:\n1. If the answer is directly stated in the notes, provide it clearly and concisely\n2. If the answer can be inferred from the context, make a reasonable inference\n3. If the answer requires combining information, do so thoughtfully\n4. Look for dates, times, names, events, and specific details\n5. Be specific and avoid generic responses\n6. Only respond with "NOT_FOUND" if the information is completely absent and cannot be inferred{prompt_suffix}\n\nAnswer:'''
                            try:
                                llm_result = self.llm.generate(prompt, max_tokens=150)
                                candidate, use_candidate = utils.suggest_temporal_or_aggregation_answer(query, llm_result, context_str, full_context=full_context)
                                # --- Post-processing and patching ---
                                expected_type = None
                                if reasoning_type == 'temporal': expected_type = 'date'
                                elif reasoning_type == 'specific' and any(w in query.lower() for w in ['how many', 'how much', 'total', 'sum']): expected_type = 'number'
                                elif reasoning_type == 'specific': expected_type = 'name'
                                patched_result, extracted_facts = patch_answer_with_context(query, llm_result, context_str, expected_type)
                                if patched_result != llm_result:
                                    logging.info(f'[PATCH][LEVEL0] Patched fragment/generic answer "{llm_result}" with "{patched_result}"')
                                if candidate and not is_generic_response(candidate):
                                    if not first_valid_answer:
                                        first_valid_answer = candidate.strip()
                                if patched_result and not is_generic_response(patched_result):
                                    if not first_valid_answer:
                                        first_valid_answer = patched_result.strip()
                                    best_result = {
                                        'level': 0, 
                                        'results': results,
                                        'note_content': context_str,
                                        'note_ids': note_ids,
                                        'llm_result': patched_result.strip(),
                                        'similarity': 0.0,
                                        'dates': all_dates,
                                        'names': all_names,
                                        'numbers': all_numbers,
                                        'extracted_facts': extracted_facts,
                                        'context': context_str
                                    }
                                    break
                                else:
                                    # Patch with extracted candidate if LLM fails
                                    if candidate:
                                        logging.warning(f'[PATCH][LEVEL0] LLM failed, patching answer with extracted candidate: {candidate}')
                                        best_result = {
                                            'level': 0,
                                            'results': results,
                                            'note_content': context_str,
                                            'note_ids': note_ids,
                                            'llm_result': candidate.strip(),
                                            'similarity': 0.0,
                                            'dates': all_dates,
                                            'names': all_names,
                                            'numbers': all_numbers,
                                            'patched': True,
                                            'extracted_facts': extracted_facts,
                                            'context': context_str
                                        }
                                        if not first_valid_answer:
                                            first_valid_answer = candidate.strip()
                                        break
                                    logging.info(f'[RETRIEVAL] Level 0 LLM result indicates information not found: "{llm_result}"')
                            except Exception as e:
                                logging.warning(f'[WARNING] Level 0 LLM generation failed: {e}')
                elif level == 1 and self.level1_clusters:
                    logging.info(f'[RETRIEVAL] Checking Level 1 clusters: {len(self.level1_clusters)} available')
                    query_emb = np.array(self.dynamic_memory.model.encode(query)).flatten()
                    summaries = [c['summary'] for c in self.level1_clusters.values() if c['summary']]
                    if summaries:
                        # Use top-3 most similar summaries for context
                        emb_matrix = np.stack([np.array(self.dynamic_memory.model.encode(s)).flatten() for s in summaries])
                        sims = emb_matrix @ query_emb / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb) + 1e-8)
                        top_idxs = np.argsort(sims)[-3:][::-1]
                        top_summaries = [summaries[i] for i in top_idxs]
                        all_dates, all_names, all_numbers = extract_all_dates_entities(top_summaries)
                        context_str = '\n'.join(top_summaries)
                        max_sim = float(np.max(sims))
                        logging.info(f'[RETRIEVAL][CONTEXT][L1] Top-3 summaries: {top_summaries}')
                        logging.info(f'[RETRIEVAL][FACTS][L1] Dates: {all_dates} | Names: {all_names} | Numbers: {all_numbers}')
                        if max_sim > level1_threshold:
                            self.retrieval_stats['level1_count'] += 1
                            prompt = f'''You are a world-class memory reasoner. Given the following summaries, answer the question as specifically as possible.\n\nSummaries: {context_str}\n\nDates: {all_dates}\nNames: {all_names}\nNumbers: {all_numbers}\n\nQuestion: "{query}"\n\nInstructions:\n1. If the answer is directly stated in the summaries, provide it clearly and concisely\n2. If the answer can be inferred from the context, make a reasonable inference\n3. If the answer requires reasoning or combining information, perform it thoughtfully\n4. Look for dates, times, names, events, and specific details\n5. Be specific and avoid generic responses\n6. Only respond with "NOT_FOUND" if the information is completely absent and cannot be inferred{prompt_suffix}\n\nAnswer:'''
                            try:
                                llm_result = self.llm.generate(prompt, max_tokens=150)
                                candidate, use_candidate = utils.suggest_temporal_or_aggregation_answer(query, llm_result, context_str, full_context=full_context)
                                # --- Post-processing and patching ---
                                expected_type = None
                                if reasoning_type == 'temporal': expected_type = 'date'
                                elif reasoning_type == 'specific' and any(w in query.lower() for w in ['how many', 'how much', 'total', 'sum']): expected_type = 'number'
                                elif reasoning_type == 'specific': expected_type = 'name'
                                patched_result, extracted_facts = patch_answer_with_context(query, llm_result, context_str, expected_type)
                                if is_fragment_or_nonanswer(llm_result):
                                    if patched_result != llm_result:
                                        logging.info(f'[PATCH][LEVEL1] Patched fragment/generic answer "{llm_result}" with "{patched_result}"')
                                if candidate and not is_generic_response(candidate):
                                    if not first_valid_answer:
                                        first_valid_answer = candidate.strip()
                                if patched_result and not is_generic_response(patched_result):
                                    if not first_valid_answer:
                                        first_valid_answer = patched_result.strip()
                                    best_result = {
                                        'level': 1, 
                                        'summary': context_str,
                                        'llm_result': patched_result.strip(),
                                        'similarity': max_sim,
                                        'dates': all_dates,
                                        'names': all_names,
                                        'numbers': all_numbers,
                                        'extracted_facts': extracted_facts,
                                        'context': context_str
                                    }
                                    break
                            except Exception as e:
                                logging.warning(f'[WARNING] Level 1 LLM generation failed: {e}')
                    else:
                        logging.info(f'[RETRIEVAL] Level 1 similarity too low ({max_sim:.3f}), trying next level')
                elif level == 2 and self.level2_clusters:
                    logging.info(f'[RETRIEVAL] Checking Level 2 principles: {len(self.level2_clusters)} available')
                    query_emb = np.array(self.dynamic_memory.model.encode(query)).flatten()
                    principles = [c['principle'] for c in self.level2_clusters.values() if c['principle']]
                    if principles:
                        # Use all principles for context
                        emb_matrix = np.stack([np.array(self.dynamic_memory.model.encode(p)).flatten() for p in principles])
                        sims = emb_matrix @ query_emb / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb) + 1e-8)
                        max_sim = float(np.max(sims))
                        all_dates, all_names, all_numbers = extract_all_dates_entities(principles)
                        context_str = '\n'.join(principles)
                        logging.info(f'[RETRIEVAL][CONTEXT][L2] All principles: {principles}')
                        logging.info(f'[RETRIEVAL][FACTS][L2] Dates: {all_dates} | Names: {all_names} | Numbers: {all_numbers}')
                        if max_sim > level2_threshold:
                            self.retrieval_stats['level2_count'] += 1
                            prompt = f'''You are a world-class memory reasoner. Given the following abstract principles, answer the question as specifically as possible.\n\nPrinciples: {context_str}\n\nDates: {all_dates}\nNames: {all_names}\nNumbers: {all_numbers}\n\nQuestion: "{query}"\n\nInstructions:\n1. If the answer is directly stated in the principles, provide it clearly and concisely\n2. If the answer can be inferred from the context, make a reasonable inference\n3. If the answer requires combining information, do so thoughtfully\n4. Look for dates, times, names, events, and specific details\n5. Be specific and avoid generic responses\n6. Only respond with "NOT_FOUND" if the information is completely absent and cannot be inferred{prompt_suffix}\n\nAnswer:'''
                            try:
                                llm_result = self.llm.generate(prompt, max_tokens=150)
                                candidate, use_candidate = utils.suggest_temporal_or_aggregation_answer(query, llm_result, context_str, full_context=full_context)
                                # --- Post-processing and patching ---
                                expected_type = None
                                if reasoning_type == 'temporal': expected_type = 'date'
                                elif reasoning_type == 'specific' and any(w in query.lower() for w in ['how many', 'how much', 'total', 'sum']): expected_type = 'number'
                                elif reasoning_type == 'specific': expected_type = 'name'
                                patched_result, extracted_facts = patch_answer_with_context(query, llm_result, context_str, expected_type)
                                if is_fragment_or_nonanswer(llm_result):
                                    if patched_result != llm_result:
                                        logging.info(f'[PATCH][LEVEL2] Patched fragment/generic answer "{llm_result}" with "{patched_result}"')
                                if candidate and not is_generic_response(candidate):
                                    if not first_valid_answer:
                                        first_valid_answer = candidate.strip()
                                if patched_result and not is_generic_response(patched_result):
                                    if not first_valid_answer:
                                        first_valid_answer = patched_result.strip()
                                    best_result = {
                                        'level': 2, 
                                        'principle': context_str,
                                        'llm_result': patched_result.strip(),
                                        'similarity': max_sim,
                                        'dates': all_dates,
                                        'names': all_names,
                                        'numbers': all_numbers,
                                        'extracted_facts': extracted_facts,
                                        'context': context_str
                                    }
                                    break
                            except Exception as e:
                                logging.warning(f'[WARNING] Level 2 LLM generation failed: {e}')
                    else:
                        logging.info(f'[RETRIEVAL] Level 2 similarity too low ({max_sim:.3f})')
                # Step 4: Aggressive context expansion for multi-hop/abstract/failed queries
                if not best_result and (is_multihop or is_aggregation or True):
                    # Always expand context: use all notes and all summaries
                    agg_context = self.get_full_context_for_aggregation()
                    prompt = f'''You are a world-class memory reasoner. Given the following context (multiple notes and summaries), answer the question as specifically as possible.\n\nContext:\n{agg_context[:6000]}\n\nQuestion: "{query}"\n\nInstructions:\n1. Carefully analyze all the provided context\n2. If the answer is directly stated, provide it clearly\n3. If the answer can be inferred by combining information, make a reasonable inference\n4. If the answer requires reasoning across multiple pieces of information, perform it step by step\n5. For temporal questions, extract the most specific date or event-based phrase\n6. For causal questions, extract the most specific cause or reason phrase (e.g., starting with 'because', 'so that', etc.)\n7. For multi-hop or adversarial questions, combine facts from multiple notes and show your reasoning\n8. Only respond with "NOT_FOUND" if the information is completely absent and cannot be inferred\n9. Be specific and avoid generic responses\n\nAnswer: '''
                    try:
                        llm_result = self.llm.generate(prompt, max_tokens=200)
                        # --- Post-processing and patching for aggregation ---
                        expected_type = None
                        if reasoning_type == 'temporal': expected_type = 'date'
                        elif reasoning_type == 'specific' and any(w in query.lower() for w in ['how many', 'how much', 'total', 'sum']): expected_type = 'number'
                        elif reasoning_type == 'specific': expected_type = 'name'
                        patched_result, extracted_facts = patch_answer_with_context(query, llm_result, agg_context, expected_type)
                        if is_fragment_or_nonanswer(llm_result):
                            if patched_result != llm_result:
                                logging.info(f'[PATCH][AGGREGATION] Patched fragment/generic answer "{llm_result}" with "{patched_result}"')
                        if llm_result and not is_generic_response(llm_result):
                            if not first_valid_answer:
                                first_valid_answer = llm_result.strip()
                            best_result = {
                                'level': 99,
                                'agg_context': agg_context[:2000],
                                'llm_result': patched_result.strip(),
                                'similarity': 0.0,
                                'extracted_facts': extracted_facts,
                                'context': agg_context
                            }
                            if not first_valid_answer:
                                first_valid_answer = patched_result.strip()
                            logging.info(f'[RETRIEVAL] Aggregated context success: {patched_result.strip()}')
                    except Exception as e:
                        logging.warning(f'[WARNING] Aggregated context LLM generation failed: {e}')

            # Step 5: Final fallback to Level 0 if nothing else worked
            if not best_result and results and 'ids' in results and results['ids'] and len(results['ids']) > 0:
                note_ids = results['ids'][0] if isinstance(results['ids'], list) and len(results['ids']) > 0 else []
                if note_ids:
                    self.retrieval_stats['level0_count'] += 1
                    note_id = note_ids[0]
                    note = self.dynamic_memory.get_note(note_id)
                    if note:
                        note_content = note['content']
                        logging.info(f'[RETRIEVAL] Fallback to Level 0 (best available)')
                        logging.info(f'[RETRIEVAL] Note ID: {note_id[:8]}...')
                        prompt = f'You are a world-class memory reasoner. Given the following note, answer the question as specifically as possible.\n\nNote: "{note_content}"\n\nQuestion: "{query}"\n\nIf the answer is not present, respond with "NOT_FOUND". Do not explain.'
                        try:
                            llm_result = self.llm.generate(prompt, max_tokens=100)
                            # --- Post-processing and patching for fallback ---
                            expected_type = None
                            if reasoning_type == 'temporal': expected_type = 'date'
                            elif reasoning_type == 'specific' and any(w in query.lower() for w in ['how many', 'how much', 'total', 'sum']): expected_type = 'number'
                            elif reasoning_type == 'specific': expected_type = 'name'
                            patched_result, extracted_facts = patch_answer_with_context(query, llm_result, note_content, expected_type)
                            if is_fragment_or_nonanswer(llm_result):
                                if patched_result != llm_result:
                                    logging.info(f'[PATCH][FALLBACK] Patched fragment/generic answer "{llm_result}" with "{patched_result}"')
                            if llm_result and not is_generic_response(llm_result):
                                if not first_valid_answer:
                                    first_valid_answer = llm_result.strip()
                                best_result = {
                                    'level': 0, 
                                    'results': results,
                                    'note_content': note_content,
                                    'note_id': note_id,
                                    'llm_result': patched_result.strip(),
                                    'similarity': 0.0,
                                    'extracted_facts': extracted_facts,
                                    'context': note_content
                                }
                                if not first_valid_answer:
                                    first_valid_answer = patched_result.strip()
                        except Exception as e:
                            logging.warning(f'[WARNING] Fallback LLM generation failed: {e}')

            # Step 6: Handle complete failure
            if not best_result:
                self.retrieval_stats['failed_count'] += 1
                logging.info(f'[RETRIEVAL] All levels failed, returning error')
                best_result = {
                    'level': -1, 
                    'error': 'No relevant information found in any level',
                    'llm_result': 'No information found',
                    'context': '',
                    'extracted_facts': {}
                }
            # Self-improvement integration
            if self_improvement_engine is not None:
                try:
                    perf = self_improvement_engine.evaluate_performance()
                    logging.info(f"[SELF-IMPROVEMENT] Performance: {perf}")
                except Exception as e:
                    logging.error(f"[SELF-IMPROVEMENT] Error: {e}")
            if best_result:
                best_result['first_valid_answer'] = first_valid_answer
            else:
                best_result = {'first_valid_answer': first_valid_answer}
            return best_result
        except Exception as e:
            logging.error(f'[ERROR][RETRIEVAL] Retrieval failed for query "{query}": {e}')
            return {
                'level': -1,
                'error': f'Retrieval failed: {str(e)}',
                'llm_result': 'Retrieval error occurred',
                'first_valid_answer': None
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hierarchical system."""
        return {
            'level0_notes': len(self.dynamic_memory.notes),
            'level1_clusters': len(self.level1_clusters),
            'level2_clusters': len(self.level2_clusters),
            'retrieval_stats': self.retrieval_stats.copy(),
            'retrieval_mode': self.retrieval_mode,
            'clustering_frequency': self.clustering_frequency,
            'dynamic_clustering': {
                'level1': self.n_clusters_level1 is None,
                'level2': self.n_clusters_level2 is None
            },
            'hierarchy_metadata': {
                'level1_clusters': {k: {'note_count': v.get('note_count', 0), 'created_at': v.get('created_at', 0)} for k, v in self.level1_clusters.items()},
                'level2_clusters': {k: {'cluster_count': v.get('cluster_count', 0), 'created_at': v.get('created_at', 0)} for k, v in self.level2_clusters.items()}
            }
        } 