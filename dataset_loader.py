"""
EVOLVE-MEM: Dataset Loader for LoCoMo Evaluation

Handles loading and processing of the LoCoMo dataset for comprehensive evaluation.
- Loads LoCoMo dataset with QA pairs
- Extracts experiences and questions for testing
- Provides evaluation metrics and SOTA comparison

"""
import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

class LoCoMoDatasetLoader:
    """
    Loader for LoCoMo dataset with comprehensive evaluation capabilities.
    """
    
    def __init__(self, dataset_path: str = "data/locomo10.json", include_adversarial: bool = False):
        """Initialize the dataset loader."""
        self.dataset_path = dataset_path
        self.data = None
        self.experiences = []
        self.qa_pairs = []
        self.adversarial_qa_pairs = []
        self.categories = {
            1: "Entity Tracking",
            2: "Temporal Reasoning", 
            3: "Causal Reasoning",
            4: "Multi-hop Reasoning",
            5: "Adversarial/Challenge"
        }
        self.include_adversarial = include_adversarial
        self.load_dataset()
    
    def load_dataset(self):
        """Load and parse the LoCoMo dataset."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            logging.info(f"[DATASET] Loaded LoCoMo dataset with {len(self.data)} stories")
            self.extract_experiences_and_qa()
            
        except Exception as e:
            logging.error(f"[DATASET] Failed to load dataset: {e}")
            raise
    
    def extract_all_strings(self, obj):
        """Recursively extract all string content from nested dicts/lists/strings."""
        strings = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self.extract_all_strings(v))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(self.extract_all_strings(item))
        return strings
    
    def extract_experiences_and_qa(self):
        """Extract experiences and QA pairs from the dataset."""
        adversarial_count = 0
        for story_idx, story in enumerate(self.data):
            # Extract story content from conversation sessions
            story_content = []
            
            if 'conversation' in story:
                conversation = story['conversation']
                
                # Extract speaker information
                speaker_a = conversation.get('speaker_a', 'Unknown')
                speaker_b = conversation.get('speaker_b', 'Unknown')
                
                # Extract all session content - FIXED: Handle both dict and list structures
                session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('_date_time')]
                session_keys.sort()  # Sort to maintain order
                
                for session_key in session_keys:
                    session_data = conversation.get(session_key, [])
                    session_date = conversation.get(f"{session_key}_date_time", "")
                    
                    # Handle session data - could be list of turns or dict
                    if isinstance(session_data, list):
                        # Process each turn in the session
                        for turn_idx, turn in enumerate(session_data):
                            if isinstance(turn, dict):
                                speaker = turn.get('speaker', 'Unknown')
                                text = turn.get('text', '')
                                
                                # Handle image captions if present
                                if 'img_url' in turn and 'blip_caption' in turn:
                                    caption = turn['blip_caption']
                                    if text:
                                        text = f"[Image: {caption}] {text}"
                                    else:
                                        text = f"[Image: {caption}]"
                                
                                if text.strip():
                                    experience_text = f"Speaker {speaker} says: {text}"
                                    if session_date:
                                        experience_text = f"[{session_date}] {experience_text}"
                                    story_content.append(experience_text)
                            elif isinstance(turn, str):
                                # If turn is just a string, create a generic experience
                                experience_text = f"Session {session_key} Turn {turn_idx}: {turn}"
                                if session_date:
                                    experience_text = f"[{session_date}] {experience_text}"
                                story_content.append(experience_text)
                    
                    elif isinstance(session_data, dict):
                        # If session is a dict, extract all string content
                        session_strings = self.extract_all_strings(session_data)
                        for text in session_strings:
                            if text.strip():
                                experience_text = f"Session {session_key}: {text}"
                                if session_date:
                                    experience_text = f"[{session_date}] {experience_text}"
                                story_content.append(experience_text)
                    
                    elif isinstance(session_data, str):
                        # If session is a string, use it directly
                        if session_data.strip():
                            experience_text = f"Session {session_key}: {session_data}"
                            if session_date:
                                experience_text = f"[{session_date}] {experience_text}"
                            story_content.append(experience_text)
            
            # Also include event_summary and session_summary if available
            if 'event_summary' in story and story['event_summary']:
                if isinstance(story['event_summary'], dict):
                    # Extract all strings from event summary dict
                    event_strings = self.extract_all_strings(story['event_summary'])
                    for text in event_strings:
                        if text.strip():
                            story_content.append(f"Event Summary: {text}")
                else:
                    story_content.append(f"Event Summary: {story['event_summary']}")
                
            if 'session_summary' in story and story['session_summary']:
                if isinstance(story['session_summary'], dict):
                    # Extract all strings from session summary dict
                    summary_strings = self.extract_all_strings(story['session_summary'])
                    for text in summary_strings:
                        if text.strip():
                            story_content.append(f"Session Summary: {text}")
                else:
                    story_content.append(f"Session Summary: {story['session_summary']}")
            
            # Create experiences from story content
            if story_content:
                # Create multiple experiences from different parts of the story
                for i, content in enumerate(story_content):
                    if content.strip():  # Only add non-empty content
                        self.experiences.append({
                            'story_id': story_idx,
                            'content': content,
                            'session_id': i,
                            'conversation': story.get('conversation', {})
                        })
            
            # Extract QA pairs
            if 'qa' in story:
                for qa in story['qa']:
                    if 'answer' in qa:
                        self.qa_pairs.append({
                            'story_id': story_idx,
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'evidence': qa.get('evidence', []),
                            'category': qa.get('category', 1),
                            'category_name': self.categories.get(qa.get('category', 1), 'Unknown')
                        })
                    elif self.include_adversarial and 'adversarial_answer' in qa:
                        self.adversarial_qa_pairs.append({
                            'story_id': story_idx,
                            'question': qa['question'],
                            'answer': qa['adversarial_answer'],
                            'evidence': qa.get('evidence', []),
                            'category': qa.get('category', 5),
                            'category_name': self.categories.get(5, 'Adversarial/Challenge')
                        })
                        adversarial_count += 1
        
        if adversarial_count > 0:
            logging.info(f"[DATASET] Found {adversarial_count} adversarial QA pairs (category 5). Include in evaluation: {self.include_adversarial}")
        logging.info(f"[DATASET] Extracted {len(self.experiences)} experiences and {len(self.qa_pairs)} QA pairs (standard)")
        if self.include_adversarial:
            logging.info(f"[DATASET] Including {len(self.adversarial_qa_pairs)} adversarial QA pairs in evaluation.")
            self.qa_pairs.extend(self.adversarial_qa_pairs)
    
    def get_experiences(self) -> List[str]:
        """Get list of experience texts for memory creation."""
        return [exp['content'] for exp in self.experiences]
    
    def get_qa_pairs(self) -> List[Dict]:
        """Get all QA pairs for evaluation."""
        return self.qa_pairs
    
    def get_qa_by_category(self, category: int) -> List[Dict]:
        """Get QA pairs filtered by category."""
        return [qa for qa in self.qa_pairs if qa['category'] == category]
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation metrics."""
        total_qa = len(self.qa_pairs)
        category_counts = {}
        
        for qa in self.qa_pairs:
            cat = qa['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            'total_stories': len(self.data),
            'total_experiences': len(self.experiences),
            'total_qa_pairs': total_qa,
            'category_distribution': category_counts,
            'categories': self.categories
        }
    
    def print_dataset_stats(self):
        """Print comprehensive dataset statistics."""
        metrics = self.get_evaluation_metrics()
        
        logging.info("=" * 60)
        logging.info("📊 LoCoMo Dataset Statistics")
        logging.info("=" * 60)
        logging.info(f"Total Stories: {metrics['total_stories']}")
        logging.info(f"Total Experiences: {metrics['total_experiences']}")
        logging.info(f"Total QA Pairs: {metrics['total_qa_pairs']}")
        logging.info("\nCategory Distribution:")
        
        for cat_id, count in metrics['category_distribution'].items():
            cat_name = self.categories.get(cat_id, 'Unknown')
            percentage = (count / metrics['total_qa_pairs']) * 100
            logging.info(f"  {cat_name} (Cat {cat_id}): {count} questions ({percentage:.1f}%)")
        
        logging.info("=" * 60)

    def _flatten_to_string(self, obj):
        """Recursively flatten dicts/lists to a single string."""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return ' '.join([self._flatten_to_string(v) for v in obj.values()])
        elif isinstance(obj, list):
            return ' '.join([self._flatten_to_string(v) for v in obj])
        else:
            return str(obj)

    def load_experiences(self) -> List[str]:
        """Get list of experience texts for memory creation."""
        return [exp['content'] for exp in self.experiences] 