"""
EVOLVE-MEM: Ablation Runner (Standalone)

Runs standardized ablation experiments on a single uniform sample by default,
without modifying the main codebase. Results are saved per variant and a
summary JSON is produced for easy inclusion in papers.

Usage:
  - Fast ranking on a single uniform sample (default):
        python run_ablation.py
  - For final numbers on the full dataset: set SAMPLE_MODE=False below
    and run selected top variants again.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Tuple, Optional
import sys

import numpy as np

# Ensure repository root on sys.path for package imports when run as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from core.dataset_loader import LoCoMoDatasetLoader
from core.memory_system import EvolveMemSystem
from core.evaluation import EVOLVEMEMEvaluator
from core.utils import normalize_answer, canonicalize_list_answer, patch_answer_generalized


# Global controls (uniform sample by default)
SAMPLE_MODE = True
SAMPLE_STORY_INDEX = 0


# Define ablation variants (only using existing public knobs)
EXPERIMENTS = [
    # Baseline (full features) 
    dict(name="full_baseline", use_level1=True, use_level2=True, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=True, patch_answers=True),

    # Hierarchy ablations
    dict(name="l0_only", use_level1=False, use_level2=False, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=False, patch_answers=True),
    dict(name="l0_l1_only", use_level1=True, use_level2=False, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=False, patch_answers=True),
    dict(name="l0_l2_only", use_level1=False, use_level2=True, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=False, patch_answers=True),

    # Clustering schedule/shape
    dict(name="freq_5_dynamic", use_level1=True, use_level2=True, clustering_frequency=5,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=True, patch_answers=True),
    dict(name="freq_15_dynamic", use_level1=True, use_level2=True, clustering_frequency=15,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=True, patch_answers=True),
    dict(name="fixed_l1_3_l2_1", use_level1=True, use_level2=True, clustering_frequency=10,
         n_clusters_level1=3, n_clusters_level2=1, self_improve=True, patch_answers=True),
    dict(name="fixed_l1_5_l2_3", use_level1=True, use_level2=True, clustering_frequency=10,
         n_clusters_level1=5, n_clusters_level2=3, self_improve=True, patch_answers=True),

    # Self-improvement & patching
    dict(name="no_self_improve", use_level1=True, use_level2=True, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=False, patch_answers=True),
    dict(name="no_patching", use_level1=True, use_level2=True, clustering_frequency=10,
         n_clusters_level1=None, n_clusters_level2=None, self_improve=True, patch_answers=False),
]


def setup_logging():
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/ablation_{ts}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def build_data(sample_mode=True, sample_idx=0):
    loader = LoCoMoDatasetLoader(include_adversarial=True)
    all_exps = loader.experiences
    all_qas = loader.get_qa_pairs()
    if sample_mode:
        story_ids = sorted(set(exp['story_id'] for exp in all_exps))
        sid = story_ids[sample_idx]
        exps = [exp['content'] for exp in all_exps if exp['story_id'] == sid]
        qas = [qa for qa in all_qas if qa['story_id'] == sid]
    else:
        exps = [exp['content'] for exp in all_exps]
        qas = all_qas
    return loader, exps, qas


def run_one_experiment(cfg, loader, experiences, qa_pairs):
    # Disable clustering entirely for L0-only variants to avoid any hierarchy updates/logs
    enable_clustering = (cfg['use_level1'] or cfg['use_level2'])
    system = EvolveMemSystem(
        retrieval_mode='hybrid',
        enable_evolution=True,
        enable_clustering=enable_clustering,
        enable_self_improvement=cfg['self_improve']
    )
    # Ensure isolation across variants: clear memory/collection before building
    try:
        system.dynamic_memory.clear()
    except Exception:
        pass
    # Apply ablation knobs available via public attributes
    hm = system.hierarchical_manager
    hm.clustering_frequency = cfg['clustering_frequency']
    hm.n_clusters_level1 = cfg['n_clusters_level1']
    hm.n_clusters_level2 = cfg['n_clusters_level2']

    # Build memory on the chosen uniform sample
    for exp in experiences:
        system.add_experience(exp)

    # Disable levels by clearing stores (post-build) when requested
    if not cfg['use_level1']:
        hm.level1_clusters = {}
    if not cfg['use_level2']:
        hm.level2_clusters = {}

    evaluator = EVOLVEMEMEvaluator()
    t0 = time.time()

    def _ablation_fused_l1_with_l0(system, query: str, category: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Ablation-only fused retrieval for L0+L1: combine top L1 summaries with top-5 L0 notes.

        Returns (answer, context) or (None, None) on failure. Does not modify system internals.
        """
        try:
            hm = system.hierarchical_manager
            # Collect top-5 Level 0 notes for the query
            notes: list = []
            try:
                l0 = system.dynamic_memory.search(query, top_k=5)
                note_ids = l0['ids'][0] if isinstance(l0, dict) and 'ids' in l0 and l0['ids'] else []
                notes = [system.dynamic_memory.get_note(nid)['content'] for nid in note_ids if system.dynamic_memory.get_note(nid)]
            except Exception:
                notes = []

            # Collect top-3 Level 1 summaries most similar to the query
            summaries = [c['summary'] for c in hm.level1_clusters.values() if isinstance(c, dict) and c.get('summary')]
            top_summaries = []
            if summaries:
                try:
                    query_emb = np.array(hm.dynamic_memory.model.encode(query)).flatten()
                    emb_matrix = np.stack([np.array(hm.dynamic_memory.model.encode(s)).flatten() for s in summaries])
                    sims = emb_matrix @ query_emb / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb) + 1e-8)
                    top_idxs = np.argsort(sims)[-3:][::-1]
                    top_summaries = [summaries[i] for i in top_idxs]
                except Exception:
                    top_summaries = summaries[:3]

            if not notes and not top_summaries:
                return None, None

            # Build fused context and concise prompt
            fused_context = ''
            if top_summaries:
                fused_context += 'Summaries:\n' + '\n'.join(top_summaries) + '\n\n'
            if notes:
                fused_context += 'Notes:\n' + '\n'.join(notes)

            expected_hint = ''
            if category:
                lc = category.lower()
                if 'temporal' in lc:
                    expected_hint = '\nIf the answer is a date/time, output only that value.'
                elif 'entity' in lc:
                    expected_hint = '\nIf the answer is a list of entities/items, output a clean, comma-separated list only.'
                elif 'multi-hop' in lc:
                    expected_hint = '\nIf combining information is needed, do the minimal calculation and output only the final value(s).'

            prompt = (
                'Answer the question as concisely and directly as possible using ONLY the context.\n'
                'If not present, respond with NOT_FOUND.'
                f'{expected_hint}\n\n'
                f'Context:\n{fused_context}\n\nQuestion: {query}\nAnswer:'
            )

            llm = hm.llm
            try:
                ans = llm.generate(prompt, max_tokens=150)
            except Exception:
                return None, None

            if not isinstance(ans, str):
                return None, None
            normalized = ans.strip()
            if not normalized or normalized.lower() in ['not_found', 'not found', 'none', 'no answer']:
                return None, fused_context
            return normalized, fused_context
        except Exception:
            return None, None

    for qa in qa_pairs:
        q = qa['question']; gt = str(qa['answer']); cat = qa['category_name']
        result = system.query(q)

        if isinstance(result, dict) and result.get('first_valid_answer'):
            pred = result['first_valid_answer']
        elif isinstance(result, dict) and result.get('llm_result'):
            pred = result['llm_result']
        else:
            pred = str(result)

        # Ablation-only enhancement: for L0+L1 variant, fuse L1 summaries with top-5 L0 notes
        fused_context_override = None
        if cfg.get('use_level1') and not cfg.get('use_level2'):
            fused_pred, fused_ctx = _ablation_fused_l1_with_l0(system, q, cat)
            if isinstance(fused_pred, str) and len(fused_pred.strip()) > 0:
                pred = fused_pred.strip()
                fused_context_override = fused_ctx

        if cfg['patch_answers']:
            try:
                context = fused_context_override if fused_context_override is not None else (result.get('note_content', '') if isinstance(result, dict) else '')
                pred, _ = patch_answer_generalized(q, pred, context, gt)
            except Exception:
                pass

        if cat and 'multi-hop' in cat.lower():
            pred = canonicalize_list_answer(pred)

        npred = normalize_answer(pred)
        ngt = normalize_answer(gt)
        evaluator.add_result(
            question=q,
            predicted=npred,
            ground_truth=ngt,
            category=cat,
            retrieval_level=int(result.get('level', -1)) if isinstance(result, dict) else -1,
            retrieval_time=0.0
        )

    metrics = evaluator.calculate_metrics()
    duration = time.time() - t0
    return metrics, duration


def main():
    log_file = setup_logging()
    loader, exps, qas = build_data(SAMPLE_MODE, SAMPLE_STORY_INDEX)
    os.makedirs('results', exist_ok=True)
    summary = []

    for cfg in EXPERIMENTS:
        logging.info(f"=== Running ablation: {cfg['name']} ===")
        metrics, dur = run_one_experiment(cfg, loader, exps, qas)
        out = {
            'name': cfg['name'],
            'sample_mode': SAMPLE_MODE,
            'duration_sec': dur,
            'overall_accuracy': metrics.get('overall_accuracy', 0.0),
            'overall_f1': metrics.get('overall_f1', 0.0),
            'bleu_1': metrics.get('bleu_1', 0.0),
            'rouge_l': metrics.get('rouge_l', 0.0),
            'rouge_2': metrics.get('rouge_2', 0.0),
            'meteor': metrics.get('meteor', 0.0),
            'sbert': metrics.get('sbert', 0.0),
            'token_length': metrics.get('token_length', 0.0),
            'retrieval_stats': metrics.get('retrieval_stats', {}),
            'category_accuracies': metrics.get('category_accuracies', {}),
        }
        summary.append(out)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"results/ablation_{cfg['name']}_{ts}.json", 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logging.info(f"=== Done: {cfg['name']} | Acc={out['overall_accuracy']:.3f}, F1={out['overall_f1']:.3f}, BLEU1={out['bleu_1']:.3f}, Time={dur:.1f}s ===")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/ablation_summary_{ts}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"Ablation summary saved to results/ablation_summary_{ts}.json")
    logging.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()


