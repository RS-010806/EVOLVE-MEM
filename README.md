# EVOLVE-MEM: Self-Evolving Hierarchical Memory for Agentic AI

## Overview

EVOLVE-MEM is a self-evolving hierarchical memory system designed for agentic AI. It organizes, retrieves, and reasons over large volumes of experiences, supporting robust performance across factual, temporal, multi-hop, and adversarial queries. The system features a three-tier memory hierarchy, dynamic clustering, LLM-powered summarization and reasoning, and a self-improvement engine. All components are evaluated with comprehensive, SOTA-aligned metrics.

---

## System Architecture

### Three-Tier Hierarchical Memory

- **Tier 1: Dynamic Memory Network**
  - Stores raw experiences as vector embeddings (SentenceTransformer + ChromaDB).
  - Supports fast semantic search and metadata tracking.

- **Tier 2: Hierarchical Memory Manager**
  - **Level 0:** Raw experiences (specific details, facts, events).
  - **Level 1:** Clustered summaries (temporal patterns, causal relationships).
  - **Level 2:** Abstract principles (multi-hop reasoning, generalizations).
  - Dynamic cluster sizing and adaptive clustering frequency.
  - LLM-powered summarization and abstraction.

- **Tier 3: Self-Improvement Engine**
  - Monitors accuracy, speed, and efficiency.
  - Triggers memory reorganization and parameter tuning based on performance.

---

## System Workflow

### 1. Data Ingestion & Storage
- **Experience Addition:**
  - Raw text experiences are embedded using a SentenceTransformer model and stored in both an in-memory dictionary and a persistent ChromaDB vector database.
  - Each note is assigned a unique ID, timestamp, and metadata.

### 2. Hierarchical Organization
- **Level 1 Clustering:**
  - Notes are clustered using KMeans on their embeddings. The number of clusters is determined dynamically.
  - Each cluster is summarized by an LLM (Google Gemini), producing concise, human-readable summaries.
- **Level 2 Abstraction:**
  - Level 1 summaries are meta-clustered into higher-level groups (principles).
  - The LLM abstracts each group into a general principle or life lesson.
- **Persistence:**
  - The hierarchy (clusters, principles, stats) is periodically saved to disk as JSON for persistence and recovery.

### 3. Query & Retrieval
- **Query Classification:**
  - Incoming queries are analyzed for complexity (e.g., specific fact, temporal, causal, multi-hop, abstract) and routed to the appropriate memory level(s).
- **Multi-Level Retrieval:**
  - **Level 0:** Semantic search in ChromaDB for relevant notes; LLM extracts the answer from note content.
  - **Level 1:** Searches cluster summaries; LLM extracts or reasons over the summary.
  - **Level 2:** For complex queries, searches principles; LLM extracts or reasons over the principle.
- **LLM Extraction & Patching:**
  - LLM is used for answer extraction, with robust patching for temporal, aggregation, and numeric/unit edge cases.
  - Fallbacks and post-processing ensure the most specific, correct, and contextually appropriate answer is returned.
- **Retrieval Path Tracking:**
  - Every query logs which level(s) were used, similarity scores, and the retrieval path for transparency.

### 4. Evaluation & Metrics
- **Comprehensive Evaluation:**
  - For each QA pair, the system compares the predicted answer to the ground truth using normalization, partial match, numeric tolerance, and special-case logic.
- **Advanced Metrics:**
  - Tracks accuracy, F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT, retrieval distribution, and average retrieval time.
- **Reporting:**
  - Generates detailed reports and SOTA comparison tables. All results and logs are saved for reproducibility.

### 5. Self-Improvement
- **Performance Monitoring:**
  - Continuously tracks system performance (accuracy, speed, efficiency, retrieval balance).
- **Reorganization:**
  - If performance drops below thresholds or retrieval is imbalanced, triggers memory reorganization and parameter tuning.

### 6. LLM Integration
- All LLM calls use the Gemini API (API key from `.env` or environment).
- Prompts are optimized for each memory level and reasoning type.
- LLM is used for summarization, abstraction, answer extraction, advanced reasoning, and fallback/post-processing.

---

## Evaluation Methodology

- **Accuracy:** Proportion of QA pairs where the system's answer matches the ground truth, using robust normalization, partial match, and numeric tolerance logic.
- **Advanced Metrics:** BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT for semantic and n-gram similarity.
- **Category Analysis:** Entity Tracking, Temporal Reasoning, Causal Reasoning, Multi-hop Reasoning, Adversarial/Challenge.
- **Retrieval Path Tracking:** Logs which memory level was used for each query.
- **SOTA Benchmarking:** Results are directly compared to published baselines (A-MEM, MemoryGPT, MEMO, etc.).
- **Reproducibility:** All evaluation results, logs, and system states are saved in the `results/` and `logs/` directories.

---

## Usage

### Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup
- Set your Gemini API key in a `.env` file:
```
  GEMINI_API_KEY=your_api_key_here
```

### Running the Pipeline
- **Full Evaluation:**
```bash
python run_comprehensive_pipeline.py
  ```
- **Performance Test Suite:**
  ```bash
  python test_system_performance.py
  ```

### Project Structure
```
EVOLVE-MEM/
├── memory_system.py            # Main system interface
├── hierarchical_manager.py     # Tier 2: Hierarchical memory management
├── dynamic_memory.py           # Tier 1: Dynamic memory network
├── self_improvement.py         # Tier 3: Self-improvement engine
├── evaluation.py               # Comprehensive evaluation framework
├── dataset_loader.py           # LoCoMo dataset loader
├── run_comprehensive_pipeline.py  # Full research pipeline
├── test_system_performance.py     # Performance test suite
├── config.py                   # Configuration management
├── llm_backend.py              # LLM integration (Gemini)
├── utils.py                    # Utility functions
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── data/                       # Datasets
├── results/                    # Evaluation outputs
├── logs/                       # Log files
```

---

## Datasets
- **LoCoMo Dataset:**
  - 10 stories, 1,986 QA pairs across 5 reasoning categories.
  - Supports custom datasets with similar structure.

---

## References & Best Practices
- This repository follows best practices for research codebases as seen in:
  - [A-MEM](https://github.com/agiresearch/A-mem)
  - [MemGPT](https://github.com/cpacker/MemGPT)
  - [LangChain](https://github.com/langchain-ai/langchain)
- For evaluation methodology and metrics, see `evaluation.py` and the generated reports in `results/`.

---

## Reproducibility
- All experiments are fully reproducible. Results, logs, and system states are saved for every run.
- Configuration is centralized in `config.py` and can be controlled via environment variables.

---

## Citation
If you use EVOLVE-MEM in your research, please cite the corresponding paper and reference this repository.

## Setup

+**Recommended Python version:** >=3.9, <3.12 for best compatibility 