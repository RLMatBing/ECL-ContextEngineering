# Local Obsidian Graph RAG experiment



Requires Python 3.10+ and Ollama. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve # only if the Ollama service is not already running
python run_experiment.py
```

Uses HotpotQA's public `distractor` validation split, takes a deterministic reservoir sample of 20 bridge questions (seed `20261022`, first 1,000 bridge examples.

Test retrieval without an answer model:

```bash
python run_experiment.py --skip-llm
```

To change only reproducible experimental controls:

```bash
python run_experiment.py --seed 20261022 --max-bridge-scan 1000 --context-tokens 1800 --ollama-model llama3.2:latest
```

## Method


## Outputs

- `Obsidian-RAG-Test/` — vault notes only; open it directly in Obsidian.
- `eval.json` — gold answers/support targets, intentionally outside the vault.
- `selection_ids.json` — deterministic selection record.
- `results.csv` — one Vector and one Graph row per question.
- `report.md` — vault topology, aggregate metric table, and per-question retrieval comparison.

The first run downloads the public dataset and MiniLM model to standard local caches. Subsequent runs reuse them. No run transmits notes, questions, embeddings, or evaluation results to an external inference/database service.

## A/B/C interactive benchmark

The Streamlit demo adds a third, no-retrieval baseline alongside the existing Vector and Graph RAG methods. It uses the same local Ollama model and temperature-zero generation settings in all modes:

- **A — No RAG:** sends only the question to Ollama.
- **B — Obsidian Search:** standalone lexical retrieval over Markdown note titles and bodies; no embedding, FAISS, or link traversal is used.
- **C — Graph RAG:** MiniLM/FAISS top-2 seeds, then one-hop traversal of only parsed Obsidian wikilinks and backlinks with NetworkX, semantic reranking, and the same evidence token ceiling.

Launch the live demo after activating the virtual environment:

```bash
streamlit run app.py
```

The installed local Ollama tag is `llama3.2:3b`. Run the optional 20-question A/B/C batch evaluation with:

```bash
python evaluate.py --ollama-model llama3.2:3b
```

Rebuild the deterministic vault and external evaluation set only when desired:

```bash
python prepare_data.py
```
