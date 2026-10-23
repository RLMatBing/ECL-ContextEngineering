#!/usr/bin/env python3
"""Reproducible local Vector RAG vs Obsidian-link Graph RAG experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
VAULT = ROOT / "Obsidian-RAG-Test"
WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "", value).strip().replace(" ", "-")
    return clean[:80] or "untitled"


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def f1(prediction: str, gold: str) -> float:
    p, g = normalize(prediction).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((__import__("collections").Counter(p) & __import__("collections").Counter(g)).values())
    return 2 * overlap / (len(p) + len(g)) if overlap else 0.0


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_bridge_questions(size: int, max_scan: int, seed: int) -> list[dict[str, Any]]:
    """A deterministic streaming reservoir sample; IDs are also persisted for replay."""
    stream = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    reservoir: list[dict[str, Any]] = []
    import random
    rng = random.Random(seed)
    bridges = 0
    for row in stream:
        if row["type"] != "bridge":
            continue
        bridges += 1
        if len(reservoir) < size:
            reservoir.append(row)
        else:
            slot = rng.randrange(bridges)
            if slot < size:
                reservoir[slot] = row
        if bridges >= max_scan:
            break
    if len(reservoir) != size:
        raise RuntimeError(f"Only found {len(reservoir)} bridge questions in {max_scan} streamed rows.")
    reservoir.sort(key=lambda x: x["id"])
    return reservoir


def paragraph_id(title: str, sentences: list[str]) -> str:
    return hashlib.sha1((title + "\n" + "\n".join(sentences)).encode()).hexdigest()[:12]


def build_vault(rows: list[dict[str, Any]], recreate: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if recreate and VAULT.exists():
        shutil.rmtree(VAULT)
    VAULT.mkdir(exist_ok=True)
    notes: dict[str, dict[str, Any]] = {}
    for row in rows:
        contexts = row["context"]
        for title, sentences in zip(contexts["title"], contexts["sentences"]):
            text = " ".join(sentences)
            ident = paragraph_id(title, sentences)
            notes.setdefault(ident, {"id": ident, "title": title, "sentences": sentences, "text": text})
    title_to_ids: dict[str, list[str]] = defaultdict(list)
    for note in notes.values():
        title_to_ids[note["title"]].append(note["id"])
    # A title that maps to multiple paragraphs is deliberately not linked: that is ambiguous.
    unique_titles = {title: ids[0] for title, ids in title_to_ids.items() if len(ids) == 1}
    title_pattern = re.compile("|".join(re.escape(t) for t in sorted(unique_titles, key=len, reverse=True)), re.I) if unique_titles else None
    edges: set[tuple[str, str]] = set()
    for note in notes.values():
        def replace(match: re.Match[str]) -> str:
            title = match.group(0)
            target = unique_titles.get(title)
            if target is None or target == note["id"]:
                return title
            edges.add((note["id"], target))
            return f"[[{slug(notes[target]['title'])}--{target}|{title}]]"
        linked = title_pattern.sub(replace, note["text"]) if title_pattern else note["text"]
        note["linked_text"] = linked
        note["path"] = f"{slug(note['title'])}--{note['id']}.md"
        (VAULT / note["path"]).write_text(f"# {note['title']}\n\n{linked}\n", encoding="utf-8")
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        outgoing[source].add(target)
        incoming[target].add(source)
    for note in notes.values():
        note["outgoing"] = sorted(outgoing[note["id"]])
        note["incoming"] = sorted(incoming[note["id"]])
    isolated = sum(not n["outgoing"] and not n["incoming"] for n in notes.values())
    stats = {"notes": len(notes), "links": len(edges), "isolated_nodes": isolated,
             "ambiguous_titles_not_linked": sum(len(x) > 1 for x in title_to_ids.values())}
    return list(notes.values()), stats


def gold_payload(rows: list[dict[str, Any]], notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title: dict[str, list[str]] = defaultdict(list)
    for note in notes:
        by_title[note["title"]].append(note["id"])
    payload = []
    for row in rows:
        support_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
        # A title can occur in multiple notes; counting any such note avoids arbitrary gold resolution.
        gold_ids = sorted({ident for title in support_titles for ident in by_title.get(title, [])})
        payload.append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                        "supporting_titles": support_titles, "supporting_note_ids": gold_ids})
    return payload


class Retriever:
    def __init__(self, notes: list[dict[str, Any]], model_name: str):
        self.notes = notes
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.embeddings = self.model.encode([n["text"] for n in notes], normalize_embeddings=True,
                                            show_progress_bar=True).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.position = {note["id"]: i for i, note in enumerate(notes)}

    def search(self, question: str, k: int) -> list[tuple[dict[str, Any], float]]:
        query = self.model.encode([question], normalize_embeddings=True).astype("float32")
        scores, positions = self.index.search(query, min(k, len(self.notes)))
        return [(self.notes[i], float(s)) for s, i in zip(scores[0], positions[0])]

    def graph_search(self, question: str, seed_k: int, budget: int) -> tuple[list[tuple[dict[str, Any], float]], list[str], list[tuple[str, str]]]:
        seeds = self.search(question, seed_k)
        candidate_ids = {n["id"] for n, _ in seeds}
        traversed: list[tuple[str, str]] = []
        for note, _ in seeds:
            for target in note["outgoing"] + note["incoming"]:
                candidate_ids.add(target)
                traversed.append((note["id"], target))
        query = self.model.encode([question], normalize_embeddings=True).astype("float32")[0]
        ranked = sorted(((self.notes[self.position[i]], float(np.dot(query, self.embeddings[self.position[i]]))) for i in candidate_ids), key=lambda x: x[1], reverse=True)
        return select_budget(ranked, self.tokenizer, budget), [n["id"] for n, _ in seeds], traversed


def select_budget(ranked: list[tuple[dict[str, Any], float]], tokenizer: Any, budget: int) -> list[tuple[dict[str, Any], float]]:
    selected, used = [], 0
    for note, score in ranked:
        cost = token_count(tokenizer, f"[{note['title']}]\n{note['text']}")
        if selected and used + cost > budget:
            continue
        if cost > budget and not selected:
            # Truncate only the displayed evidence, retaining its source identity.
            words = note["text"].split()
            clipped = dict(note, text=" ".join(words[: max(1, budget * 3)]))
            selected.append((clipped, score))
            break
        selected.append((note, score)); used += cost
    return selected


def ollama_answer(question: str, docs: list[tuple[dict[str, Any], float]], model: str) -> tuple[str, float]:
    evidence = "\n\n".join(f"NOTE: {n['title']}\n{n['text']}" for n, _ in docs)
    prompt = ("Answer the question using only the evidence below. Give a short answer, then a new line "
              "starting with 'Citations:' and list the exact note titles used. If evidence is insufficient, say 'Insufficient evidence'.\n\n"
              f"Question: {question}\n\nEvidence:\n{evidence}")
    started = time.perf_counter()
    response = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt,
                             "stream": False, "options": {"temperature": 0}}, timeout=600)
    response.raise_for_status()
    return response.json()["response"].strip(), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-bridge-scan", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--ollama-model", default="llama3.2:latest")
    parser.add_argument("--context-tokens", type=int, default=1800)
    parser.add_argument("--skip-llm", action="store_true", help="Run retrieval metrics only.")
    parser.add_argument("--keep-vault", action="store_true")
    args = parser.parse_args()
    rows = sample_bridge_questions(args.sample_size, args.max_bridge_scan, args.seed)
    (ROOT / "selection_ids.json").write_text(json.dumps({"seed": args.seed, "max_bridge_scan": args.max_bridge_scan,
        "ids": [r["id"] for r in rows]}, indent=2) + "\n")
    notes, vault_stats = build_vault(rows, recreate=not args.keep_vault)
    gold = gold_payload(rows, notes)
    (ROOT / "eval.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n")
    retriever = Retriever(notes, args.model)
    results: list[dict[str, Any]] = []
    for row, target in zip(rows, gold):
        for system in ("vector", "graph"):
            started = time.perf_counter()
            if system == "vector":
                ranked = retriever.search(row["question"], 5)
                docs = select_budget(ranked, retriever.tokenizer, args.context_tokens)
                seeds, edges = [], []
            else:
                docs, seeds, edges = retriever.graph_search(row["question"], 2, args.context_tokens)
            retrieval_s = time.perf_counter() - started
            doc_ids = [n["id"] for n, _ in docs]
            retrieved_titles = [n["title"] for n, _ in docs]
            gold_ids = set(target["supporting_note_ids"])
            recall = len(gold_ids & set(doc_ids)) / len(gold_ids) if gold_ids else 0.0
            llm_text, llm_s = ("", 0.0) if args.skip_llm else ollama_answer(row["question"], docs, args.ollama_model)
            answer = llm_text.split("Citations:", 1)[0].strip() if llm_text else ""
            results.append({"question_id": row["id"], "system": system, "question": row["question"], "gold_answer": row["answer"],
                "answer": answer, "answer_exact_match": int(normalize(answer) == normalize(row["answer"])), "answer_f1": round(f1(answer, row["answer"]), 4),
                "supporting_doc_recall": round(recall, 4), "all_supporting_docs_retrieved": int(bool(gold_ids) and gold_ids <= set(doc_ids)),
                "retrieved_document_ids": json.dumps(doc_ids), "retrieved_titles": json.dumps(retrieved_titles),
                "similarity_scores": json.dumps([round(s, 6) for _, s in docs]), "seed_node_ids": json.dumps(seeds), "traversed_edges": json.dumps(edges),
                "retrieval_latency_s": round(retrieval_s, 4), "llm_latency_s": round(llm_s, 4), "total_latency_s": round(retrieval_s + llm_s, 4),
                "retrieved_token_count": sum(token_count(retriever.tokenizer, n["text"]) for n, _ in docs), "llm_raw_response": llm_text})
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    by_q = defaultdict(dict)
    for result in results: by_q[result["question_id"]][result["system"]] = result
    lines = ["# Local Obsidian Graph RAG experiment", "", "## Vault", "", f"- Notes: {vault_stats['notes']}", f"- Natural wikilinks: {vault_stats['links']}", f"- Isolated nodes: {vault_stats['isolated_nodes']}", f"- Ambiguous titles deliberately not linked: {vault_stats['ambiguous_titles_not_linked']}", "", "## Aggregate results", "", "| System | EM | F1 | Supporting recall | All support | Mean total latency (s) | Mean tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for system in ("vector", "graph"):
        group = [x for x in results if x["system"] == system]
        mean = lambda key: sum(float(x[key]) for x in group) / len(group)
        lines.append(f"| {system} | {mean('answer_exact_match'):.3f} | {mean('answer_f1'):.3f} | {mean('supporting_doc_recall'):.3f} | {mean('all_supporting_docs_retrieved'):.3f} | {mean('total_latency_s'):.3f} | {mean('retrieved_token_count'):.1f} |")
    lines += ["", "## Per-question comparison", "", "| Question ID | Vector support recall | Graph support recall | Change | Interpretation |", "|---|---:|---:|---|---|"]
    for qid, pair in by_q.items():
        v, g = pair["vector"], pair["graph"]; delta = g["supporting_doc_recall"] - v["supporting_doc_recall"]
        reason = "Graph expansion added linked evidence." if delta > 0 else ("Graph candidate ranking/context budget excluded vector evidence." if delta < 0 else "No supporting-document change from available links.")
        lines.append(f"| {qid} | {v['supporting_doc_recall']:.2f} | {g['supporting_doc_recall']:.2f} | {delta:+.2f} | {reason} |")
    density = vault_stats['links'] / vault_stats['notes'] if vault_stats['notes'] else 0
    verdict = "enough" if vault_stats['links'] >= args.sample_size and vault_stats['isolated_nodes'] < vault_stats['notes'] * .5 else "not enough"
    lines += ["", "## Graph validity", "", f"The vault has {density:.2f} links per note. It is **{verdict}** natural link structure for a meaningful Graph RAG comparison under this sample; inspect the per-question deltas because sparse explicit title mentions limit graph expansion.", "", "All links were derived only from explicit article-title mentions in context paragraphs. Gold answers and supporting facts are stored separately and never create edges."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Complete: {len(results)} runs. Wrote Obsidian-RAG-Test/, eval.json, results.csv, report.md")

if __name__ == "__main__":
    main()
