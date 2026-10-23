"""Local A/B/C RAG implementations over the Obsidian Markdown vault."""
from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Any
import faiss
import networkx as nx
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
VAULT = ROOT / "Obsidian-RAG-Test"

def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())

def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter
    p, r = normalized(prediction).split(), normalized(reference).split()
    common = sum((Counter(p) & Counter(r)).values())
    return 2 * common / (len(p) + len(r)) if p and r else float(p == r)

def token_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    """Normalized answer-token precision, recall, and F1 against a gold answer."""
    from collections import Counter
    p, r = normalized(prediction).split(), normalized(reference).split()
    overlap = sum((Counter(p) & Counter(r)).values())
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def document_prf(retrieved: set[str], gold: set[str]) -> tuple[float, float, float]:
    overlap = len(retrieved & gold)
    precision = overlap / len(retrieved) if retrieved else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

class LocalRAG:
    """One shared MiniLM/FAISS index and an edge-faithful Obsidian graph."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.notes = self._load_notes()
        if not self.notes: raise RuntimeError("Vault is empty. Run `python prepare_data.py` first.")
        self.by_id = {n["id"]: n for n in self.notes}
        self.model = SentenceTransformer(model_name); self.tokenizer = self.model.tokenizer
        self.embeddings = self.model.encode([n["text"] for n in self.notes], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1]); self.index.add(self.embeddings)
        self.pos = {n["id"]: i for i, n in enumerate(self.notes)}
        self.graph = nx.DiGraph(); self.graph.add_nodes_from(self.by_id)
        self.graph.add_edges_from((n["id"], target) for n in self.notes for target in n["outgoing"] if target in self.by_id)

    def _load_notes(self) -> list[dict[str, Any]]:
        notes = []
        for path in sorted(VAULT.glob("*.md")):
            raw = path.read_text(encoding="utf-8").strip(); title, _, body = raw.partition("\n")
            targets = re.findall(r"\[\[([^]|]+)(?:\|[^]]+)?\]\]", body)
            outgoing = [x.rsplit("--", 1)[-1] for x in targets if "--" in x]
            clean = re.sub(r"\[\[([^]|]+)\|([^]]+)\]\]", r"\2", body)
            notes.append({"id":path.stem.rsplit("--",1)[-1], "title":title.removeprefix("# "), "text":clean, "path":path.name, "outgoing":outgoing})
        return notes

    def search(self, question: str, k: int) -> list[tuple[dict[str, Any], float]]:
        q = self.model.encode([question], normalize_embeddings=True).astype("float32")
        scores, pos = self.index.search(q, min(k, len(self.notes)))
        return [(self.notes[i], float(s)) for s,i in zip(scores[0],pos[0])]

    def tokens(self, text: str) -> int: return len(self.tokenizer.encode(text, add_special_tokens=False))
    def select_budget(self, ranked: list[tuple[dict[str, Any],float]], budget: int):
        out=[]; used=0
        for note, score in ranked:
            cost=self.tokens(f"[{note['title']}]\n{note['text']}")
            if used+cost<=budget: out.append((note,score)); used+=cost
        return out
    def _trace(self, mode, docs, latency, seeds=(), edges=(), expanded=()):
        return {"mode":mode,"documents":[{"id":n["id"],"title":n["title"],"score":round(s,6),"text":n["text"]} for n,s in docs],"seed_ids":list(seeds),"traversed_edges":list(edges),"expanded_ids":list(expanded),"retrieval_latency_s":latency,"context_tokens":sum(self.tokens(n["text"]) for n,_ in docs)}
    def vector(self, question: str, budget=1800):
        started=time.perf_counter(); docs=self.select_budget(self.search(question,5),budget)
        return self._trace("vector",docs,time.perf_counter()-started)
    def obsidian_search(self, question: str, budget=1800):
        """Standalone lexical Markdown search, deliberately without embeddings or links.

        This mirrors the reproducible core of an Obsidian vault text search: note titles
        and bodies are matched by terms. It does not call Obsidian's UI/search index.
        """
        started=time.perf_counter(); terms=set(normalized(question).split())
        ranked=[]
        for note in self.notes:
            title_terms=normalized(note['title']).split(); body_terms=normalized(note['text']).split()
            # Title hits are weighted above body hits, like a useful vault search.
            score=3*sum(t in terms for t in title_terms)+sum(t in terms for t in body_terms)
            if score: ranked.append((note,float(score)))
        ranked.sort(key=lambda x:(-x[1],x[0]['title']))
        return self._trace("obsidian",self.select_budget(ranked[:5],budget),time.perf_counter()-started)
    def graph_rag(self, question: str, budget=1800):
        started=time.perf_counter(); pairs=self.search(question,2); seeds=[n["id"] for n,_ in pairs]; candidates=set(seeds); edges=[]
        for seed in seeds:
            for node in set(self.graph.successors(seed)) | set(self.graph.predecessors(seed)):
                candidates.add(node); edges.append((seed,node))
        q=self.model.encode([question],normalize_embeddings=True).astype("float32")[0]
        ranked=sorted(((self.by_id[i],float(np.dot(q,self.embeddings[self.pos[i]]))) for i in candidates),key=lambda x:x[1],reverse=True)
        return self._trace("graph",self.select_budget(ranked,budget),time.perf_counter()-started,seeds,edges,sorted(candidates-set(seeds)))
    def answer(self, question: str, trace: dict | None, model="llama3.2:3b"):
        evidence="" if trace is None else "\n\nEvidence:\n"+"\n\n".join(f"NOTE: {d['title']}\n{d['text']}" for d in trace["documents"])
        instruction="Use only the provided evidence and cite note titles on a line starting Citations:." if trace else "Use your own knowledge. Do not cite documents."
        started=time.perf_counter()
        try:
            r=requests.post("http://localhost:11434/api/generate",json={"model":model,"prompt":f"Answer concisely. {instruction}\n\nQuestion: {question}{evidence}","stream":False,"options":{"temperature":0}},timeout=600); r.raise_for_status(); raw=r.json()["response"].strip()
        except requests.RequestException as exc: raise RuntimeError(f"Ollama generation failed: {exc}. Start `ollama serve`.") from exc
        return {"answer":raw.split("Citations:",1)[0].strip(),"raw":raw,"generation_latency_s":time.perf_counter()-started}
    def run_modes(self, question: str, model="llama3.2:3b", budget=1800):
        base=self.answer(question,None,model); obsidian=self.obsidian_search(question,budget); obsidian.update(self.answer(question,obsidian,model)); graph=self.graph_rag(question,budget); graph.update(self.answer(question,graph,model)); return {"baseline":base,"obsidian":obsidian,"graph":graph}

def load_eval():
    path=ROOT/"eval.json"
    if not path.exists(): raise RuntimeError("Missing eval.json. Run `python prepare_data.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))

def local_ollama_models() -> list[str]:
    """Return only models exposed by the user's local Ollama server."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
    except requests.RequestException:
        return ["llama3.2:3b"]
