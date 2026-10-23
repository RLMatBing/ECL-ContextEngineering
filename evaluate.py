
import argparse,csv
from collections import defaultdict
from rag import ROOT,LocalRAG,load_eval,normalized,token_f1
p=argparse.ArgumentParser();p.add_argument('--ollama-model',default='llama3.2:3b');p.add_argument('--context-tokens',type=int,default=1800);a=p.parse_args();rag=LocalRAG();rows=[]
for item in load_eval():
 for mode,result in rag.run_modes(item['question'],a.ollama_model,a.context_tokens).items():
  ids={d['id'] for d in result.get('documents',[])};gold=set(item['supporting_note_ids'])
  rows.append({'question_id':item['id'],'mode':mode,'question':item['question'],'reference_answer':item['answer'],'answer':result['answer'],'exact_match':int(normalized(result['answer'])==normalized(item['answer'])),'answer_f1':round(token_f1(result['answer'],item['answer']),4),'supporting_doc_recall':round(len(ids&gold)/len(gold),4) if gold else '', 'all_supporting_docs':int(bool(gold) and gold<=ids) if mode!='baseline' else '','retrieval_latency_s':round(result.get('retrieval_latency_s',0),4),'generation_latency_s':round(result['generation_latency_s'],4),'total_latency_s':round(result.get('retrieval_latency_s',0)+result['generation_latency_s'],4),'context_tokens':result.get('context_tokens',0),'source_titles':' | '.join(d['title'] for d in result.get('documents',[])),'seed_ids':' | '.join(result.get('seed_ids',[])),'traversed_edges':str(result.get('traversed_edges',[]))})
with (ROOT/'results.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
g=defaultdict(list)
for r in rows:g[r['mode']].append(r)
lines=['# A/B/C local RAG benchmark','', '| Mode | EM | F1 | Support recall | All support | Mean latency (s) | Mean context tokens |','|---|---:|---:|---:|---:|---:|---:|']
for mode,x in g.items():
 avg=lambda k:sum(float(i[k]) for i in x)/len(x); support='—' if mode=='baseline' else f'{avg("supporting_doc_recall"):.3f}'; all_docs='—' if mode=='baseline' else f'{avg("all_supporting_docs"):.3f}'
 lines.append(f'| {mode} | {avg("exact_match"):.3f} | {avg("answer_f1"):.3f} | {support} | {all_docs} | {avg("total_latency_s"):.3f} | {avg("context_tokens"):.1f} |')
lines+=['','A: no retrieval. B: standalone lexical search of Obsidian Markdown note titles/bodies, with no embeddings or link traversal. C: FAISS top-2 followed by only existing Obsidian wikilinks/backlinks using NetworkX, then semantic reranking.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n');print(f'Wrote {len(rows)} rows')
