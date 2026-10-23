"""Run with: streamlit run app.py"""
import streamlit as st
import pandas as pd
from rag import LocalRAG,document_prf,load_eval,local_ollama_models,normalized,token_prf
st.set_page_config(page_title='Local Obsidian RAG Lab',layout='wide');st.title('Local A/B/C HotpotQA RAG Benchmark');st.caption('A: Ollama only • B: standalone Obsidian-style Markdown keyword search • C: vector-seeded Graph RAG over actual Obsidian links/backlinks')
@st.cache_resource
def get_rag(): return LocalRAG()
@st.cache_data
def get_items(): return load_eval()
def graph_dot(seeds, edges):
    """Small dependency-free DOT rendering of actual vault traversal edges."""
    nodes=set(seeds)
    for source,target in edges: nodes.update((source,target))
    lines=['digraph vault {', 'rankdir=LR;', 'node [shape=box, style=rounded, fontname="Arial"];']
    for node in nodes:
        attrs='color="#7c3aed", penwidth=3' if node in seeds else 'color="#64748b"'
        lines.append(f'"{node}" [{attrs}];')
    lines += [f'"{source}" -> "{target}";' for source,target in edges]
    return '\n'.join(lines+['}'])
try: rag=get_rag();items=get_items()
except Exception as e: st.error(str(e));st.stop()
options={f"{x['id']} — {x['question']}":x for x in items};item=options[st.selectbox('HotpotQA bridge question',options)];custom=st.text_input('Optional custom question (no reference answer)');models=local_ollama_models();default=models.index('llama3.2:3b') if 'llama3.2:3b' in models else 0;model=st.selectbox('Local Ollama model',models,index=default);budget=st.number_input('Maximum retrieved evidence tokens',256,4096,1800,128);question=custom.strip() or item['question']
if st.button('Run All Three Modes',type='primary'):
 try:
  with st.spinner('Running locally through Ollama…'): st.session_state['run']=rag.run_modes(question,model,int(budget))
 except Exception as e: st.error(str(e))
if 'run' in st.session_state:
 run=st.session_state['run']
 if not custom.strip():
  st.info(f"**Correct HotpotQA answer:** {item['answer']}")
 for col,key,title in zip(st.columns(3),['baseline','obsidian','graph'],['A — No RAG','B — Obsidian Search','C — Graph RAG']):
  r=run[key]
  with col:
   st.subheader(title);st.write(r['answer']);st.caption(f"Generation {r['generation_latency_s']:.2f}s"+(f" · Retrieval {r['retrieval_latency_s']:.3f}s · {r['context_tokens']} tokens" if key!='baseline' else ' · no evidence'))
   if key!='baseline':
    st.dataframe(pd.DataFrame([{k:d[k] for k in ['title','score','id']} for d in r['documents']]),hide_index=True,use_container_width=True)
    with st.expander('Exact evidence sent to Ollama'):st.text('\n\n'.join(f"NOTE: {d['title']}\n{d['text']}" for d in r['documents']))
   if key=='graph':st.caption(f"Seeds: {', '.join(r['seed_ids']) or 'none'}; expanded: {', '.join(r['expanded_ids']) or 'none'}")
 if not custom.strip():
  st.caption('Exact match: A '+('✓' if normalized(run['baseline']['answer'])==normalized(item['answer']) else '✗')+' · B '+('✓' if normalized(run['obsidian']['answer'])==normalized(item['answer']) else '✗')+' · C '+('✓' if normalized(run['graph']['answer'])==normalized(item['answer']) else '✗'))
  st.subheader('Benchmark metrics')
  answer_rows=[]; doc_rows=[]; gold=set(item['supporting_note_ids'])
  for key,label in [('baseline','A — No RAG'),('obsidian','B — Obsidian Search'),('graph','C — Graph RAG')]:
   precision,recall,f1=token_prf(run[key]['answer'],item['answer']);answer_rows.append({'Mode':label,'Precision':round(precision,3),'Recall':round(recall,3),'F1':round(f1,3)})
   if key!='baseline':
    precision,recall,f1=document_prf({d['id'] for d in run[key]['documents']},gold);doc_rows.append({'Mode':label,'Precision':round(precision,3),'Recall':round(recall,3),'F1':round(f1,3),'All gold documents retrieved':gold <= {d['id'] for d in run[key]['documents']}})
  left,right=st.columns(2)
  with left: st.caption('Answer tokens versus the HotpotQA reference answer');st.dataframe(pd.DataFrame(answer_rows),hide_index=True,use_container_width=True)
  with right: st.caption('Retrieved documents versus HotpotQA gold supporting notes');st.dataframe(pd.DataFrame(doc_rows),hide_index=True,use_container_width=True)
 edges=run['graph']['traversed_edges'];seeds=run['graph']['seed_ids']
 if edges or seeds: st.subheader('Mode C actual vault links traversed');st.graphviz_chart(graph_dot(seeds,edges));st.caption('Purple nodes are vector seed notes. This diagram contains only Markdown wikilinks/backlinks; no model-inferred relationships.')
