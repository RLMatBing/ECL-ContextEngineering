"""Create/recreate the deterministic HotpotQA vault and external gold file."""
import argparse, json
from run_experiment import ROOT, build_vault, gold_payload, sample_bridge_questions
parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=20261022); parser.add_argument('--sample-size',type=int,default=20); parser.add_argument('--max-bridge-scan',type=int,default=1000); args=parser.parse_args()
rows=sample_bridge_questions(args.sample_size,args.max_bridge_scan,args.seed); notes,stats=build_vault(rows,recreate=True)
(ROOT/'eval.json').write_text(json.dumps(gold_payload(rows,notes),indent=2,ensure_ascii=False)+'\n')
(ROOT/'selection_ids.json').write_text(json.dumps({'seed':args.seed,'max_bridge_scan':args.max_bridge_scan,'ids':[r['id'] for r in rows]},indent=2)+'\n')
print(json.dumps({**stats,'graph_nodes':len(notes)},indent=2))
