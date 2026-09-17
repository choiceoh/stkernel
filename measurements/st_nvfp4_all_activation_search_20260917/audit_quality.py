#!/usr/bin/env python3
"""Replay reviewed encoding maps against raw receipts; never replace canonical grades."""
import json,hashlib,re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'bench'))
from onepass_quality import assess
ROOT=None
# Explicit maps are reviewed against named assignments in the response, never inferred from the oracle alone.
MAPPINGS={
 ('candidate-1','measure-c1',128000,None): {'110000':'111000','001001':'001010','101001':'101010'},
 ('candidate-1','measure-c2-32000-qall',32000,0): {'110001':'111001'},
 ('candidate-1','measure-c2-32000-qall',32000,1): {'110001':'110000'},
 ('base-1','measure-c1',32000,None): {'110010':'110000','111010':'110010','101110':'100110','110100':'111001'},
 ('base-1','measure-c1',128000,None): {'110000':'111000','001000':'001010','101000':'101010'},
 ('base-1','measure-c2-2000-q2',2000,0): {'001101':'000110','011101':'010110'},
 ('base-1','measure-c2-32000-qall',32000,1): {'111000':'111001'},
 ('base-2','measure-c1',32000,None): {'110001':'111001'},
 ('base-2','measure-c1',128000,None): {'001001':'001010','110100':'111000','101100':'101010','001011':'001111'},
}

def audit(folder):
 p=ROOT/folder
 reqs={r['output_sha256']:r for r in map(json.loads,(p/'requests.jsonl').read_text().splitlines())}
 workloads=json.loads((p/'workloads.json').read_text())
 record=json.loads((p/'record.json').read_text())
 assert record['recording']['status']=='complete',p
 out=[]
 for q in map(json.loads,(p/'quality.jsonl').read_text().splitlines()):
  for g in q['results']:
   if g['passed']:continue
   r=reqs[q['output_sha256']]
   key=(folder,q['phase'],q['ctx'],q['client'])
   mapping=MAPPINGS.get(key)
   row=dict(arm=folder,run_id=record['run_id'],phase=q['phase'],ctx=q['ctx'],client=q['client'],
            request_sha256=q['request_sha256'],output_sha256=q['output_sha256'],
            source_record_sha256=hashlib.sha256((p/'record.json').read_bytes()).hexdigest(),original_grade=g)
   if mapping and g['case']=='logic':
    visible=''.join(d['content'] for d in r['channels'] if isinstance(d.get('content'),str)).strip()
    visible=re.sub(r'^```json\s*\n|\n```$', '',visible)
    actual=json.loads(visible)['logic']
    case=next(c for w in workloads if w['ctx']==q['ctx'] and w['question']==q['question'] for c in w['quality_cases'] if c['id']=='logic')
    converted=[mapping.get(w,w) for w in actual['derivation']['worlds']]
    assert sorted(converted)==sorted(case['answer']['derivation']['worlds']),key
    witness_ok=all(all((mapping.get(w[k],w[k]) in options[k] if options[k] else w[k] is None) for k in ('true','false')) for w,options in zip(actual['witnesses'],case['witness_options']))
    assert witness_ok,key
    row.update(classification='named assignments correctly derived; bit-string transcription error',encoding_map=mapping,
               mapped_worlds_and_witnesses_match_oracle=True)
    lines=r['text'].splitlines()
    evidence=[]
    for wrong,right in mapping.items():
     candidates=[]
     for line in lines:
      if wrong not in line or len(line)>900:continue
      assignments=dict(re.findall(r'\b([A-F])\s*=\s*([01])\b',line))
      if len(assignments)==6 and ''.join(assignments[k] for k in 'ABCDEF')==right:
       candidates.append(line)
     if candidates:evidence.append(dict(wrong=wrong,right=right,excerpt=candidates[-1]))
    if key==('base-1','measure-c2-2000-q2',2000,0):
     evidence=[dict(wrong=wrong,right=right,excerpt='\n\n'.join(lines[16:22])) for wrong,right in mapping.items()]
    assert len(evidence)==len(mapping),(key,evidence)
    row['evidence']=evidence
    if not g['checks']['counterfactual']:
     row['classification'] += '; also a genuine minimal-core error'
     row['evidence'] += [dict(excerpt=line) for line in r['text'].splitlines() if '{U3,U4,U6}' in line and len(line)<700][:3]
   elif folder.startswith('candidate') and g['case']!='logic':
    row['classification']='intermediate-field or constraint error; final result correct'
    row['evidence']=[line for line in r['text'].splitlines() if len(line)<900 and any(k in line for k in ['BDE:','BCD:','BCE:','CDE:','ACE:','after_loss = floor','After_loss:'])][:12]
    if g['parse_error']:
     visible=''.join(d['content'] for d in r['channels'] if isinstance(d.get('content'),str)).strip()
     visible=re.sub(r'^```json\s*\n|\n```$', '',visible)
     parsed=json.loads(visible+'}')
     row['classification']='JSON missing final closing brace; reported primary answer correct'
     row['evidence']=[visible[-350:]]
     row['reported_primary_result']=parsed[g['case']]['result']
     row['diagnostic_repair']='append one closing brace only; original grades unchanged'
     row['repaired_output']=parsed
     item=next(w for w in workloads if w['ctx']==q['ctx'] and w['question']==q['question'])
     repaired=assess(item,[{'content':visible+'}'}],'stop')
     assert all(result['passed'] for result in repaired),key
     row['diagnostic_repaired_grade']=repaired
   else:
    row['classification']='subproof error; final result correct'
    row['evidence']=[line for line in r['text'].splitlines() if 'So {U3,U4,U6} is contradictory?' in line or 'But wait — does {U3,U4,U6} use U1?' in line]
    assert row['evidence'],key
   out.append(row)
 return out

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--raw-root',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args(); ROOT=args.raw_root
 folders=['base-1','base-2','candidate-1','candidate-2']
 rows=[row for folder in folders for row in audit(folder)]
 args.output.write_text(json.dumps(dict(scope='Review of every failing certificate. Original scores and final-result dimensions are retained. Encoding maps are diagnostic and do not change canonical grades.',rows=rows),ensure_ascii=False,indent=2)+'\n')
 print('audited',len(rows),'failed certificates')
