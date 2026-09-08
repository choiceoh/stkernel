"""Canonical Korean requests for the owned observation boot; raw SSE retained."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import urllib.request

import onepass
from onepass_fresh import FreshRequests
from prefill_observation import PrivateObserverAPI, collect_request, idle_observers
from window_metrics import metric_sum, traffic_state, exclusive_errors

BASE='http://127.0.0.1:18000'


def validate_baseline(record, requests):
    expected=[(2000,0),(2000,1),(2000,2),(32000,'all'),(128000,'all')]
    if ([(r['ctx'],r['question']) for r in record['requests']]!=expected
            or record['quality']!={'ok':9,'total':9} or record['korean']['dirty']
            or record.get('evidence_issues') or len(requests)!=5
            or any(r.get('issues') or r.get('error') for r in requests)):
        raise RuntimeError('canonical baseline quality, request coverage or traffic gate failed')


def prefill_identity(wire):
    payload=json.loads(wire)
    # Token limit differs in instrumentation; the complete prompt and sampling
    # identity must still agree with the corresponding canonical request.
    return hashlib.sha256(json.dumps({k:v for k,v in payload.items()
        if k not in ('cache_salt','max_tokens')},sort_keys=True).encode()).hexdigest()


class CapturedResponse:
    def __init__(self, response, chunks):self.response,self.chunks=response,chunks
    def __enter__(self):self.response.__enter__();return self
    def __exit__(self,*args):return self.response.__exit__(*args)
    def __iter__(self):
        for line in self.response:self.chunks.append(line);yield line


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('baseline','profile','routes'),required=True)
    ap.add_argument('--name',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--ctx',type=int,choices=(2000,32000,128000))
    ap.add_argument('--observer-sha',required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit(143)))
    if os.environ.get('HEAD_URL')!=BASE or os.environ.get('GLM53_API_PORT')!='18000':
        raise ValueError('private endpoint environment required for every benchmark helper')
    api=PrivateObserverAPI(BASE)
    idle_observers(api.post('/glm53/prefill-observe',{'op':'status'}),args.observer_sha)
    opener=urllib.request.urlopen
    with opener(BASE+'/openapi.json',timeout=10) as response:spec=json.load(response)
    if 'cache_salt' not in spec['components']['schemas']['ChatCompletionRequest']['properties']:
        raise RuntimeError('request-level cache isolation unavailable')
    def metrics():
        with opener(BASE+'/metrics',timeout=10) as response:text=response.read().decode()
        return dict(traffic=traffic_state(text),prefix_hits=metric_sum(text,'vllm:prefix_cache_hits_total'),
                    prefix_queries=metric_sum(text,'vllm:prefix_cache_queries_total'))
    raw=[]
    def captured_open(request,*pos,**kw):
        record=dict(wire=request.data,chunks=[]);raw.append(record)
        return CapturedResponse(opener(request,*pos,**kw),record['chunks'])
    original=onepass.ask_stream
    fresh=FreshRequests(BASE+'/v1/chat/completions',original,captured_open,metrics)
    # FreshRequests.open delegates non-model traffic to its opener as well.
    def open_request(request,*pos,**kw):
        if getattr(request,'full_url',request)!=fresh.url:return opener(request,*pos,**kw)
        return fresh.open(request,*pos,**kw)
    old_argv=sys.argv
    report=dict(mode=args.mode,name=args.name,complete=False,performance_acceptance=False)
    code=1
    try:
        urllib.request.urlopen=open_request;onepass.ask_stream=fresh.call
        if args.mode=='baseline':
            sys.argv=['onepass.py','--name',args.name,'--ctx','2000,32000,128000',
                      '--combine-min-ctx','32000','--fixed-decode-tokens','0',
                      '--require-exclusive','--seed','7','--out',str(args.out/'onepass.jsonl')]
            code=onepass.main() or 0
            record=json.loads((args.out/'onepass.jsonl').read_text().strip())
            report['onepass']=record
            validate_baseline(record,fresh.records)
        else:
            if args.ctx is None:raise ValueError('instrumented context required')
            cq=onepass._load('check-quality.py','observation_quality')
            br=onepass._load('bracket.py','observation_bracket')
            bd=onepass._load('bench-dec.py','observation_bench')
            doc=cq.build(args.ctx,7+args.ctx)
            questions='\n'.join(f'{i+1}. {q}' for i,(_,q,_) in enumerate(cq.FACTS))
            suffix=(onepass.INSTRUCTION+cq.FACTS[0][1] if args.ctx==2000
                    else onepass.INSTRUCTION_COMBINED+questions)
            content='문서:\n'+doc+'\n\n'+suffix
            def request():
                with br._StepWindows(bd,period=1.0) as watch:
                    result=fresh.call(fresh.url,'glm-5.3-flash',content,1)
                item=fresh.records[-1]
                issues=exclusive_errors(item['before']['traffic'],item['after']['traffic'],watch.traffic_samples,1)
                if issues:raise RuntimeError(str(issues))
                return dict(output=result[0],prompt_tokens=result[2],completion_tokens=result[3],
                            finish_reason=result[4],traffic_samples=watch.traffic_samples)
            report['instrumented']=collect_request(api=api,mode=args.mode,request_id=args.name,
                source_sha256=args.observer_sha,request=request)
            if not report['instrumented']['complete']:raise RuntimeError('instrumented request incomplete')
            if len(fresh.records)!=1:raise RuntimeError('exactly one instrumented request required')
            code=0
        idle_observers(api.post('/glm53/prefill-observe',{'op':'status'}),args.observer_sha)
        report['complete']=True
    except BaseException as exc:
        report['error']=repr(exc)
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
    finally:
        urllib.request.urlopen=opener;onepass.ask_stream=original;sys.argv=old_argv
        report['fresh_requests']=fresh.records
        report['raw']=[]
        for i,item in enumerate(raw):
            wire=item['wire'];sse=b''.join(item['chunks'])
            (args.out/f'{i}.request.json').write_bytes(wire)
            (args.out/f'{i}.sse.gz').write_bytes(gzip.compress(sse,mtime=0))
            report['raw'].append(dict(request_sha256=hashlib.sha256(wire).hexdigest(),
                sse_sha256=hashlib.sha256(sse).hexdigest(),prefill_identity_sha256=prefill_identity(wire)))
        (args.out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return code if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
