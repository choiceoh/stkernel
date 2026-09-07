"""One isolated instrumented request, with cleanup after partial RPC failures.

The outer runner must attest and own the private four-rank serving boot, capture
trace inventories and collect fresh files. This helper does not boot a server,
measure uninstrumented TTFT or authorize deployment. There is no standalone CLI.
"""
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
from glm53_prefill_observer import validate_ranks
from glm53_offline_checks import check_holder


class PrivateObserverAPI:
    def __init__(self, base):
        if base != 'http://127.0.0.1:18000':
            raise ValueError('owned loopback diagnostic endpoint on port 18000 required')
        self.base = base

    def post(self, path, payload=None):
        check_holder()
        if path not in ('/glm53/prefill-observe', '/start_profile', '/stop_profile'):
            raise ValueError('unsupported observation endpoint')
        data = b'' if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base+path, data=data,
                                         headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if path == '/glm53/prefill-observe' else {'status':response.status}


def idle_observers(reply, source_sha256):
    ranks = reply.get('ranks', [])
    if (len(ranks) != 4 or {r.get('rank') for r in ranks} != {0,1,2,3}
            or any(type(r.get('rank')) is not int or r.get('active') is not False
                   or r.get('source_sha256') != source_sha256 for r in ranks)):
        raise ValueError('all four exact-source observers must be idle')


def collect_request(*, api, mode, request_id, source_sha256, request):
    """Run one already-attested request; return incomplete evidence on failure.

Always attempt cleanup after a begin/start *attempt*, even if the RPC response
was lost. The outer runner must still destroy its diagnostic boot and recover
incoming serving when cleanup cannot be proven.
    """
    if mode not in ('profile','routes'):
        raise ValueError('only separately instrumented profile/routes requests are supported')
    evidence = dict(schema=1, request_id=request_id, mode=mode, complete=False, errors=[],
                    performance_acceptance=False, observation=None, request=None)
    begin_attempted = profile_attempted = False
    try:
        evidence['before'] = api.post('/glm53/prefill-observe', {'op':'status'})
        idle_observers(evidence['before'], source_sha256)
        begin_attempted = True
        begun = api.post('/glm53/prefill-observe', dict(op='begin',mode=mode,request_id=request_id))
        ranks = begun.get('ranks', [])
        if (len(ranks) != 4 or {r.get('rank') for r in ranks} != {0,1,2,3}
                or any(type(r.get('rank')) is not int or r.get('active') is not True
                       or r.get('source_sha256') != source_sha256
                       or r.get('request_id') != request_id or r.get('mode') != mode for r in ranks)):
            raise ValueError('four exact-source observer begin acknowledgments required')
        evidence['begin'] = begun
        if mode == 'profile':
            profile_attempted = True
            api.post('/start_profile')
        evidence['request'] = request()
    except Exception as exc:
        evidence['errors'].append('collection: '+repr(exc))
    finally:
        # stop first: include all model annotations while observers are live.
        if profile_attempted:
            try:
                api.post('/stop_profile')
                evidence['profiler_stopped'] = True
            except Exception as exc:
                evidence['errors'].append('profiler cleanup: '+repr(exc))
                evidence['profiler_stopped'] = False
        if begin_attempted:
            try:
                reply = api.post('/glm53/prefill-observe', {'op':'end'})
                evidence['observation'] = reply
                validate_ranks(reply.get('ranks', []), request_id=request_id, mode=mode, source_sha256=source_sha256)
            except Exception as exc:
                evidence['errors'].append('observer completion: '+repr(exc))
            try:
                evidence['after'] = api.post('/glm53/prefill-observe', {'op':'status'})
                idle_observers(evidence['after'], source_sha256)
            except Exception as exc:
                evidence['errors'].append('observer cleanup: '+repr(exc))
    evidence['complete'] = not evidence['errors'] and evidence['request'] is not None
    return evidence
