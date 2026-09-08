#!/usr/bin/env python3
"""Record SSE channels around the unchanged onepass request and quality gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request


class Channels:
    def __init__(self, response, request):
        self.response = response
        self.request_sha256 = hashlib.sha256(request.data).hexdigest()
        self.parts = {k: [] for k in ('content', 'reasoning_content', 'reasoning')}
        self.combined = []
        self.finish_reason = None

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def __iter__(self):
        for raw in self.response:
            # Let onepass timestamp the original chunk before our accounting.
            yield raw
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            try:
                data = json.loads(line[5:].strip())
            except ValueError:
                continue
            for choice in data.get('choices') or []:
                delta = choice.get('delta') or {}
                for key in self.parts:
                    if delta.get(key):
                        self.parts[key].append(delta[key])
                self.combined.append(delta.get('content') or delta.get('reasoning_content')
                                     or delta.get('reasoning') or '')
                if choice.get('finish_reason'):
                    self.finish_reason = choice['finish_reason']

    def record(self, timing, text):
        combined = ''.join(self.combined)
        assert combined == text, 'channel recorder must preserve the exact onepass text'
        output_sha = hashlib.sha256(text.encode()).hexdigest()
        if timing is not None:
            assert timing['request_sha256'] == self.request_sha256
            assert timing['output_sha256'] == output_sha
        return dict(request_sha256=self.request_sha256, output_sha256=output_sha,
                    channels={k: ''.join(v) for k, v in self.parts.items()},
                    timing=timing, finish_reason=self.finish_reason)


def recorder(ask, output):
    def traced(url, model, content, max_tokens, timing=None, **kwargs):
        traces = []
        original = urllib.request.urlopen
        def opening(request, *args, **options):
            response = original(request, *args, **options)
            # The step sampler concurrently calls urlopen with a metrics URL.
            # Wrap only this completion request; a metrics response is not SSE.
            if (not isinstance(request, urllib.request.Request)
                    or request.full_url != url or request.get_method() != 'POST'):
                return response
            trace = Channels(response, request)
            traces.append(trace)
            return trace
        urllib.request.urlopen = opening
        try:
            result = ask(url, model, content, max_tokens, timing, **kwargs)
        finally:
            urllib.request.urlopen = original
        assert len(traces) == 1
        row = traces[0].record(timing, result[0])
        # Disk I/O is outside onepass's measured request interval.
        with Path(output).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return result
    return traced


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'bench'))
    spec = importlib.util.spec_from_file_location('input_channels_onepass', root/'bench/onepass.py')
    onepass = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(onepass)
    output = Path(os.environ['INPUT_REUSE_CHANNELS_OUT'])
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'fresh channel evidence required'
    onepass.ask_stream = recorder(onepass.ask_stream, output)
    return onepass.main()


if __name__ == '__main__':
    raise SystemExit(main())
