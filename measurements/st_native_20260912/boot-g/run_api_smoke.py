"""Bounded API checks for the exact serving image after onepass, before rollback."""
import base64
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib

root = Path(__file__).resolve().parent
base = 'http://127.0.0.1:' + os.environ.get('GLM53_API_PORT', '8001')
records = []


def ask(name, messages, **options):
    body = dict(model='glm-5.3-flash', messages=messages, max_tokens=96,
                temperature=0, chat_template_kwargs={'enable_thinking': False})
    body.update(options)
    begin = time.monotonic()
    request = urllib.request.Request(base + '/v1/chat/completions',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        answer = json.load(response)
    records.append(dict(name=name, seconds=time.monotonic() - begin, request=body, response=answer))
    (root / 'api-smoke.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    assert 'error' not in answer, answer
    assert answer['usage']['completion_tokens'] > 0, answer
    for choice in answer['choices']:
        assert choice['message'].get('content', '').strip(), answer
        assert choice['finish_reason'] == 'stop', answer
    print(json.dumps({'name': name, 'seconds': records[-1]['seconds'], 'response': answer}, ensure_ascii=False), flush=True)
    return answer


ask('sampling-two-choices', [{'role': 'user', 'content': '대한민국의 수도 이름만 짧게 답하세요.'}],
    temperature=0.7, top_p=0.9, seed=17, n=2, max_tokens=48)
schema = {'type': 'object', 'properties': {'answer': {'type': 'integer', 'enum': [4]}},
          'required': ['answer'], 'additionalProperties': False}
answer = ask('structured-json', [{'role': 'user', 'content': 'Return the answer to 2+2 in the answer field.'}],
             response_format={'type': 'json_schema', 'json_schema': {'name': 'sum', 'strict': True, 'schema': schema}},
             max_tokens=48)
assert json.loads(answer['choices'][0]['message']['content']) == {'answer': 4}, answer


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


pixels = (b'\x00' + b'\xff\x00\x00' * 112) * 112
png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 112, 112, 8, 2, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
url = 'data:image/png;base64,' + base64.b64encode(png).decode()
answer = ask('vision-red-image', [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': url}},
    {'type': 'text', 'text': 'Name the dominant color in this image. Answer with one English word.'}]}], max_tokens=32)
assert 'red' in answer['choices'][0]['message']['content'].lower(), answer
print('API_SMOKE_PASS 3/3', flush=True)
