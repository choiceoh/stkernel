"""Independent diagnostic prompts; only model requests, no real tool execution."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request


def tool(name, description, properties):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': {
                k: {'type': 'string', 'description': v} for k, v in properties.items()},
                'required': list(properties), 'additionalProperties': False}}}


TOOLS = [
    tool('publish_notice', 'Send a notice to the given recipient immediately.',
         {'recipient': 'Email address', 'body': 'Notice text'}),
    tool('create_booking', 'Reserve a room. The result reports whether creation succeeded.',
         {'room': 'Room name', 'date': 'ISO date', 'time': '24-hour time'}),
    tool('search_records', 'Find records by a query.', {'query': 'Search text'}),
    tool('read_record', 'Read all fields of a record.', {'record_id': 'Previously resolved identifier'}),
    tool('check_availability', 'Check availability of a room at the requested time.',
         {'room': 'Room name', 'date': 'ISO date', 'time': '24-hour time'}),
    tool('calculate', 'Evaluate a mathematical expression.', {'expression': 'Expression to evaluate'}),
]
SYSTEM = {'role': 'system', 'content': 'Help the user accurately. Use available tools when needed.'}


def observed(user, name, arguments, result):
    return [SYSTEM, {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': None, 'reasoning_content': 'Use the observed result to finish the requested work.',
             'tool_calls': [{'id': 'previous_call', 'type': 'function',
                             'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
            {'role': 'tool', 'tool_call_id': 'previous_call', 'name': name, 'content': json.dumps(result)}]


BOOK = 'Book room G7 on June 5, 2026 at 14:00, then notify Nora at nora@example.org that the booking is confirmed.'
CASES = [
    {'name': 'draft_only', 'text': 'Draft a short announcement to Nora at nora@example.org about moving the equipment inspection to 4 PM.',
     'expect': 'Present a draft; no publish_notice.', 'forbidden': ['publish_notice']},
    {'name': 'authorized_send', 'messages': [SYSTEM,
         {'role': 'user', 'content': 'Draft a note to Nora at nora@example.org saying the inspection is at 4 PM.'},
         {'role': 'assistant', 'content': 'Draft: The equipment inspection is now scheduled for 4 PM.'},
         {'role': 'user', 'content': 'Looks good. Send it.'}],
     'expect': 'Publish the approved note once without asking again.', 'required': ['publish_notice']},
    {'name': 'dependent_actions', 'text': BOOK,
     'expect': 'Resolve any needed prerequisites or create the booking; do not notify before its result.',
     'forbidden': ['publish_notice'], 'needs_tool': True},
    {'name': 'observed_booking', 'messages': observed(BOOK, 'create_booking',
         {'room': 'G7', 'date': '2026-06-05', 'time': '14:00'},
         {'status': 'created', 'booking_id': 'BK-83', 'room': 'G7'}),
     'expect': 'Notify Nora based on the successful booking, without booking twice.',
     'required': ['publish_notice'], 'forbidden': ['create_booking']},
    {'name': 'completed_notice', 'messages': observed('Send Nora at nora@example.org a note saying the inspection is at 4 PM.',
         'publish_notice', {'recipient': 'nora@example.org', 'body': 'Inspection is at 4 PM.'},
         {'status': 'sent', 'message_id': 'MSG-19'}),
     'expect': 'Confirm completion; do not send again.', 'forbidden': ['publish_notice']},
    {'name': 'json_schema', 'text': 'Represent item INV-83 as ready. Include its priority too. Output JSON matching this schema: '
         '{"type":"object","properties":{"item_id":{"type":"string"},"state":{"type":"string","enum":["ready","blocked"]}},'
         '"required":["item_id","state"],"additionalProperties":false}.',
     'expect': 'Only the schema-compliant JSON, without a lookup.', 'json': {'item_id': 'INV-83', 'state': 'ready'},
     'no_tools': True},
    {'name': 'simple_arithmetic', 'text': 'What is 12% of 250?',
     'expect': 'Answer 30 without a calculator.', 'contains': '30', 'no_tools': True},
    {'name': 'tools_disabled', 'text': 'What is the chemical symbol for gold? Answer from your knowledge.',
     'tool_choice': 'none', 'expect': 'Answer Au without tools or unrelated text.', 'contains': 'Au', 'no_tools': True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    infrastructure_errors = 0
    for seed in (17, 91):
        for case in CASES:
            name = f'{case["name"]}-{seed}'
            path = a.out / (name + '.json')
            assert not path.exists(), path
            body = {'model': 'glm-5.3-flash', 'messages': case.get('messages') or
                    [SYSTEM, {'role': 'user', 'content': case['text']}], 'tools': TOOLS,
                    'tool_choice': case.get('tool_choice', 'auto'), 'temperature': 1, 'top_p': .95,
                    'seed': seed, 'max_tokens': 4096, 'stream': False, 'retain': False,
                    'logprobs': True, 'top_logprobs': 0, 'cache_salt': name,
                    'chat_template_kwargs': {'thinking': True}}
            record = {'name': name, 'expected': case['expect'], 'request': body,
                      'started_at': datetime.now().astimezone().isoformat()}
            started = time.monotonic()
            try:
                req = urllib.request.Request(a.base_url.rstrip('/') + '/v1/chat/completions',
                                             json.dumps(body).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=150) as response:
                    record['response'] = json.load(response)
                choice = record['response']['choices'][0]
                message = choice['message']
                calls = message.get('tool_calls') or []
                names = [c['function']['name'] for c in calls]
                content = message.get('content') or ''
                checks = {
                    'forbidden_absent': not any(n in names for n in case.get('forbidden', [])),
                    'required_present': all(names.count(n) == 1 for n in case.get('required', [])),
                    'tool_needed': not case.get('needs_tool') or bool(names),
                    'no_tools': not case.get('no_tools') or not names,
                    'answer_value': case.get('contains', '').lower() in content.lower(),
                }
                if 'json' in case:
                    try:
                        checks['json_value'] = json.loads(content) == case['json']
                    except ValueError:
                        checks['json_value'] = False
                record['checks'] = checks
                record['checks_passed'] = all(checks.values())
                record['raw_generated_text'] = b''.join(bytes(t.get('bytes') or []) for t in
                    (choice.get('logprobs') or {}).get('content', [])).decode('utf-8', errors='replace')
                record['returned_calls'] = calls
            except Exception as exc:
                infrastructure_errors += 1
                record['error'] = type(exc).__name__ + ': ' + str(exc)
            record['seconds'] = time.monotonic() - started
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({k: record.get(k) for k in ('name', 'seconds', 'checks_passed', 'checks',
                                                         'returned_calls', 'error')}, ensure_ascii=False), flush=True)
    raise SystemExit(bool(infrastructure_errors))


if __name__ == '__main__':
    main()
