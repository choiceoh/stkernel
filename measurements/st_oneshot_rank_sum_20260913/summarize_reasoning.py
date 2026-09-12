"""Count retained SSE channels on the CPU; never query the running engine."""
import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--phase', default='measure-c1')
    args = parser.parse_args()
    record = json.loads((args.run / 'record.json').read_text())
    tok = Tokenizer.from_file(str(args.tokenizer))
    configured_truncation = tok.truncation
    # The checkpoint serializes a 2048-token truncation default. Count the
    # complete output, as the server does, rather than clipping it a second time.
    tok.no_truncation()
    tok.no_padding()
    report = dict(source=record['arm_sha'], run_id=record['run_id'],
        method='CPU re-tokenization of complete retained SSE channel text with truncation/padding '
               'disabled; not original generated token IDs. Original usage.reasoning_tokens '
               'was not retained by this harness.',
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        tokenizers_version=tokenizers.__version__,
        original_tokenizer_truncation=configured_truncation, requests=[])
    with (args.run / 'requests.jsonl').open() as stream:
        for line in stream:
            request = json.loads(line)
            if request['phase'] != args.phase:
                continue
            row = {key: request.get(key) for key in (
                'ctx', 'question', 'phase', 'reasoning_budget', 'completion_tokens',
                'finish_reason', 'response_id', 'output_sha256')}
            for channel, key in (('reasoning_content', 'reasoning'), ('content', 'content')):
                text = ''.join(delta.get(channel) or '' for delta in request['channels'])
                row[key + '_retokenized'] = len(tok.encode(text, add_special_tokens=False).ids)
                if key == 'reasoning':
                    row['reasoning_tail'] = text[-250:]
            row['reasoning_at_cap'] = row['reasoning_retokenized'] == row['reasoning_budget']
            report['requests'].append(row)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(requests=len(report['requests']), output=str(args.output))))


if __name__ == '__main__':
    main()
