"""Report independent probe checks and raw-token/API call agreement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.profiles.glm53.tools import parse_tool_calls

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=Path)
a = ap.parse_args()
rows = []
for path in sorted(a.directory.glob('*.json')):
    data = json.loads(path.read_text())
    row = {k: data.get(k) for k in ('name', 'seconds', 'checks_passed', 'checks', 'error')}
    if 'response' in data:
        raw = data['raw_generated_text']
        message = data['response']['choices'][0]['message']
        # The opener belongs to the prompt; generated reasoning may begin
        # without an opening tag, but its closing tag separates assistant text.
        assistant = raw.rsplit('</think>', 1)[-1]
        known_boundary = '</think>' in raw or not message.get('reasoning_content')
        parsed = parse_tool_calls(assistant, tools=data['request']['tools']) or []
        actual = message.get('tool_calls') or []
        canonical = lambda calls: [(name, json.loads(arguments)) for name, arguments in calls]
        row['raw_and_api_calls_agree'] = known_boundary and canonical(parsed) == canonical([
            (call['function']['name'], call['function']['arguments']) for call in actual])
        row['tool_names'] = [call['function']['name'] for call in actual]
        row['content'] = message.get('content')
    rows.append(row)
report = {'diagnostic_only': True, 'requests': len(rows),
          'checks_passed': sum(row.get('checks_passed') is True for row in rows),
          'errors': [row['name'] for row in rows if row.get('error')],
          'all_raw_and_api_calls_agree': bool(rows) and all(row.get('raw_and_api_calls_agree') for row in rows),
          'rows': rows}
target = a.directory.parent.parent / 'diagnostic-analysis.json'
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
