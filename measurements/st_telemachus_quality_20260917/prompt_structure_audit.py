"""Read the incident prompt's turn frame off the private record, with the tokenizer the door serves.

Written because the prompt's structure decided three claims that were made about this incident, and
none of them had been read off the bytes. Two were wrong: the request already ends in an explicit
assistant turn that opens a thinking block (its chat template writes exactly `<|assistant|><think>`),
and the "first decisive miss" attributed to the thinking block's third token is the draw that produced
the intended Korean continuation. The third held: the turn before this request is already degraded
inside this very prompt, so this request inherits a corrupted assistant turn rather than being the
first thing to break.

Decoding a token stream is the part that fools people, so the decode here runs in the served image
with `tokenizers` on the same tokenizer file the door serves. A hand-rolled ByteLevel map is easy to
get subtly wrong (map 0xAD to U+00AD instead of U+0143 and every `항`/`청`/`삭` in the text turns into
a fake replacement character), and a decoder that invents corruption in the system prompt is worse
than no decoder. Do not replace this with a byte map.

Only counts, positions, frame tokens and a handful of short fragments leave here: the prompt IDs, the
complete prompt and the complete output text stay private.

Usage (inside the served ST image, with the private record staged in the mounted log directory):

    python3 prompt_structure_audit.py \
        --record /home/choiceoh/glm53-logs/original-engine-record.json \
        --tokenizer /repo/st-glm53-meta/tokenizer.json \
        --template /repo/st-glm53-meta/chat_template.jinja \
        --replay /home/choiceoh/glm53-logs/case00-mode20-original-seed7.json \
        --out /home/choiceoh/glm53-logs/prompt-structure-evidence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import tokenizers

ASSISTANT = '<|assistant|>'
THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'
PROMPT_TOKENS = 50005
PROMPT_SHA256 = 'e8aefde846b0c67641f9285d2646ca3977ab790534761306422ba81dcc4bb1dc'
# Short fragments of the prior answer that no longer read as Korean. They are what makes "the turn
# before this request was already broken" a reading of the bytes instead of a memory of the screen.
DEGRADED_PRIOR_ANSWER = ('그리섬 기준으로', '고담 시적 상징', '대상도끼보다', '걸사 재단')


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def frame_tokens(tokenizer) -> dict:
    """The vocabulary's own frame tokens, id -> text, for the frame this prompt uses."""
    out = {}
    for token, index in tokenizer.get_vocab().items():
        if (token.startswith('<|') and token.endswith('|>')) or token in (
                THINK_OPEN, THINK_CLOSE, '[gMASK]', '<sop>', '\n<tool_call>'.strip(), '</tool_call>',
                '<tool_response>', '</tool_response>', '<arg_key>', '</arg_key>', '<arg_value>', '</arg_value>'):
            out.setdefault(index, token)
    return out


def positions(ids, index):
    return [position for position, value in enumerate(ids) if value == index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', required=True, type=pathlib.Path, help='private tokens/prompt_len record')
    ap.add_argument('--tokenizer', required=True, type=pathlib.Path)
    ap.add_argument('--template', type=pathlib.Path, default=None, help='the checkpoint chat template')
    ap.add_argument('--replay', type=pathlib.Path, default=None, help='one arm result for the output tail')
    ap.add_argument('--out', required=True, type=pathlib.Path)
    a = ap.parse_args()

    tokenizer = tokenizers.Tokenizer.from_file(str(a.tokenizer))
    marks = frame_tokens(tokenizer)
    for name in (ASSISTANT, '<|user|>', '<|observation|>', '<tool_call>', '</tool_call>', THINK_CLOSE):
        assert name in marks.values(), (name, 'not in the served vocabulary')
    by_name = {text: index for index, text in marks.items()}

    record = json.loads(a.record.read_text())
    prompt = record['tokens'][:record['prompt_len']]
    output = record['tokens'][record['prompt_len']:]
    assert len(prompt) == PROMPT_TOKENS, (len(prompt), 'prompt length moved')
    assert digest(prompt) == PROMPT_SHA256, 'the private record is not the prompt these captures used'

    tail = prompt[-2:]
    assert tail == [by_name[ASSISTANT], by_name[THINK_OPEN]], 'the prompt does not end in an assistant think block'

    users = positions(prompt, by_name['<|user|>'])
    assistants = positions(prompt, by_name[ASSISTANT])
    systems = positions(prompt, by_name['<|system|>'])
    prior = assistants[-2]  # the turn before the request's own assistant turn
    assert prior < users[-1] < assistants[-1], 'the last assistant turn is not the request being served'

    prior_turn = prompt[prior:users[-1]]
    closes = [position for position, value in enumerate(prior_turn) if value == by_name[THINK_CLOSE]]
    assert closes, 'the prior turn never closed its thinking block'
    prior_answer = tokenizer.decode(prior_turn[closes[0] + 1:], skip_special_tokens=False)
    degraded = [fragment for fragment in DEGRADED_PRIOR_ANSWER if fragment in prior_answer]
    assert len(degraded) == len(DEGRADED_PRIOR_ANSWER), ('prior answer fragments moved', degraded)

    output_text = tokenizer.decode(output, skip_special_tokens=False)
    assert output_text.startswith('User continues the Ithaca conversation'), 'the recorded output moved'
    output_close = positions(output, by_name[THINK_CLOSE])
    assert len(output_close) == 1, 'the recorded output does not close exactly one thinking block'

    evidence = dict(
        prompt=dict(tokens=len(prompt), sha256=digest(prompt), chars=len(tokenizer.decode(prompt)),
                    record='private; prompt IDs, full prompt and full output text stay out of git'),
        tokenizer=dict(path=str(a.tokenizer), sha256=sha256(a.tokenizer), decoder='ByteLevel'),
        frame=dict(head=[marks[i] for i in prompt[:3]], tail=[marks[i] for i in tail],
                   last_assistant_index=assistants[-1], thinking_open_index=assistants[-1] + 1),
        turn_layout=dict(system=len(systems), users=users, assistants=assistants,
                         unmarked_tokens_after_last_user=len(prompt) - users[-1] - 1,
                         note='no role token is written between the last user turn and '
                              'the assistant turn this request is served as'),
        prior_turn=dict(assistant_index=prior, tokens=len(prior_turn), thinking_closes_at_index=closes[0],
                        thinking='a coherent English plan, as in the request under audit',
                        visible_answer_degraded_fragments=degraded,
                        reading='the turn before this request is already degraded inside this prompt; this '
                                'request inherits it, so the first breakage is upstream of this request'),
        outputs=dict(original=dict(tokens=len(output), thinking_closes_at_index=output_close[0],
                                   tail_tokens=[marks[i] for i in output[-3:]])),
        corrections=dict(
            assistant_turn_at_tail='the prompt already ends in <|assistant|><think>; nothing is missing there',
            last_korean_instruction='the last user turn is inside this prompt, not 50,000 tokens back',
            gen3_branch='the thinking block\'s third token is a benign bilingual branch, not the corruption onset',
            isolation='the recall block is already tagged untrusted and is not instructions; what it quotes '
                      'is the assistant\'s own degraded prior answer',
        ),
    )
    if a.template is not None:
        template = a.template.read_text()
        line = [row.strip() for row in template.splitlines() if row.strip().startswith('<|assistant|>')]
        assert any(THINK_OPEN in row for row in line), 'the template no longer opens a think block'
        evidence['template'] = dict(path=str(a.template), sha256=sha256(a.template), generation_prompt=line[-1],
                                    note='the frame this request was served with ships with the checkpoint')
    if a.replay is not None:
        arm = json.loads(a.replay.read_text())
        ids = arm['response']['ids']
        arm_closes = positions(ids, by_name[THINK_CLOSE])
        evidence['outputs']['replay'] = dict(case=arm['receipt']['case'], mode=arm['receipt']['mode'],
                                             tokens=len(ids), thinking_closes_at_index=arm_closes[0],
                                             tail_tokens=[marks[i] for i in ids[-1:]],
                                             text=arm['text'][:120])
        assert len(ids) == arm['receipt']['completion_tokens'], 'the arm result is truncated'
        assert not any(i == by_name[ASSISTANT] for i in ids), 'an arm emitted a second assistant turn'
        print('  arm %d mode %d closes thinking at %d, ends %s'
              % (arm['receipt']['case'], arm['receipt']['mode'], arm_closes[0], [marks[i] for i in ids[-1:]]),
              flush=True)

    a.out.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k: evidence[k] for k in ('frame', 'prior_turn')}, indent=2), flush=True)
    print('VERIFIED: the prompt frame, the pre-degraded prior turn and the recorded output tails', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
