# dsv41_encoding

DeepSeek-V4.1's prompt format, which is not V4's in three places.

1. **DSML tag names carry a leading space** -- `<|DSML| calls>` with
   `<|DSML| invoke>` / `<|DSML| parameter>`, where V4 wrote `<|DSML|tool_calls>`.
   A V4 parser does not degrade on this; it matches nothing.
2. **Reasoning effort is a number, 1-100**, rendered only in thinking mode and
   only at turn 0. Aliases: low 25, high 50 (default), xhigh 75, max 100.
3. **Mid-conversation system messages** via `<|System|>` behave like a user
   message for appending the generation header.

The grammar is transcribed from the checkpoint's own `encoding/encoding.py`
rather than from the README's prose, because of one detail the prose does not
show: every parameter carries `string="true|false"`, and `false` means the value
is JSON. A parser that ignores it returns `"42"` where the tool wants `42`,
silently, for every non-string argument. This module decodes it; the first draft
here did not, and its unit test passed anyway until the real grammar was read.

A NEW file rather than an edit of `deepseek_reasoning` / `deepseek_tool_parser`:
production serves V4 from those, and the compose refuses two modules claiming
one source name -- which is the rule doing its job, not an obstacle to it.

`split_thinking` is unconditional by design. GLM-5.3 leaked reasoning into
content when a template ended in an open `<think>` and the request said
thinking=false, because the server skips the parser for such requests; here
`thinking=False` yields no reasoning instead of routing it to the answer.

**Verified against the reference** (`probes/dsv41_encoding_diff.py`, no model,
no image): all five of the checkpoint's golden fixtures re-encode exactly
through its own `load_cases`/`encode_case`, and for six completions built by its
`encode_messages` -- thinking and chat, with and without tool calls, with and
without prose -- our reasoning/content split and our tool names and decoded
argument values match `parse_message_from_completion_text`.

Four mutations were checked to fail it: skipping the `string="false"` JSON
decode, dropping the leading space from the DSML tag names (V4's form), leaking
reasoning into content when thinking is off, and leaving the call markup in the
content.

One thing that finding cost: the wire format carries a newline after every tag,
which the README does not say. Hand-written markup without them is rejected by
the reference outright, so the first version of this probe was testing our
parser against a string the model never emits.

Also verified by unit test (26 assertions): effort
resolution and its two suppression conditions, role anchoring, typed DSML
parameters including JSON decode and malformed-JSON fallback, rejection of V4's
spaceless form, and the thinking split including an unterminated span.
