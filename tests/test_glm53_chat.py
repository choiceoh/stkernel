"""CPU regressions: uv run --with jinja2 python tests/test_glm53_chat.py."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "glm53_chat", ROOT / "overlay/modules/glm53_runtime/glm53_chat.py")
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def render(messages, **kwargs):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda obj, **options: json.dumps(obj, **options)
    def fail(message):
        raise TemplateError(message)
    env.globals["raise_exception"] = fail
    return env.from_string((ROOT / "launchers/chat_template_mm_v2.jinja").read_text()).render(
        messages=messages, tools=[], add_generation_prompt=True, **kwargs)


class TemplateTests(unittest.TestCase):
    def test_null_tool_call_body_is_empty(self):
        output = render([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "function": {"name": "lookup", "arguments": {"q": "서울"}}}]}])
        self.assertNotIn("None", output)
        self.assertIn("<think></think><tool_call>lookup", output)

    def test_literal_tags_and_whitespace_are_preserved(self):
        for content in ("Use </think> literally; keep middle </think> and tail.",
                        "<think>an example</think> then </think> literally",
                        "  ```python\n  pass\n```\n  "):
            with self.subTest(content=content):
                output = render([{"role": "assistant", "content": content}])
                self.assertIn("<think></think>" + content + "<|assistant|>", output)

    def test_legacy_decode_requires_opt_in_and_a_leading_block(self):
        output = render([{"role": "assistant", "content": "<think>reason</think>answer </think> suffix"}],
                        legacy_reasoning_content=True)
        self.assertIn("<think>reason</think>answer </think> suffix", output)
        text = "Here is </think> literally"
        self.assertIn("<think></think>" + text, render(
            [{"role": "assistant", "content": text}], legacy_reasoning_content=True))

    def test_explicit_reasoning_field_wins(self):
        content = "<think>literal</think>answer"
        output = render([{"role": "assistant", "content": content, "reasoning_content": "actual"}],
                        legacy_reasoning_content=True)
        self.assertIn("<think>actual</think>" + content, output)

    def test_clear_thinking_retains_current_tool_round(self):
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer", "reasoning_content": "old reason"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": None, "reasoning_content": "current reason",
             "tool_calls": [{"id": "a", "function": {"name": "lookup", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "a", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        output = render(messages, clear_thinking=True)
        self.assertNotIn("old reason", output)
        self.assertEqual(output.count("current reason"), 1)
        self.assertIn("<think></think>done", output)

    def test_reordered_tool_results_preserve_both_payloads(self):
        calls = [{"id": key, "function": {"name": "lookup", "arguments": {}}} for key in ("a", "b")]
        output = render([
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "b", "content": "B </think> text"},
            {"role": "tool", "tool_call_id": "a", "content": "A text"},
        ])
        self.assertIn("<tool_response>A text</tool_response><tool_response>B </think> text</tool_response>", output)

    def test_default_on_and_compatibility_off(self):
        self.assertTrue(render([]).endswith("<|assistant|><think>"))
        for key in ("thinking", "enable_thinking"):
            self.assertTrue(render([], **{key: False}).endswith("<|assistant|><think></think>"))
        for effort in ("low", "high", "max"):
            self.assertIn("Reasoning Effort: " + effort.capitalize(), render([], reasoning_effort=effort))

    def test_direct_template_users_get_option_validation(self):
        for kwargs in ({"thinking": "false"}, {"enable_thinking": None},
                       {"clear_thinking": 0}, {"legacy_reasoning_content": "true"},
                       {"thinking": False, "enable_thinking": True}, {"reasoning_effort": "medium"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TemplateError):
                render([], **kwargs)


class OptionTests(unittest.TestCase):
    def test_explicit_options_are_canonical_without_mutating_history(self):
        body = {"messages": [{"role": "user", "content": "test"}],
                "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}
        original = copy.deepcopy(body)
        result = chat.normalize_chat_options(body)
        self.assertEqual(body, original)
        self.assertEqual(result["chat_template_kwargs"],
                         {"thinking": False, "enable_thinking": False, "reasoning_effort": "low"})

    def test_omissions_preserve_server_defaults(self):
        self.assertEqual(chat.normalize_chat_options({}), {})
        self.assertEqual(chat.normalize_chat_options({"chat_template_kwargs": None}),
                         {"chat_template_kwargs": None})

    def test_bad_types_conflicts_and_efforts_are_rejected(self):
        cases = [{"chat_template_kwargs": []}, {"reasoning_effort": "medium"},
                 {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}},
                 {"chat_template_kwargs": {"thinking": False, "enable_thinking": True}}]
        for key in ("thinking", "enable_thinking", "clear_thinking", "legacy_reasoning_content"):
            cases += [{"chat_template_kwargs": {key: value}} for value in ("false", 0, 1, None, [])]
        for body in cases:
            with self.subTest(body=body), self.assertRaises(chat.ChatContractError):
                chat.normalize_chat_options(body)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunked_input_and_streamed_output_and_disconnect(self):
        body = json.dumps({"messages": [{"role": "user", "content": "서울"}],
                           "chat_template_kwargs": {"thinking": False}}, ensure_ascii=False).encode()
        messages = iter([{"type": "http.request", "body": body[:17], "more_body": True},
                         {"type": "http.request", "body": body[17:], "more_body": False},
                         {"type": "http.disconnect"}])
        async def receive():
            return next(messages)
        sent = []
        async def send(message):
            sent.append(message)
        response = [{"type": "http.response.start", "status": 200, "headers": []},
                    {"type": "http.response.body", "body": b'data: {"finish_reason":"length"}\n\n', "more_body": True},
                    {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}]
        async def app(scope, recv, emit):
            normalized = (await recv())["body"]
            self.assertEqual(int(dict(scope["headers"])[b"content-length"]), len(normalized))
            self.assertFalse(json.loads(normalized)["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual((await recv())["type"], "http.disconnect")
            for message in response:
                await emit(message)
        await chat.ChatContractMiddleware(app)(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"content-length", str(len(body)).encode())]}, receive, send)
        self.assertEqual(sent, response)

    async def test_errors_never_reach_generation(self):
        for raw in (b'{"chat_template_kwargs":{"thinking":"false"}}', b"[]", b"not-json"):
            async def app(*args):
                self.fail("invalid input reached generation")
            async def receive():
                return {"type": "http.request", "body": raw}
            sent = []
            async def send(message):
                sent.append(message)
            await chat.ChatContractMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/v1/chat/completions"}, receive, send)
            self.assertEqual(sent[0]["status"], 400)
            self.assertEqual(json.loads(sent[1]["body"])["error"]["type"], "invalid_request_error")

    async def test_unrelated_routes_and_lifespan_pass_through(self):
        for scope in ({"type": "lifespan"}, {"type": "http", "method": "POST", "path": "/glm53/lab"}):
            async def receive():
                self.fail("middleware consumed unrelated input")
            async def send(message):
                pass
            async def app(actual, recv, emit):
                self.assertIs(actual, scope)
                self.assertIs(recv, receive)
                self.assertIs(emit, send)
            await chat.ChatContractMiddleware(app)(scope, receive, send)


STAGE_MOCK = r'''
import hashlib, os
from pathlib import Path
import subprocess, sys
root = Path(os.environ['CHAT_TEST_ROOT'])
tool = Path(sys.argv[0]).name
if tool == 'sha256sum':
    for name in sys.argv[1:]:
        print(hashlib.sha256(Path(name).read_bytes()).hexdigest() + '  ' + name)
elif tool == 'ssh':
    node = sys.argv[-2].split('@')[-1]
    if node == os.environ.get('CHAT_FAIL_NODE'):
        sys.exit(255)
    command = sys.argv[-1].replace(str(root/'head'), str(root/node))
    sys.exit(subprocess.call([os.environ['CHAT_BASH'], '-c', command]))
elif tool == 'docker':
    if os.environ.get('CHAT_BAD_MOUNT'):
        print('0' * 64 + '  wrong-template')
    else:
        name = Path(sys.argv[-1]).name
        print(hashlib.sha256((root/'head'/name).read_bytes()).hexdigest() + '  ' + name)
'''


class StagingTests(unittest.TestCase):
    def run_stage(self, *, dry_run=False, **settings):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("bin", "repo", "head", "w1", "w2", "w3"):
                (root / name).mkdir()
            for tool in ("ssh", "docker", "sha256sum"):
                path = root / "bin" / tool
                path.write_text("#!" + sys.executable + "\n" + STAGE_MOCK)
                path.chmod(0o755)
            (root / "repo/chat_template_mm_v2.jinja").write_text("new template")
            for node in ("head", "w1", "w2", "w3"):
                (root / node / "chat_template_mm_v2.jinja").write_text("stale template")
            bash = shutil.which("bash")
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       CHAT_TEST_ROOT=str(root), CHAT_BASH=bash, **settings)
            script = f'''set -euo pipefail
. {shlex.quote(str(ROOT / 'launchers/lib/glm53-chat.sh'))}
MODEL_HOST_PATH={shlex.quote(str(root / 'head'))}
MODEL_PATH=/models/test
HEAD_IP=head
WORKER_IPS=(w1 w2 w3)
REASONING_PARSER=glm45
SSHOPT=''
IMAGE=test
DRY_RUN={int(dry_run)}
ct_prepare_glm53_chat {shlex.quote(str(root / 'repo'))}
printf 'verified\n'
'''
            result = subprocess.run([bash, "-c", script], env=env, capture_output=True, text=True)
            files = {str(p.relative_to(root)): p.read_text() for node in ("head", "w1", "w2", "w3")
                     for p in (root / node).iterdir()}
            return result, files

    def test_repo_revision_reaches_every_node_without_overwriting_old_copy(self):
        result, files = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        digest = hashlib.sha256(b"new template").hexdigest()
        for node in ("head", "w1", "w2", "w3"):
            self.assertEqual(files[f"{node}/chat_template.{digest}.jinja"], "new template")
            self.assertEqual(files[f"{node}/chat_template_mm_v2.jinja"], "stale template")

    def test_worker_failure_or_wrong_container_bytes_stop_launch(self):
        for settings in ({"CHAT_FAIL_NODE": "w2"}, {"CHAT_BAD_MOUNT": "1"}):
            result, _ = self.run_stage(**settings)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("verified", result.stdout)

    def test_dry_run_does_not_stage_or_contact_nodes(self):
        result, files = self.run_stage(dry_run=True, CHAT_FAIL_NODE="w1", CHAT_BAD_MOUNT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(files), 4)


if __name__ == "__main__":
    unittest.main()
