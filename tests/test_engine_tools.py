"""Tool-call wire formats -> OpenAI tool calls (engine/base/tool_formats): the arg_key/arg_value layout GLM-5.3 writes
(engine/profiles/glm53/tools.py keeps its names), the function/parameter XML layout Qwen3.8 writes, the grammars that
hold calls lazily, eagerly (tool_choice required or named) or to one call, and the detection that reads a template's
layout off the template."""
import importlib.util
import json
import unittest
from pathlib import Path

from engine.base import tool_formats
from engine.base.tool_formats import (ARG_PAIRS, FUNCTION_XML, detect, grammar_arg_pairs, grammar_function_xml,
                                      parse_function_xml, partial_function_xml)
from engine.profiles.glm53.tools import (parse_tool_calls, partial_tool_calls,
                                         tool_call_token, tool_grammar)


class ToolCallTests(unittest.TestCase):
    def test_calls_in_order_with_json_and_text_values(self):
        text = ('Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Seoul</arg_value>'
                '<arg_key>days</arg_key><arg_value>3</arg_value><arg_key>opts</arg_key><arg_value>{"units": "C"}</arg_value></tool_call>'
                '<tool_call>noop</tool_call>')
        calls = parse_tool_calls(text)
        self.assertEqual([name for name, _ in calls], ["get_weather", "noop"])
        self.assertEqual(json.loads(calls[0][1]), {"city": "Seoul", "days": 3, "opts": {"units": "C"}})
        self.assertEqual(calls[1][1], "{}")

    def test_text_without_a_complete_call_is_not_a_call(self):
        self.assertIsNone(parse_tool_calls("plain answer"))
        self.assertIsNone(parse_tool_calls("<tool_call>get_weather<arg_key>city</arg_key>"))
        self.assertIsNone(parse_tool_calls("<tool_call></tool_call>"))

    def test_korean_values_survive(self):
        calls = parse_tool_calls("<tool_call>search<arg_key>q</arg_key><arg_value>서울 날씨</arg_value></tool_call>")
        self.assertEqual(json.loads(calls[0][1]), {"q": "서울 날씨"})


class PartialToolCallTests(unittest.TestCase):
    """`arguments` is streamed in fragments, so what this returns may only ever grow (45차 §44)."""

    SHAPES = {
        "korean and a number": '<tool_call>get_weather<arg_key>city</arg_key><arg_value>서울특별시</arg_value>'
                               '<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>',
        "whitespace around a value": '<tool_call>write<arg_key>text</arg_key><arg_value>  안녕  하세요  </arg_value></tool_call>',
        "a value that is json": '<tool_call>send<arg_key>body</arg_key><arg_value>{"a": [1, 2]}</arg_value></tool_call>',
        "no arguments at all": "<tool_call>ping</tool_call>",
        "two calls": '<tool_call>a<arg_key>k</arg_key><arg_value>v1</arg_value></tool_call>'
                     '<tool_call>b<arg_key>k</arg_key><arg_value>한글</arg_value></tool_call>',
        "quotes and escapes": '<tool_call>echo<arg_key>s</arg_key><arg_value>he said "hi"\\n끝</arg_value></tool_call>',
    }

    def test_every_prefix_only_ever_grows(self):
        """Character by character, over the whole call: a client concatenates these fragments,
        so a fragment that contradicts one already sent cannot be taken back."""
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                seen = {}
                for n in range(len(text) + 1):
                    for i, (_, args, _done) in enumerate(partial_tool_calls(text[:n])):
                        self.assertTrue(args.startswith(seen.get(i, "")),
                                        f"at {n}: {args!r} does not continue {seen.get(i, '')!r}")
                        seen[i] = args

    def test_the_end_of_the_stream_is_what_the_whole_parse_says(self):
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                streamed = [(name, args) for name, args, done in partial_tool_calls(text) if done]
                self.assertEqual(streamed, [tuple(c) for c in (parse_tool_calls(text) or [])])
                for _, args in streamed:
                    json.loads(args)                        # and each one is JSON, whole

    def test_a_value_whose_type_is_not_settled_yet_waits(self):
        """`3` could become `3`, `30` or `3.5`, and `{` could become anything: a number or an
        object is only rendered once `</arg_value>` says what it was. A plain string cannot
        change its type, so it streams."""
        opening = '<tool_call>f<arg_key>n</arg_key><arg_value>'
        self.assertEqual(partial_tool_calls(opening + "3")[0][1], "{")
        self.assertEqual(partial_tool_calls(opening + '{"a"')[0][1], "{")
        self.assertEqual(partial_tool_calls(opening + "서울")[0][1], '{"n": "서울')

    def test_a_closing_tag_halfway_here_is_not_value_text(self):
        opening = '<tool_call>f<arg_key>n</arg_key><arg_value>서울'
        for cut in range(len("</arg_value>")):
            self.assertEqual(partial_tool_calls(opening + "</arg_value>"[:cut])[0][1], '{"n": "서울')

    def test_a_name_is_not_reported_until_it_is_whole(self):
        self.assertEqual(partial_tool_calls("<tool_call>get_wea"), [])
        self.assertEqual(partial_tool_calls("<tool_call>get_weather<arg_k"), [])
        self.assertEqual(partial_tool_calls("<tool_call>get_weather<arg_key>")[0][0], "get_weather")
        self.assertEqual(partial_tool_calls("<tool_call>ping</tool_call>"), [("ping", "{}", True)])

    def test_korean_is_not_escaped_into_six_bytes(self):
        text = '<tool_call>search<arg_key>q</arg_key><arg_value>서울 날씨'
        self.assertNotIn("\\u", partial_tool_calls(text)[0][1])


class ToolGrammarTests(unittest.TestCase):
    """Nothing held a call to the tools that were declared (45차 §45)."""

    TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
                 "type": "object", "properties": {"days": {}, "city": {}}}}},
             {"type": "function", "function": {"name": "ping"}}]

    def test_it_binds_the_names_the_keys_and_the_shape(self):
        text = tool_grammar(self.TOOLS)
        self.assertIn('call0 ::= "get_weather" pairs0 "</tool_call>"', text)
        self.assertIn('key0 ::= "city" | "days"', text)                 # sorted, so the same tools are one cache entry
        self.assertIn('pairs1 ::= ""', text)                            # a tool that takes nothing takes nothing
        self.assertIn('root ::= call ("<tool_call>" call)*', text)

    def test_a_value_may_be_anything(self):
        """A value is raw text and may hold `<`, quotes, code. Its rule admits every character,
        and the ambiguity with the closing tag is the point: inside a value nothing is forbidden."""
        self.assertIn("value ::= [^\\u0000]*", tool_grammar(self.TOOLS))

    def test_nothing_to_bind_is_no_grammar(self):
        self.assertIsNone(tool_grammar([]))
        self.assertIsNone(tool_grammar(None))
        self.assertIsNone(tool_grammar([{"type": "function", "function": {}}]))
        self.assertIsNone(tool_grammar([{"type": "function"}]))

    def test_the_trigger_is_a_token_or_there_is_no_grammar(self):
        class Whole:
            def encode(self, text, add_special_tokens=True):
                return type("E", (), {"ids": [154843]})()
            def decode(self, ids, skip_special_tokens=True):
                return "<tool_call>"

        class InPieces(Whole):
            def encode(self, text, add_special_tokens=True):
                return type("E", (), {"ids": [1, 2, 3]})()

        class PlainList(Whole):
            def encode(self, text, add_special_tokens=True):
                return [154843]

        self.assertEqual(tool_call_token(Whole()), 154843)
        self.assertEqual(tool_call_token(PlainList()), 154843)          # a transformers tokenizer answers this way
        self.assertIsNone(tool_call_token(InPieces()))                  # armed in the middle of a marker: no
        self.assertIsNone(tool_call_token(None))



def qwen_call(name, arguments, first=True):
    """One call as Qwen3.8's chat_template.jinja writes it: a string value raw, any other value as JSON, each between
    the layout newlines; a call after the first opens with a newline."""
    out = ("" if first else "\n") + "<tool_call>\n<function=" + name + ">\n"
    for key, value in arguments.items():
        out += "<parameter=" + key + ">\n" + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "\n</parameter>\n"
    return out + "</function>\n</tool_call>"


def glm_call(name, arguments):
    """One call as GLM-5.3's chat_template_mm_v2.jinja writes it."""
    return ("<tool_call>" + name + "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</arg_value>"
        for k, v in arguments.items()) + "</tool_call>")


class GlmFormatTests(unittest.TestCase):
    def test_the_glm_profile_binds_the_base_format(self):
        self.assertIs(parse_tool_calls, ARG_PAIRS.parse)
        self.assertIs(partial_tool_calls, ARG_PAIRS.partial)
        self.assertIs(tool_grammar, ARG_PAIRS.grammar)
        self.assertIs(tool_call_token, tool_formats.tool_call_token)


class QwenXmlToolCallTests(unittest.TestCase):
    CODE = "def f(x):\n    if x:\n\n        return '<ok>'\n    return None"

    def test_calls_in_order_with_json_and_text_values(self):
        text = ("Let me check." + qwen_call("get_weather", {"city": "Seoul", "days": 3, "opts": {"units": "C"}})
                + qwen_call("noop", {}, first=False))
        calls = parse_function_xml(text)
        self.assertEqual([name for name, _ in calls], ["get_weather", "noop"])
        self.assertEqual(json.loads(calls[0][1]), {"city": "Seoul", "days": 3, "opts": {"units": "C"}})
        self.assertEqual(calls[1][1], "{}")

    def test_a_value_comes_back_exactly_as_written(self):
        """The layout newlines go, nothing else: code keeps its indentation and blank lines, Korean stays Korean."""
        calls = parse_function_xml(qwen_call("write", {"code": self.CODE, "note": "  서울 날씨  ", "empty": ""}))
        self.assertEqual(json.loads(calls[0][1]), {"code": self.CODE, "note": "  서울 날씨  ", "empty": ""})
        self.assertNotIn("\\u", calls[0][1])

    def test_text_without_a_complete_call_is_not_a_call(self):
        self.assertIsNone(parse_function_xml("plain answer"))
        self.assertIsNone(parse_function_xml("<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul"))
        self.assertIsNone(parse_function_xml("<tool_call>\n</tool_call>"))
        self.assertIsNone(parse_function_xml("<tool_call>\n<function=>\n</function>\n</tool_call>"))

    def test_a_call_closed_without_its_function_tag_still_counts(self):
        calls = parse_function_xml("<tool_call>\n<function=ping>\n<parameter=n>\n2\n</parameter>\n</tool_call>")
        self.assertEqual(calls, [("ping", '{"n": 2}')])


class QwenXmlPartialToolCallTests(unittest.TestCase):
    """The GLM contract on the XML layout: what streams may only ever grow, and ends as the whole parse."""

    SHAPES = {
        "korean and a number": qwen_call("get_weather", {"city": "서울특별시", "days": 3}),
        "whitespace around a value": qwen_call("write", {"text": "  안녕  하세요  "}),
        "code with blank lines": qwen_call("write", {"code": QwenXmlToolCallTests.CODE}),
        "a value that ends in newlines": qwen_call("write", {"text": "line\n\n"}),
        "a value that is json": qwen_call("send", {"body": {"a": [1, 2]}}),
        "no arguments at all": qwen_call("ping", {}),
        "two calls": qwen_call("a", {"k": "v1"}) + qwen_call("b", {"k": "한글"}, first=False),
        "quotes and escapes": qwen_call("echo", {"s": 'he said "hi"\\n끝'}),
        "a tag-like value": qwen_call("echo", {"s": "a <b> </param c"}),
    }

    def test_every_prefix_only_ever_grows(self):
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                seen = {}
                for n in range(len(text) + 1):
                    for i, (_, args, _done) in enumerate(partial_function_xml(text[:n])):
                        self.assertTrue(args.startswith(seen.get(i, "")),
                                        f"at {n}: {args!r} does not continue {seen.get(i, '')!r}")
                        seen[i] = args

    def test_the_end_of_the_stream_is_what_the_whole_parse_says(self):
        for label, text in self.SHAPES.items():
            with self.subTest(label):
                streamed = [(name, args) for name, args, done in partial_function_xml(text) if done]
                self.assertEqual(streamed, [tuple(c) for c in (parse_function_xml(text) or [])])
                for _, args in streamed:
                    json.loads(args)

    def test_a_value_whose_type_is_not_settled_yet_waits(self):
        opening = "<tool_call>\n<function=f>\n<parameter=n>\n"
        self.assertEqual(partial_function_xml(opening + "3")[0][1], "{")
        self.assertEqual(partial_function_xml(opening + '{"a"')[0][1], "{")
        self.assertEqual(partial_function_xml(opening + "   ")[0][1], "{")         # blank so far: no type yet
        self.assertEqual(partial_function_xml(opening + "서울")[0][1], '{"n": "서울')

    def test_a_closing_tag_halfway_here_is_not_value_text(self):
        for layout in ("\n</parameter>", "</parameter>"):
            opening = "<tool_call>\n<function=f>\n<parameter=n>\n서울"
            for cut in range(len(layout)):
                with self.subTest(layout=layout, cut=cut):
                    self.assertEqual(partial_function_xml(opening + layout[:cut])[0][1], '{"n": "서울')

    def test_a_name_is_not_reported_until_it_is_whole(self):
        self.assertEqual(partial_function_xml("<tool_call>\n<function=get_wea"), [])
        self.assertEqual(partial_function_xml("<tool_call>\n<function=get_weather>")[0][:2], ("get_weather", "{"))
        self.assertEqual(partial_function_xml(qwen_call("ping", {})), [("ping", "{}", True)])


class QwenXmlGrammarTests(unittest.TestCase):
    TOOLS = ToolGrammarTests.TOOLS

    def test_it_binds_the_names_the_keys_and_the_layout(self):
        text = grammar_function_xml(self.TOOLS)
        self.assertIn('call0 ::= "\\n<function=" "get_weather" ">\\n" params0 "</function>\\n</tool_call>"', text)
        self.assertIn('params0 ::= ("<parameter=" key0 ">\\n" value "\\n</parameter>\\n")*', text)
        self.assertIn('key0 ::= "city" | "days"', text)
        self.assertIn('params1 ::= ""', text)
        self.assertIn('root ::= call ("\\n<tool_call>" call)*', text)
        self.assertIn("value ::= [^\\u0000]*", text)

    def test_nothing_to_bind_is_no_grammar(self):
        for tools in ([], None, [{"type": "function", "function": {}}], [{"type": "function"}]):
            self.assertIsNone(grammar_function_xml(tools))


class GrammarShapeTests(unittest.TestCase):
    """Lazily a grammar begins after the marker; eagerly (tool_choice required or named) the answer is the calls from
    its first token, blank space allowed before the marker; with parallel=False it is one call."""

    TOOLS = ToolGrammarTests.TOOLS

    def test_both_layouts_open_eagerly_with_the_marker_and_close_after_one_call_when_asked(self):
        for grammar, more in ((grammar_arg_pairs, '"<tool_call>" call'), (grammar_function_xml, '"\\n<tool_call>" call')):
            with self.subTest(grammar=grammar.__name__):
                self.assertTrue(grammar(self.TOOLS).startswith(f"root ::= call ({more})*\n"))
                self.assertTrue(grammar(self.TOOLS, lazy=False).startswith(f'root ::= [ \\n]* "<tool_call>" call ({more})*\n'))
                self.assertTrue(grammar(self.TOOLS, parallel=False).startswith("root ::= call\n"))
                self.assertTrue(grammar(self.TOOLS, lazy=False, parallel=False).startswith('root ::= [ \\n]* "<tool_call>" call\n'))
                self.assertEqual(grammar(self.TOOLS, lazy=False).split("\n")[1:], grammar(self.TOOLS).split("\n")[1:])


class SchemaTests(unittest.TestCase):
    """Given the request's tools, an argument the schema types as a string is the text written, whatever it looks like;
    and a tool whose parameters declare no properties takes any key."""

    TOOLS = [{"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}, "mode": {"enum": ["1", "2"]}, "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "n": {"type": "integer"}}}}},
             {"type": "function", "function": {"name": "anything", "parameters": {"type": "object"}}},
             {"type": "function", "function": {"name": "closed", "parameters": {"type": "object", "additionalProperties": False}}},
             {"type": "function", "function": {"name": "none"}}]
    ARGS = {"code": "123", "mode": "2", "note": "true", "n": 7}

    def test_both_layouts_read_strings_as_written_and_numbers_as_numbers(self):
        for parse, partial, write in ((parse_tool_calls, partial_tool_calls, glm_call),
                                      (parse_function_xml, partial_function_xml, qwen_call)):
            with self.subTest(layout=parse.__name__):
                text = write("run", self.ARGS)
                self.assertEqual(json.loads(parse(text, tools=self.TOOLS)[0][1]), self.ARGS)
                self.assertEqual(json.loads(parse(text)[0][1]), {"code": 123, "mode": 2, "note": True, "n": 7})
                seen = ""
                for n in range(len(text) + 1):                        # typed strings stream from their first character
                    calls = partial(text[:n], tools=self.TOOLS)
                    if calls:
                        self.assertTrue(calls[0][1].startswith(seen))
                        seen = calls[0][1]
                self.assertEqual(json.loads(seen), self.ARGS)
                self.assertTrue(getattr(parse, "reads_tools", False) and getattr(partial, "reads_tools", False))

    def test_a_free_form_tool_takes_any_key_and_a_closed_or_bare_one_none(self):
        for grammar, anykey in ((grammar_arg_pairs, "key1 ::= [^<]+"), (grammar_function_xml, "key1 ::= [^>\\n]+")):
            with self.subTest(grammar=grammar.__name__):
                text = grammar(self.TOOLS)
                self.assertIn(anykey, text)
                self.assertIn("key0 ::= " + " | ".join(json.dumps(k) for k in sorted(self.ARGS)), text)
                self.assertTrue('pairs2 ::= ""' in text or 'params2 ::= ""' in text)
                self.assertTrue('pairs3 ::= ""' in text or 'params3 ::= ""' in text)


class DetectTests(unittest.TestCase):
    """The format is read off the template: the one whose parser reads back the call the template wrote."""

    @staticmethod
    def template(write, example=""):
        def render(messages, kwargs, *, generation_prompt=True, continue_final=False):
            out = example
            for m in messages:
                out += f"<|{m['role']}|>" + (m.get("content") or "")
                for call in m.get("tool_calls") or []:
                    arguments = call["function"]["arguments"]
                    if not isinstance(arguments, dict):
                        raise TypeError("Can only get item pairs from a mapping.")
                    out += write(call["function"]["name"], arguments)
            return out + ("<|assistant|>" if generation_prompt else "")
        return render

    def test_each_layout_finds_its_format(self):
        self.assertIs(detect(self.template(glm_call)), ARG_PAIRS)
        # Qwen3.8's tools block carries an example call; the probe's own call is what decides
        example = qwen_call("example_function_name", {"example_parameter_1": "value_1"})
        self.assertIs(detect(self.template(qwen_call, example=example)), FUNCTION_XML)

    def test_a_template_that_writes_no_call_or_refuses_has_none(self):
        self.assertIsNone(detect(self.template(lambda name, arguments: "")))
        self.assertIsNone(detect(None))

        def refuses(messages, kwargs, **_):
            raise ValueError("tools are not supported")
        self.assertIsNone(detect(refuses))

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "renders the real template with transformers")
    def test_glm53s_own_template_is_glm(self):
        from transformers.utils.chat_template_utils import render_jinja_template
        template = (Path(__file__).resolve().parents[1] / "launchers" / "chat_template_mm_v2.jinja").read_text()

        def render(messages, kwargs, *, generation_prompt=True, continue_final=False):
            kwargs = dict(kwargs)
            tools = kwargs.pop("tools", None)
            out, _ = render_jinja_template(conversations=[messages], tools=tools, chat_template=template,
                                           add_generation_prompt=generation_prompt, **kwargs)
            return out[0]
        self.assertIs(detect(render), ARG_PAIRS)
        self.assertEqual(ARG_PAIRS.start_token(None), None)


if __name__ == "__main__":
    unittest.main()
