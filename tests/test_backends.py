import json

import httpx
import pytest

import hierophant as hp
from hierophant.backends import Message, backend


def mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_request_and_usage():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )

    b = hp.OpenAICompat("m", "http://x/v1", client=mock(handler))
    r = b.complete([Message("user", "q")], json=True)
    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert (r.text, r.input_tokens, r.output_tokens) == ("hi", 7, 2)


def test_anthropic_splits_system():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    b = hp.Anthropic("claude-x", api_key="k", client=mock(handler))
    r = b.complete([Message("system", "be brief"), Message("user", "q")])
    assert seen["body"]["system"] == "be brief"
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert r.text == "ok"


def test_specs():
    assert backend("ollama:qwen3:8b").model == "qwen3:8b"
    assert backend("ollama:qwen3:8b").base_url == "http://localhost:11434/v1"
    assert backend("anthropic:claude-haiku-4-5").model == "claude-haiku-4-5"
    with pytest.raises(ValueError):
        backend("nope:m")
    with pytest.raises(ValueError):
        backend("ollama")


def test_record_then_replay(tmp_path):
    path = tmp_path / "run.jsonl"
    rec = hp.Recorder(hp.Replay(["a", "b"]), path)
    rec.complete([Message("user", "1")])
    rec.complete([Message("user", "2")])
    again = hp.Replay(path)
    assert [again.complete([]).text for _ in range(2)] == ["a", "b"]
