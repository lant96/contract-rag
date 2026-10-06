from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, InternalServerError, RateLimitError

from contract_rag.llm import LLMClient, LLMError, LLMSettings, settings_from_env

SETTINGS = LLMSettings(
    base_url="https://example.com/v1",
    api_key="secret",
    model="test-model",
    extra_body={"reasoning_effort": "low"},
)
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
MESSAGES = [{"role": "user", "content": "Capital of Greece?"}]


def rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    """An error like the one a provider sends when we ask too often (HTTP 429)."""
    headers = {"retry-after": retry_after} if retry_after else {}
    request: Any = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response: Any = httpx.Response(429, headers=headers, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def connection_error(cause: str = "dns failed") -> APIConnectionError:
    """An error like the one we get when the provider cannot be reached at all."""
    request: Any = httpx.Request("POST", "https://example.com/v1/chat/completions")
    error = APIConnectionError(request=request)
    error.__cause__ = OSError(cause)
    return error


def server_error() -> InternalServerError:
    """An error like the one we get when the provider itself has a problem (HTTP 500)."""
    request: Any = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response: Any = httpx.Response(500, request=request)
    return InternalServerError("server broke", response=response, body=None)


def fake_reply(text: str = '{"city": "Athens"}', finish_reason: str = "stop") -> SimpleNamespace:
    """A reply shaped like the real one, with just the fields our client reads."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    """Plays back the given outcomes (a reply, or an error to raise), one per call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(tmp_path: Path, *outcomes, max_retries: int = 3):
    fake = FakeClient(*outcomes)
    waits = []
    client = LLMClient(
        SETTINGS,
        cache_dir=tmp_path / "cache",
        client=fake,
        max_retries=max_retries,
        sleep=waits.append,
    )
    return client, fake, waits


def test_reply_has_the_text_and_token_counts(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, fake_reply())
    reply = client.complete_json(MESSAGES, SCHEMA, "capital")
    assert reply.text == '{"city": "Athens"}'
    assert (reply.prompt_tokens, reply.completion_tokens) == (11, 7)
    assert not reply.cached


def test_the_request_carries_the_schema_and_provider_options(tmp_path: Path) -> None:
    client, fake, _ = make_client(tmp_path, fake_reply())
    client.complete_json(MESSAGES, SCHEMA, "capital")

    request = fake.requests[0]
    assert request["model"] == "test-model"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA
    assert request["extra_body"] == {"reasoning_effort": "low"}


def test_a_repeated_call_comes_from_the_cache(tmp_path: Path) -> None:
    client, fake, _ = make_client(tmp_path, fake_reply())
    first = client.complete_json(MESSAGES, SCHEMA, "capital")
    second = client.complete_json(MESSAGES, SCHEMA, "capital")

    assert len(fake.requests) == 1  # the second call never reached the provider
    assert second.cached
    assert second.text == first.text


def test_different_messages_are_cached_separately(tmp_path: Path) -> None:
    client, fake, _ = make_client(tmp_path, fake_reply(), fake_reply('{"city": "Rome"}'))
    client.complete_json(MESSAGES, SCHEMA, "capital")
    other = client.complete_json([{"role": "user", "content": "Capital of Italy?"}], SCHEMA, "x")
    assert len(fake.requests) == 2
    assert other.text == '{"city": "Rome"}'


def test_it_waits_as_long_as_the_provider_asks(tmp_path: Path) -> None:
    client, fake, waits = make_client(tmp_path, rate_limit_error("3"), fake_reply())
    reply = client.complete_json(MESSAGES, SCHEMA, "capital")
    assert reply.text == '{"city": "Athens"}'
    assert waits == [4.0]
    assert len(fake.requests) == 2


def test_without_a_retry_after_header_the_pause_grows(tmp_path: Path) -> None:
    client, _, waits = make_client(tmp_path, rate_limit_error(), rate_limit_error(), fake_reply())
    client.complete_json(MESSAGES, SCHEMA, "capital")
    assert waits == [2.0, 4.0]


def test_it_gives_up_after_the_maximum_number_of_retries(tmp_path: Path) -> None:
    errors = [rate_limit_error() for _ in range(3)]
    client, _, waits = make_client(tmp_path, *errors, max_retries=2)
    with pytest.raises(LLMError):
        client.complete_json(MESSAGES, SCHEMA, "capital")
    assert len(waits) == 2


def test_an_unreachable_provider_stops_early_and_explains_why(tmp_path: Path, capsys) -> None:
    errors = [connection_error("dns failed") for _ in range(3)]
    client, fake, waits = make_client(tmp_path, *errors)
    with pytest.raises(LLMError, match="Cannot reach https://example.com/v1"):
        client.complete_json(MESSAGES, SCHEMA, "capital")
    assert waits == [1.0, 2.0]  # two short retries, then it gives up
    assert len(fake.requests) == 3
    assert "dns failed" in capsys.readouterr().out


def test_server_errors_are_retried(tmp_path: Path) -> None:
    client, _, waits = make_client(tmp_path, server_error(), fake_reply())
    reply = client.complete_json(MESSAGES, SCHEMA, "capital")
    assert reply.text == '{"city": "Athens"}'
    assert waits == [1.0]


def test_a_very_long_pause_is_not_waited_for(tmp_path: Path) -> None:
    client, fake, waits = make_client(tmp_path, rate_limit_error("3600"))
    with pytest.raises(LLMError, match="daily limit"):
        client.complete_json(MESSAGES, SCHEMA, "capital")
    assert waits == []
    assert len(fake.requests) == 1


def test_a_cut_off_answer_is_an_error_and_is_not_cached(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, fake_reply('{"cit', finish_reason="length"))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        client.complete_json(MESSAGES, SCHEMA, "capital")
    assert not (tmp_path / "cache").exists()


def test_settings_are_read_from_environment_variables() -> None:
    env = {
        "LLM_BASE_URL": "https://example.com/v1",
        "LLM_API_KEY": "secret",
        "LLM_MODEL": "some-model",
        "LLM_TEMPERATURE": "0.5",
        "LLM_MAX_TOKENS": "500",
        "LLM_EXTRA_BODY": '{"reasoning_effort": "low"}',
    }
    settings = settings_from_env(env)
    assert settings.model == "some-model"
    assert settings.temperature == 0.5
    assert settings.max_tokens == 500
    assert settings.extra_body == {"reasoning_effort": "low"}


def test_missing_or_broken_settings_give_a_clear_error() -> None:
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        settings_from_env({"LLM_BASE_URL": "x", "LLM_MODEL": "y"})
    good = {"LLM_BASE_URL": "x", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
    with pytest.raises(LLMError, match="not valid JSON"):
        settings_from_env({**good, "LLM_EXTRA_BODY": "{oops"})
