import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError, Timeout

MAX_WAIT_SECONDS = 120
MAX_CONNECTION_RETRIES = 2
CONNECT_TIMEOUT_SECONDS = 10.0
REQUEST_TIMEOUT_SECONDS = 30.0


class LLMError(Exception):
    """Something went wrong that retrying will not fix."""


def describe_error(error: Exception) -> str:
    """A readable description of what went wrong, including the underlying cause."""
    text = f"{type(error).__name__}: {error}"
    cause = error.__cause__
    if cause is not None:
        text += f" (caused by {type(cause).__name__}: {cause})"
    return text


@dataclass
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1000
    extra_body: dict = field(default_factory=dict)


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False


def settings_from_env(env: Mapping[str, str]) -> LLMSettings:
    """Build the settings from a dictionary of environment variables."""
    missing = [name for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not env.get(name)]
    if missing:
        raise LLMError(f"Missing settings: {', '.join(missing)}. Copy .env.example to .env first.")

    try:
        extra_body = json.loads(env.get("LLM_EXTRA_BODY") or "{}")
    except json.JSONDecodeError as error:
        raise LLMError(f"LLM_EXTRA_BODY is not valid JSON: {error}") from error

    return LLMSettings(
        base_url=env["LLM_BASE_URL"],
        api_key=env["LLM_API_KEY"],
        model=env["LLM_MODEL"],
        temperature=float(env.get("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(env.get("LLM_MAX_TOKENS", "1000")),
        extra_body=extra_body,
    )


def load_settings() -> LLMSettings:
    """Read the .env file (if there is one) and build the settings from the environment."""
    load_dotenv()
    return settings_from_env(os.environ)


class LLMClient:
    """Sends chat requests, retries when the provider is busy, and caches the answers.

    `client` can be any object that behaves like the OpenAI client; tests pass a fake one.
    `sleep` is the function used to wait; tests pass one that does not wait.
    """

    def __init__(
        self,
        settings: LLMSettings,
        cache_dir: Path = Path("data/llm_cache"),
        client=None,
        max_retries: int = 5,
        sleep=time.sleep,
    ):
        self.settings = settings
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.sleep = sleep
        timeout = Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
        self.client = client or OpenAI(
            base_url=settings.base_url, api_key=settings.api_key, max_retries=0, timeout=timeout
        )

    def complete_json(self, messages: list[dict], schema: dict, schema_name: str) -> LLMReply:
        """Ask for an answer that follows a JSON schema. The reply text is a JSON string.

        With `strict` the provider guarantees the JSON has the right shape. It cannot
        guarantee the content is true, so the caller still has to check that.
        """
        request = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_completion_tokens": self.settings.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        if self.settings.extra_body:
            request["extra_body"] = self.settings.extra_body

        cache_file = self.cache_dir / f"{self.cache_key(request)}.json"
        if cache_file.exists():
            reply = LLMReply(**json.loads(cache_file.read_text(encoding="utf-8")))
            reply.cached = True
            reply.seconds = 0.0
            return reply

        reply = self.call_with_retries(request)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(reply.__dict__), encoding="utf-8")
        return reply

    def cache_key(self, request: dict) -> str:
        """Same provider and same request give the same key."""
        text = json.dumps({"base_url": self.settings.base_url, **request}, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def call_with_retries(self, request: dict) -> LLMReply:
        connection_failures = 0
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self.client.chat.completions.create(**request)
            except RateLimitError as error:
                wait = self.wait_time(error, attempt)
                reason = "rate limit reached"
                if wait > MAX_WAIT_SECONDS:
                    raise LLMError(
                        f"The provider asks for a pause of {wait:.0f} s. The daily limit is "
                        "probably used up. Try again later; answers already received are cached."
                    ) from error
            except APIConnectionError as error:
                connection_failures += 1
                reason = describe_error(error)
                if connection_failures > MAX_CONNECTION_RETRIES:
                    raise LLMError(
                        f"Cannot reach {self.settings.base_url}. {reason}. Check your internet "
                        "connection, any VPN, proxy or firewall, and LLM_BASE_URL in .env."
                    ) from error
                wait = 2.0 ** (connection_failures - 1)
            except InternalServerError as error:
                reason = describe_error(error)
                wait = 2.0**attempt
            else:
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise LLMError("The answer was cut off. Raise LLM_MAX_TOKENS in .env.")
                usage = response.usage
                return LLMReply(
                    text=choice.message.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    seconds=time.time() - started,
                )

            if attempt == self.max_retries:
                raise LLMError(
                    f"The provider kept failing after {self.max_retries} retries: {reason}"
                )
            print(f"  {reason}; waiting {wait:.0f} s (attempt {attempt + 1}) ...")
            self.sleep(wait)

        raise LLMError("unreachable")

    def wait_time(self, error: RateLimitError, attempt: int) -> float:
        """How long to wait after a 429: what the provider asks for, else a growing pause."""
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after) + 1.0
            except ValueError:
                pass
        return 2.0 ** (attempt + 1)
