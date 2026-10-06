"""Check the LLM connection step by step.

Steps 1 to 4 cost no tokens. Steps 5 and 6 send two tiny chat requests (about 200 tokens).
Each step says what works. The first step that fails shows where the problem is.
"""

import os
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from openai import OpenAI, Timeout

from contract_rag.llm import CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS, load_settings

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}


def try_chat(client: OpenAI, model: str, messages: list, **options) -> None:
    """Send one small chat request and print what happened."""
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=400, **options
        )
    except Exception as error:
        print(f"   FAILED after {time.time() - started:.1f} s: {type(error).__name__}: {error}")
        if error.__cause__ is not None:
            print(f"   caused by: {type(error.__cause__).__name__}: {error.__cause__}")
        return

    choice = response.choices[0]
    usage = response.usage
    tokens = f"{usage.prompt_tokens} in, {usage.completion_tokens} out" if usage else "unknown"
    print(f"ok in {time.time() - started:.1f} s | finish_reason: {choice.finish_reason}")
    print(f"tokens: {tokens}")
    print(f"reply: {(choice.message.content or '')[:100]!r}")


def main() -> None:
    settings = load_settings()
    url = urlparse(settings.base_url)
    host = url.hostname or ""
    port = url.port or (443 if url.scheme == "https" else 80)

    print(f"base url : {settings.base_url}")
    print(f"model    : {settings.model}")
    print(f"api key  : set, {len(settings.api_key)} characters")  # we never print the key itself
    print(f"options  : {settings.extra_body}")
    proxies = [
        n for n in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy") if os.environ.get(n)
    ]
    print(f"proxy variables set: {', '.join(proxies) or 'none'}")

    print(f"\n1. DNS lookup for {host}")
    try:
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(host, port)}
        print("   ok:", ", ".join(sorted(addresses)))
    except OSError as error:
        print(f"   FAILED: {error}")
        print("   The name could not be resolved: check your internet connection, VPN or DNS.")
        return

    print(f"\n2. TCP connection to {host}:{port}")
    try:
        socket.create_connection((host, port), timeout=10).close()
        print("   ok")
    except OSError as error:
        print(f"   FAILED: {error}")
        print("   The server is not reachable: a firewall, VPN or proxy may be blocking it.")
        return

    print("\n3. List the models with plain Python (urllib). Costs no tokens.")
    request = urllib.request.Request(
        settings.base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {settings.api_key}", "User-Agent": "contract-rag-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(f"   ok: HTTP {response.status}")
    except urllib.error.HTTPError as error:
        body = error.read()[:200].decode("utf-8", errors="replace")
        print(f"   HTTP {error.code}: {body}")
        if error.code == 401:
            print("   401 means the key is wrong or missing. Check LLM_API_KEY in .env.")
            return
        if error.code == 403:
            print("   403 here is usually the provider's bot filter refusing Python's plain")
            print("   urllib. It does not matter; the next step uses the real library.")
    except OSError as error:
        print(f"   FAILED: {error}")
        return

    print("\n4. List the models through the OpenAI library, as the real calls do")
    timeout = Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    client = OpenAI(
        base_url=settings.base_url, api_key=settings.api_key, max_retries=0, timeout=timeout
    )
    try:
        models = [model.id for model in client.models.list()]
    except Exception as error:
        print(f"   FAILED: {type(error).__name__}: {error}")
        if error.__cause__ is not None:
            print(f"   caused by: {type(error.__cause__).__name__}: {error.__cause__}")
        print("   If this says 401, the key is wrong. Otherwise the library cannot connect.")
        return
    print(f"   ok: {len(models)} models available")
    if settings.model in models:
        print(f"model '{settings.model}' is in the list")
    else:
        print(f"   WARNING: '{settings.model}' is NOT in the list. Some that are: {models[:10]}")

    print("\n5. A plain chat request: no schema, no special options")
    plain = [{"role": "user", "content": "Reply with one word: hello"}]
    try_chat(client, settings.model, plain)

    print("\n6. The same kind of request with the settings (strict JSON schema + options)")
    question = [{"role": "user", "content": "What is the capital of Greece? City and country."}]
    try_chat(
        client,
        settings.model,
        question,
        temperature=settings.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "capital", "strict": True, "schema": SCHEMA},
        },
        extra_body=settings.extra_body,
    )
    print("\nIf step 5 works and step 6 fails, one of the options is the problem.")


if __name__ == "__main__":
    main()
