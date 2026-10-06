"""Check that the LLM connection works: two identical requests with a JSON schema.

The second request should come from the cache and use no tokens.
If something fails, run scripts/check_llm.py to find out where.
"""

from contract_rag.llm import LLMClient, LLMError, load_settings

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}
MESSAGES = [
    {"role": "user", "content": "What is the capital of Greece? Answer with the city and country."}
]


def main() -> None:
    try:
        settings = load_settings()
        client = LLMClient(settings)
        print(f"model: {settings.model}")

        for call_number in (1, 2):
            reply = client.complete_json(MESSAGES, SCHEMA, "capital")
            print(f"call {call_number}: {reply.text}")
            print(
                f"  tokens: {reply.prompt_tokens} in, {reply.completion_tokens} out"
                f" | {reply.seconds:.1f} s | from cache: {reply.cached}"
            )
    except LLMError as error:
        print(f"\nFAILED: {error}")
        print("Run 'uv run python scripts/check_llm.py' to see which step breaks.")


if __name__ == "__main__":
    main()
