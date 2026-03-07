"""
shared/llm.py
Anthropic API call wrapper — shared across pipeline steps.

Used by:
  - step_0k_event_calendar (with web_search tool)
  - step7_execution_advisor (no tools)
  - Future: any step needing LLM

Source: Trading Desk Spec Teil 6 §32
"""

import logging
import os

import anthropic

logger = logging.getLogger("shared.llm")


def call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    tools: list = None,
    timeout: int = 120,
) -> dict:
    """
    Anthropic API call wrapper.

    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        model: Model name (e.g. "claude-sonnet-4-5-20250929")
        temperature: 0.0-1.0
        max_tokens: Max response tokens
        tools: Optional list of tool configs (e.g. web_search)
        timeout: Timeout in seconds

    Returns:
        Normalized response dict:
        {
            "content": [{"type": "text", "text": "..."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": N, "output_tokens": N},
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=timeout,
    )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
    }

    if tools:
        kwargs["tools"] = tools

    logger.info(
        f"API call: model={model}, temp={temperature}, "
        f"max_tokens={max_tokens}, tools={bool(tools)}"
    )

    response = client.messages.create(**kwargs)

    logger.info(
        f"API response: {response.stop_reason}, "
        f"usage={response.usage.input_tokens}in/"
        f"{response.usage.output_tokens}out"
    )

    # Normalize content blocks
    content = []
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        else:
            content.append({"type": block.type, "text": getattr(block, "text", "")})

    return {
        "content": content,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
