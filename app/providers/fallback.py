"""An LLM provider that falls back to a second one when the first fails.

Resume upload must not fail because the network is down. Groq turns a 101-second
local extraction into ~3 seconds, which is worth having as the default — but it
is a *hosted* dependency in a project whose entire premise (DECISIONS 13.3) is
that everything runs locally, and a hosted dependency that can take the feature
down with it would be a straight downgrade.

So: try Groq, and on any :class:`LLMError` fall through to Ollama for that call,
logging at WARNING. The user waits longer and gets their resume parsed.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.providers.llm import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class FallbackLLMProvider(LLMProvider):
    """Delegates to ``primary``; on failure, retries the call on ``fallback``.

    Per *call*, not per process: a transient Groq outage should not pin the
    application to the slow path until it restarts, and a persistent one shows
    up as a WARNING on every request rather than one that scrolled away hours
    ago.
    """

    name = "fallback"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        #: Incremented whenever the fallback path is taken. Tests assert on it,
        #: and it is the number to expose if this ever needs a metric.
        self.fallback_count = 0

    @property
    def model(self) -> str:
        return getattr(self.primary, "model", self.primary.name)

    def _run(self, operation: str, call: Any) -> Any:
        try:
            return call(self.primary)
        except LLMError as exc:
            self.fallback_count += 1
            # WARNING, not ERROR: the call is about to succeed. But it must be
            # visible, because silent degradation to a 100x slower path is
            # exactly the kind of thing that goes unnoticed until a demo.
            logger.warning(
                "%s via %s failed (%s: %s); falling back to %s",
                operation,
                self.primary.name,
                type(exc).__name__,
                exc,
                self.fallback.name,
            )
            return call(self.fallback)

    def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        return self._run(
            "complete",
            lambda provider: provider.complete(prompt, schema, system=system),
        )

    def complete_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return self._run(
            "complete_text",
            lambda provider: provider.complete_text(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

    def close(self) -> None:
        for provider in (self.primary, self.fallback):
            close = getattr(provider, "close", None)
            if callable(close):
                close()
