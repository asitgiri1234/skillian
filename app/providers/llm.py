"""The LLM provider contract.

Callers depend on this ABC, never on a concrete provider. Swapping Ollama for a
hosted model is a config flag plus one new file in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMUnavailableError(LLMError):
    """The backend could not be reached, or the model is not available.

    Separate from LLMResponseError because it is an operator problem (daemon
    down, model not pulled) rather than something a retry with a better prompt
    could fix.
    """


class LLMResponseError(LLMError):
    """The backend replied, but not with something usable (e.g. invalid JSON)."""


class LLMProvider(ABC):
    """Structured completion against a language model."""

    #: Stable identifier, matching the key in the provider registry.
    name: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Return a JSON object conforming to ``schema``.

        ``schema`` is a pydantic model *class*; the provider is expected to use
        its JSON schema to constrain decoding rather than asking the model to
        produce JSON in the prompt.

        Returns the decoded object as a plain dict and deliberately does **not**
        validate it against ``schema``. Validation is the caller's job: a
        provider that validated would have to choose a retry policy, and that
        policy belongs with the caller that knows what a good answer looks like
        (see app/structure.py).

        Raises:
            LLMUnavailableError: backend unreachable or model missing.
            LLMResponseError: reply could not be decoded as a JSON object.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return a free-text completion, with no schema constraining decoding.

        Added on day 3 for match explanations, which are prose read by a human.
        Forcing them through :meth:`complete` with a one-field wrapper schema
        would constrain the decoder for no benefit and make the model spend
        tokens on JSON punctuation.

        Abstract rather than a default implemented in terms of ``complete``:
        every real backend has a plain completion call, and a default would let
        a provider silently inherit a worse one.

        Raises:
            LLMUnavailableError: backend unreachable or model missing.
            LLMResponseError: the reply was empty.
        """
        raise NotImplementedError
