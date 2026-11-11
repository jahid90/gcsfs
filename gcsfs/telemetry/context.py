"""Internal thread-safe and coroutine-safe context management for telemetry state."""

from __future__ import annotations

import contextvars
import os
import re
from enum import StrEnum
from typing import Dict, Optional, Union

# Allowed characters in RFC 9110 product-version tokens: ASCII alphanumeric, '_', '.', '-'
_TOKEN_SANITIZE_REGEX = re.compile(r"[^a-zA-Z0-9_.-]")


class Dimension(StrEnum):
    """Standard telemetry dimension keys."""

    FRAMEWORK = "fw"
    ENVIRONMENT = "env"
    METHOD = "m"


ALL_DIMENSIONS: frozenset[Dimension] = frozenset(Dimension)
USER_AGENT_DIMENSIONS: frozenset[Dimension] = frozenset(
    {Dimension.ENVIRONMENT, Dimension.FRAMEWORK}
)


def sanitize_token(token: Optional[str], max_len: int = 64) -> Optional[str]:
    """Sanitize a bare token value to comply with RFC 9110 product-version grammar."""
    if not token or not isinstance(token, str):
        return None

    clean = _TOKEN_SANITIZE_REGEX.sub("_", token.strip())[:max_len]
    return clean if clean.strip("_.-") else None


# ContextVar for propagating multi-dimensional telemetry tokens across async event loop and thread boundaries
_current_telemetry: contextvars.ContextVar[Optional[Dict[str, str]]] = (
    contextvars.ContextVar("gcsfs_current_telemetry", default=None)
)


if hasattr(os, "register_at_fork"):

    def _reset_telemetry_in_child():
        _current_telemetry.set(None)

    os.register_at_fork(after_in_child=_reset_telemetry_in_child)


def has_telemetry_context(dimension: Optional[Dimension] = None) -> bool:
    """Check if a telemetry context (or a specific dimension within it) is active."""
    val = _current_telemetry.get()
    if val is None:
        return False
    return True if dimension is None else dimension.value in val


def get_telemetry_context() -> Dict[str, str]:
    """Retrieve active telemetry tokens mapping from context."""
    val = _current_telemetry.get()
    return dict(val) if val is not None else {}


def set_telemetry_context(
    tokens: Union[Dict[str, str], Dimension], value: Optional[str] = None
) -> contextvars.Token:
    """Set telemetry tokens mapping (or update a single dimension) in context."""
    if isinstance(tokens, Dimension):
        current = get_telemetry_context()
        current[tokens.value] = str(value) if value is not None else ""
        return _current_telemetry.set(current)
    normalized = {
        (k.value if isinstance(k, Dimension) else str(k)): (
            str(v) if v is not None else ""
        )
        for k, v in tokens.items()
    }
    return _current_telemetry.set(normalized)


def reset_telemetry_context(token: contextvars.Token) -> None:
    """Reset telemetry context to the state associated with the given token."""
    _current_telemetry.reset(token)
