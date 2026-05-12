"""Jumpy denoising rollout (one-shot completion) for MCTS evaluation."""

from __future__ import annotations

from typing import Any, Dict


def one_shot_jumpy_rollout(state: Dict[str, Any], adapter) -> Any:
    """Run a one-step jumpy rollout from `state` and return final tokens.

    The state is cloned first so the caller's state is not mutated.
    Returns the `output_tokens` tensor from the completed state.
    """
    cloned = adapter.clone_state(state)
    completed = adapter.one_shot_complete(cloned)
    return adapter.extract_output_tokens(completed)
