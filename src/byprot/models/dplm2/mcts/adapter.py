"""Variant-agnostic adapter over DPLM2 generate primitives.

Each DPLM2 variant (baseline / hybrid / noise) exposes the same five
primitives after the MCTS-friendly refactor:

    init_decoding_state(input_tokens, max_iter, **gen_kwargs) -> state
    clone_decoding_state(state) -> state
    decoding_step(state) -> state           # one denoising iter (Expansion)
    one_shot_complete(state) -> state       # fill all remaining masks (Jumpy)

This module wraps them in a small uniform `ModelAdapter` so the search
loop never has to branch on variant type. Variant-specific
hyperparameters that should not be exposed to the search loop (e.g.
`sample_noise_every_step` for noise) are baked into `gen_kwargs` at
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


VARIANTS = ("baseline", "hybrid", "noise")


@dataclass
class ModelAdapter:
    """Bundle of callables the MCTS loop uses to interact with a model.

    All callables operate on opaque ``state`` dicts owned by the model.
    """

    variant: str
    init_state: Callable[..., Dict[str, Any]]
    clone_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    step: Callable[[Dict[str, Any]], Dict[str, Any]]
    one_shot_complete: Callable[[Dict[str, Any]], Dict[str, Any]]
    extract_output_tokens: Callable[[Dict[str, Any]], Any]
    is_terminal: Callable[[Dict[str, Any]], bool]
    gen_kwargs: Dict[str, Any]

    def make_root(self, input_tokens, partial_masks, max_iter):
        """Initialize a fresh root state for `input_tokens`."""
        return self.init_state(
            input_tokens=input_tokens,
            max_iter=max_iter,
            partial_masks=partial_masks,
            **self.gen_kwargs,
        )


def _baseline_extract_tokens(state: Dict[str, Any]):
    return state["prev_decoder_out"]["output_tokens"]


def _baseline_is_terminal(state: Dict[str, Any]) -> bool:
    """A baseline state is terminal once all maskable positions are filled
    or the configured number of steps has been reached."""
    pdo = state["prev_decoder_out"]
    if pdo["step"] >= state["max_iter"]:
        return True
    masks = pdo.get("output_masks")
    if masks is None:
        return False
    return not bool(masks.any().item())


def _embedding_extract_tokens(state: Dict[str, Any]):
    return state["output_tokens"]


def _embedding_is_terminal(state: Dict[str, Any]) -> bool:
    if state["step"] >= state["max_iter"]:
        return True
    still_corrupted = ~state["clean_mask"] & state["output_masks"]
    return not bool(still_corrupted.any().item())


def build_adapter(variant: str, model, **gen_kwargs) -> ModelAdapter:
    """Build an adapter for `variant` bound to a loaded `model` instance.

    Variant-specific gen_kwargs (e.g. ``sample_noise_every_step`` for
    noise) should be passed here; they will flow into every
    ``init_decoding_state`` call.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")

    if variant == "baseline":
        return ModelAdapter(
            variant=variant,
            init_state=model.init_decoding_state,
            clone_state=model.clone_decoding_state,
            step=model.decoding_step,
            one_shot_complete=model.one_shot_complete,
            extract_output_tokens=_baseline_extract_tokens,
            is_terminal=_baseline_is_terminal,
            gen_kwargs=dict(gen_kwargs),
        )

    # hybrid / noise share the same state shape (top-level dict with
    # output_tokens / clean_mask / output_masks / step / max_iter ...).
    return ModelAdapter(
        variant=variant,
        init_state=model.init_decoding_state,
        clone_state=model.clone_decoding_state,
        step=model.decoding_step,
        one_shot_complete=model.one_shot_complete,
        extract_output_tokens=_embedding_extract_tokens,
        is_terminal=_embedding_is_terminal,
        gen_kwargs=dict(gen_kwargs),
    )
