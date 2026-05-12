"""ProtInvTree-style Monte Carlo Tree Search.

Implements Algorithm 1 (Sec. 4 of the paper) on top of the variant-
agnostic ``ModelAdapter``. The four MCTS stages are:

  1. Selection      — descend the tree from root via UCT (Eq. 3) until
                      reaching an expandable node.
  2. Expansion      — sample K children by stepping the policy
                      (model.decoding_step) K times from the same state,
                      each with a different RNG seed for diversity.
  3. Evaluation     — for every new child, run a one-shot jumpy rollout
                      (model.one_shot_complete) and score the resulting
                      sequence with the reward model (Eq. 11).
  4. Backpropagation — update visit counts and value sums along the
                       path back to the root (Eqs. 5–6).

After M iterations (or once a child reward exceeds ``reward_threshold``)
the search returns ``mcts_num_outputs`` (sequence, reward) pairs ranked
by reward.

The search operates on **a single target at a time** (B=1). Caller
should batch externally if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from byprot.models.dplm2.mcts.adapter import ModelAdapter
from byprot.models.dplm2.mcts.node import TreeNode
from byprot.models.dplm2.mcts.rollout import one_shot_jumpy_rollout


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class MCTSConfig:
    """Hyperparameters for ``mcts_search``.

    All values can be overridden per-call. Defaults follow ProtInvTree
    Sec. 5.1 (M=50, c_uct=0.01) with K=4 as the per-node fan-out.
    """

    iterations: int = 50          # M: total MCTS iterations
    expansions_per_node: int = 4  # K: child count expanded per Selection
    c_uct: float = 0.01           # exploration weight in UCT (Eq. 3)
    reward_threshold: float = 0.99  # τ: early-accept threshold for leaves
    max_depth: Optional[int] = None  # default = adapter's max_iter
    num_outputs: int = 1          # how many top-reward sequences to return
    rng_seed_base: int = 0        # base for per-step seeding (added to global step counter)
    verbose: bool = False


# ----------------------------------------------------------------------
# Selection (Eq. 3)
# ----------------------------------------------------------------------
def _select(root: TreeNode, k_max: int, c_uct: float) -> TreeNode:
    """Descend via UCT until an expandable (or terminal) node is reached.

    Mirrors the standard MCTS selection: at each interior node, pick the
    child with the highest UCT score; stop as soon as we reach a node
    that still has unexpanded children (or that is terminal).
    """
    node = root
    while not node.is_expandable(k_max) and not node.terminal:
        if not node.children:
            return node  # nothing to descend into; caller handles
        parent_visits = node.visit_count
        node = max(
            node.children,
            key=lambda c: c.uct(c_uct, parent_visits),
        )
    return node


# ----------------------------------------------------------------------
# Backprop (Eqs. 5–6)
# ----------------------------------------------------------------------
def _backprop(
    leaf: TreeNode,
    reward: float,
    tokens: Any,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Walk parent pointers, updating N and V; track best-of-subtree."""
    node: Optional[TreeNode] = leaf
    while node is not None:
        node.visit_count += 1
        node.value_sum += reward
        if reward > node.best_reward:
            node.best_reward = reward
            node.best_tokens = tokens
            node.best_metrics = metrics
        node = node.parent


# ----------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------
def mcts_search(
    *,
    adapter: ModelAdapter,
    input_tokens: torch.Tensor,
    partial_masks: Optional[torch.Tensor],
    max_iter: int,
    reward_fn: Callable[[Any], Any],
    decode_to_str: Callable[[Any], str],
    config: MCTSConfig,
) -> List[Tuple[Any, float, str, Dict[str, float]]]:
    """Run MCTS for one target and return top-N (tokens, reward, seq, metrics).

    Args:
        adapter: ``ModelAdapter`` bound to the loaded model.
        input_tokens: [1, L] token tensor for the single target.
        partial_masks: [1, L] bool tensor of conditioning positions
            (True = never unmask). Same as DPLM2 generate.
        max_iter: number of denoising steps the model uses to fully
            decode a sequence; also acts as the natural tree depth bound.
        reward_fn: callable(tokens) -> float OR (float, dict). When the
            caller wants per-output metrics recorded alongside the reward
            (e.g. ``mean_plddt``, ``ca_rmsd``, ``inv_fold_seq_recovery``),
            it should return ``(reward, metrics_dict)``. Plain float is
            still accepted and yields an empty metrics dict.
        decode_to_str: callable(tokens) -> str; used to materialize the
            amino-acid sequence for fold/tm-score.
        config: MCTSConfig with M / K / c_uct / etc.

    Returns:
        List of (output_tokens, reward, aa_str, metrics_dict) tuples,
        length ``config.num_outputs``, ranked by reward (descending).
    """
    if input_tokens.shape[0] != 1:
        raise ValueError(
            f"mcts_search expects B=1 inputs, got {tuple(input_tokens.shape)}"
        )

    max_depth = config.max_depth if config.max_depth is not None else max_iter

    root_state = adapter.make_root(
        input_tokens=input_tokens,
        partial_masks=partial_masks,
        max_iter=max_iter,
    )
    root = TreeNode(state=root_state, parent=None, depth=0)
    # Score the root with a jumpy rollout so it has an initial value.
    _evaluate_and_record(root, adapter, reward_fn, decode_to_str)

    completed: List[Tuple[Any, float, str, Dict[str, float]]] = []
    step_counter = 0

    for it in range(config.iterations):
        # 1. Selection
        node = _select(root, config.expansions_per_node, config.c_uct)
        if node.terminal and not node.children:
            # Only the root is terminal (e.g. zero unmasked) — nothing to do.
            break

        # 2. Expansion: K children from `node`
        for k in range(config.expansions_per_node):
            if node.terminal:
                break
            child_seed = config.rng_seed_base + step_counter
            step_counter += 1

            # Seed RNG so each child gets a distinct trajectory. This is
            # what gives baseline / hybrid / noise their per-child diversity.
            torch.manual_seed(child_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(child_seed)

            child_state = adapter.clone_state(node.state)
            child_state = adapter.step(child_state)
            child_depth = node.depth + 1
            child = TreeNode(
                state=child_state,
                parent=node,
                depth=child_depth,
                terminal=adapter.is_terminal(child_state) or child_depth >= max_depth,
            )
            node.children.append(child)

            # 3. Evaluation: jumpy rollout + reward
            tokens, reward, aa_str, metrics = _evaluate_and_record(
                child, adapter, reward_fn, decode_to_str
            )
            if config.verbose:
                print(
                    f"[mcts] iter={it} k={k} depth={child_depth} reward={reward:.4f}"
                )

            if reward >= config.reward_threshold or child.terminal:
                completed.append((tokens, reward, aa_str, metrics))

            if reward >= config.reward_threshold:
                # Early-accept: stop expanding this iteration; outer loop
                # may still continue collecting more above-threshold leaves
                # if config.iterations allows.
                break

    # Final ranking: include leaves we accepted plus the best-of-tree fallback.
    # Best-of-subtree is tracked at every node via _backprop, so root.best_*
    # gives us the best leaf the search ever evaluated.
    if root.best_tokens is not None:
        completed.append((
            root.best_tokens,
            root.best_reward,
            decode_to_str(root.best_tokens),
            root.best_metrics or {},
        ))

    completed.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate by AA string so we don't return the same seq twice.
    seen = set()
    deduped: List[Tuple[Any, float, str, Dict[str, float]]] = []
    for tokens, reward, aa_str, metrics in completed:
        if aa_str in seen:
            continue
        seen.add(aa_str)
        deduped.append((tokens, reward, aa_str, metrics))
        if len(deduped) >= config.num_outputs:
            break

    if not deduped:
        # Should not happen because we evaluated the root; but as a safety net,
        # rollout the root state.
        tokens = one_shot_jumpy_rollout(root.state, adapter)
        deduped.append((tokens, 0.0, decode_to_str(tokens), {}))

    return deduped[: config.num_outputs]


def _evaluate_and_record(
    node: TreeNode,
    adapter: ModelAdapter,
    reward_fn: Callable[[Any], Any],
    decode_to_str: Callable[[Any], str],
) -> Tuple[Any, float, str, Dict[str, float]]:
    """Jumpy-rollout `node`, score it, and backprop the reward.

    ``reward_fn`` may return either a plain ``float`` (legacy) or a
    ``(reward, metrics_dict)`` tuple. In the latter case the metrics
    dict is propagated alongside the best-of-subtree tokens so that
    callers can record per-output metrics in the final results.
    """
    tokens = one_shot_jumpy_rollout(node.state, adapter)
    aa_str = decode_to_str(tokens)
    result = reward_fn(tokens)
    if isinstance(result, tuple):
        reward, metrics = result
        reward = float(reward)
    else:
        reward, metrics = float(result), {}
    _backprop(node, reward, tokens, metrics)
    return tokens, reward, aa_str, metrics
