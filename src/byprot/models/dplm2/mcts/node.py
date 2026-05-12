"""Tree node + UCT selection for ProtInvTree-style MCTS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    """One node in the MCTS search tree.

    Each node owns a `state` dict (an opaque DPLM2 decoding state owned
    by a `ModelAdapter`), tracks visit/value statistics for UCT, and
    keeps a list of expanded children. Leaf vs. non-leaf is implicit:
    a node is "expandable" if `not terminal and len(children) < K_max`.
    """

    state: Dict[str, Any]
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    depth: int = 0
    terminal: bool = False
    # Best (tokens, reward) seen at this subtree; lifted up by backprop.
    best_tokens: Any = None
    best_reward: float = -float("inf")
    best_metrics: Optional[Dict[str, float]] = None

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    def uct(self, c_uct: float, parent_visits: int) -> float:
        """Eq. 3 in ProtInvTree."""
        if self.visit_count == 0:
            return float("inf")
        return self.value + c_uct * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visit_count
        )

    def is_expandable(self, k_max: int) -> bool:
        return (not self.terminal) and len(self.children) < k_max
