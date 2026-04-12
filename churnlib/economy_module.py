from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    margin: float   # m
    cost: float     # C
    success: float  # s


DEFAULT_SCENARIOS = [
    Scenario("conservative", margin=0.30, cost=3.0, success=0.10),
    Scenario("base", margin=0.50, cost=2.0, success=0.20),
    Scenario("optimistic", margin=0.70, cost=1.5, success=0.30),
]


def expected_value(p: np.ndarray, V: np.ndarray, sc: Scenario) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    V = np.asarray(V, dtype=float)
    return p * sc.success * (V * sc.margin) - sc.cost


def profit_curve(p: np.ndarray, V: np.ndarray, sc: Scenario) -> np.ndarray:
    order = np.argsort(-p)
    ev = expected_value(p[order], V[order], sc)
    return np.cumsum(ev)


def best_k(curve: np.ndarray) -> Tuple[int, float]:
    k = int(np.argmax(curve) + 1)
    return k, float(curve[k - 1])
