from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
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


def _default_base_scenario() -> Scenario:
    return next(s for s in DEFAULT_SCENARIOS if s.name == "base")


def scenario_from_params(params: Dict[str, Any] | None, *, name: str = "base") -> Scenario:
    params = params or {}
    default = _default_base_scenario()
    margin = float(params.get("business_margin", default.margin))
    cost = float(params.get("business_cost", default.cost))
    success = float(params.get("business_success", default.success))
    margin = min(max(margin, 0.0), 1.0)
    success = min(max(success, 0.0), 1.0)
    cost = max(cost, 0.0)
    return Scenario(name=name, margin=margin, cost=cost, success=success)


def build_scenarios(params: Dict[str, Any] | None) -> List[Scenario]:
    base = scenario_from_params(params, name="base")
    conservative = Scenario(
        "conservative",
        margin=min(max(base.margin * 0.8, 0.0), 1.0),
        cost=max(base.cost * 1.25, 0.0),
        success=min(max(base.success * 0.7, 0.0), 1.0),
    )
    optimistic = Scenario(
        "optimistic",
        margin=min(max(base.margin * 1.15, 0.0), 1.0),
        cost=max(base.cost * 0.8, 0.0),
        success=min(max(base.success * 1.35, 0.0), 1.0),
    )
    return [conservative, base, optimistic]


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


def profit_curve_from_ev(ev: np.ndarray) -> np.ndarray:
    ev = np.asarray(ev, dtype=float)
    if ev.size == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(-ev)
    return np.cumsum(ev[order])
