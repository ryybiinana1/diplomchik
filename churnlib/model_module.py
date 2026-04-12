# churnlib/model_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


@dataclass(frozen=True)
class TrainConfig:
    label_col: str = "target"
    time_col: str = "anchor_time"
    id_col: str = "entity_id"
    n_splits: int = 4
    test_size_frac: float = 0.2
    random_state: int = 42
    model_kind: str = "lightgbm"
    use_class_weight: bool = False
    model_params: Optional[Dict[str, Any]] = None


def compute_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg / pos)


def eval_probs(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def time_train_test_split(
    df: pd.DataFrame,
    time_col: str,
    test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).copy()
    cut = int((1.0 - test_frac) * len(df))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def prepare_X(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for col in X.columns:
        if str(X[col].dtype).startswith("datetime"):
            X[col] = pd.to_datetime(X[col]).view("int64") // 10**9

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def _get_model(kind: str, random_state: int, params: Optional[Dict] = None, pos_weight: float = 1.0):
    params = params or {}
    kind = kind.lower()

    if kind == "logreg":
        base = dict(
            C=1.0,
            max_iter=2000,
            random_state=random_state,
        )
        base.update(params)
        return LogisticRegression(**base)

    if kind == "sklearn_gbdt":
        base = dict(
            max_iter=500,
            learning_rate=0.05,
            random_state=random_state,
        )
        base.update(params)
        return HistGradientBoostingClassifier(**base)

    if kind == "random_forest":
        base = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        base.update(params)
        return RandomForestClassifier(**base)

    if kind == "lightgbm":
        import lightgbm as lgb
        base = dict(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_lambda=1.0,
            random_state=random_state,
        )
        if params.get("use_pos_weight", False):
            base["scale_pos_weight"] = pos_weight
        base.update({k: v for k, v in params.items() if k != "use_pos_weight"})
        return lgb.LGBMClassifier(**base)

    if kind == "catboost":
        from catboost import CatBoostClassifier
        base = dict(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=10,
            loss_function="Logloss",
            verbose=False,
            random_seed=random_state,
        )
        if params.get("use_pos_weight", False):
            base["scale_pos_weight"] = pos_weight
        base.update({k: v for k, v in params.items() if k != "use_pos_weight"})
        return CatBoostClassifier(**base)

    raise ValueError(f"Unknown model kind: {kind}")


def train_time_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: TrainConfig,
) -> Dict:
    train_df, test_df = time_train_test_split(df, cfg.time_col, cfg.test_size_frac)

    X_train = prepare_X(train_df, feature_cols)
    X_test = prepare_X(test_df, feature_cols)

    y_train = train_df[cfg.label_col].astype(int).values
    y_test = test_df[cfg.label_col].astype(int).values

    pos_weight = compute_pos_weight(y_train)

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    cv_metrics = []

    for tr_idx, va_idx in tscv.split(X_train):
        model = _get_model(
            kind=cfg.model_kind,
            random_state=cfg.random_state,
            params=cfg.model_params,
            pos_weight=pos_weight,
        )
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        p_va = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        cv_metrics.append(eval_probs(y_train[va_idx], p_va))

    cv_mean = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    cv_std = {k: float(np.std([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}

    model = _get_model(
        kind=cfg.model_kind,
        random_state=cfg.random_state,
        params=cfg.model_params,
        pos_weight=pos_weight,
    )
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    test_metrics = eval_probs(y_test, p_test)

    return {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "cv_metrics_mean": cv_mean,
        "cv_metrics_std": cv_std,
        "test_metrics": test_metrics,
        "p_test": p_test,
        "y_test": y_test,
        "pos_weight": pos_weight,
    }

def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: TrainConfig,
    min_train_splits: int = 2,
) -> Dict:
    df = df.sort_values(cfg.time_col).copy()
    X_all = prepare_X(df, feature_cols)
    y_all = df[cfg.label_col].astype(int).values

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    fold_rows = []

    for fold_id, (tr_idx, te_idx) in enumerate(tscv.split(X_all), start=1):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        X_train = X_all.iloc[tr_idx]
        X_test = X_all.iloc[te_idx]
        y_train = y_all[tr_idx]
        y_test = y_all[te_idx]

        pos_weight = compute_pos_weight(y_train)

        model = _get_model(
            kind=cfg.model_kind,
            random_state=cfg.random_state,
            params=cfg.model_params,
            pos_weight=pos_weight,
        )
        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]
        metrics = eval_probs(y_test, p_test)

        fold_rows.append({
            "fold": fold_id,
            "train_size": int(len(tr_idx)),
            "test_size": int(len(te_idx)),
            **metrics,
        })

    folds_df = pd.DataFrame(fold_rows)
    if folds_df.empty:
        raise ValueError("Walk-forward backtest produced no folds")

    mean_metrics = folds_df[["roc_auc", "pr_auc", "brier"]].mean().to_dict()
    std_metrics = folds_df[["roc_auc", "pr_auc", "brier"]].std().fillna(0).to_dict()

    return {
        "folds_df": folds_df,
        "mean_metrics": {k: float(v) for k, v in mean_metrics.items()},
        "std_metrics": {k: float(v) for k, v in std_metrics.items()},
    }