from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib


def save_model_bundle(
    bundle_dir: Path,
    model,
    calibrator,
    feature_cols: List[str],
    mapping: Dict[str, str],
    template: str,
    params: Dict[str, Any],
    training_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    joblib.dump(calibrator, bundle_dir / "calibrator.joblib")

    (bundle_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / "config.json").write_text(
        json.dumps(
            {"template": template, "params": params},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "training_schema.json").write_text(
        json.dumps(training_schema or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"bundle_dir": str(bundle_dir)}


def load_model_bundle(bundle_dir: str) -> Dict[str, Any]:
    d = Path(bundle_dir)
    model = joblib.load(d / "model.joblib")
    calibrator = joblib.load(d / "calibrator.joblib")
    feature_cols = json.loads((d / "feature_cols.json").read_text(encoding="utf-8"))
    mapping = json.loads((d / "mapping.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    schema_path = d / "training_schema.json"
    training_schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    return {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "mapping": mapping,
        "config": config,
        "training_schema": training_schema,
    }


def compare_input_to_schema(
    input_columns: List[str],
    training_schema: Dict[str, Any] | None,
    mapping: Dict[str, str] | None = None,
    score_mapping: Dict[str, str] | None = None,
    *,
    require_full_mapping: bool = False,
    reject_extra_columns: bool = False,
) -> Dict[str, Any]:
    schema = training_schema or {}
    overlay = {str(k): str(v) for k, v in (score_mapping or {}).items() if k and v}
    expected_training = list(schema.get("expected_source_columns") or schema.get("source_columns") or [v for v in (mapping or {}).values() if v])
    required_training = list(schema.get("required_source_columns") or [v for v in (mapping or {}).values() if v])
    input_set = set(input_columns)

    invalid_mapping_keys = sorted(set(overlay) - set(expected_training))
    missing_mapping = sorted([col for col in expected_training if not overlay.get(col)]) if require_full_mapping else []

    expected_source = {overlay.get(col, col) for col in expected_training}
    required_source = {overlay.get(col, col) for col in required_training}

    missing_required = sorted(required_source - input_set)
    missing_expected = sorted(expected_source - input_set)
    extra_columns = sorted(input_set - expected_source)

    expected_dtypes = schema.get("source_dtypes") or {}
    input_dtypes = schema.get("input_dtypes") or {}
    dtype_mismatch = []
    for train_col, expected_dtype in expected_dtypes.items():
        actual_dtype = input_dtypes.get(overlay.get(train_col, train_col))
        if actual_dtype and expected_dtype and actual_dtype != expected_dtype:
            dtype_mismatch.append(
                {
                    "column": overlay.get(train_col, train_col),
                    "expected_column": train_col,
                    "expected_dtype": str(expected_dtype),
                    "actual_dtype": str(actual_dtype),
                }
            )

    problems = [*missing_required, *invalid_mapping_keys, *missing_mapping]
    if reject_extra_columns:
        problems.extend(extra_columns)
    ok = not problems
    status = "ok" if ok else "error"
    if ok and (missing_expected or extra_columns or dtype_mismatch):
        status = "warning"

    return {
        "status": status,
        "ok": ok,
        "missing_required": missing_required,
        "missing_expected": missing_expected,
        "extra_columns": extra_columns,
        "dtype_mismatch": dtype_mismatch,
        "effective_mapping": overlay,
        "missing_mapping": missing_mapping,
        "invalid_mapping_keys": invalid_mapping_keys,
        "require_full_mapping": require_full_mapping,
        "reject_extra_columns": reject_extra_columns,
    }
