from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self, base_dir: str = "outputs/jobs"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def new_job(self) -> str:
        job_id = str(uuid.uuid4())
        job_dir = self.base / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "raw").mkdir(exist_ok=True)
        (job_dir / "run").mkdir(exist_ok=True)
        (job_dir / "artifacts").mkdir(exist_ok=True)
        self.write_status(job_id, {"status": "created"})
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.base / job_id

    def write_status(self, job_id: str, payload: Dict) -> None:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / "status.json"
        tmp = job_dir / "status.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)

    def read_status(self, job_id: str) -> Dict:
        p = self.job_dir(job_id) / "status.json"
        if not p.exists():
            return {"status": "unknown", "job_id": job_id}

        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "status": "running",
                "job_id": job_id,
                "stage": "updating_status",
                "progress": None,
            }
        except Exception as e:
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"read_status_failed: {e}",
            }

    def list_jobs(self) -> List[Dict]:
        rows: List[Dict] = []
        for p in sorted(self.base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue

            job_id = p.name
            status = self.read_status(job_id)
            rows.append(
                {
                    "job_id": job_id,
                    "status": status.get("status"),
                    "stage": status.get("stage"),
                    "progress": status.get("progress"),
                    "result": status.get("result"),
                }
            )
        return rows

    def list_model_bundles(self) -> List[Dict]:
        rows: List[Dict] = []

        for p in sorted(self.base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue

            job_id = p.name
            status = self.read_status(job_id)
            result = status.get("result", {})

            if status.get("status") != "done":
                continue

            bundle_dir = None

            single_bundle = p / "artifacts" / "single_run" / "bundle"
            if single_bundle.exists():
                bundle_dir = single_bundle

            if not bundle_dir:
                try:
                    bundle_candidate = result.get("artifacts", {}).get("bundle_dir") or result.get("artifacts", {}).get(
                        "bundle", {}
                    ).get("bundle_dir")
                    if bundle_candidate and Path(bundle_candidate).exists():
                        bundle_dir = Path(bundle_candidate)
                except Exception as e:
                    logger.debug("bundle_dir from result skipped for %s: %s", job_id, e)

            if not bundle_dir:
                continue

            config_path = bundle_dir / "config.json"
            config = {}
            params_cfg = {}

            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    params_cfg = config.get("params", {}) or {}
                except Exception:
                    config = {}
                    params_cfg = {}

            result_params = result.get("params_used", {}) or {}
            merged_params = {**params_cfg, **result_params}

            rows.append(
                {
                    "job_id": job_id,
                    "bundle_dir": str(bundle_dir),
                    "status": status.get("status"),
                    "template": result.get("template") or config.get("template"),
                    "model_name": merged_params.get("model_name", job_id),
                    "model_kind": merged_params.get("model_kind"),
                    "horizon_days": merged_params.get("horizon_days"),
                    "history_days": merged_params.get("history_days"),
                    "params_used": merged_params,
                    "training_schema": result.get("training_schema", {}),
                    "test_metrics": result.get("test_metrics", {}),
                    "test_metrics_cal": result.get("test_metrics_cal", {}),
                }
            )

        return rows