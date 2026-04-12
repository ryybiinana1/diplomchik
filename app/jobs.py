from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List


class JobStore:
    def __init__(self, base_dir: str = "outputs/jobs"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def new_job(self) -> str:
        job_id = str(uuid.uuid4())
        (self.base / job_id).mkdir(parents=True, exist_ok=True)
        (self.base / job_id / "raw").mkdir(exist_ok=True)
        (self.base / job_id / "run").mkdir(exist_ok=True)
        (self.base / job_id / "artifacts").mkdir(exist_ok=True)
        self.write_status(job_id, {"status": "created"})
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.base / job_id

    def write_status(self, job_id: str, payload: Dict):
        p = self.job_dir(job_id) / "status.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_status(self, job_id: str) -> Dict:
        p = self.job_dir(job_id) / "status.json"
        if not p.exists():
            return {"status": "unknown"}
        return json.loads(p.read_text(encoding="utf-8"))

    def list_jobs(self) -> List[Dict]:
        rows: List[Dict] = []
        for p in sorted(self.base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_dir():
                continue
            job_id = p.name
            status = self.read_status(job_id)
            rows.append({
                "job_id": job_id,
                "status": status.get("status"),
                "stage": status.get("stage"),
                "progress": status.get("progress"),
                "result": status.get("result"),
            })
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
                    bundle_candidate = result.get("artifacts", {}).get("bundle", {}).get("bundle_dir")
                    if bundle_candidate and Path(bundle_candidate).exists():
                        bundle_dir = Path(bundle_candidate)
                except Exception:
                    pass

            if not bundle_dir:
                continue

            rows.append({
                "job_id": job_id,
                "bundle_dir": str(bundle_dir),
                "status": status.get("status"),
                "template": result.get("template"),
                "params_used": result.get("params_used", {}),
                "test_metrics_cal": result.get("test_metrics_cal", {}),
                "business_metrics": result.get("business_metrics", {}),
            })

        return rows