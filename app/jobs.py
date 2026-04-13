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

        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
                    bundle_candidate = result.get("artifacts", {}).get("bundle_dir")
                    if bundle_candidate and Path(bundle_candidate).exists():
                        bundle_dir = Path(bundle_candidate)
                except Exception:
                    pass

            if not bundle_dir:
                continue

            rows.append(
                {
                    "job_id": job_id,
                    "bundle_dir": str(bundle_dir),
                    "status": status.get("status"),
                    "template": result.get("template"),
                    "params_used": result.get("params_used", {}),
                    "test_metrics": result.get("test_metrics", {}),
                }
            )
        return rows