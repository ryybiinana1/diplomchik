from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class ApiClient:
    base_url: str

    def __post_init__(self):
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> "ApiClient":
        base_url = os.getenv("API_URL", "http://localhost:8000")
        return cls(base_url=base_url.rstrip("/"))

    def contracts(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/contracts", timeout=60)
        r.raise_for_status()
        return r.json()

    def inspect(self, file_bytes: bytes, template: str) -> Dict[str, Any]:
        files = {"file": ("input.csv", file_bytes, "text/csv")}
        data = {"template": template}
        r = self.session.post(f"{self.base_url}/inspect", files=files, data=data, timeout=180)
        r.raise_for_status()
        return r.json()

    def create_job(
        self,
        file_bytes: bytes,
        template: str,
        mapping: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        files = {"file": ("input.csv", file_bytes, "text/csv")}
        data = {
            "template": template,
            "mapping_json": json.dumps(mapping, ensure_ascii=False),
            "params_json": json.dumps(params, ensure_ascii=False),
        }
        r = self.session.post(f"{self.base_url}/jobs", files=files, data=data, timeout=600)
        r.raise_for_status()
        return r.json()

    def job_status(self, job_id: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}", timeout=60)
        if r.status_code >= 400:
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"API returned {r.status_code}: {r.text}",
            }
        return r.json()

    def job_result(self, job_id: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}/result", timeout=60)
        if r.status_code >= 400:
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"API returned {r.status_code}: {r.text}",
            }
        return r.json()

    def list_models(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/models", timeout=60)
        r.raise_for_status()
        return r.json()

    def score(self, file_bytes: bytes, bundle_dir: str) -> Dict[str, Any]:
        files = {"file": ("score.csv", file_bytes, "text/csv")}
        data = {"bundle_dir": bundle_dir}
        r = self.session.post(f"{self.base_url}/score", files=files, data=data, timeout=600)
        r.raise_for_status()
        return r.json()