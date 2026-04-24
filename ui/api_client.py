from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import requests


def _json_safe(obj: Any) -> Any:
    """Превращает mapping/params в то, что гарантированно сериализуется в JSON (numpy и т.д.)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, float) and obj != obj:  # NaN
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


def _normalize_mapping_columns(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Пробелы в именах колонок часто ломают проверку на сервере."""
    out: Dict[str, Any] = {}
    for k, v in mapping.items():
        if v is None:
            out[str(k)] = None
        elif isinstance(v, str):
            s = v.strip()
            out[str(k)] = s if s else None
        else:
            out[str(k)] = v
    return out


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
        mapping_payload = _json_safe(_normalize_mapping_columns(mapping))
        params_payload = _json_safe(params)
        data = {
            "template": str(template),
            "mapping_json": json.dumps(mapping_payload, ensure_ascii=False),
            "params_json": json.dumps(params_payload, ensure_ascii=False),
        }
        r = self.session.post(f"{self.base_url}/jobs", files=files, data=data, timeout=600)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text or r.reason
            raise RuntimeError(f"{r.status_code}: {detail}") from None
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
        r.raise_for_status()
        return r.json()

    def job_experiments(self, job_id: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}/experiments", timeout=60)
        r.raise_for_status()
        return r.json()

    def download_job_artifact(self, job_id: str, artifact_key: str) -> bytes:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}/artifacts/{artifact_key}", timeout=600)
        r.raise_for_status()
        return r.content

    def download_job_report(self, job_id: str) -> bytes:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}/download", timeout=600)
        r.raise_for_status()
        return r.content

    def list_models(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/models", timeout=60)
        r.raise_for_status()
        return r.json()

    def start_score(
        self,
        file_bytes: bytes,
        bundle_dir: str,
        score_mapping: Dict[str, Any] | None = None,
        scenario_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        files = {"file": ("score.csv", file_bytes, "text/csv")}
        data = {
            "bundle_dir": bundle_dir,
            "score_mapping_json": json.dumps(_json_safe(_normalize_mapping_columns(score_mapping or {})), ensure_ascii=False),
            "scenario_params_json": json.dumps(_json_safe(scenario_params or {}), ensure_ascii=False),
        }
        r = self.session.post(f"{self.base_url}/score", files=files, data=data, timeout=120)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text or r.reason
            raise RuntimeError(f"{r.status_code}: {detail}") from None
        return r.json()

    def score_schema_check(
        self,
        file_bytes: bytes,
        bundle_dir: str,
        score_mapping: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        files = {"file": ("score.csv", file_bytes, "text/csv")}
        data = {
            "bundle_dir": bundle_dir,
            "score_mapping_json": json.dumps(_json_safe(_normalize_mapping_columns(score_mapping or {})), ensure_ascii=False),
        }
        r = self.session.post(f"{self.base_url}/score/schema-check", files=files, data=data, timeout=120)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text or r.reason
            raise RuntimeError(f"{r.status_code}: {detail}") from None
        return r.json()

    def score_job_status(self, score_id: str) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/scores/{score_id}", timeout=60)
        if r.status_code >= 400:
            return {
                "status": "failed",
                "score_id": score_id,
                "error": f"API returned {r.status_code}: {r.text}",
            }
        return r.json()

    def download_score_csv(self, score_id: str) -> bytes:
        r = self.session.get(f"{self.base_url}/scores/{score_id}/download", timeout=600)
        r.raise_for_status()
        return r.content

    def download_score_artifact(self, score_id: str, artifact_name: str) -> bytes:
        r = self.session.get(f"{self.base_url}/scores/{score_id}/artifacts/{artifact_name}", timeout=600)
        r.raise_for_status()
        return r.content