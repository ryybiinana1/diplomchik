# app/main.py
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.jobs import JobStore
from app.mapping import CONTRACTS, detect_mapping, validate_mapping
from app.pipeline import run_pipeline, run_scoring_pipeline
from churnlib.validation_module import profile_dataset

app = FastAPI(title="Churn/Retention Self-Serve (Self-Hosted)")

store = JobStore("outputs/jobs")
score_store = JobStore("outputs/scores")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/contracts")
def get_contracts():
    return {"contracts": CONTRACTS}


@app.post("/inspect")
async def inspect_file(
    file: UploadFile = File(...),
    template: str = Form(...),
):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {e}")

    suggested_mapping = detect_mapping(list(df.columns), template)
    quality = profile_dataset(df, template)
    preview = df.head(20).fillna("").to_dict(orient="records")

    return {
        "template": template,
        "columns": list(df.columns),
        "suggested_mapping": suggested_mapping,
        "quality_report": quality,
        "preview": preview,
    }


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template: str = Form(...),
    mapping_json: str = Form(...),
    params_json: str = Form("{}"),
):
    job_id = store.new_job()
    job_dir = store.job_dir(job_id)
    raw_path = job_dir / "raw" / "input.csv"

    with raw_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    try:
        mapping = json.loads(mapping_json)
        params = json.loads(params_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad JSON: {e}")

    try:
        df_check = pd.read_csv(raw_path)
        validate_mapping(mapping, template, list(df_check.columns))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mapping error: {e}")

    store.write_status(job_id, {"status": "queued", "job_id": job_id})

    def _run():
        try:
            store.write_status(
                job_id,
                {"status": "running", "job_id": job_id, "stage": "starting", "progress": 1},
            )

            def _status_callback(stage: str, progress: int, extra: dict | None = None):
                payload = {"status": "running", "job_id": job_id, "stage": stage, "progress": progress}
                if extra:
                    payload["extra"] = extra
                store.write_status(job_id, payload)

            res = run_pipeline(
                input_csv=str(raw_path),
                template=template,
                mapping=mapping,
                params=params,
                out_dir=str(job_dir / "artifacts"),
                status_callback=_status_callback,
            )

            store.write_status(
                job_id,
                {"status": "done", "job_id": job_id, "stage": "done", "progress": 100, "result": res},
            )
        except Exception as e:
            store.write_status(job_id, {"status": "failed", "job_id": job_id, "error": str(e)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return store.read_status(job_id)
    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "error": f"job_status_endpoint_failed: {e}",
        }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    status = store.read_status(job_id)
    return status.get("result", {})


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job_dir = store.job_dir(job_id)
    artifacts = job_dir / "artifacts"
    if not artifacts.exists():
        raise HTTPException(status_code=404, detail="No artifacts")

    zip_path = job_dir / "report.zip"
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", artifacts)
    return FileResponse(zip_path, filename="report.zip")


@app.get("/models")
def list_models():
    return {"models": store.list_model_bundles()}


@app.post("/score")
async def score_with_model(
    file: UploadFile = File(...),
    bundle_dir: str = Form(...),
):
    score_id = score_store.new_job()
    score_dir = score_store.job_dir(score_id)
    raw_path = score_dir / "raw" / "score_input.csv"

    with raw_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    score_store.write_status(score_id, {"status": "running", "score_id": score_id})

    try:
        res = run_scoring_pipeline(
            input_csv=str(raw_path),
            bundle_dir=bundle_dir,
            out_dir=str(score_dir / "artifacts"),
        )
        scored_df = pd.read_csv(res["output_csv"])
        preview = scored_df.head(50).fillna("").to_dict(orient="records")
        payload = {**res, "preview": preview, "score_id": score_id, "download_url": f"/scores/{score_id}/download"}
        score_store.write_status(score_id, {"status": "done", "score_id": score_id, "result": payload})
        return payload
    except Exception as e:
        score_store.write_status(score_id, {"status": "failed", "score_id": score_id, "error": str(e)})
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/scores/{score_id}/download")
def download_score(score_id: str):
    score_dir = score_store.job_dir(score_id)
    out_csv = score_dir / "artifacts" / "scored_clients.csv"
    if not out_csv.exists():
        raise HTTPException(status_code=404, detail="No scored file")
    return FileResponse(out_csv, filename="scored_clients.csv")























'''from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException, File, Form
from fastapi.responses import FileResponse

from .jobs import JobStore
from .pipeline import run_pipeline, run_scoring_pipeline
from .mapping import detect_mapping, validate_mapping
from churnlib.validation_module import profile_dataset


app = FastAPI(title="Churn/Retention Self-Serve (Self-Hosted)")
store = JobStore("outputs/jobs")


@app.post("/inspect")
async def inspect_file(
    file: UploadFile = File(...),
    template: str = Form(...),
):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {e}")

    suggested_mapping = detect_mapping(list(df.columns), template)
    quality = profile_dataset(df, template)
    preview = df.head(20).fillna("").to_dict(orient="records")

    return {
        "template": template,
        "columns": list(df.columns),
        "suggested_mapping": suggested_mapping,
        "quality_report": quality,
        "preview": preview,
    }


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template: str = Form(...),
    mapping_json: str = Form(...),
    params_json: str = Form("{}"),
):
    job_id = store.new_job()
    job_dir = store.job_dir(job_id)

    raw_path = job_dir / "raw" / "input.csv"
    with raw_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    try:
        mapping = json.loads(mapping_json)
        params = json.loads(params_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad JSON: {e}")

    try:
        df_check = pd.read_csv(raw_path)
        validate_mapping(mapping, template, list(df_check.columns))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mapping error: {e}")

    store.write_status(job_id, {"status": "queued", "job_id": job_id})

    def _run():
        try:
            store.write_status(job_id, {
                "status": "running",
                "job_id": job_id,
                "stage": "starting",
                "progress": 1,
            })

            def _status_callback(stage: str, progress: int, extra: dict | None = None):
                payload = {
                    "status": "running",
                    "job_id": job_id,
                    "stage": stage,
                    "progress": progress,
                }
                if extra:
                    payload.update(extra)
                store.write_status(job_id, payload)

            res = run_pipeline(
                input_csv=str(raw_path),
                template=template,
                mapping=mapping,
                params=params,
                out_dir=str(job_dir / "artifacts"),
                status_callback=_status_callback,
            )

            store.write_status(job_id, {
                "status": "done",
                "job_id": job_id,
                "result": res,
                "stage": "finished",
                "progress": 100,
            })

        except Exception as e:
            store.write_status(job_id, {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
            })

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return store.read_status(job_id)


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    status = store.read_status(job_id)
    return status.get("result", {})


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job_dir = store.job_dir(job_id)
    artifacts = job_dir / "artifacts"
    if not artifacts.exists():
        raise HTTPException(status_code=404, detail="No artifacts")
    zip_path = job_dir / "report.zip"
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", artifacts)
    return FileResponse(zip_path, filename="report.zip")


@app.get("/models")
def list_models():
    return {"models": store.list_model_bundles()}


@app.post("/score")
async def score_with_model(
    file: UploadFile = File(...),
    bundle_dir: str = Form(...),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "score_input.csv"

        with tmp_csv.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        res = run_scoring_pipeline(
            input_csv=str(tmp_csv),
            bundle_dir=bundle_dir,
            out_dir=str(Path(tmpdir) / "scoring_output"),
        )

        scored_path = Path(res["output_csv"])
        scored_df = pd.read_csv(scored_path)

        return {
            **res,
            "preview": scored_df.head(50).fillna("").to_dict(orient="records"),
        }
'''