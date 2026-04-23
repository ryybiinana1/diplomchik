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
from app.pipeline.bundle import compare_input_to_schema, load_model_bundle
from app.pipeline.profiling import profile_dataset

app = FastAPI(title="Churn/Retention Self-Serve (Self-Hosted)")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_JOBS_OUTPUT = (_PROJECT_ROOT / "outputs" / "jobs").resolve()

store = JobStore(str(_JOBS_OUTPUT))
score_store = JobStore(str(_PROJECT_ROOT / "outputs" / "scores"))


def _resolve_allowed_bundle_dir(bundle_dir: str) -> Path:
    """Разрешены только каталоги bundle внутри outputs/jobs (защита от path traversal)."""
    p = Path(bundle_dir).expanduser()
    try:
        resolved = p.resolve()
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Некорректный путь bundle_dir: {e}") from e

    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=400, detail="bundle_dir должен существовать и быть каталогом")

    try:
        resolved.relative_to(_JOBS_OUTPUT)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="bundle_dir должен находиться внутри каталога outputs/jobs",
        ) from e

    return resolved


def _dir_tree_max_mtime(root: Path) -> float:
    mt = 0.0
    if not root.exists():
        return mt
    for path in root.rglob("*"):
        if path.is_file():
            mt = max(mt, path.stat().st_mtime)
    return mt


def _validate_score_mapping(score_mapping: dict) -> None:
    values = [str(v) for v in score_mapping.values() if v]
    dup = sorted({v for v in values if values.count(v) > 1})
    if dup:
        raise HTTPException(
            status_code=400,
            detail=f"Одна и та же колонка скоринга сопоставлена несколько раз: {dup}",
        )


def _schema_error_detail(payload: dict) -> str:
    parts = []
    if payload.get("missing_mapping"):
        parts.append(f"не сопоставлены ожидаемые колонки: {', '.join(payload['missing_mapping'])}")
    if payload.get("missing_required"):
        parts.append(f"отсутствуют обязательные колонки: {', '.join(payload['missing_required'])}")
    if payload.get("missing_expected"):
        parts.append(f"не хватает колонок из обучающего набора: {', '.join(payload['missing_expected'])}")
    if payload.get("extra_columns"):
        parts.append(f"есть лишние колонки: {', '.join(payload['extra_columns'])}")
    if payload.get("invalid_mapping_keys"):
        parts.append(f"есть недопустимые ключи сопоставления: {', '.join(payload['invalid_mapping_keys'])}")
    return "; ".join(parts) if parts else "схема файла не совпадает с обучающим набором"


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
        df = pd.read_csv(file.file, encoding="utf-8-sig")
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
        if not params.get("model_name"):
            params["model_name"] = f"model_{job_id[:8]}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad JSON: {e}")

    try:
        df_check = pd.read_csv(raw_path, encoding="utf-8-sig")
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
        raise HTTPException(status_code=500, detail=f"job_status_failed: {e}") from e


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    status = store.read_status(job_id)
    return status.get("result", {})


@app.get("/jobs/{job_id}/experiments")
def job_experiments(job_id: str):
    """Сводная таблица вариантов для режима сравнения (grid_search)."""
    status = store.read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(status_code=404, detail="Обучение ещё не завершено")

    result = status.get("result") or {}
    if result.get("mode") != "grid_search":
        raise HTTPException(status_code=404, detail="Для этого задания нет таблицы сравнения вариантов")

    csv_path = Path(str(result.get("all_results_csv") or ""))
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="Файл с результатами сравнения не найден")

    df = pd.read_csv(csv_path)
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return {
        "selection_metric": result.get("selection_metric"),
        "n_experiments": result.get("n_experiments"),
        "rows": rows,
    }


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job_dir = store.job_dir(job_id)
    artifacts = job_dir / "artifacts"
    if not artifacts.exists():
        raise HTTPException(status_code=404, detail="No artifacts")

    zip_path = job_dir / "report.zip"
    artifacts_mtime = _dir_tree_max_mtime(artifacts)
    need_archive = True
    if zip_path.exists() and artifacts_mtime <= zip_path.stat().st_mtime:
        need_archive = False
    if need_archive:
        shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", artifacts)
    return FileResponse(zip_path, filename="report.zip")


@app.get("/models")
def list_models():
    return {"models": store.list_model_bundles()}


@app.post("/score")
async def score_with_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    bundle_dir: str = Form(...),
    score_mapping_json: str = Form("{}"),
):
    bundle_path = _resolve_allowed_bundle_dir(bundle_dir)
    bundle_str = str(bundle_path)
    try:
        bundle = load_model_bundle(bundle_str)
        score_mapping = json.loads(score_mapping_json or "{}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некорректный JSON сопоставления: {e}") from e
    _validate_score_mapping(score_mapping)

    score_id = score_store.new_job()
    score_dir = score_store.job_dir(score_id)
    raw_path = score_dir / "raw" / "score_input.csv"

    with raw_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    try:
        df_score = pd.read_csv(raw_path, encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV для прогноза: {e}") from e

    schema_payload = compare_input_to_schema(
        input_columns=list(df_score.columns),
        training_schema={
            **(bundle.get("training_schema") or {}),
            "input_dtypes": {col: str(df_score[col].dtype) for col in df_score.columns},
        },
        mapping=bundle.get("mapping") or {},
        score_mapping=score_mapping,
        require_full_mapping=True,
        reject_extra_columns=True,
    )
    if not schema_payload.get("ok"):
        raise HTTPException(status_code=400, detail=_schema_error_detail(schema_payload))

    score_store.write_status(score_id, {"status": "queued", "score_id": score_id})

    def _run():
        try:
            score_store.write_status(
                score_id,
                {"status": "running", "score_id": score_id, "stage": "scoring", "progress": 10},
            )
            res = run_scoring_pipeline(
                input_csv=str(raw_path),
                bundle_dir=bundle_str,
                out_dir=str(score_dir / "artifacts"),
                score_mapping=score_mapping,
            )
            scored_df = pd.read_csv(res["output_csv"])
            preview = scored_df.head(50).fillna("").to_dict(orient="records")
            payload = {
                **res,
                "preview": preview,
                "score_id": score_id,
                "download_url": f"/scores/{score_id}/download",
            }
            score_store.write_status(
                score_id,
                {
                    "status": "done",
                    "score_id": score_id,
                    "stage": "done",
                    "progress": 100,
                    "result": payload,
                },
            )
        except Exception as e:
            score_store.write_status(
                score_id,
                {"status": "failed", "score_id": score_id, "error": str(e)},
            )

    background_tasks.add_task(_run)
    return {"score_id": score_id, "status": "queued"}


@app.post("/score/schema-check")
async def score_schema_check(
    file: UploadFile = File(...),
    bundle_dir: str = Form(...),
    score_mapping_json: str = Form("{}"),
):
    bundle_path = _resolve_allowed_bundle_dir(bundle_dir)
    try:
        bundle = load_model_bundle(str(bundle_path))
        df = pd.read_csv(file.file, encoding="utf-8-sig")
        score_mapping = json.loads(score_mapping_json or "{}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось проверить схему: {e}") from e
    _validate_score_mapping(score_mapping)

    training_schema = bundle.get("training_schema") or {}
    payload = compare_input_to_schema(
        input_columns=list(df.columns),
        training_schema={
            **training_schema,
            "input_dtypes": {col: str(df[col].dtype) for col in df.columns},
        },
        mapping=bundle.get("mapping") or {},
        score_mapping=score_mapping,
        require_full_mapping=True,
        reject_extra_columns=True,
    )
    payload["input_columns"] = list(df.columns)
    return payload


@app.get("/scores/{score_id}")
def score_status(score_id: str):
    try:
        payload = dict(score_store.read_status(score_id))
        payload.setdefault("score_id", score_id)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"score_status_failed: {e}") from e


@app.get("/scores/{score_id}/download")
def download_score(score_id: str):
    score_dir = score_store.job_dir(score_id)
    out_csv = score_dir / "artifacts" / "scored_clients.csv"
    if not out_csv.exists():
        raise HTTPException(status_code=404, detail="No scored file")
    return FileResponse(out_csv, filename="scored_clients.csv")
