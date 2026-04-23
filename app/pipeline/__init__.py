"""
Оркестрация обучения и скоринга.

Публичный API пакета совпадает с бывшим модулем ``app.pipeline``:
``run_pipeline``, ``run_scoring_pipeline``.
"""

from app.pipeline.scoring import run_scoring_pipeline
from app.pipeline.train import run_pipeline

__all__ = ["run_pipeline", "run_scoring_pipeline"]
