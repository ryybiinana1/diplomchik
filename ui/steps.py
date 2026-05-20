"""Ключи шагов мастера (стабильные строки для session_state)."""

STEP_1_DATA = "step_1_data"
STEP_2_ANALYTICS = "step_2_analytics"
STEP_3_TRAIN = "step_3_train"
STEP_4_QUALITY = "step_4_quality"
STEP_5_FORECAST = "step_5_forecast"

STEPS_ORDER = [
    STEP_1_DATA,
    STEP_2_ANALYTICS,
    STEP_3_TRAIN,
    STEP_4_QUALITY,
    STEP_5_FORECAST,
]

STEP_LABELS = {
    STEP_1_DATA: "1. Данные и колонки",
    STEP_2_ANALYTICS: "2. Анализ датасета",
    STEP_3_TRAIN: "3. Обучение",
    STEP_4_QUALITY: "4. Качество и сравнение",
    STEP_5_FORECAST: "5. Прогноз",
}

STEP_NAV_SHORT = {
    STEP_1_DATA: "Данные",
    STEP_2_ANALYTICS: "Анализ",
    STEP_3_TRAIN: "Обучение",
    STEP_4_QUALITY: "Качество",
    STEP_5_FORECAST: "Прогноз",
}

TRAINING_STAGE_LABELS = {
    "loading_input": "Загрузка и чтение файла",
    "profiling_and_validation": "Проверка структуры данных",
    "experiment_grid": "Подготовка сценариев обучения",
    "grid_search": "Перебор вариантов модели",
    "best_model_interpretation": "Интерпретация лучшей модели",
    "finalizing_artifacts": "Сохранение отчётов и модели",
    "starting": "Запуск",
    "done": "Готово",
}

MODEL_KIND_LABELS = {
    "lightgbm": "LightGBM",
    "logreg": "Логистическая регрессия",
    "random_forest": "Случайный лес",
    "catboost": "CatBoost",
    "sklearn_gbdt": "Gradient Boosting (sklearn)",
    "mlp": "Нейронная сеть (экспериментально)",
}

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "precision": "Точность (Precision)",
    "recall": "Полнота (Recall)",
}


def format_model_kind(kind: str | None) -> str:
    if not kind:
        return "—"
    return MODEL_KIND_LABELS.get(str(kind), str(kind))
