from __future__ import annotations

from typing import List

import pandas as pd

from churnlib.model_module import prepare_X


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    return prepare_X(df, feature_cols)
