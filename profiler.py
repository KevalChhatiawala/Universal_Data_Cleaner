import pandas as pd
from typing import Dict, Optional
from utils.helpers import infer_column_type, calculate_iqr_bounds


class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self) -> Dict:
        profile = {
            "shape": self.df.shape,
            "columns": {}
        }

        for col in self.df.columns:
            series = self.df[col]

            profile["columns"][col] = {
                "dtype": str(series.dtype),
                "detected_type": infer_column_type(series),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(series.isna().mean() * 100, 2),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_value": (
                    series.dropna().iloc[0]
                    if not series.dropna().empty
                    else None
                ),
                "is_constant": series.nunique(dropna=True) == 1,
                "outlier_info": self._outlier_info(series),
            }

        return profile

    def _outlier_info(self, series: pd.Series) -> Optional[Dict]:
        if series.dtype.kind not in "iuf":
            return None

        lower, upper = calculate_iqr_bounds(series.dropna())

        outliers = series[(series < lower) | (series > upper)]
        return {
            "outlier_count": int(outliers.count()),
            "outlier_percent": round(len(outliers) / len(series), 4) * 100
        }