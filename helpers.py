import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def calculate_iqr_bounds(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def try_parse_datetime(series: pd.Series, threshold: float = 0.7) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() >= threshold


def infer_column_type(series: pd.Series) -> str:
    if try_parse_datetime(series):
        return "datetime"
    if series.dtype.kind in "iuf":
        return "numeric"
    if series.nunique(dropna=True) <= 20:
        return "categorical"
    return "text"