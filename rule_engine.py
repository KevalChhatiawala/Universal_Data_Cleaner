import pandas as pd
from utils.helpers import calculate_iqr_bounds, safe_to_numeric, standardize_column_names


class RuleEngine:
    def __init__(self, df: pd.DataFrame, rules: dict):
        self.df = df.copy()
        self.rules = rules
        self.audit_log = []

    def apply(self):
        self._apply_global_rules()

        for col, rule in self.rules.get("columns", {}).items():
            if col not in self.df.columns:
                continue

            if rule.get("drop"):
                self.df.drop(columns=[col], inplace=True)
                self.audit_log.append(f"Dropped column: {col}")
                continue

            self._handle_type_cast(col, rule)
            self._handle_missing(col, rule)
            self._handle_outliers(col, rule)

        return self.df, self.audit_log

    def _apply_global_rules(self):
        if self.rules["global_rules"].get("standardize_column_names"):
            self.df = standardize_column_names(self.df)
            self.audit_log.append("Standardized column names")

        if self.rules["global_rules"].get("drop_duplicates"):
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            self.audit_log.append(f"Dropped {before - len(self.df)} duplicate rows")

    def _handle_type_cast(self, col, rule):
        target_type = rule.get("type")

        if target_type == "numeric":
            self.df[col] = safe_to_numeric(self.df[col])
            self.audit_log.append(f"{col}: cast to numeric")

        elif target_type == "datetime":
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            self.audit_log.append(f"{col}: cast to datetime")

        elif target_type == "categorical":
            self.df[col] = self.df[col].astype("category")
            self.audit_log.append(f"{col}: cast to categorical")

    def _handle_missing(self, col, rule):
        strategy = rule.get("missing", {}).get("strategy")
        if not strategy:
            return

        before = self.df[col].isna().sum()

        if strategy == "mean":
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif strategy == "median":
            self.df[col] = self.df[col].fillna(self.df[col].median())
        elif strategy == "mode":
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == "constant":
            value = rule["missing"].get("value", "Unknown")
            self.df[col] = self.df[col].fillna(value)
        elif strategy == "drop":
            self.df = self.df.dropna(subset=[col])

        after = self.df[col].isna().sum()
        self.audit_log.append(
            f"{col}: missing handled ({before} → {after}) using {strategy}"
        )

    def _handle_outliers(self, col, rule):
        outlier_rule = rule.get("outliers")
        if not outlier_rule:
            return

        lower, upper = calculate_iqr_bounds(self.df[col].dropna())

        before = len(self.df)
        self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        removed = before - len(self.df)

        self.audit_log.append(
            f"{col}: removed {removed} outliers using IQR"
        )