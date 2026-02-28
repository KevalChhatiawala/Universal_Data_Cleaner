import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

from profiler.profiler import DataProfiler
from rules.rule_engine import RuleEngine

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Universal Data Cleaner",
    layout="wide"
)

st.title("🧹 Universal Data Cleaner & EDA Tool")

# =========================================================
# Performance Helpers
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return pd.read_json(file, orient="records")


@st.cache_data(show_spinner=False)
def run_profiling(df):
    profiler = DataProfiler(df)
    return profiler.profile()


def get_active_df():
    """Always return the active dataframe (cleaned if exists, else raw)"""
    return st.session_state.get("clean_df", st.session_state["raw_df"])


# =========================================================
# File Upload
# =========================================================
uploaded_file = st.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "json"]
)

if not uploaded_file:
    st.stop()

df = load_data(uploaded_file)

# Persist raw + active dataframe
st.session_state["raw_df"] = df
st.session_state.setdefault("clean_df", df.copy())

st.success("Dataset loaded successfully")

# =========================================================
# Tabs
# =========================================================
tab_overview, tab_cleaning, tab_eda, tab_query = st.tabs(
    ["📋 Overview", "🧹 Cleaning", "📊 EDA", "🧠 SQL Query"]
)

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab_overview:
    st.header("🔍 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Total Missing", int(df.isna().sum().sum()))

    st.dataframe(df.head(20), use_container_width=True)

    # ---------------- Missing Values ----------------
    st.subheader("🧩 Missing Values Summary")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2)
    }).reset_index(drop=True)

    st.dataframe(
        missing_df.sort_values("Missing Count", ascending=False),
        use_container_width=True
    )

    # ---------------- Table Definition ----------------
    st.subheader("📘 Table Definition (Data Dictionary)")

    table_def = []
    for col in df.columns:
        sample_vals = df[col].dropna().unique()[:3]
        sample_vals = ", ".join(map(str, sample_vals))

        table_def.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].notna().sum()),
            "Missing Count": int(df[col].isna().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample Values": sample_vals
        })

    st.dataframe(pd.DataFrame(table_def), use_container_width=True)

# =========================================================
# TAB 2 — CLEANING
# =========================================================
with tab_cleaning:
    profile = run_profiling(df)

    st.header("🤖 Auto Rule Suggestions")

    suggested_rules = {
        "global_rules": {"drop_duplicates": True},
        "columns": {}
    }

    for col, info in profile["columns"].items():
        rule = {}

        if info["missing_percent"] > 40:
            rule["drop"] = True
        else:
            if info["detected_type"] == "numeric":
                if info["missing_percent"] < 5:
                    rule["missing"] = {"strategy": "mean"}
                elif info["missing_percent"] < 30:
                    rule["missing"] = {"strategy": "median"}
            else:
                if info["missing_percent"] > 0:
                    rule["missing"] = {"strategy": "mode"}

        if (
            info["detected_type"] == "numeric"
            and info.get("outlier_info", {}).get("outlier_count", 0) > 0
        ):
            rule["outliers"] = {"method": "iqr"}

        if rule:
            suggested_rules["columns"][col] = rule

    st.json(suggested_rules)
    use_suggestions = st.checkbox("Apply suggested rules")

    # ---------------- Manual Rules ----------------
    st.header("🧹 Cleaning Rules")

    rules = suggested_rules if use_suggestions else {
        "global_rules": {
            "drop_duplicates": st.checkbox("Drop duplicates"),
            "standardize_column_names": st.checkbox("Standardize column names")
        },
        "columns": {}
    }

    for col in df.columns:
        with st.expander(f"⚙️ {col}"):
            rule = {}

            t = st.selectbox(
                "Convert type",
                ["", "numeric", "categorical", "datetime"],
                key=f"type_{col}"
            )
            if t:
                rule["type"] = t

            m = st.selectbox(
                "Missing strategy",
                ["", "mean", "median", "mode", "constant", "drop"],
                key=f"miss_{col}"
            )
            if m:
                rule["missing"] = {"strategy": m}

            if profile["columns"][col]["detected_type"] == "numeric":
                if st.checkbox("Remove outliers (IQR)", key=f"out_{col}"):
                    rule["outliers"] = {"method": "iqr"}

            if st.checkbox("Drop column", key=f"drop_{col}"):
                rule = {"drop": True}

            if rule:
                rules["columns"][col] = rule

    if st.button("🚀 Apply Cleaning"):
        engine = RuleEngine(df, rules)
        clean_df, audit = engine.apply()
        st.session_state["clean_df"] = clean_df

        st.success("Cleaning completed")

        st.subheader("🧾 Audit Log")
        for a in audit:
            st.write("•", a)

        st.dataframe(clean_df.head(20), use_container_width=True)

# =========================================================
# TAB 3 — EDA (GRAPHS)
# =========================================================
with tab_eda:
    st.header("📊 Exploratory Data Analysis")

    active_df = get_active_df()
    num_cols = active_df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        st.warning("No numeric columns available for EDA.")
    else:
        col = st.selectbox("Select numeric column", num_cols)

        st.plotly_chart(
            px.histogram(active_df, x=col, marginal="box"),
            use_container_width=True
        )

        st.plotly_chart(
            px.box(active_df, y=col),
            use_container_width=True
        )

        if len(num_cols) > 1:
            st.subheader("📈 Correlation Heatmap")
            corr = active_df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 — SQL QUERY EDITOR
# =========================================================
with tab_query:
    st.header("🧠 SQL-Style Query Editor")

    st.markdown(
        """
        **Examples:**
        ```
        age > 30
        salary >= 50000 and city == 'Mumbai'
        ```
        """
    )

    query_text = st.text_area(
        "Write Pandas-SQL style query",
        height=150,
        placeholder="e.g. age > 30 and salary < 80000"
    )

    if st.button("▶ Run Query"):
        try:
            result = get_active_df().query(query_text)
            st.success(f"Returned {len(result)} rows")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"Query error: {e}")

# =========================================================
# Export
# =========================================================
st.header("⬇ Export Cleaned Data")

active_df = get_active_df()

st.download_button(
    "Download CSV",
    active_df.to_csv(index=False),
    "cleaned.csv"
)

st.download_button(
    "Download JSON",
    active_df.to_json(orient="records"),
    "cleaned.json"
)