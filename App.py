import os
import uuid
import tempfile
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Data Analytics Studio", layout="wide", page_icon="📊")

# ================= STYLING (glass + Power BI style KPI cards) =================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1026 0%, #131a3a 50%, #1a2456 100%); }
h1, h2, h3, p, label, span, div { color: #e8ecff !important; }
section[data-testid="stSidebar"] { background: rgba(10,14,35,0.9); }
.kpi-card {
    background: rgba(255,255,255,0.07);
    border-left: 5px solid #6c8bff;
    border-radius: 12px;
    padding: 16px 18px;
    backdrop-filter: blur(10px);
    margin-bottom: 8px;
}
.kpi-value { font-size: 26px; font-weight: 800; margin: 0; }
.kpi-label { font-size: 13px; opacity: 0.75; margin: 0; }
.story-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 22px 26px;
    backdrop-filter: blur(10px);
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Data Analytics Studio")
st.caption("Upload CSV / Parquet / Excel (scales to crores of rows via DuckDB, out-of-core) → Clean → Power BI-style Dashboard → Executive Data Story")

CURRENCY = st.sidebar.selectbox("Currency symbol (used in KPIs & story)", ["₹", "$", "€", "£"], index=0)
st.sidebar.caption("Big files: prefer CSV/Parquet over Excel. Streamlit Cloud's default upload cap is 200MB — raise it via .streamlit/config.toml (maxUploadSize) if you need more.")

# ================= HELPERS =================
def to_float(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def qi(col):
    return '"' + str(col).replace('"', '""') + '"'

def ql(val):
    return "'" + str(val).replace("'", "''") + "'"

def format_big_number(value, symbol=""):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"{symbol}{value/1e9:.2f} Billion"
    elif abs_v >= 1e7 and symbol == "₹":
        return f"{symbol}{value/1e7:.2f} Crore"
    elif abs_v >= 1e6:
        return f"{symbol}{value/1e6:.2f} Million"
    elif abs_v >= 1e3:
        return f"{symbol}{value/1e3:.2f} K"
    else:
        return f"{symbol}{value:,.2f}"

def kpi_card(label, value, color="#6c8bff"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color};">
        <p class="kpi-value">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)

def get_column_types(con, table):
    info = con.execute(f"DESCRIBE {table}").df()
    numeric_types = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
                      "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL", "REAL"}
    date_types = {"DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "TIME", "TIMESTAMP_S",
                  "TIMESTAMP_MS", "TIMESTAMP_NS"}
    num_cols, cat_cols, date_cols = [], [], []
    for _, row in info.iterrows():
        name, dtype = row["column_name"], str(row["column_type"]).upper()
        base = dtype.split("(")[0].strip()
        if base in numeric_types:
            num_cols.append(name)
        elif base in date_types:
            date_cols.append(name)
        else:
            cat_cols.append(name)
    return num_cols, cat_cols, date_cols

def detect_col(cols, keywords, exclude=None):
    exclude = exclude or []
    cols = [c for c in cols if c not in exclude]
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None

# ================= SESSION STATE (per-user isolated DuckDB) =================
if "db_id" not in st.session_state:
    st.session_state.db_id = str(uuid.uuid4())
if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(database=":memory:")
if "raw_table" not in st.session_state:
    st.session_state.raw_table = None
if "clean_table" not in st.session_state:
    st.session_state.clean_table = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []

con = st.session_state.con

# ================= UPLOAD =================
uploaded = st.file_uploader("Upload CSV / Parquet / Excel", type=["csv", "parquet", "xlsx", "xls"])

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{st.session_state.db_id}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    full_scan = st.checkbox("Scan full file for accurate column types (slower, recommended for messy data)", value=False)
    sample_size = -1 if full_scan else 200000

    try:
        if suffix == ".csv":
            con.execute(f"""
                CREATE OR REPLACE VIEW raw_data AS
                SELECT * FROM read_csv_auto('{tmp_path}', SAMPLE_SIZE={sample_size}, IGNORE_ERRORS=TRUE)
            """)
        elif suffix == ".parquet":
            con.execute(f"CREATE OR REPLACE VIEW raw_data AS SELECT * FROM read_parquet('{tmp_path}')")
        else:
            df_xl = pd.read_excel(tmp_path)
            con.register("raw_data_xl", df_xl)
            con.execute("CREATE OR REPLACE VIEW raw_data AS SELECT * FROM raw_data_xl")

        st.session_state.raw_table = "raw_data"
        row_count = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        st.success(f"Loaded **{row_count:,}** rows — ready for cleaning & analysis.")
    except Exception as e:
        st.error(f"Couldn't read file: {e}")

tab1, tab2, tab3 = st.tabs(["🧹 AI Data Cleaner", "📈 Power BI-Style Dashboard", "📝 Executive Data Story"])

# ================= TAB 1: AI DATA CLEANER =================
with tab1:
    if st.session_state.raw_table is None:
        st.info("Upload a file above to begin.")
    else:
        num_cols, cat_cols, date_cols = get_column_types(con, "raw_data")
        all_cols = num_cols + cat_cols + date_cols
        total_rows = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        dup_rows = total_rows - con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw_data)").fetchone()[0]

        # single-pass profiling query (fast even at huge scale)
        select_parts = []
        for c in all_cols:
            select_parts.append(f'COUNT(*) FILTER (WHERE {qi(c)} IS NULL) AS {qi(c+"__missing")}')
            select_parts.append(f'COUNT(DISTINCT {qi(c)}) AS {qi(c+"__unique")}')
        profile_row = con.execute(f"SELECT {', '.join(select_parts)} FROM raw_data").fetchone()

        missing_map, unique_map = {}, {}
        for i, c in enumerate(all_cols):
            missing_map[c] = profile_row[i * 2]
            unique_map[c] = profile_row[i * 2 + 1]
        missing_total = sum(missing_map.values())

        st.subheader("Raw Data Profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Columns", len(all_cols))
        c3.metric("Missing cells", f"{missing_total:,}")
        c4.metric("Duplicate rows", f"{dup_rows:,}")

        with st.expander("Column-level profile", expanded=True):
            profile_df = pd.DataFrame({
                "column": all_cols,
                "missing": [missing_map[c] for c in all_cols],
                "missing_%": [round(missing_map[c] / total_rows * 100, 2) if total_rows else 0 for c in all_cols],
                "unique": [unique_map[c] for c in all_cols],
            })
            st.dataframe(profile_df, use_container_width=True)

        st.divider()
        st.subheader("Auto-Clean Settings")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            dedupe = st.checkbox("Remove duplicate rows", value=True)
        with cc2:
            num_strategy = st.selectbox("Fill missing numeric", ["Median", "Mean", "Zero", "Leave as-is"])
        with cc3:
            cat_strategy = st.selectbox("Fill missing categorical", ["Mode", "'Unknown'", "Leave as-is"])
        trim_strings = st.checkbox("Trim whitespace on text columns", value=True)

        if st.button("🚀 Run AI Cleaning", type="primary"):
            log = []
            with st.spinner("Cleaning — computed via SQL, scales to very large files..."):
                num_fill = {}
                if num_strategy in ("Median", "Mean") and num_cols:
                    fn = "MEDIAN" if num_strategy == "Median" else "AVG"
                    agg = ", ".join(f'{fn}({qi(c)}) AS {qi(c)}' for c in num_cols)
                    row = con.execute(f"SELECT {agg} FROM raw_data").fetchone()
                    num_fill = dict(zip(num_cols, row))

                cat_fill = {}
                if cat_strategy == "Mode" and cat_cols:
                    for c in cat_cols:
                        r = con.execute(
                            f'SELECT {qi(c)} FROM raw_data WHERE {qi(c)} IS NOT NULL '
                            f'GROUP BY {qi(c)} ORDER BY COUNT(*) DESC LIMIT 1'
                        ).fetchone()
                        if r:
                            cat_fill[c] = r[0]

                select_exprs = []
                for c in num_cols:
                    expr = qi(c)
                    if num_strategy in ("Median", "Mean") and num_fill.get(c) is not None:
                        expr = f'COALESCE({expr}, {num_fill[c]})'
                        log.append(f"Filled missing values in '{c}' using {num_strategy}.")
                    elif num_strategy == "Zero":
                        expr = f'COALESCE({expr}, 0)'
                    select_exprs.append(f'{expr} AS {qi(c)}')

                for c in cat_cols:
                    expr = qi(c)
                    if trim_strings:
                        expr = f'TRIM({expr})'
                    if cat_strategy == "Mode" and c in cat_fill:
                        expr = f'COALESCE({expr}, {ql(cat_fill[c])})'
                        log.append(f"Filled missing values in '{c}' using mode.")
                    elif cat_strategy == "'Unknown'":
                        expr = f"COALESCE({expr}, 'Unknown')"
                    select_exprs.append(f'{expr} AS {qi(c)}')

                for c in date_cols:
                    select_exprs.append(qi(c))

                if trim_strings:
                    log.append("Trimmed whitespace on all text columns.")

                distinct_kw = "DISTINCT" if dedupe else ""
                con.execute(
                    f'CREATE OR REPLACE TABLE clean_data AS '
                    f'SELECT {distinct_kw} {", ".join(select_exprs)} FROM raw_data'
                )
                if dedupe:
                    log.insert(0, f"Removed {dup_rows:,} duplicate rows.")

                st.session_state.clean_table = "clean_data"
                st.session_state.clean_log = log

            st.success("Cleaning complete!")

        if st.session_state.clean_table is not None:
            st.subheader("Cleaning Log")
            for entry in st.session_state.clean_log:
                st.write(f"✔️ {entry}")

            clean_count = con.execute("SELECT COUNT(*) FROM clean_data").fetchone()[0]
            st.subheader(f"Cleaned Data Preview ({clean_count:,} rows total)")
            st.dataframe(con.execute("SELECT * FROM clean_data LIMIT 50").df(), use_container_width=True)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                export_parquet = os.path.join(tempfile.gettempdir(), f"clean_{st.session_state.db_id}.parquet")
                con.execute(f"COPY clean_data TO '{export_parquet}' (FORMAT PARQUET)")
                with open(export_parquet, "rb") as f:
                    st.download_button("⬇️ Download Cleaned Data (Parquet, full — best for big data)",
                                        f, "cleaned_data.parquet")
            with dcol2:
                if clean_count <= 1_000_000:
                    csv_bytes = con.execute("SELECT * FROM clean_data").df().to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Cleaned CSV (full)", csv_bytes, "cleaned_data.csv", "text/csv")
                else:
                    st.caption("Dataset > 1M rows — CSV download disabled to avoid browser memory issues. Use the Parquet download instead.")

# ================= TAB 2: POWER BI-STYLE DASHBOARD =================
with tab2:
    active_table = st.session_state.clean_table or st.session_state.raw_table
    if active_table is None:
        st.info("Upload a file (and ideally clean it in Tab 1) to see the dashboard.")
    else:
        num_cols, cat_cols, date_cols = get_column_types(con, active_table)

        st.sidebar.header("🔎 Slicers")
        where_clauses = []

        for c in num_cols:
            lo, hi = con.execute(f"SELECT MIN({qi(c)}), MAX({qi(c)}) FROM {active_table}").fetchone()
            if lo is not None and hi is not None and lo != hi:
                sel = st.sidebar.slider(f"{c}", float(lo), float(hi), (float(lo), float(hi)))
                if sel != (float(lo), float(hi)):
                    where_clauses.append(f"{qi(c)} BETWEEN {sel[0]} AND {sel[1]}")

        for c in cat_cols:
            uniq_count = con.execute(f"SELECT COUNT(DISTINCT {qi(c)}) FROM {active_table}").fetchone()[0]
            if 1 < uniq_count <= 200:
                opts = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {qi(c)} FROM {active_table} WHERE {qi(c)} IS NOT NULL ORDER BY 1"
                ).fetchall()]
                sel = st.sidebar.multiselect(f"{c}", opts, default=opts)
                if sel and len(sel) < len(opts):
                    in_list = ", ".join(ql(v) for v in sel)
                    where_clauses.append(f"{qi(c)} IN ({in_list})")

        for c in date_cols:
            lo, hi = con.execute(f"SELECT MIN({qi(c)}), MAX({qi(c)}) FROM {active_table}").fetchone()
            if lo is not None and hi is not None:
                sel = st.sidebar.date_input(f"{c} range", (lo, hi))
                if isinstance(sel, tuple) and len(sel) == 2:
                    where_clauses.append(f"{qi(c)} BETWEEN {ql(sel[0])} AND {ql(sel[1])}")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        filtered_count = con.execute(f"SELECT COUNT(*) FROM {active_table} WHERE {where_sql}").fetchone()[0]
        total_count = con.execute(f"SELECT COUNT(*) FROM {active_table}").fetchone()[0]

        st.subheader(f"Showing {filtered_count:,} of {total_count:,} rows")

        primary_metric = detect_col(num_cols, ["revenue", "sales", "amount", "total"]) or (num_cols[0] if num_cols else None)
        secondary_metric = detect_col(num_cols, ["profit", "margin", "net"], exclude=[primary_metric]) or \
                            (num_cols[1] if len(num_cols) > 1 else None)

        k1, k2, k3, k4 = st.columns(4)
        kpi_card_col_map = [(k1, "#6c8bff"), (k2, "#33d69f"), (k3, "#ffb020"), (k4, "#ff6b6b")]
        with kpi_card_col_map[0][0]:
            kpi_card("Total Records", f"{filtered_count:,}", kpi_card_col_map[0][1])
        if primary_metric:
            total_primary = to_float(con.execute(f'SELECT SUM({qi(primary_metric)}) FROM {active_table} WHERE {where_sql}').fetchone()[0])
            with kpi_card_col_map[1][0]:
                kpi_card(f"Total {primary_metric}", format_big_number(total_primary, CURRENCY), kpi_card_col_map[1][1])
        if secondary_metric:
            total_secondary = to_float(con.execute(f'SELECT SUM({qi(secondary_metric)}) FROM {active_table} WHERE {where_sql}').fetchone()[0])
            with kpi_card_col_map[2][0]:
                kpi_card(f"Total {secondary_metric}", format_big_number(total_secondary, CURRENCY), kpi_card_col_map[2][1])
            if primary_metric and total_primary:
                margin = total_secondary / total_primary * 100
                with kpi_card_col_map[3][0]:
                    kpi_card("Overall Margin %", f"{margin:.2f}%", kpi_card_col_map[3][1])

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            if cat_cols and primary_metric:
                bar_cat = st.selectbox("Bar chart — breakdown by", cat_cols, key="barcat")
                bar_df = con.execute(f"""
                    SELECT {qi(bar_cat)} AS category, SUM({qi(primary_metric)}) AS value
                    FROM {active_table} WHERE {where_sql}
                    GROUP BY 1 ORDER BY value DESC LIMIT 10
                """).df()
                bar_df["value"] = bar_df["value"].astype(float)
                st.plotly_chart(px.bar(bar_df, x="category", y="value", template="plotly_dark",
                                        title=f"Top {bar_cat} by {primary_metric}"), use_container_width=True)
        with colB:
            channel_col = detect_col(cat_cols, ["channel", "mode", "type of sale", "online"])
            donut_cat = channel_col or (cat_cols[0] if cat_cols else None)
            if donut_cat and primary_metric:
                donut_df = con.execute(f"""
                    SELECT {qi(donut_cat)} AS category, SUM({qi(primary_metric)}) AS value
                    FROM {active_table} WHERE {where_sql}
                    GROUP BY 1 ORDER BY value DESC LIMIT 8
                """).df()
                donut_df["value"] = donut_df["value"].astype(float)
                st.plotly_chart(px.pie(donut_df, names="category", values="value", hole=0.5,
                                        template="plotly_dark", title=f"{primary_metric} split by {donut_cat}"),
                                 use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            if date_cols and primary_metric:
                trend_date = date_cols[0]
                trend_df = con.execute(f"""
                    SELECT DATE_TRUNC('month', {qi(trend_date)}) AS period, SUM({qi(primary_metric)}) AS value
                    FROM {active_table} WHERE {where_sql}
                    GROUP BY 1 ORDER BY 1
                """).df()
                trend_df["value"] = trend_df["value"].astype(float)
                st.plotly_chart(px.line(trend_df, x="period", y="value", template="plotly_dark",
                                         title=f"{primary_metric} trend over time"), use_container_width=True)
        with colD:
            if len(num_cols) >= 2:
                sample_df = con.execute(f"""
                    SELECT * FROM {active_table} WHERE {where_sql} USING SAMPLE 20000 ROWS (reservoir)
                """).df()
                st.plotly_chart(px.scatter(sample_df, x=num_cols[0], y=num_cols[1], template="plotly_dark",
                                            title=f"{num_cols[0]} vs {num_cols[1]} (sampled)"),
                                 use_container_width=True)

        if len(num_cols) >= 2:
            st.subheader("Correlation Heatmap (sampled)")
            sample_corr = con.execute(f"""
                SELECT * FROM {active_table} WHERE {where_sql} USING SAMPLE 50000 ROWS (reservoir)
            """).df()
            corr = sample_corr[num_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=".2f", template="plotly_dark",
                                       color_continuous_scale="RdBu_r"), use_container_width=True)

# ================= TAB 3: EXECUTIVE DATA STORY =================
with tab3:
    story_table = st.session_state.clean_table or st.session_state.raw_table
    if story_table is None:
        st.info("Upload and clean data first for the best story.")
    else:
        num_cols, cat_cols, date_cols = get_column_types(con, story_table)
        total_records = con.execute(f"SELECT COUNT(*) FROM {story_table}").fetchone()[0]

        revenue_col = detect_col(num_cols, ["revenue", "sales", "total_amount",
