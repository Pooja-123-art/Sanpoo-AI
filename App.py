import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="AI Data Analytics Studio", layout="wide", page_icon="📊")

# ---------------- Glassmorphism styling ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1026 0%, #131a3a 50%, #1a2456 100%); }
h1, h2, h3, p, label, span, div { color: #e8ecff !important; }
[data-testid="stMetric"], .glass {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 14px;
    backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] { background: rgba(10,14,35,0.85); }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Data Analytics Studio")
st.caption("Upload a file → Auto-clean → Explore with an interactive dashboard → Get a plain-English data story")

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.session_state.df_raw = df
    except Exception as e:
        st.error(f"Couldn't read file: {e}")

tab1, tab2, tab3 = st.tabs(["🧹 AI Data Cleaner", "📈 Interactive Dashboard", "📝 Auto Data Story"])

# ================= TAB 1: AI DATA CLEANER =================
with tab1:
    if st.session_state.df_raw is None:
        st.info("Upload a file above to begin.")
    else:
        df = st.session_state.df_raw.copy()
        st.subheader("Raw Data Profile")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        c4.metric("Duplicate rows", int(df.duplicated().sum()))

        with st.expander("Column-level profile", expanded=True):
            profile = pd.DataFrame({
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "missing_%": (df.isna().mean() * 100).round(1),
                "unique": df.nunique()
            })
            st.dataframe(profile, use_container_width=True)

        st.divider()
        st.subheader("Auto-Clean Settings")

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            dedupe = st.checkbox("Remove duplicate rows", value=True)
        with cc2:
            num_strategy = st.selectbox("Fill missing numeric", ["Median", "Mean", "Zero", "Leave as-is"])
        with cc3:
            cat_strategy = st.selectbox("Fill missing categorical", ["Mode", "'Unknown'", "Leave as-is"])

        trim_strings = st.checkbox("Trim whitespace & standardize text case", value=True)
        fix_dtypes = st.checkbox("Auto-detect & fix date / numeric columns stored as text", value=True)

        if st.button("🚀 Run AI Cleaning", type="primary"):
            log = []
            cleaned = df.copy()

            before = cleaned.shape[0]
            if dedupe:
                cleaned = cleaned.drop_duplicates()
                log.append(f"Removed {before - cleaned.shape[0]} duplicate rows.")

            if fix_dtypes:
                for col in cleaned.columns:
                    if cleaned[col].dtype == object:
                        converted = pd.to_numeric(cleaned[col], errors="coerce")
                        if converted.notna().mean() > 0.8:
                            cleaned[col] = converted
                            log.append(f"Converted '{col}' to numeric.")
                            continue
                        try:
                            dt = pd.to_datetime(cleaned[col], errors="coerce")
                            if dt.notna().mean() > 0.8:
                                cleaned[col] = dt
                                log.append(f"Converted '{col}' to datetime.")
                        except Exception:
                            pass

            if trim_strings:
                for col in cleaned.select_dtypes(include="object").columns:
                    cleaned[col] = cleaned[col].astype(str).str.strip()
                log.append("Trimmed whitespace on text columns.")

            num_cols = cleaned.select_dtypes(include=np.number).columns
            for col in num_cols:
                miss = cleaned[col].isna().sum()
                if miss == 0 or num_strategy == "Leave as-is":
                    continue
                if num_strategy == "Median":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                elif num_strategy == "Mean":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                elif num_strategy == "Zero":
                    cleaned[col] = cleaned[col].fillna(0)
                log.append(f"Filled {miss} missing values in '{col}' using {num_strategy}.")

            cat_cols = cleaned.select_dtypes(include="object").columns
            for col in cat_cols:
                miss = cleaned[col].isna().sum()
                if miss == 0 or cat_strategy == "Leave as-is":
                    continue
                if cat_strategy == "Mode" and not cleaned[col].mode().empty:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
                elif cat_strategy == "'Unknown'":
                    cleaned[col] = cleaned[col].fillna("Unknown")
                log.append(f"Filled {miss} missing values in '{col}' using {cat_strategy}.")

            st.session_state.df_clean = cleaned
            st.session_state.clean_log = log
            st.success("Cleaning complete! Check the log and preview below, then move to the Dashboard tab.")

        if st.session_state.df_clean is not None:
            st.subheader("Cleaning Log")
            for entry in st.session_state.clean_log:
                st.write(f"✔️ {entry}")

            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head(50), use_container_width=True)

            csv_buf = st.session_state.df_clean.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Cleaned CSV", csv_buf, "cleaned_data.csv", "text/csv")

# ================= TAB 2: INTERACTIVE DASHBOARD =================
with tab2:
    active_df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

    if active_df is None:
        st.info("Upload a file (and ideally clean it in Tab 1) to see the dashboard.")
    else:
        df = active_df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        st.sidebar.header("🔎 Filters")
        filtered = df.copy()

        for col in num_cols:
            if df[col].nunique() > 1:
                min_v, max_v = float(df[col].min()), float(df[col].max())
                sel = st.sidebar.slider(f"{col}", min_v, max_v, (min_v, max_v))
                filtered = filtered[(filtered[col] >= sel[0]) & (filtered[col] <= sel[1])]

        for col in cat_cols:
            if 1 < df[col].nunique() <= 50:
                opts = sorted(df[col].dropna().unique().tolist())
                sel = st.sidebar.multiselect(f"{col}", opts, default=opts)
                if sel:
                    filtered = filtered[filtered[col].isin(sel)]

        for col in date_cols:
            min_d, max_d = df[col].min(), df[col].max()
            sel = st.sidebar.date_input(f"{col} range", (min_d, max_d))
            if isinstance(sel, tuple) and len(sel) == 2:
                filtered = filtered[(filtered[col] >= pd.to_datetime(sel[0])) & (filtered[col] <= pd.to_datetime(sel[1]))]

        st.subheader(f"Showing {filtered.shape[0]} of {df.shape[0]} rows")

        m1, m2, m3, m4 = st.columns(4)
        if num_cols:
            m1.metric(f"Avg {num_cols[0]}", round(filtered[num_cols[0]].mean(), 2))
            m2.metric(f"Max {num_cols[0]}", round(filtered[num_cols[0]].max(), 2))
        m3.metric("Rows", filtered.shape[0])
        m4.metric("Columns", filtered.shape[1])

        colA, colB = st.columns(2)
        with colA:
            if num_cols:
                col_h = st.selectbox("Histogram column", num_cols, key="hist")
                st.plotly_chart(px.histogram(filtered, x=col_h, template="plotly_dark"), use_container_width=True)
        with colB:
            if cat_cols:
                col_c = st.selectbox("Category breakdown", cat_cols, key="cat")
                counts = filtered[col_c].value_counts().reset_index()
                counts.columns = [col_c, "count"]
                st.plotly_chart(px.bar(counts, x=col_c, y="count", template="plotly_dark"), use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            if len(num_cols) >= 2:
                x_ax = st.selectbox("Scatter X", num_cols, index=0, key="sx")
                y_ax = st.selectbox("Scatter Y", num_cols, index=1, key="sy")
                st.plotly_chart(px.scatter(filtered, x=x_ax, y=y_ax, template="plotly_dark"), use_container_width=True)
        with colD:
            if len(num_cols) >= 2:
                corr = filtered[num_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=".2f", template="plotly_dark", color_continuous_scale="RdBu_r"), use_container_width=True)

        if date_cols and num_cols:
            st.subheader("Time Trend")
            d_col = st.selectbox("Date column", date_cols, key="tdate")
            v_col = st.selectbox("Value column", num_cols, key="tval")
            trend = filtered.groupby(d_col)[v_col].mean().reset_index()
            st.plotly_chart(px.line(trend, x=d_col, y=v_col, template="plotly_dark"), use_container_width=True)

        st.session_state.df_filtered = filtered

# ================= TAB 3: AUTO DATA STORY =================
with tab3:
    story_df = st.session_state.get("df_filtered")
    if story_df is None:
        story_df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

    if story_df is None:
        st.info("Upload and explore data first.")
    else:
        df = story_df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("📝 Auto-Generated Data Story")

        insights = []
        insights.append(f"This dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns** after your filters.")

        for col in num_cols[:5]:
            mean, std, mn, mx = df[col].mean(), df[col].std(), df[col].min(), df[col].max()
            skew = df[col].skew()
            trend_word = "fairly balanced" if abs(skew) < 0.5 else ("skewed toward higher values" if skew > 0 else "skewed toward lower values")
            insights.append(f"**{col}** averages **{mean:.2f}** (range {mn:.2f}–{mx:.2f}), and the distribution looks {trend_word}.")

        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            top_pair = corr.stack().idxmax()
            top_val = corr.loc[top_pair]
            if top_val > 0.5:
                insights.append(f"**{top_pair[0]}** and **{top_pair[1]}** show a notably strong relationship (correlation {top_val:.2f}).")

        for col in cat_cols[:3]:
            top_cat = df[col].value_counts().idxmax()
            top_pct = df[col].value_counts(normalize=True).max() * 100
            insights.append(f"In **{col}**, '{top_cat}' is the most common value, making up {top_pct:.1f}% of records.")

        for line in insights:
            st.markdown(f"- {line}")

        st.divider()
        st.subheader("✨ Polish with local Ollama (optional)")
        use_ollama = st.toggle("Send insights to local Ollama for polished prose")

        if use_ollama:
            model_name = st.text_input("Ollama model", value="llama3")
            if st.button("Generate polished story"):
                try:
                    import requests
                    prompt = "Turn these bullet-point data insights into a short, engaging analyst-style narrative:\n" + "\n".join(insights)
                    resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model_name, "prompt": prompt, "stream": False},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.write(resp.json().get("response", "No response text."))
                    else:
                        st.warning("Ollama call failed, showing plain insights above instead.")
                except Exception:
                    st.warning("Couldn't reach local Ollama — showing plain-English insights above instead.")