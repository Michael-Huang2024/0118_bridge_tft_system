# app.py
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------- Prophet ----------
from prophet import Prophet

# -------- RAG / FAISS -----------
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------
load_dotenv()

ROOT = Path(__file__).resolve().parents[0]
DEFAULT_INDEX_DIR = ROOT / "artifacts" / "index" / "faiss"
DEFAULT_CSV_PATH  = ROOT / "artifacts" / "extracted" / "bridge_summary_all_years.csv"

st.set_page_config(page_title="Bridge Report RAG + Prophet", page_icon="🛠️", layout="wide")

# -----------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 尝试规范列名（你之前的 extract 脚本基本就是这些字段）
    df.columns = [c.strip().lower() for c in df.columns]
    # year -> datetime（用当年 6 月 1 日做代表）
    if "year" in df.columns:
        df = df.sort_values("year")
        df["ds"] = pd.to_datetime(df["year"].astype(int), format="%Y") + pd.offsets.MonthBegin(6)
    return df

def pick_metric_columns(df: pd.DataFrame) -> dict:
    """
    返回可选字段的映射: {展示名: 列名}
    你可按自己的 CSV 列名调整这里
    """
    mapping = {}
    colset = set(df.columns)

    # 常见别名兜底
    def has_any(*cands):
        for c in cands:
            if c in colset:
                return c
        return None

    m = has_any("deck_condition_rating","deck_condition","deck")
    if m: mapping["Deck Condition Rating"] = m

    m = has_any("superstructure_condition_rating","superstructure_condition","superstructure")
    if m: mapping["Superstructure Condition Rating"] = m

    m = has_any("substructure_condition_rating","substructure_condition","substructure")
    if m: mapping["Substructure Condition Rating"] = m

    m = has_any("average_daily_traffic","adt","average_daily_traffic_(adt)")
    if m: mapping["Average Daily Traffic (ADT)"] = m

    m = has_any("load_rating","operatingrating_us_tons","inventoryrating_us_tons")
    if m: mapping["Load Rating (if present)"] = m

    m = has_any("structural_evaluation_appraisal","structure_evaluation_rating","structureappraisal")
    if m: mapping["Structural Evaluation/Appraisal"] = m

    return mapping

def fit_prophet_forecast(
    df_hist: pd.DataFrame,
    metric_col: str,
    horizon_years: int = 5,
    yearly_seasonality: bool = False,
    changepoint_prior_scale: float = 0.5,
) -> pd.DataFrame:
    """
    用 Prophet 训练并预测，返回带 yhat 的结果（含历史与预测）
    df_hist: 必须包含列 ['ds', metric_col]
    """
    ts = df_hist.dropna(subset=[metric_col, "ds"])[["ds", metric_col]].copy()
    ts = ts.rename(columns={metric_col: "y"})
    # Prophet 模型：桥梁评分/载荷一般是年度缓慢变化，季节性可关掉
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    m.fit(ts)
    # 预测未来 horizon_years 年（按年）
    future = m.make_future_dataframe(periods=horizon_years, freq="Y")
    forecast = m.predict(future)
    # 合并真实值
    out = pd.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                   ts, on="ds", how="left")
    return out

def plot_history_forecast(df_forecast: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    # 历史真实值
    hist = df_forecast.dropna(subset=["y"])
    if len(hist):
        fig.add_trace(go.Scatter(
            x=hist["ds"], y=hist["y"], mode="lines+markers",
            name="History", line=dict(color="#2E86C1", width=3)
        ))
    # 预测 yhat
    fig.add_trace(go.Scatter(
        x=df_forecast["ds"], y=df_forecast["yhat"], mode="lines",
        name="Forecast (Prophet)", line=dict(color="#27AE60", width=3, dash="dash")
    ))
    # 置信区间
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast["ds"], df_forecast["ds"][::-1]]),
        y=pd.concat([df_forecast["yhat_upper"], df_forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(39,174,96,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

# ------------------------ RAG (same logic as before) -----------------------------
def get_embeddings_backend():
    """
    与索引一致的后端：有 OPENAI_API_KEY 就用 OpenAI embeddings，否则用本地 huggingface
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    backend = os.getenv("EMBED_BACKEND", "auto").lower()
    if backend == "openai" or (backend == "auto" and key):
        st.info("🔌 Using OpenAI text-embedding-3-small for FAISS", icon="ℹ️")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        st.info("💻 Using local sentence-transformers/all-MiniLM-L6-v2 for FAISS", icon="ℹ️")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_db(index_dir: str):
    emb = get_embeddings_backend()
    return FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)

def pretty_sources(docs: List):
    out = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        where = f"{md.get('source_file','')} | page={md.get('page','?')} | chunk={md.get('chunk','?')}"
        out.append(f"[{i}] {where}\n{d.page_content[:500]}...")
    return "\n\n".join(out)

def answer_with_llm(query: str, docs: List):
    system = (
        "You are a helpful assistant for bridge maintenance. "
        "Answer strictly using the provided context. "
        "If you are not sure, say you don't know."
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 使用 gpt-4o-mini
    msg = llm([
        SystemMessage(content=system),
        HumanMessage(content=f"Question: {query}\n\nContext:\n" + "\n\n---\n\n".join(d.page_content for d in docs))
    ])
    return msg.content

# -----------------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------------
st.title("🛠️ Bridge Report - RAG + Prophet Forecast")

tab1, tab2 = st.tabs(["🔎 RAG QA", "📈 Trends & Forecast (Prophet)"])

with tab1:
    st.subheader("RAG Q&A (FAISS)")
    index_dir = st.text_input("Index directory", value=str(DEFAULT_INDEX_DIR))
    q = st.text_input("Your question", "Traffic flow on the bridge")
    if st.button("Ask", type="primary"):
        try:
            db = load_db(index_dir)
        except Exception as e:
            st.error(f"Failed to load FAISS: {e}")
        else:
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(q)
            if not docs:
                st.warning("No relevant chunks found.")
            else:
                if os.getenv("OPENAI_API_KEY", "").strip():
                    ans = answer_with_llm(q, docs)
                    st.success(ans)
                    st.markdown("**Sources**")
                    st.code(pretty_sources(docs))
                else:
                    st.info("No OPENAI_API_KEY. Showing retrieved chunks:", icon="ℹ️")
                    st.code(pretty_sources(docs))

with tab2:
    st.subheader("Historical Trends & Prophet Forecast")

    csv_path = st.text_input("All-years CSV path", value=str(DEFAULT_CSV_PATH))
    left, right = st.columns([1,1])
    horizon = left.number_input("Forecast horizon (years)", min_value=1, max_value=20, value=5, step=1)
    cps     = right.slider("Changepoint prior scale (smoothness, higher = more flexible)",
                           min_value=0.05, max_value=1.0, value=0.5, step=0.05)

    if not Path(csv_path).exists():
        st.error("CSV not found. Please run extract_fields_all.py first.")
        st.stop()

    df = load_csv(csv_path)
    metric_map = pick_metric_columns(df)
    if not metric_map:
        st.error("No known metrics found in CSV. Please check column names.")
        st.stop()

    metric_label = st.selectbox("Select a metric", list(metric_map.keys()), index=0)
    metric_col = metric_map[metric_label]

    st.markdown(f"**Using column:** `{metric_col}`")
    hist_cols = ["year", "ds", metric_col]
    st.dataframe(df[hist_cols].rename(columns={metric_col: metric_label}), use_container_width=True, height=280)

    # 训练 Prophet & 预测
    try:
        forecast = fit_prophet_forecast(
            df_hist=df,
            metric_col=metric_col,
            horizon_years=horizon,
            yearly_seasonality=False,            # 年度评分/ADT 通常不需要季节性
            changepoint_prior_scale=cps,
        )
    except Exception as e:
        st.error(f"Prophet failed: {e}")
        st.stop()

    # 画图
    fig = plot_history_forecast(forecast, title=f"{metric_label} - History & Prophet Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # 预测表格（只显示未来）
    future_only = forecast[forecast["y"].isna()][["ds","yhat","yhat_lower","yhat_upper"]].copy()
    future_only["year"] = future_only["ds"].dt.year
    st.markdown("**Forecast table (future years):**")
    st.dataframe(
        future_only[["year","yhat","yhat_lower","yhat_upper"]].round(3),
        use_container_width=True, height=260
    )

    # 下载结果
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name="prophet_forecast.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download filtered (history only) CSV",
            data=forecast.dropna(subset=["y"])[["ds","y"]].to_csv(index=False).encode("utf-8"),
            file_name="history_only.csv",
            mime="text/csv"
        )
