"""
Phase 9: ACMGS Dashboard

Modern real-time control center for the Autonomous Carbon-Aware
Manufacturing Genome System.

Run:
    cd C:\\Users\\HP\\ACMGS
    streamlit run src/dashboard/app.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests
from collections import deque
import threading

# ─── Path setup (works regardless of cwd) ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH, CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD
from src.carbon_scheduler import classify_carbon_zone, get_recommendation
from src.optimization.recommender import DynamicRecommender
from src.knowledge_graph import SemanticGraph, BatchForensics, EvidenceRecommender

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="ACMGS | Control Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ACMGS v9.0 — Autonomous Carbon-Aware Manufacturing Genome System"},
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ─────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 100%) !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* ── Header banner ───────────────────────────────────────── */
.acmgs-header {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.1) 0%,
        rgba(0,255,136,0.06) 50%,
        rgba(0,80,200,0.08) 100%
    );
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 16px;
}
.acmgs-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff 0%, #00ff88 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}
.acmgs-header p {
    color: rgba(255,255,255,0.5);
    font-size: 0.875rem;
    margin: 6px 0 10px 0;
}
.hbadge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 6px;
}
.hbadge-green  { background: rgba(0,255,136,0.12); border-color: rgba(0,255,136,0.3); color: #00ff88; }
.hbadge-yellow { background: rgba(255,214,0,0.1);  border-color: rgba(255,214,0,0.3);  color: #ffd600; }
.hbadge-red    { background: rgba(255,75,75,0.12); border-color: rgba(255,75,75,0.3);  color: #ff6b6b; }

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,212,255,0.35) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    width: 100%;
    box-sizing: border-box;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.83rem;
    padding: 7px 16px;
    transition: all 0.2s;
    flex: 1 1 0;
    justify-content: center;
    text-align: center;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.16) !important;
    color: #00d4ff !important;
    font-weight: 600 !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1426 0%, #090d1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Section label ───────────────────────────────────────── */
.slabel {
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 5px;
    margin: 18px 0 10px 0;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.22); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.4); }

/* ── Inputs / Sliders ────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(0,212,255,0.18) !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Expander ────────────────────────────────────────────── */
details > summary {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.7) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
GENOME_LABELS = (
    ["Temp", "Pressure", "Speed", "FeedRate", "Humidity"]
    + ["Density", "Hardness", "Grade"]
    + [f"EdNA{i:02d}" for i in range(16)]
    + ["CarbonInt"]
)

ZONE_COLORS = {"LOW": "#00ff88", "MEDIUM": "#ffd600", "HIGH": "#ff4b4b"}
ZONE_BG     = {"LOW": "rgba(0,255,136,0.08)", "MEDIUM": "rgba(255,214,0,0.08)",  "HIGH": "rgba(255,75,75,0.08)"}
ZONE_BORDER = {"LOW": "rgba(0,255,136,0.3)",  "MEDIUM": "rgba(255,214,0,0.3)",   "HIGH": "rgba(255,75,75,0.3)"}
ZONE_EMOJI  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
ZONE_TITLE  = {"LOW": "CLEAN GRID", "MEDIUM": "MIXED GRID", "HIGH": "DIRTY GRID"}
ZONE_DESC   = {
    "LOW":    "Renewable energy dominant — Maximize production output at full capacity.",
    "MEDIUM": "Balanced energy mix — Optimize for efficiency and sustainability.",
    "HIGH":   "Heavy fossil fuel load — Activate conservation mode, minimize energy.",
}

TABLE_ICONS = {
    "batches":           "📦",   # product batches
    "energy_embeddings": "🔋",   # energy vectors / stored embeddings
    "genome_vectors":    "🧬",   # DNA genome
    "predictions":       "🔮",   # forecasting / ML predictions
    "pareto_solutions":  "⚖️",   # multi-objective trade-off balance
    "carbon_schedules":  "🌍",   # carbon / climate scheduling
    "pipeline_runs":     "🚀",   # pipeline execution
}

_CYAN    = "#00d4ff"
_GREEN   = "#00ff88"
_YELLOW  = "#ffd600"
_RED     = "#ff4b4b"
_PURPLE  = "#a855f7"
_ORANGE  = "#f97316"

# Demo carbon intensity profile (gCO2/kWh) - 24 hourly values
# Replace with real grid data from carbon scheduler in production
CARBON_24H = [120, 100, 85, 75, 70, 65, 60, 55, 50, 55, 65, 80,
              100, 130, 160, 200, 260, 320, 420, 500, 460, 380, 280, 180]


# ─── Data loading (all cached) ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_batches() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM batches ORDER BY batch_id", conn)
    conn.close()
    df["zone"] = df["carbon_intensity"].apply(classify_carbon_zone)
    return df


@st.cache_data(ttl=300)
def _load_pareto() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pareto_solutions ORDER BY pred_yield DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_predictions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_schedules() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM carbon_schedules ORDER BY id", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_pipeline_runs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def _load_db_summary() -> dict:
    conn = sqlite3.connect(DB_PATH)
    tables = ["batches", "energy_embeddings", "genome_vectors", "predictions",
              "pareto_solutions", "carbon_schedules", "pipeline_runs"]
    summary = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    summary["db_size_mb"] = round(os.path.getsize(DB_PATH) / 1_048_576, 2)
    conn.close()
    return summary


@st.cache_data(ttl=600)
def _load_genomes(n: int = 80) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT batch_id, genome FROM genome_vectors LIMIT {n}", conn)
    conn.close()
    return df


# ─── Chart helpers ────────────────────────────────────────────────────────────
def dark_layout(fig: go.Figure, height: int = None, margin: dict = None) -> go.Figure:
    """Apply the dashboard dark theme to any Plotly figure."""
    kw: dict = {}
    if height:
        kw["height"] = height
    if margin:
        kw["margin"] = margin
    else:
        kw["margin"] = dict(l=16, r=16, t=44, b=16)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="rgba(255,255,255,0.75)", family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            linecolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            linecolor="rgba(255,255,255,0.08)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
        ),
        title=dict(font=dict(size=13, color="rgba(255,255,255,0.7)")),
        **kw,
    )
    return fig


def make_gauge(val: float, zone: str) -> go.Figure:
    """Build a Plotly radial gauge for carbon intensity."""
    col = ZONE_COLORS[zone]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={
            "text": (
                "Grid Carbon Intensity<br>"
                "<span style='font-size:0.8em;color:rgba(255,255,255,0.45)'>gCO₂ / kWh</span>"
            ),
            "font": {"size": 14, "color": "rgba(255,255,255,0.6)"},
        },
        number={
            "font": {"size": 54, "color": col, "family": "JetBrains Mono, monospace"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 600],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 9},
                "nticks": 7,
            },
            "bar": {"color": col, "thickness": 0.2},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0,   150], "color": "rgba(0,255,136,0.12)"},
                {"range": [150, 400], "color": "rgba(255,214,0,0.10)"},
                {"range": [400, 600], "color": "rgba(255,75,75,0.13)"},
            ],
            "threshold": {
                "line": {"color": col, "width": 3},
                "thickness": 0.82,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "rgba(255,255,255,0.65)", "family": "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ─── Load all data once ───────────────────────────────────────────────────────
df_batches   = _load_batches()
df_pareto    = _load_pareto()
df_preds     = _load_predictions()
df_schedules = _load_schedules()
df_runs      = _load_pipeline_runs()
db_summary   = _load_db_summary()
df_genomes   = _load_genomes(80)

# Merge batches + predictions for combined analytics
df_merged = df_batches.merge(df_preds, on="batch_id", how="left")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:18px 0 14px 0;
            border-bottom:1px solid rgba(255,255,255,0.08);
            margin-bottom:18px;">
  <svg width="72" height="72" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="lg1" x1="0" y1="0" x2="68" y2="68" gradientUnits="userSpaceOnUse">
        <stop stop-color="#00d4ff"/><stop offset="1" stop-color="#00ff88"/>
      </linearGradient>
      <filter id="fw" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <!-- Hexagon = Carbon (C) -->
    <polygon points="34,3 62,19 62,49 34,65 6,49 6,19"
             stroke="url(#lg1)" stroke-width="1.5" fill="rgba(0,212,255,0.04)"/>
    <!-- Vertex dots = Manufacturing (M) gear nodes -->
    <circle cx="34" cy="3"  r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <circle cx="62" cy="19" r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <circle cx="62" cy="49" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="34" cy="65" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="6"  cy="49" r="2.5" fill="#00ff88" filter="url(#fw)"/>
    <circle cx="6"  cy="19" r="2.5" fill="#00d4ff" filter="url(#fw)"/>
    <!-- DNA strands = Genome (G) -->
    <path d="M24,13 C22,21 26,25 24,34 C22,43 26,47 24,55"
          stroke="#00d4ff" stroke-width="2" fill="none"/>
    <path d="M44,13 C46,21 42,25 44,34 C46,43 42,47 44,55"
          stroke="#00ff88" stroke-width="2" fill="none"/>
    <!-- DNA rungs -->
    <line x1="24" y1="19" x2="44" y2="19" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="27" x2="44" y2="27" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="34" x2="44" y2="34" stroke="rgba(255,255,255,0.30)" stroke-width="1.5"/>
    <line x1="24" y1="41" x2="44" y2="41" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <line x1="24" y1="49" x2="44" y2="49" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
    <!-- Orbital ellipse = System (S) integration layer -->
    <ellipse cx="34" cy="34" rx="12" ry="6"
             stroke="rgba(0,212,255,0.32)" stroke-width="1" fill="none"
             transform="rotate(-35 34 34)"/>
    <!-- Central node = Autonomous (A) AI core -->
    <circle cx="34" cy="34" r="5.5" fill="url(#lg1)" filter="url(#fw)"/>
    <circle cx="34" cy="34" r="2.8" fill="#050e1f"/>
  </svg>
  <div style="font-size:1.3rem;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#00ff88);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:0.08em;margin-top:2px;">ACMGS</div>
  <div style="font-size:0.63rem;color:rgba(255,255,255,0.28);
              letter-spacing:0.15em;margin-top:3px;">CONTROL CENTER  v9.0</div>
</div>
""", unsafe_allow_html=True)

    # ── Carbon intensity slider ────────────────────────────────────────────────
    st.markdown('<div class="slabel">Live Carbon Monitor</div>', unsafe_allow_html=True)
    carbon_val = st.slider(
        "Grid Carbon Intensity (gCO₂/kWh)",
        min_value=0, max_value=600, value=220, step=5,
        label_visibility="collapsed",
    )
    zone = classify_carbon_zone(float(carbon_val))

    # Zone indicator card
    st.markdown(
        f'<div style="background:{ZONE_BG[zone]};border:1px solid {ZONE_BORDER[zone]};'
        'border-radius:12px;padding:14px;text-align:center;margin:8px 0 14px 0;">'
        f'<div style="font-size:1.9rem;line-height:1.2;">{ZONE_EMOJI[zone]}</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{ZONE_COLORS[zone]};'
        'letter-spacing:0.05em;margin-top:2px;">'
        f'{zone} CARBON</div>'
        '<div style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-top:4px;">'
        f'{carbon_val} gCO₂/kWh</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # AI recommendation mini-panel
    try:
        rec = get_recommendation(float(carbon_val))
        sched = rec.get("recommended_schedule", {})
        rec_yield  = sched.get("pred_yield", 0.0)
        rec_energy = sched.get("pred_energy", 0.0)
        rec_carbon = sched.get("pred_carbon", 0.0)

        st.markdown(
            '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
            'border-radius:8px;padding:13px;margin-bottom:14px;">'
            '<div style="font-size:0.69rem;color:rgba(255,255,255,0.28);text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:7px;">AI Recommendation</div>'
            f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.8);line-height:1.5;">'
            f'{ZONE_DESC[zone]}</div>'
            '<div style="display:flex;gap:7px;margin-top:11px;">'
            '<div style="flex:1;background:rgba(0,212,255,0.08);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#00d4ff;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_yield:.4f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">Yield</div></div>'
            '<div style="flex:1;background:rgba(0,255,136,0.06);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#00ff88;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_energy:.0f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">kWh</div></div>'
            '<div style="flex:1;background:rgba(255,75,75,0.06);border-radius:6px;padding:7px 4px;text-align:center;">'
            f'<div style="font-size:0.95rem;font-weight:600;color:#ff6b6b;font-family:\'JetBrains Mono\',monospace;">'
            f'{rec_carbon:.0f}</div>'
            '<div style="font-size:0.63rem;color:rgba(255,255,255,0.38);text-transform:uppercase;">kg CO₂</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);'
            'padding:10px;background:rgba(255,255,255,0.02);border-radius:6px;">'
            f'{ZONE_DESC[zone]}</div>',
            unsafe_allow_html=True,
        )

    # ── DB row counts ──────────────────────────────────────────────────────────
    st.markdown('<div class="slabel">Database Status</div>', unsafe_allow_html=True)
    for key, count in [(k, v) for k, v in db_summary.items() if k != "db_size_mb"]:
        icon = TABLE_ICONS.get(key, "📄")
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:5px 2px;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:0.77rem;color:rgba(255,255,255,0.44);">'
            f'{icon} {key.replace("_"," ").title()}</span>'
            f'<span style="font-size:0.77rem;font-weight:600;color:#00d4ff;'
            f'font-family:\'JetBrains Mono\',monospace;">{count:,}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="margin-top:9px;padding:8px;background:rgba(255,255,255,0.02);'
        'border-radius:6px;display:flex;justify-content:space-between;">'
        '<span style="font-size:0.71rem;color:rgba(255,255,255,0.3);">DB Size</span>'
        f'<span style="font-size:0.71rem;font-weight:600;color:rgba(255,255,255,0.5);">'
        f'{db_summary["db_size_mb"]} MB</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.67rem;color:rgba(255,255,255,0.2);'
        'text-align:center;margin:12px 0 8px 0;">'
        f'{datetime.now().strftime("%b %d, %Y  ·  %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄  Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main header ─────────────────────────────────────────────────────────────
badge_zone_cls = {"LOW": "hbadge-green", "MEDIUM": "hbadge-yellow", "HIGH": "hbadge-red"}[zone]
st.markdown(
    '<div class="acmgs-header"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
    '<div>'
    '<h1>🧬 ACMGS Control Center</h1>'
    '<p>Autonomous Carbon-Aware Manufacturing Genome System — Real-Time Intelligence Dashboard</p>'
    '<div>'
    '<span class="hbadge">Phase 9</span>'
    '<span class="hbadge hbadge-green">● Live</span>'
    f'<span class="hbadge">{len(df_batches):,} Batches</span>'
    f'<span class="hbadge">⚖️ Pareto Solutions</span>'
    f'<span class="hbadge {badge_zone_cls}">{ZONE_EMOJI[zone]} {zone} Zone</span>'
    '</div></div>'
    f'<div style="text-align:right;font-size:0.73rem;color:rgba(255,255,255,0.3);padding-top:4px;">'
    f'<div style="font-size:1.4rem;">{ZONE_EMOJI[zone]}</div>'
    f'Updated {datetime.now().strftime("%H:%M:%S")}'
    '</div></div></div>',
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎛️  Command Center",
    "📈  Production Analytics",
    "⚖️  Pareto Intelligence",
    "🧬  Genome Explorer",
    "🩺  System Health",
    "🤖  Digital Twin",
    "📡  ESP32 Real-Time",
    "⚡  Optimization Insights",
    "🔬 Track A Forensics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Batches", f"{len(df_batches):,}", f"2,000 loaded")
    with k2:
        avg_yield = df_batches["yield"].mean()
        st.metric("Avg Batch Yield", f"{avg_yield:.4f}", f"σ = {df_batches['yield'].std():.4f}")
    with k3:
        avg_carbon = df_batches["carbon_intensity"].mean()
        st.metric("Avg Carbon Intensity", f"{avg_carbon:.1f}", "gCO₂/kWh", delta_color="inverse")
    with k4:
        best_yield = df_pareto["pred_yield"].max() if len(df_pareto) > 0 else 0.0
        st.metric("Best Pareto Yield", f"{best_yield:.4f}", f"{len(df_pareto)} solutions")

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + schedule recommendation
    left, right = st.columns([4, 6])

    with left:
        fig_gauge = make_gauge(carbon_val, zone)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        # Zone description strip below gauge
        st.markdown(
            f'<div style="background:{ZONE_BG[zone]};border:1px solid {ZONE_BORDER[zone]};'
            'border-radius:10px;padding:12px 16px;margin-top:-10px;text-align:center;">'
            f'<span style="font-size:1.1rem;font-weight:700;color:{ZONE_COLORS[zone]};">'
            f'{ZONE_EMOJI[zone]}  {ZONE_TITLE[zone]}</span><br>'
            f'<span style="font-size:0.8rem;color:rgba(255,255,255,0.55);margin-top:4px;display:block;">'
            f'{ZONE_DESC[zone]}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        try:
            rec = get_recommendation(float(carbon_val))
            sched = rec["recommended_schedule"]

            st.markdown(
                '<div class="slabel" style="margin-top:4px;">Optimal Manufacturing Schedule</div>',
                unsafe_allow_html=True,
            )

            # Process parameters (4 per row)
            process_params = [
                ("Temperature", f"{sched.get('temperature', 0):.1f}", "°C",   _CYAN),
                ("Pressure",    f"{sched.get('pressure', 0):.2f}",    "bar",  _CYAN),
                ("Speed",       f"{sched.get('speed', 0):.0f}",       "rpm",  _CYAN),
                ("Feed Rate",   f"{sched.get('feed_rate', 0):.2f}",   "kg/h", _CYAN),
            ]
            material_params = [
                ("Density",     f"{sched.get('material_density', 0):.3f}",   "g/cm³",  _GREEN),
                ("Hardness",    f"{sched.get('material_hardness', 0):.1f}",  "HV",     _GREEN),
                ("Mat. Grade",  f"{int(sched.get('material_grade', 0))}",    "",       _GREEN),
                ("Humidity",    f"{sched.get('humidity', 0):.1f}",           "%",      _GREEN),
            ]
            outcome_params = [
                ("Pred Yield",   f"{sched.get('pred_yield', 0):.4f}",  "",      _YELLOW),
                ("Pred Quality", f"{sched.get('pred_quality', 0):.4f}", "",     _YELLOW),
                ("Pred Energy",  f"{sched.get('pred_energy', 0):.1f}",  "kWh",  _ORANGE),
                ("Pred Carbon",  f"{sched.get('pred_carbon', 0):.1f}",  "kg",   _RED),
            ]

            def _param_grid(params):
                cols = st.columns(4)
                for i, (label, val_str, unit, col_hex) in enumerate(params):
                    with cols[i]:
                        st.markdown(
                            '<div style="background:rgba(255,255,255,0.04);'
                            'border:1px solid rgba(255,255,255,0.08);border-radius:9px;'
                            'padding:11px 8px;text-align:center;margin-bottom:8px;">'
                            f'<div style="font-size:1.05rem;font-weight:600;color:{col_hex};'
                            "font-family:'JetBrains Mono',monospace;\">"
                            f'{val_str}'
                            f'<span style="font-size:0.62rem;color:rgba(255,255,255,0.35);'
                            f'margin-left:2px;">{unit}</span></div>'
                            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.38);'
                            'text-transform:uppercase;letter-spacing:0.05em;margin-top:3px;">'
                            f'{label}</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

            _param_grid(process_params)
            _param_grid(material_params)
            _param_grid(outcome_params)

        except Exception as e:
            st.warning(f"Recommendation unavailable: {e}")

    # Schedule history
    if len(df_schedules) > 0:
        st.markdown('<div class="slabel" style="margin-top:24px;">Historical Schedule Decisions</div>',
                    unsafe_allow_html=True)
        disp = df_schedules[[
            "carbon_intensity", "zone",
            "schedule_pred_yield", "schedule_pred_quality",
            "schedule_pred_energy", "schedule_pred_carbon",
        ]].copy()
        disp.columns = ["Carbon Int.", "Zone", "Pred Yield", "Pred Quality", "Pred Energy (kWh)", "Pred Carbon (kg)"]
        disp = disp.round(4)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCTION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Yield",   f"{df_batches['yield'].mean():.4f}",
                  f"σ={df_batches['yield'].std():.4f}")
    with m2:
        st.metric("Avg Quality", f"{df_batches['quality'].mean():.4f}",
                  f"σ={df_batches['quality'].std():.4f}")
    with m3:
        st.metric("Avg Energy",  f"{df_batches['energy_consumption'].mean():.0f} kWh",
                  f"Max {df_batches['energy_consumption'].max():.0f}")
    with m4:
        st.metric("Avg Carbon",  f"{df_batches['carbon_intensity'].mean():.1f}",
                  "gCO₂/kWh", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: yield histogram + zone pie
    c1, c2 = st.columns([6, 4])

    with c1:
        fig_hist = px.histogram(
            df_batches, x="yield", nbins=60,
            title="Yield Distribution — 2,000 Batches",
            labels={"yield": "Batch Yield", "count": "Frequency"},
            color_discrete_sequence=[_CYAN],
        )
        fig_hist.update_traces(
            marker_line_color="rgba(0,212,255,0.5)",
            marker_line_width=0.5,
        )
        dark_layout(fig_hist, height=320)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with c2:
        zone_counts = df_batches["zone"].value_counts().reset_index()
        zone_counts.columns = ["zone", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=zone_counts["zone"],
            values=zone_counts["count"],
            hole=0.5,
            marker=dict(colors=[ZONE_COLORS.get(z, _CYAN) for z in zone_counts["zone"]],
                        line=dict(color="rgba(0,0,0,0.4)", width=2)),
            textfont=dict(color="rgba(255,255,255,0.8)", size=11),
            hovertemplate="<b>%{label}</b><br>Batches: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=dict(text="Carbon Zone Distribution", font=dict(size=13, color="rgba(255,255,255,0.7)")),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.12)", borderwidth=1),
            height=320,
            margin=dict(l=16, r=16, t=44, b=16),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Row 3: full-width scatter of all 2000 batches
    fig_scatter = px.scatter(
        df_batches, x="carbon_intensity", y="energy_consumption",
        color="zone",
        color_discrete_map=ZONE_COLORS,
        title="Carbon Intensity vs Energy Consumption — Full Production Fleet (2,000 Batches)",
        labels={
            "carbon_intensity": "Carbon Intensity (gCO₂/kWh)",
            "energy_consumption": "Energy Consumption (kWh)",
            "zone": "Carbon Zone",
        },
        hover_data=["batch_id", "yield", "quality"],
        opacity=0.6,
        category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
    )
    fig_scatter.add_vline(
        x=CARBON_LOW_THRESHOLD, line_dash="dash",
        line_color="rgba(0,255,136,0.45)",
        annotation_text=f"LOW (<{CARBON_LOW_THRESHOLD})",
        annotation_position="top left",
        annotation_font=dict(color="rgba(0,255,136,0.7)", size=10),
    )
    fig_scatter.add_vline(
        x=CARBON_HIGH_THRESHOLD, line_dash="dash",
        line_color="rgba(255,75,75,0.45)",
        annotation_text=f"HIGH (>{CARBON_HIGH_THRESHOLD})",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,75,75,0.7)", size=10),
    )
    dark_layout(fig_scatter, height=380)
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # Row 4: two correlation scatters
    c3, c4 = st.columns(2)

    with c3:
        fig_tv = px.scatter(
            df_batches, x="temperature", y="yield",
            color="zone", color_discrete_map=ZONE_COLORS,
            title="Temperature vs Yield",
            labels={"temperature": "Temperature (°C)", "yield": "Yield"},
            opacity=0.55,
            hover_data=["batch_id"],
            category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
        )
        dark_layout(fig_tv, height=320)
        st.plotly_chart(fig_tv, use_container_width=True, config={"displayModeBar": False})

    with c4:
        fig_se = px.scatter(
            df_batches, x="speed", y="energy_consumption",
            color="zone", color_discrete_map=ZONE_COLORS,
            title="Production Speed vs Energy Consumption",
            labels={"speed": "Speed (rpm)", "energy_consumption": "Energy (kWh)"},
            opacity=0.55,
            hover_data=["batch_id"],
            category_orders={"zone": ["LOW", "MEDIUM", "HIGH"]},
        )
        dark_layout(fig_se, height=320)
        st.plotly_chart(fig_se, use_container_width=True, config={"displayModeBar": False})

    # Row 5: Correlation heatmap + Batch explorer
    c5, c6 = st.columns([5, 5])

    with c5:
        st.markdown('<div class="slabel">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = ["temperature", "pressure", "speed", "feed_rate",
                    "humidity", "yield", "quality", "energy_consumption", "carbon_intensity"]
        corr = df_batches[num_cols].corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=[c.replace("_", " ").title() for c in corr.columns],
            y=[c.replace("_", " ").title() for c in corr.index],
            colorscale=[[0,"#ff4b4b"], [0.5,"rgba(255,255,255,0.05)"], [1,"#00d4ff"]],
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9, color="rgba(255,255,255,0.8)"),
            hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                thickness=12, len=0.9,
            ),
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter", size=9),
            xaxis=dict(tickangle=40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            height=380,
            margin=dict(l=80, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with c6:
        st.markdown('<div class="slabel">Batch Explorer</div>', unsafe_allow_html=True)
        search_term = st.text_input("Search by Batch ID prefix",
                                    placeholder="e.g.  BATCH_00",
                                    label_visibility="collapsed")
        df_search = (df_batches[df_batches["batch_id"].str.startswith(search_term)]
                     if search_term else df_batches.head(100))
        show_cols = ["batch_id", "temperature", "pressure", "speed",
                     "yield", "quality", "energy_consumption", "carbon_intensity", "zone"]
        st.dataframe(
            df_search[show_cols].round(4),
            use_container_width=True,
            hide_index=True,
            height=330,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PARETO INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df_pareto) == 0:
        st.warning("No Pareto solutions in the database. Run Phase 5 first.")
    else:
        # Filter bar
        f1, f2, f3 = st.columns([3, 3, 4])
        with f1:
            min_yield = st.slider("Min Pred Yield", 0.0, 0.99, 0.0, 0.01)
        with f2:
            _c_min = float(df_pareto["pred_carbon"].min())
            _c_max = float(df_pareto["pred_carbon"].max())
            max_carbon = st.slider("Max Pred Carbon (kgCO₂)", _c_min, _c_max,
                                   _c_max, (_c_max - _c_min) / 20 or 1.0)
        with f3:
            color_by = st.selectbox(
                "Color dimension",
                ["pred_yield", "pred_quality", "pred_energy", "pred_carbon"],
                index=0,
            )

        df_fp = df_pareto[
            (df_pareto["pred_yield"]  >= min_yield) &
            (df_pareto["pred_carbon"] <= max_carbon)
        ].copy()

        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.38);margin-bottom:12px;">'
            f'Showing <b style="color:#00d4ff;">{len(df_fp)}</b> of '
            f'<b style="color:rgba(255,255,255,0.6);">{len(df_pareto)}</b> Pareto solutions</div>',
            unsafe_allow_html=True,
        )

        # 3D scatter ──────────────────────────────────────────────────────────
        color_scale_map = {
            "pred_yield":   [[0, _RED],    [0.5, _YELLOW], [1, _GREEN]],
            "pred_quality": [[0, _CYAN],   [0.5, _PURPLE], [1, _GREEN]],
            "pred_energy":  [[0, _GREEN],  [0.5, _YELLOW], [1, _RED]],
            "pred_carbon":  [[0, _GREEN],  [0.5, _YELLOW], [1, _RED]],
        }

        fig_3d = px.scatter_3d(
            df_fp,
            x="pred_yield", y="pred_energy", z="pred_carbon",
            color=color_by,
            color_continuous_scale=color_scale_map[color_by],
            title="Pareto Frontier — 3-Objective Trade-off Space",
            labels={
                "pred_yield":   "Yield",
                "pred_energy":  "Energy (kWh)",
                "pred_carbon":  "Carbon (kgCO₂)",
            },
            hover_data=["pred_quality", "temperature", "pressure", "speed"],
        )
        fig_3d.update_traces(marker=dict(size=7, opacity=0.9))
        fig_3d.update_layout(
            paper_bgcolor="rgba(13,17,23,1)",
            scene=dict(
                bgcolor="rgb(13,17,23)",
                xaxis=dict(
                    backgroundcolor="rgba(0,212,255,0.05)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,255,136,0.02)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
                zaxis=dict(
                    backgroundcolor="rgba(255,75,75,0.02)",
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    title=dict(font=dict(color="rgba(255,255,255,0.5)", size=10)),
                ),
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                thickness=12,
            ),
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            height=480,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Pareto front + top-10 bar
        p1, p2 = st.columns([6, 4])

        with p1:
            fig_pareto2 = px.scatter(
                df_fp,
                x="pred_energy", y="pred_yield",
                color="pred_carbon",
                color_continuous_scale=[[0, _GREEN], [0.5, _YELLOW], [1, _RED]],
                title="Pareto Front — Yield vs Energy (size = quality)",
                labels={
                    "pred_energy": "Predicted Energy (kWh)",
                    "pred_yield":  "Predicted Yield",
                    "pred_carbon": "Carbon (kgCO₂)",
                },
                size="pred_quality",
                size_max=14,
                hover_data=["temperature", "pressure", "speed", "pred_quality"],
            )
            dark_layout(fig_pareto2, height=360)
            fig_pareto2.update_coloraxes(
                colorbar=dict(
                    tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                    thickness=11, len=0.9,
                    title=dict(text="Carbon", font=dict(size=10)),
                )
            )
            st.plotly_chart(fig_pareto2, use_container_width=True,
                            config={"displayModeBar": False})

        with p2:
            top10 = df_pareto.head(10).copy()
            top10["rank"] = [f"#{i+1}" for i in range(len(top10))]
            fig_bar10 = go.Figure(go.Bar(
                x=top10["pred_yield"],
                y=top10["rank"],
                orientation="h",
                marker=dict(
                    color=top10["pred_yield"],
                    colorscale=[[0, "rgba(0,212,255,0.5)"], [1, "#00ff88"]],
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Yield: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig_bar10.update_layout(
                title=dict(text="Top 10 Solutions by Yield",
                           font=dict(size=13, color="rgba(255,255,255,0.7)")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                xaxis=dict(title="Predicted Yield",
                           gridcolor="rgba(255,255,255,0.07)",
                           tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10)),
                height=360,
                margin=dict(l=50, r=16, t=44, b=30),
            )
            st.plotly_chart(fig_bar10, use_container_width=True,
                            config={"displayModeBar": False})

        # Filtered data table
        st.markdown('<div class="slabel">Filtered Pareto Solutions</div>', unsafe_allow_html=True)
        show_p_cols = [
            "temperature", "pressure", "speed", "feed_rate",
            "material_density", "material_hardness", "material_grade",
            "pred_yield", "pred_quality", "pred_energy", "pred_carbon",
        ]
        st.dataframe(
            df_fp[show_p_cols].reset_index(drop=True).round(4),
            use_container_width=True,
            hide_index=True,
            height=300,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GENOME EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(df_genomes) == 0:
        st.warning("No genome data available in the database.")
    else:
        # Parse all loaded genomes into matrix
        genome_matrix = np.array([json.loads(g) for g in df_genomes["genome"]])  # (N, 25)
        batch_ids_gm  = df_genomes["batch_id"].tolist()

    # Row 6: Batch selector
        st.markdown('<div class="slabel">Individual Batch Analysis</div>', unsafe_allow_html=True)
        g1, g2 = st.columns([3, 7])
        with g1:
            all_ids = df_batches["batch_id"].tolist()
            selected_batch = st.selectbox(
                "Select Batch ID",
                all_ids,
                index=0,
                label_visibility="collapsed",
            )

        # Resolve genome for selected batch
        sel_genome = None
        if selected_batch in batch_ids_gm:
            sel_idx    = batch_ids_gm.index(selected_batch)
            sel_genome = genome_matrix[sel_idx]
        else:
            conn = sqlite3.connect(DB_PATH)
            row_ = conn.execute(
                "SELECT genome FROM genome_vectors WHERE batch_id=?", (selected_batch,)
            ).fetchone()
            conn.close()
            if row_:
                sel_genome = np.array(json.loads(row_[0]))

        with g2:
            if sel_genome is not None:
                batch_row = df_batches[df_batches["batch_id"] == selected_batch]
                if len(batch_row) > 0:
                    br = batch_row.iloc[0]
                    st.markdown(
                        '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                        + "".join([
                            f'<div style="background:rgba(255,255,255,0.04);border:1px solid '
                            f'rgba(255,255,255,0.09);border-radius:8px;padding:8px 14px;'
                            f'text-align:center;min-width:80px;">'
                            f'<div style="font-size:0.95rem;font-weight:600;color:{c};'
                            f'font-family:\'JetBrains Mono\',monospace;">{v}</div>'
                            f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.38);'
                            f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">{lbl}</div>'
                            '</div>'
                            for lbl, v, c in [
                                ("Yield",   f"{br['yield']:.4f}",           _GREEN),
                                ("Quality", f"{br['quality']:.4f}",         _CYAN),
                                ("Energy",  f"{br['energy_consumption']:.0f} kWh", _YELLOW),
                                ("Carbon",  f"{br['carbon_intensity']:.1f}", _RED),
                                ("Zone",    br["zone"],                      ZONE_COLORS[br["zone"]]),
                            ]
                        ])
                        + '</div>',
                        unsafe_allow_html=True,
                    )

        # Individual batch charts
        if sel_genome is not None:
            ga, gb = st.columns([4, 6])

            with ga:
                # Radar chart of 5 process parameters (normalized to 0-100)
                batch_row = df_batches[df_batches["batch_id"] == selected_batch]
                if len(batch_row) > 0:
                    br = batch_row.iloc[0]
                    radar_cats = ["Temperature", "Pressure", "Speed", "Feed Rate", "Humidity"]
                    raw_vals   = [br["temperature"], br["pressure"], br["speed"],
                                  br["feed_rate"],  br["humidity"]]
                    feat_cols   = ["temperature", "pressure", "speed", "feed_rate", "humidity"]
                    norm_vals   = [
                        (v - df_batches[c].min()) / (df_batches[c].max() - df_batches[c].min() + 1e-9) * 100
                        for v, c in zip(raw_vals, feat_cols)
                    ]
                    # close the loop
                    r_vals  = norm_vals  + [norm_vals[0]]
                    r_theta = radar_cats + [radar_cats[0]]

                    fig_radar = go.Figure(go.Scatterpolar(
                        r=r_vals, theta=r_theta,
                        fill="toself",
                        fillcolor="rgba(0,212,255,0.15)",
                        line=dict(color=_CYAN, width=2),
                        marker=dict(color=_CYAN, size=6),
                        name=selected_batch,
                        hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.1f}%<extra></extra>",
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(255,255,255,0.02)",
                            radialaxis=dict(
                                visible=True, range=[0, 100],
                                tickfont=dict(size=8, color="rgba(255,255,255,0.3)"),
                                gridcolor="rgba(255,255,255,0.08)",
                                linecolor="rgba(255,255,255,0.1)",
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=10, color="rgba(255,255,255,0.6)"),
                                gridcolor="rgba(255,255,255,0.08)",
                                linecolor="rgba(255,255,255,0.1)",
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                        title=dict(text=f"Process Profile<br><span style='font-size:0.8em'>{selected_batch}</span>",
                                   font=dict(size=12, color="rgba(255,255,255,0.6)")),
                        height=360,
                        margin=dict(l=30, r=30, t=55, b=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True,
                                    config={"displayModeBar": False})

            with gb:
                # Bar chart of all 25 genome dimensions
                seg_colors = (
                    [_CYAN]   * 5   # Process
                    + [_GREEN]  * 3   # Material
                    + [_PURPLE] * 16  # EnergyDNA
                    + [_YELLOW] * 1   # Carbon
                )
                fig_gbar = go.Figure(go.Bar(
                    x=GENOME_LABELS,
                    y=sel_genome.tolist(),
                    marker=dict(
                        color=seg_colors,
                        opacity=0.88,
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{x}</b><br>z-score: %{y:.4f}<extra></extra>",
                ))
                fig_gbar.add_hline(
                    y=0,
                    line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1),
                )
                # Segment dividers
                for xpos, label, col in [(4.5, "Process", _CYAN),
                                         (7.5, "Material", _GREEN),
                                         (23.5, "EdNA", _PURPLE),
                                         (24.5, "Ci", _YELLOW)]:
                    fig_gbar.add_vline(
                        x=xpos,
                        line=dict(color="rgba(255,255,255,0.12)", dash="dot", width=1),
                    )
                fig_gbar.update_layout(
                    title=dict(
                        text=f"25-Dimension Genome Vector — {selected_batch}",
                        font=dict(size=12, color="rgba(255,255,255,0.6)"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                    xaxis=dict(
                        tickangle=60, tickfont=dict(size=8),
                        gridcolor="rgba(255,255,255,0.04)",
                        linecolor="rgba(255,255,255,0.08)",
                    ),
                    yaxis=dict(
                        title="z-score",
                        gridcolor="rgba(255,255,255,0.07)",
                        zerolinecolor="rgba(255,255,255,0.15)",
                        linecolor="rgba(255,255,255,0.08)",
                    ),
                    height=360,
                    margin=dict(l=50, r=16, t=50, b=70),
                    showlegend=False,
                    bargap=0.25,
                )
                st.plotly_chart(fig_gbar, use_container_width=True,
                                config={"displayModeBar": False})

        # Population heatmap
        st.markdown(
            f'<div class="slabel">Genome Population Heatmap — '
            f'{len(df_genomes)} Batches × 25 Dimensions</div>',
            unsafe_allow_html=True,
        )
        fig_hm = go.Figure(go.Heatmap(
            z=genome_matrix.T,             # shape (25, N) — dims as rows
            x=[bid[-4:] for bid in batch_ids_gm],
            y=GENOME_LABELS,
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.3, "rgba(200,80,80,0.5)"],
                [0.5, "rgba(255,255,255,0.06)"],
                [0.7, "rgba(0,160,255,0.5)"],
                [1.0, "#00d4ff"],
            ],
            zmid=0,
            hovertemplate="Batch: %{x}<br>Dimension: %{y}<br>z-score: %{z:.4f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                outlinecolor="rgba(255,255,255,0.08)",
                outlinewidth=1,
                thickness=12,
                len=0.9,
                title=dict(text="z-score", font=dict(size=10, color="rgba(255,255,255,0.4)")),
            ),
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
            xaxis=dict(
                title="Batch ID (last 4 chars)",
                tickfont=dict(size=7),
                tickangle=90,
                gridcolor="rgba(255,255,255,0.03)",
            ),
            yaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.03)"),
            height=500,
            margin=dict(l=90, r=20, t=20, b=70),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="slabel">Database Overview</div>', unsafe_allow_html=True)

    # 7 metric cards
    h_cols = st.columns(7)
    for i, (key, count) in enumerate([(k, v) for k, v in db_summary.items() if k != "db_size_mb"]):
        icon = TABLE_ICONS.get(key, "📄")
        with h_cols[i]:
            st.metric(f"{icon} {key.replace('_', ' ').title()}", f"{count:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: DB bar chart + prediction accuracy
    ha, hb = st.columns([4, 6])

    with ha:
        df_counts = pd.DataFrame([
            {"Table": k.replace("_", " ").title(), "Rows": v}
            for k, v in db_summary.items() if k != "db_size_mb"
        ]).sort_values("Rows", ascending=True)

        fig_dbbar = go.Figure(go.Bar(
            x=df_counts["Rows"],
            y=df_counts["Table"],
            orientation="h",
            marker=dict(
                color=df_counts["Rows"],
                colorscale=[[0, "rgba(0,212,255,0.4)"], [1, "#00ff88"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Rows: %{x:,}<extra></extra>",
            text=df_counts["Rows"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.6)", size=10),
        ))
        fig_dbbar.update_layout(
            title=dict(text="Table Row Counts", font=dict(size=12, color="rgba(255,255,255,0.6)")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.015)",
            font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=9)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9)),
            height=360,
            margin=dict(l=130, r=55, t=44, b=16),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_dbbar, use_container_width=True, config={"displayModeBar": False})

    with hb:
        # Model prediction accuracy (yield pred vs actual, if data exists)
        df_pred_notnull = df_preds.dropna(subset=["pred_yield", "actual_yield"]) \
            if len(df_preds) > 0 and "actual_yield" in df_preds.columns else pd.DataFrame()

        if len(df_pred_notnull) > 0:
            mn = min(df_pred_notnull["actual_yield"].min(), df_pred_notnull["pred_yield"].min())
            mx = max(df_pred_notnull["actual_yield"].max(), df_pred_notnull["pred_yield"].max())
            fig_acc = px.scatter(
                df_pred_notnull,
                x="actual_yield", y="pred_yield",
                title="Yield Prediction Accuracy (Predicted vs Actual)",
                labels={"actual_yield": "Actual Yield", "pred_yield": "Predicted Yield"},
                color_discrete_sequence=[_CYAN],
                opacity=0.6,
            )
            fig_acc.add_shape(
                type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                line=dict(color="rgba(255,214,0,0.55)", dash="dash", width=2),
            )
            dark_layout(fig_acc, height=360)
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})
        else:
            # Empty state — show a note + DB info card
            st.markdown(
                '<div style="background:rgba(255,255,255,0.03);border:1px solid '
                'rgba(255,255,255,0.09);border-radius:12px;padding:24px;'
                'text-align:center;height:360px;display:flex;flex-direction:column;'
                'justify-content:center;align-items:center;">'
                '<div style="font-size:2rem;margin-bottom:10px;">🎯</div>'
                '<div style="font-size:0.9rem;color:rgba(255,255,255,0.6);">'
                'Prediction accuracy plot available after<br>'
                'running Phase 4 and loading <code>predictions</code> table.</div>'
                f'<div style="margin-top:16px;font-size:0.8rem;color:{_CYAN};">'
                f'Database: {db_summary["db_size_mb"]} MB  ·  '
                f'{db_summary["batches"]:,} batches loaded</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # Pareto multi-objective metrics
    if len(df_pareto) > 0:
        st.markdown('<div class="slabel">Pareto Frontier Statistics</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Best Yield",    f"{df_pareto['pred_yield'].max():.4f}",
                      f"Avg {df_pareto['pred_yield'].mean():.4f}")
        with p2:
            st.metric("Best Quality",  f"{df_pareto['pred_quality'].max():.4f}",
                      f"Avg {df_pareto['pred_quality'].mean():.4f}")
        with p3:
            st.metric("Min Energy",    f"{df_pareto['pred_energy'].min():.1f} kWh",
                      f"Avg {df_pareto['pred_energy'].mean():.1f}", delta_color="inverse")
        with p4:
            st.metric("Min Carbon",    f"{df_pareto['pred_carbon'].min():.1f} kg",
                      f"Avg {df_pareto['pred_carbon'].mean():.1f}", delta_color="inverse")

    # Pipeline run audit log
    st.markdown('<div class="slabel">Pipeline Execution Log</div>', unsafe_allow_html=True)
    if len(df_runs) > 0:
        run_cols = [c for c in ["phase", "phase_name", "status", "details", "started_at", "finished_at"]
                    if c in df_runs.columns]
        st.dataframe(df_runs[run_cols], use_container_width=True, hide_index=True, height=260)
    else:
        st.info("No pipeline runs recorded yet.")

    # Footer
    st.markdown(
        '<div style="text-align:center;margin-top:30px;padding:18px;'
        'border-top:1px solid rgba(255,255,255,0.06);">'
        '<div style="font-size:0.77rem;color:rgba(255,255,255,0.22);">'
        'ACMGS v9.0 &nbsp;·&nbsp; Autonomous Carbon-Aware Manufacturing Genome System'
        '&nbsp;·&nbsp; Phase 9: Streamlit Dashboard<br>'
        f'Database: {db_summary.get("db_size_mb", 0)} MB &nbsp;·&nbsp; '
        f'{db_summary.get("batches", 0):,} Batches &nbsp;·&nbsp; '
        f'{db_summary.get("pareto_solutions", 0)} Pareto Solutions &nbsp;·&nbsp; '
        f'Rendered: {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    # ─────────────────────────────────────────────────────────────────────────
    # DATA LOAD — Digital Twin
    # ─────────────────────────────────────────────────────────────────────────
    try:
        _dt_conn = sqlite3.connect(DB_PATH)
        _dt_anomaly_count = _dt_conn.execute(
            "SELECT COUNT(*) FROM energy_embeddings WHERE is_anomaly=1"
        ).fetchone()[0]
        df_dt_health = pd.read_sql_query(
            "SELECT batch_id, recon_error, is_anomaly FROM energy_embeddings ORDER BY rowid",
            _dt_conn,
        )
        _dt_conn.close()
        df_dt_health["recon_error"] = pd.to_numeric(df_dt_health["recon_error"], errors="coerce")
        df_dt_health["is_anomaly"]  = (
            pd.to_numeric(df_dt_health["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )
    except Exception:
        _dt_anomaly_count = 0
        df_dt_health = pd.DataFrame(columns=["batch_id", "recon_error", "is_anomaly"])

    _nb          = len(df_batches)
    _dt_total    = db_summary.get("batches", 0)
    _anom_rate   = _dt_anomaly_count / max(_dt_total, 1) * 100
    _avg_yield   = float(df_preds["pred_yield"].mean())            if len(df_preds) > 0   else 0.0
    _avg_quality = float(df_preds["pred_quality"].mean())          if len(df_preds) > 0   else 0.0
    _avg_energy  = float(df_batches["energy_consumption"].mean())  if _nb > 0             else 0.0
    _latest      = df_batches.iloc[-1] if _nb > 0 else None

    # Factory health level
    if _anom_rate > 15:
        _fstate, _fc, _fbg, _fbd, _fpulse, _ficon = (
            "CRITICAL", _RED,
            "rgba(255,75,75,0.09)", "rgba(255,75,75,0.38)",
            "#ff4b4b", "&#128308;",   # red circle HTML entity
        )
    elif _anom_rate > 7:
        _fstate, _fc, _fbg, _fbd, _fpulse, _ficon = (
            "DEGRADED", _YELLOW,
            "rgba(255,214,0,0.09)", "rgba(255,214,0,0.38)",
            "#ffd600", "&#128993;",   # yellow circle
        )
    else:
        _fstate, _fc, _fbg, _fbd, _fpulse, _ficon = (
            "ONLINE", _GREEN,
            "rgba(0,255,136,0.09)", "rgba(0,255,136,0.38)",
            "#00ff88", "&#128994;",   # green circle
        )

    _lat_id  = str(_latest["batch_id"])           if _latest is not None else "N/A"
    _lat_ci  = float(_latest["carbon_intensity"]) if _latest is not None else 0.0
    _lat_zone = classify_carbon_zone(_lat_ci)     if _latest is not None else zone

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 &#10143; FACTORY HEALTH CARD
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        f"""
<style>
@keyframes dt_pulse {{
  0%,100%{{box-shadow:0 0 0 0 {_fpulse}66;}}
  50%{{box-shadow:0 0 0 12px {_fpulse}00;}}
}}
.dt_kpi_pill {{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.11);
  border-radius:20px;padding:5px 16px;
  font-size:0.73rem;font-weight:600;
  color:rgba(255,255,255,0.7);margin:3px 4px;
  letter-spacing:0.04em;
}}
.dt_kpi_dot {{width:7px;height:7px;border-radius:50%;display:inline-block;}}
</style>
<div style="background:linear-gradient(135deg,{_fbg},{_fbg.replace('0.09','0.03')});
     border:1px solid {_fbd};border-radius:16px;padding:24px 28px;margin-bottom:22px;">
  <div style="display:flex;justify-content:space-between;align-items:center;
       flex-wrap:wrap;gap:14px;">
    <div style="display:flex;align-items:center;gap:18px;">
      <div style="width:56px;height:56px;border-radius:50%;
           background:{_fbg};border:2px solid {_fbd};
           display:flex;align-items:center;justify-content:center;
           font-size:1.8rem;animation:dt_pulse 2.2s infinite;">{_ficon}</div>
      <div>
        <div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.18em;
             color:rgba(255,255,255,0.3);margin-bottom:3px;">
          ACMGS Digital Twin &nbsp;&#183;&nbsp; Live Factory Mirror</div>
        <div style="font-size:1.9rem;font-weight:800;color:{_fc};
             letter-spacing:0.05em;line-height:1.1;">FACTORY {_fstate}</div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.38);margin-top:4px;">
          Latest batch&nbsp;{_lat_id}&nbsp;&#183;&nbsp;
          Carbon&nbsp;{_lat_ci:.0f}&nbsp;gCO&#8322;/kWh&nbsp;&#183;&nbsp;
          Zone&nbsp;{_lat_zone}</div>
      </div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:2px;">
      <div class="dt_kpi_pill">
        <span class="dt_kpi_dot" style="background:{_GREEN};"></span>
        {_dt_total:,} Batches</div>
      <div class="dt_kpi_pill">
        <span class="dt_kpi_dot" style="background:{_RED};"></span>
        {_dt_anomaly_count} Anomalies&nbsp;({_anom_rate:.1f}%)</div>
      <div class="dt_kpi_pill">
        <span class="dt_kpi_dot" style="background:{_CYAN};"></span>
        Avg Yield&nbsp;{_avg_yield:.4f}</div>
      <div class="dt_kpi_pill">
        <span class="dt_kpi_dot" style="background:{_PURPLE};"></span>
        Avg Quality&nbsp;{_avg_quality:.4f}</div>
      <div class="dt_kpi_pill">
        <span class="dt_kpi_dot" style="background:{_YELLOW};"></span>
        Avg Energy&nbsp;{_avg_energy:.0f}&nbsp;kWh</div>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Simulate New Batch button ─────────────────────────────────────────
    _sim_col, _ = st.columns([1, 5])
    with _sim_col:
        if st.button("+ Simulate New Batch", key="dt_sim_batch",
                     help="Insert a new simulated batch into the DB and refresh"):
            import random as _rnd
            with sqlite3.connect(DB_PATH) as _sim_conn:
                _last_id  = _sim_conn.execute(
                    "SELECT batch_id FROM batches ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                _next_num = int(_last_id[0].split("_")[1]) + 1 if _last_id else 2000
            _new_id   = f"BATCH_{_next_num}"
            _sim_conn.execute(
                """INSERT INTO batches
                   (batch_id, temperature, pressure, speed, feed_rate, humidity,
                    material_density, material_hardness, material_grade,
                    yield, quality, energy_consumption, carbon_intensity, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                (
                    _new_id,
                    round(_rnd.uniform(150.0, 350.0), 4),
                    round(_rnd.uniform(1.0,   10.0),  4),
                    round(_rnd.uniform(500.0, 3000.0),4),
                    round(_rnd.uniform(0.1,   1.0),   4),
                    round(_rnd.uniform(20.0,  80.0),  4),
                    round(_rnd.uniform(2.0,   5.0),   4),
                    round(_rnd.uniform(10.0,  60.0),  4),
                    _rnd.randint(1, 5),
                    round(_rnd.uniform(0.3,   1.0),   4),
                    round(_rnd.uniform(0.3,   1.0),   4),
                    round(_rnd.uniform(100.0, 500.0), 4),
                    round(_rnd.uniform(200.0, 600.0), 4),
                ),
            )
            _sim_conn.commit()
            _sim_conn.close()
            _load_batches.clear()
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 &#10143; LIVE MACHINE GAUGES + TREND
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#9889; Live Machine State &#8212; Current Batch Gauges + 300-Batch Trend</div>',
        unsafe_allow_html=True,
    )

    def _dt_gauge(val, lo, hi, label, unit, col):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title=dict(
                text=f"{label}<br><span style='font-size:0.72em;"
                     f"color:rgba(255,255,255,0.35);'>{unit}</span>",
                font=dict(size=12, color="rgba(255,255,255,0.5)"),
            ),
            number=dict(font=dict(size=40, color=col, family="JetBrains Mono,monospace")),
            gauge=dict(
                axis=dict(range=[lo, hi], tickwidth=1,
                          tickcolor="rgba(255,255,255,0.15)",
                          tickfont=dict(color="rgba(255,255,255,0.28)", size=7),
                          nticks=5),
                bar=dict(color=col, thickness=0.21),
                bgcolor="rgba(255,255,255,0.02)",
                borderwidth=1, bordercolor="rgba(255,255,255,0.06)",
                steps=[
                    dict(range=[lo,                    lo+(hi-lo)*0.4],  color="rgba(0,255,136,0.07)"),
                    dict(range=[lo+(hi-lo)*0.4,        lo+(hi-lo)*0.75], color="rgba(255,214,0,0.06)"),
                    dict(range=[lo+(hi-lo)*0.75, hi],                    color="rgba(255,75,75,0.08)"),
                ],
            ),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
            height=230, margin=dict(l=20, r=20, t=30, b=8),
        )
        return fig

    _gc1, _gc2, _gc3, _gtrd = st.columns([2, 2, 2, 4])
    if _latest is not None:
        with _gc1:
            st.plotly_chart(
                _dt_gauge(float(_latest["temperature"]), 100, 350, "Temperature", "degC", _CYAN),
                use_container_width=True, config={"displayModeBar": False},
            )
        with _gc2:
            st.plotly_chart(
                _dt_gauge(float(_latest["speed"]), 500, 3000, "Speed", "rpm", _GREEN),
                use_container_width=True, config={"displayModeBar": False},
            )
        with _gc3:
            st.plotly_chart(
                _dt_gauge(float(_latest["pressure"]), 1, 10, "Pressure", "bar", _ORANGE),
                use_container_width=True, config={"displayModeBar": False},
            )
        with _gtrd:
            _tdf = df_batches.tail(300).reset_index(drop=True)
            _ftrd = go.Figure()
            _ftrd.add_trace(go.Scatter(
                x=list(range(len(_tdf))), y=_tdf["yield"].tolist(),
                mode="lines", name="Yield",
                line=dict(color=_CYAN, width=1.4),
                hovertemplate="Batch +%{x}<br>Yield: %{y:.4f}<extra></extra>",
            ))
            _ftrd.add_trace(go.Scatter(
                x=list(range(len(_tdf))), y=_tdf["quality"].tolist(),
                mode="lines", name="Quality",
                line=dict(color=_GREEN, width=1.4),
                hovertemplate="Batch +%{x}<br>Quality: %{y:.4f}<extra></extra>",
            ))
            _ftrd.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
                font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
                title=dict(text="300-Batch Yield & Quality Trend",
                           font=dict(size=11, color="rgba(255,255,255,0.45)")),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=8)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(size=8)),
                legend=dict(bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.1)",
                            borderwidth=1, orientation="h", yanchor="bottom", y=1.08, x=1, xanchor="right"),
                height=230, margin=dict(l=10, r=10, t=52, b=10),
            )
            st.plotly_chart(_ftrd, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No batch data available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 &#10143; FACTORY PIPELINE VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#127981; Factory Pipeline &#8212; Current Batch State Across All Stations</div>',
        unsafe_allow_html=True,
    )

    def _station(name, icon, value, unit, bg_col, border_col, badge_html=""):
        return (
            f'<div style="flex:1;min-width:110px;background:{bg_col};'
            f'border:1px solid {border_col};border-radius:12px;'
            f'padding:14px 10px;text-align:center;">'
            f'<div style="font-size:1.45rem;">{icon}</div>'
            f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);'
            f'text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 2px 0;">{name}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{border_col};'
            f'font-family:JetBrains Mono,monospace;">{value}'
            f'<span style="font-size:0.6rem;color:rgba(255,255,255,0.28);'
            f'margin-left:2px;">{unit}</span></div>'
            f'{badge_html}</div>'
        )

    _arr = '<div style="display:flex;align-items:center;color:rgba(255,255,255,0.18);font-size:1.2rem;padding:0 4px;">&#8594;</div>'

    if _latest is not None:
        _lat_anom_row = df_dt_health[df_dt_health["batch_id"] == _latest["batch_id"]]
        _lat_is_anom  = bool(int(_lat_anom_row.iloc[0]["is_anomaly"])) if len(_lat_anom_row) > 0 else False
        _qc_badge = (
            '<div style="font-size:0.58rem;margin-top:4px;padding:2px 8px;'
            'border-radius:10px;background:rgba(255,75,75,0.18);color:#ff4b4b;">&#9888; ANOMALY</div>'
            if _lat_is_anom else
            '<div style="font-size:0.58rem;margin-top:4px;padding:2px 8px;'
            'border-radius:10px;background:rgba(0,255,136,0.12);color:#00ff88;">&#10003; NORMAL</div>'
        )
        st.markdown(
            '<div style="display:flex;gap:6px;flex-wrap:nowrap;overflow-x:auto;margin-bottom:10px;">'
            + _station("Raw Input",  "&#128230;", int(_latest["material_grade"]),
                       "grade", "rgba(168,85,247,0.08)", _PURPLE)
            + _arr
            + _station("Heating",    "&#128293;", f"{_latest['temperature']:.0f}",
                       "degC",  "rgba(249,115,22,0.08)", _ORANGE)
            + _arr
            + _station("Pressure",   "&#9881;",   f"{_latest['pressure']:.1f}",
                       "bar",   "rgba(0,212,255,0.08)", _CYAN)
            + _arr
            + _station("Production", "&#127959;", f"{_latest['speed']:.0f}",
                       "rpm",   "rgba(0,255,136,0.08)", _GREEN)
            + _arr
            + _station("QC Check",   "&#127919;", f"{_latest['quality']:.4f}",
                       "",      "rgba(255,214,0,0.08)", _YELLOW, _qc_badge)
            + _arr
            + _station("Dispatch",   "&#128666;", f"{_latest['energy_consumption']:.0f}",
                       "kWh",   "rgba(255,75,75,0.08)",  _RED)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No current batch to display.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 &#10143; ANOMALY HEARTBEAT
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#128168; Anomaly Heartbeat &#8212; LSTM Reconstruction Error vs Threshold</div>',
        unsafe_allow_html=True,
    )

    if len(df_dt_health) > 0:
        _hs = df_dt_health.copy().reset_index(drop=True)
        _hs["idx"] = range(1, len(_hs) + 1)
        _thresh = 0.199084
        _nrm = _hs[_hs["is_anomaly"] == 0]
        _anm = _hs[_hs["is_anomaly"] == 1]

        # Current batch risk score
        _cur_recon = float(df_dt_health.iloc[-1]["recon_error"]) if len(df_dt_health) > 0 else 0.0
        if _cur_recon != _cur_recon:  # NaN check
            _cur_recon = 0.0
        _risk_pct  = min(100, int(_cur_recon / _thresh * 100))
        _risk_col  = _RED if _cur_recon > _thresh else (_YELLOW if _cur_recon > _thresh * 0.7 else _GREEN)
        _risk_lbl  = "HIGH RISK" if _cur_recon > _thresh else ("ELEVATED" if _cur_recon > _thresh * 0.7 else "NORMAL")

        # Verdict strip
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;'
            f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:10px;padding:12px 18px;margin-bottom:12px;">'
            f'<div style="flex:1;">'
            f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.3);'
            f'text-transform:uppercase;letter-spacing:0.12em;">ML Verdict &#8212; Latest Batch</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{_risk_col};">'
            f'{_risk_lbl}</div></div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.3);">Recon Error</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:{_risk_col};'
            f'font-family:JetBrains Mono,monospace;">{_cur_recon:.5f}</div>'
            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.3);">'
            f'threshold = {_thresh}</div></div>'
            f'<div style="width:90px;">'
            f'<div style="height:8px;background:rgba(255,255,255,0.1);border-radius:4px;">'
            f'<div style="height:8px;width:{_risk_pct}%;background:{_risk_col};'
            f'border-radius:4px;transition:width 0.5s;"></div></div>'
            f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.28);'
            f'margin-top:3px;text-align:center;">{_risk_pct}% of threshold</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        _fhb = go.Figure()
        _fhb.add_trace(go.Scatter(
            x=_nrm["idx"].tolist(), y=_nrm["recon_error"].tolist(),
            mode="markers", name="Normal",
            marker=dict(color=_CYAN, size=2.5, opacity=0.5),
            hovertemplate="Batch #%{x}<br>Error: %{y:.5f}<extra></extra>",
        ))
        if len(_anm) > 0:
            _fhb.add_trace(go.Scatter(
                x=_anm["idx"].tolist(), y=_anm["recon_error"].tolist(),
                mode="markers", name="Anomaly",
                marker=dict(color=_RED, size=7, symbol="x",
                            line=dict(color=_RED, width=1.5)),
                hovertemplate="&#9888; Batch #%{x}<br>Error: %{y:.5f}<extra></extra>",
            ))
        _fhb.add_hline(y=_thresh, line=dict(color=_YELLOW, dash="dash", width=1.5),
                       annotation_text=f"Threshold {_thresh}",
                       annotation_font=dict(color=_YELLOW, size=9),
                       annotation_position="top right")
        _fhb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.012)",
            font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
            xaxis=dict(title=dict(text="Batch Number", font=dict(size=9)),
                       gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=8)),
            yaxis=dict(title=dict(text="Reconstruction Error", font=dict(size=9)),
                       gridcolor="rgba(255,255,255,0.07)", tickfont=dict(size=8)),
            legend=dict(bgcolor="rgba(0,0,0,0.28)", bordercolor="rgba(255,255,255,0.1)",
                        borderwidth=1, orientation="h", yanchor="bottom", y=1.01, x=0),
            height=300, margin=dict(l=55, r=12, t=38, b=42),
        )
        st.plotly_chart(_fhb, use_container_width=True, config={"displayModeBar": False})

        _ah1, _ah2, _ah3, _ah4 = st.columns(4)
        with _ah1: st.metric("Batches Analysed", f"{len(_hs):,}")
        with _ah2: st.metric("Anomalies", f"{_dt_anomaly_count:,}", delta_color="inverse")
        with _ah3: st.metric("Anomaly Rate", f"{_anom_rate:.1f}%", delta_color="inverse")
        with _ah4: st.metric("Avg Recon Error", f"{float(_hs['recon_error'].mean()):.5f}")
    else:
        st.info("No anomaly data &#8212; run Phase 2 (Energy DNA Model).")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 &#10143; CARBON OPPORTUNITY WINDOW
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#127757; Carbon Opportunity Window &#8212; When to Run Expensive Batches Today</div>',
        unsafe_allow_html=True,
    )

    _hours  = list(range(24))
    _h_cols = []
    _h_adv  = []
    for _v in CARBON_24H:
        if _v < CARBON_LOW_THRESHOLD:
            _h_cols.append(_GREEN);  _h_adv.append("RUN NOW")
        elif _v < CARBON_HIGH_THRESHOLD:
            _h_cols.append(_YELLOW); _h_adv.append("OK")
        else:
            _h_cols.append(_RED);    _h_adv.append("AVOID")

    _now_h  = datetime.now().hour
    _best_h = [h for h, a in enumerate(_h_adv) if a == "RUN NOW"]
    _avoid_h = [h for h, a in enumerate(_h_adv) if a == "AVOID"]
    _best_str  = (f"{_best_h[0]:02d}:00&#8211;{_best_h[-1]+1:02d}:00"
                  if _best_h else "None today")
    _avoid_str = (f"{_avoid_h[0]:02d}:00&#8211;{_avoid_h[-1]+1:02d}:00"
                  if _avoid_h else "None")
    _now_adv   = _h_adv[_now_h]
    _now_col   = _h_cols[_now_h]

    # AI verdict strip
    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
        f'<div style="background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.25);'
        f'border-radius:10px;padding:10px 18px;flex:1;min-width:140px;">'
        f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;'
        f'letter-spacing:0.12em;">Right Now ({_now_h:02d}:00)</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{_now_col};">{_now_adv}</div></div>'
        f'<div style="background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.22);'
        f'border-radius:10px;padding:10px 18px;flex:1;min-width:140px;">'
        f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;'
        f'letter-spacing:0.12em;">Optimal Window</div>'
        f'<div style="font-size:1.0rem;font-weight:700;color:{_GREEN};">{_best_str}</div></div>'
        f'<div style="background:rgba(255,75,75,0.07);border:1px solid rgba(255,75,75,0.22);'
        f'border-radius:10px;padding:10px 18px;flex:1;min-width:140px;">'
        f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;'
        f'letter-spacing:0.12em;">Avoid Window</div>'
        f'<div style="font-size:1.0rem;font-weight:700;color:{_RED};">{_avoid_str}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _fcw = go.Figure()
    _fcw.add_trace(go.Bar(
        x=_hours, y=CARBON_24H,
        marker=dict(color=_h_cols, opacity=0.85, line=dict(width=0)),
        text=_h_adv,
        textposition="outside",
        textfont=dict(size=7, color="rgba(255,255,255,0.4)"),
        hovertemplate="Hour %{x}:00<br>%{y} gCO&#8322;/kWh<br>Advice: %{text}<extra></extra>",
    ))
    _fcw.add_hline(y=CARBON_LOW_THRESHOLD,
                   line=dict(color=_GREEN, dash="dash", width=1.2),
                   annotation_text=f"LOW &lt;{CARBON_LOW_THRESHOLD}",
                   annotation_font=dict(color=_GREEN, size=9))
    _fcw.add_hline(y=CARBON_HIGH_THRESHOLD,
                   line=dict(color=_RED, dash="dash", width=1.2),
                   annotation_text=f"HIGH &gt;{CARBON_HIGH_THRESHOLD}",
                   annotation_font=dict(color=_RED, size=9))
    _fcw.add_vline(x=_now_h, line=dict(color=_CYAN, dash="dot", width=2),
                   annotation_text="NOW",
                   annotation_font=dict(color=_CYAN, size=10))
    _fcw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
        xaxis=dict(title=dict(text="Hour of Day", font=dict(size=9)),
                   tickvals=list(range(0, 24, 3)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                   tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title=dict(text="gCO&#8322;/kWh", font=dict(size=9)),
                   tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.07)"),
        height=280, margin=dict(l=55, r=12, t=20, b=42), showlegend=False,
    )
    st.plotly_chart(_fcw, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 &#10143; BATCH REPLAY (TIME MACHINE)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#9654; Batch Replay (Time Machine) &#8212; Step Through 2,000-Batch History</div>',
        unsafe_allow_html=True,
    )

    if _nb > 0:
        # Ensure session state key exists
        if "dt_replay_pos" not in st.session_state:
            st.session_state["dt_replay_pos"] = _nb  # start at latest

        # Nav buttons
        _nb0, _nb1, _nb2, _nb3, _nb4 = st.columns([3, 1, 1, 1, 1])
        with _nb0:
            st.markdown(
                '<div style="font-size:0.74rem;color:rgba(255,255,255,0.35);padding-top:9px;">'
                'Rewind to any batch &#8212; factory state fully reconstructed from DB history.</div>',
                unsafe_allow_html=True,
            )
        with _nb1:
            if st.button("&#9198; First", key="dt_rb_first", use_container_width=True):
                st.session_state["dt_replay_pos"] = 1
        with _nb2:
            if st.button("&#9664; Prev", key="dt_rb_prev", use_container_width=True):
                st.session_state["dt_replay_pos"] = max(1, st.session_state["dt_replay_pos"] - 1)
        with _nb3:
            if st.button("Next &#9654;", key="dt_rb_next", use_container_width=True):
                st.session_state["dt_replay_pos"] = min(_nb, st.session_state["dt_replay_pos"] + 1)
        with _nb4:
            if st.button("Last &#9197;", key="dt_rb_last", use_container_width=True):
                st.session_state["dt_replay_pos"] = _nb

        _rpos = st.slider(
            f"Batch Position (1 &#8594; {_nb:,})",
            min_value=1, max_value=_nb,
            key="dt_replay_pos",
        )
        _ridx = _rpos - 1
        _rrow = df_batches.iloc[_ridx]
        _rzone = classify_carbon_zone(float(_rrow["carbon_intensity"]))

        # Anomaly lookup
        _ranom_r = df_dt_health[df_dt_health["batch_id"] == _rrow["batch_id"]]
        _r_is_anom = bool(int(_ranom_r.iloc[0]["is_anomaly"])) if len(_ranom_r) > 0 else False
        _r_recon   = float(_ranom_r.iloc[0]["recon_error"])    if len(_ranom_r) > 0 else 0.0
        _r_anom_col = _RED if _r_is_anom else _GREEN
        _r_anom_lbl = "&#9888; ANOMALY DETECTED" if _r_is_anom else "&#10003; NORMAL"

        # Prediction lookup
        _rpred_r = df_preds[df_preds["batch_id"] == _rrow["batch_id"]] if len(df_preds) > 0 else pd.DataFrame()

        _rl, _rr = st.columns([3, 7])

        with _rl:
            st.markdown(
                f'<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);'
                f'border-radius:13px;padding:18px 16px;">'
                f'<div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.14em;'
                f'color:rgba(255,255,255,0.25);margin-bottom:4px;">BATCH {_rpos} OF {_nb:,}</div>'
                f'<div style="font-size:1.0rem;font-weight:700;color:{_CYAN};'
                f'font-family:JetBrains Mono,monospace;margin-bottom:10px;">{_rrow["batch_id"]}</div>'
                + "".join([
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
                    f'<span style="font-size:0.7rem;color:rgba(255,255,255,0.35);">{lbl}</span>'
                    f'<span style="font-size:0.76rem;font-weight:600;color:{vc};'
                    f'font-family:JetBrains Mono,monospace;">{val}</span></div>'
                    for lbl, val, vc in [
                        ("Temp",      f"{_rrow['temperature']:.1f} degC",         _ORANGE),
                        ("Pressure",  f"{_rrow['pressure']:.2f} bar",             _CYAN),
                        ("Speed",     f"{_rrow['speed']:.0f} rpm",               _GREEN),
                        ("Feed Rate", f"{_rrow['feed_rate']:.2f} kg/h",          _CYAN),
                        ("Humidity",  f"{_rrow['humidity']:.1f}%",               "rgba(255,255,255,0.5)"),
                        ("Density",   f"{_rrow['material_density']:.3f} g/cm3",  _PURPLE),
                        ("Hardness",  f"{_rrow['material_hardness']:.1f} HV",    _PURPLE),
                        ("Grade",     f"{int(_rrow['material_grade'])}",          _PURPLE),
                    ]
                ])
                + f'<div style="margin-top:10px;padding:8px 10px;border-radius:8px;'
                f'background:rgba(255,255,255,0.04);'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:0.7rem;color:rgba(255,255,255,0.35);">Zone</span>'
                f'<span style="font-weight:600;color:{ZONE_COLORS[_rzone]};">{_rzone}</span></div>'
                f'<div style="margin-top:6px;padding:8px 10px;border-radius:8px;'
                f'background:rgba(255,255,255,0.04);'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:0.7rem;color:rgba(255,255,255,0.35);">ML Verdict</span>'
                f'<span style="font-size:0.8rem;font-weight:600;color:{_r_anom_col};">'
                f'{_r_anom_lbl}</span></div>'
                f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.22);'
                f'margin-top:8px;">Recon Error: {_r_recon:.5f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with _rr:
            # Replay pipeline
            _r_qc_badge = (
                '<div style="font-size:0.58rem;margin-top:4px;padding:2px 8px;border-radius:10px;'
                'background:rgba(255,75,75,0.18);color:#ff4b4b;">&#9888; ANOMALY</div>'
                if _r_is_anom else
                '<div style="font-size:0.58rem;margin-top:4px;padding:2px 8px;border-radius:10px;'
                'background:rgba(0,255,136,0.12);color:#00ff88;">&#10003; NORMAL</div>'
            )
            st.markdown(
                '<div style="font-size:0.68rem;color:rgba(255,255,255,0.3);'
                'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">'
                'Pipeline Replay</div>'
                '<div style="display:flex;gap:5px;flex-wrap:nowrap;overflow-x:auto;">'
                + _station("Raw Input",  "&#128230;", int(_rrow["material_grade"]),
                           "grade", "rgba(168,85,247,0.08)", _PURPLE)
                + _arr
                + _station("Heating",    "&#128293;", f"{_rrow['temperature']:.0f}",
                           "degC",  "rgba(249,115,22,0.08)", _ORANGE)
                + _arr
                + _station("Pressure",   "&#9881;",   f"{_rrow['pressure']:.1f}",
                           "bar",   "rgba(0,212,255,0.08)", _CYAN)
                + _arr
                + _station("Production", "&#127959;", f"{_rrow['speed']:.0f}",
                           "rpm",   "rgba(0,255,136,0.08)", _GREEN)
                + _arr
                + _station("QC Check",   "&#127919;", f"{_rrow['quality']:.4f}",
                           "",      "rgba(255,214,0,0.08)", _YELLOW, _r_qc_badge)
                + _arr
                + _station("Dispatch",   "&#128666;", f"{_rrow['energy_consumption']:.0f}",
                           "kWh",   ZONE_BG[_rzone], ZONE_COLORS[_rzone])
                + "</div>",
                unsafe_allow_html=True,
            )

            # Outcome cards
            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            _oc1, _oc2, _oc3, _oc4 = st.columns(4)
            for _oc, (_lbl, _vstr, _vc) in zip(
                [_oc1, _oc2, _oc3, _oc4],
                [
                    ("Actual Yield",   f"{_rrow['yield']:.4f}",                  _GREEN),
                    ("Actual Quality", f"{_rrow['quality']:.4f}",                _CYAN),
                    ("Energy Used",    f"{_rrow['energy_consumption']:.0f} kWh", _YELLOW),
                    ("Carbon CI",      f"{_rrow['carbon_intensity']:.1f}",       _RED),
                ],
            ):
                with _oc:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.04);'
                        f'border:1px solid rgba(255,255,255,0.08);'
                        f'border-radius:9px;padding:10px 8px;text-align:center;">'
                        f'<div style="font-size:0.6rem;color:rgba(255,255,255,0.32);'
                        f'text-transform:uppercase;letter-spacing:0.08em;">{_lbl}</div>'
                        f'<div style="font-size:1.05rem;font-weight:700;color:{_vc};'
                        f'font-family:JetBrains Mono,monospace;margin-top:3px;">{_vstr}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            # Pred vs actual mini chart (if predictions exist)
            if len(_rpred_r) > 0:
                _fcmp = go.Figure()
                _cats = ["Yield", "Quality"]
                _acts = [float(_rrow["yield"]), float(_rrow["quality"])]
                _prds = [float(_rpred_r.iloc[0]["pred_yield"]),
                         float(_rpred_r.iloc[0]["pred_quality"])]
                _fcmp.add_trace(go.Bar(
                    name="Actual", x=_cats, y=_acts,
                    marker=dict(color=[_GREEN, _CYAN], opacity=0.85),
                ))
                _fcmp.add_trace(go.Bar(
                    name="Predicted", x=_cats, y=_prds,
                    marker=dict(color=[_YELLOW, _ORANGE], opacity=0.7),
                ))
                _fcmp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
                    font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
                    title=dict(text=f"Actual vs Predicted &#8212; Batch {_rpos}",
                               font=dict(size=10, color="rgba(255,255,255,0.45)")),
                    barmode="group", bargap=0.3,
                    xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.07)"),
                    legend=dict(bgcolor="rgba(0,0,0,0.25)",
                                bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                                orientation="h", yanchor="bottom", y=1.01, x=0),
                    height=210, margin=dict(l=42, r=8, t=36, b=30),
                )
                st.plotly_chart(_fcmp, use_container_width=True,
                                config={"displayModeBar": False})

        # Context window chart
        _wz = 40
        _wlo = max(0, _ridx - _wz // 2)
        _whi = min(_nb, _wlo + _wz)
        _wdf = df_batches.iloc[_wlo:_whi].reset_index(drop=True)
        _wpos = _ridx - _wlo

        _fwin = go.Figure()
        _fwin.add_trace(go.Scatter(
            x=list(range(len(_wdf))), y=_wdf["yield"].tolist(),
            mode="lines+markers", name="Yield",
            line=dict(color="rgba(0,212,255,0.35)", width=1.3),
            marker=dict(size=3, color="rgba(0,212,255,0.35)"),
        ))
        _fwin.add_trace(go.Scatter(
            x=list(range(len(_wdf))), y=_wdf["quality"].tolist(),
            mode="lines+markers", name="Quality",
            line=dict(color="rgba(0,255,136,0.35)", width=1.3),
            marker=dict(size=3, color="rgba(0,255,136,0.35)"),
        ))
        _fwin.add_vline(x=_wpos, line=dict(color=_YELLOW, width=2, dash="dot"),
                        annotation_text="Selected",
                        annotation_font=dict(color=_YELLOW, size=9))
        _fwin.add_scatter(
            x=[_wpos],
            y=[float(_wdf.iloc[_wpos]["yield"]) if _wpos < len(_wdf) else 0],
            mode="markers",
            marker=dict(color=_YELLOW, size=11, symbol="diamond"),
            name="Selected",
            hovertemplate="Selected batch<br>Yield: %{y:.4f}<extra></extra>",
        )
        _fwin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.012)",
            font=dict(color="rgba(255,255,255,0.6)", family="Inter"),
            title=dict(text=f"Context: batches {_wlo+1}&#8211;{_whi} (selected = {_rpos})",
                       font=dict(size=10, color="rgba(255,255,255,0.42)")),
            xaxis=dict(title=dict(text="Window position", font=dict(size=8)),
                       tickfont=dict(size=7), gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(255,255,255,0.07)"),
            legend=dict(bgcolor="rgba(0,0,0,0.35)",
                        bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                        orientation="h", yanchor="top", y=0.99, x=1, xanchor="right"),
            height=200, margin=dict(l=42, r=8, t=36, b=35),
        )
        st.plotly_chart(_fwin, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("No batch data available for replay.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7 &#10143; WHAT-IF SIMULATION LAB
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="slabel">&#128300; What-If Simulation Lab &#8212; Run the Full ML Pipeline In-Memory</div>',
        unsafe_allow_html=True,
    )

    _wl, _wr = st.columns([2, 3], gap="large")
    with _wl:
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.1em;margin-bottom:6px;">PROCESS PARAMETERS</div>',
            unsafe_allow_html=True,
        )
        _wi_temp     = st.slider("Temperature (degC)",      100,  300, 180,       key="wi_temp")
        _wi_pressure = st.slider("Pressure (bar)",          1.0, 10.0,  5.0, 0.1, key="wi_pressure")
        _wi_speed    = st.slider("Speed (rpm)",              50,  300, 150,       key="wi_speed")
        _wi_feed     = st.slider("Feed Rate (kg/h)",         5.0, 50.0, 20.0, 0.5,key="wi_feed")
        _wi_humidity = st.slider("Humidity (%)",             20,   80,  45,       key="wi_humidity")
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.1em;margin:10px 0 6px 0;">MATERIAL PROPERTIES</div>',
            unsafe_allow_html=True,
        )
        _wi_density  = st.slider("Density (g/cm3)",         0.8,  3.5,  2.0, 0.1, key="wi_density")
        _wi_hardness = st.slider("Hardness (HRC)",           10,  100,  55,       key="wi_hardness")
        _wi_grade    = st.slider("Grade (1-10)",              1,   10,   5,       key="wi_grade")
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.1em;margin:10px 0 6px 0;">ENVIRONMENT</div>',
            unsafe_allow_html=True,
        )
        _wi_carbon = st.slider("Carbon Intensity (gCO2/kWh)", 0, 600, 220, key="wi_carbon")
        st.markdown("<br>", unsafe_allow_html=True)
        _wi_run = st.button("&#9654; Run Full Pipeline Simulation",
                            key="wi_run", use_container_width=True)

    with _wr:
        if _wi_run:
            _wi_stat = st.empty()
            _wi_prog = st.progress(0)
            _wi_out  = st.empty()
            try:
                import pickle, torch
                from src.energy_dna.model import LSTMAutoencoder
                from config.settings import (
                    MODELS_DIR, ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
                    ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS,
                )

                _wi_stat.markdown(
                    '<div style="color:#00d4ff;font-size:0.82rem;">'
                    '&#9881; Stage 1/5 &#8212; Generating energy signal...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(12)
                time.sleep(0.3)
                _rng   = np.random.default_rng(seed=int(_wi_temp * 100 + _wi_speed))
                _base  = 50 + (_wi_temp/300)*80 + (_wi_speed/300)*50 + (_wi_feed/50)*20
                _sig   = _base + 10*np.sin(np.linspace(0, 4*np.pi, 128)) + _rng.normal(0, 5, 128)

                _wi_stat.markdown(
                    '<div style="color:#00d4ff;font-size:0.82rem;">'
                    '&#9889; Stage 2/5 &#8212; Encoding Energy DNA via LSTM Autoencoder...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(30)
                time.sleep(0.4)
                _sm, _ss = _sig.mean(), max(float(_sig.std()), 1e-9)
                _snorm   = (_sig - _sm) / _ss

                _ae_path = os.path.join(MODELS_DIR, "lstm_autoencoder.pth")
                if os.path.exists(_ae_path):
                    _ae = LSTMAutoencoder(ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM,
                                         ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS)
                    _ae.load_state_dict(torch.load(_ae_path, map_location="cpu", weights_only=True))
                    _ae.eval()
                    with torch.no_grad():
                        _xt = torch.tensor(_snorm, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                        _rt, _lt = _ae(_xt)
                        _emb_wi  = _lt.squeeze(0).numpy()
                        _recon_wi = float(torch.mean((_rt - _xt) ** 2).item())
                else:
                    _emb_wi = np.zeros(16, dtype=np.float32); _recon_wi = 0.0

                _wi_stat.markdown(
                    '<div style="color:#00d4ff;font-size:0.82rem;">'
                    '&#129516; Stage 3/5 &#8212; Assembling Batch Genome (25D)...</div>',
                    unsafe_allow_html=True,
                )
                _wi_prog.progress(52)
                time.sleep(0.35)
                _genome_wi = np.array(
                    [float(_wi_temp), float(_wi_pressure), float(_wi_speed),
                     float(_wi_feed), float(_wi_humidity),
                     float(_wi_density), float(_wi_hardness), float(_wi_grade),
                     *_emb_wi.tolist(), float(_wi_carbon)],
                    dtype=np.float32,
                )
            except Exception as e:
                st.error(f"Pipeline simulation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ESP32 REAL-TIME
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown(
        '<div class="acmgs-header"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        '<div>'
        '<h1>📡 ESP32 Real-Time Monitor</h1>'
        '<p>Live Current & Temperature Data Streaming from IoT Sensors</p>'
        '<div>'
        '<span class="hbadge">📡 IoT Device</span>'
        '<span class="hbadge hbadge-green">● WebSocket</span>'
        '<span class="hbadge">Real-Time </span>'
        '</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for ESP32 data
    if 'esp32_data' not in st.session_state:
        st.session_state.esp32_data = deque(maxlen=300)  # Last 300 readings
    if 'esp32_latest' not in st.session_state:
        st.session_state.esp32_latest = {
            'temperature': 0.0,
            'humidity': 0.0,
            'current': 0.0,
            'power_watts': 0.0,
            'timestamp': datetime.now().isoformat()
        }

    # ─── Connection Status & Settings ─────────────────────────────────
    st.markdown('<div class="slabel">Connection Settings</div>', unsafe_allow_html=True)
    
    col_host, col_port, col_connect = st.columns([2, 1, 1])
    with col_host:
        esp32_host = st.text_input("ESP32 Server Host", value="localhost", label_visibility="collapsed")
    with col_port:
        esp32_port = st.number_input("Port", value=8001, min_value=1, max_value=65535, label_visibility="collapsed")
    with col_connect:
        esp32_url = f"http://{esp32_host}:{esp32_port}"
        if st.button("🔌 Connect", use_container_width=True):
            try:
                resp = requests.get(f"{esp32_url}/api/health", timeout=3)
                if resp.status_code == 200:
                    st.success("✓ Connected to ESP32 Server")
                else:
                    st.error(f"Server returned {resp.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Live Gauges (Current, Temperature, Humidity) ─────────────────
    st.markdown('<div class="slabel">⚡ Live Sensor Gauges</div>', unsafe_allow_html=True)
    
    try:
        # Fetch latest data from API
        latest_resp = requests.get(f"{esp32_url}/api/latest", timeout=3)
        if latest_resp.status_code == 200:
            latest = latest_resp.json()
            st.session_state.esp32_latest = latest
            
            g1, g2, g3, g4 = st.columns(4)
            
            # Current Gauge
            with g1:
                fig_current = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest.get('current', 0),
                    title={
                        "text": "Current<br><span style='font-size:0.75em;color:rgba(255,255,255,0.35);'>Amperes</span>",
                        "font": {"size": 12, "color": "rgba(255,255,255,0.5)"},
                    },
                    number={"font": {"size": 36, "color": _CYAN, "family": "JetBrains Mono,monospace"}},
                    gauge={
                        "axis": {"range": [0, 150], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)",
                                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 8}, "nticks": 6},
                        "bar": {"color": _CYAN, "thickness": 0.2},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "borderwidth": 1, "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(0,255,136,0.12)"},
                            {"range": [50, 100], "color": "rgba(255,214,0,0.10)"},
                            {"range": [100, 150], "color": "rgba(255,75,75,0.13)"},
                        ],
                    },
                ))
                fig_current.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
                                         height=240, margin=dict(l=20, r=20, t=30, b=8))
                st.plotly_chart(fig_current, use_container_width=True, config={"displayModeBar": False})
            
            # Temperature Gauge
            with g2:
                fig_temp = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest.get('temperature', 0),
                    title={
                        "text": "Temperature<br><span style='font-size:0.75em;color:rgba(255,255,255,0.35);'>°C</span>",
                        "font": {"size": 12, "color": "rgba(255,255,255,0.5)"},
                    },
                    number={"font": {"size": 36, "color": _ORANGE, "family": "JetBrains Mono,monospace"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)",
                                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 8}, "nticks": 6},
                        "bar": {"color": _ORANGE, "thickness": 0.2},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "borderwidth": 1, "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0,255,136,0.12)"},
                            {"range": [30, 60], "color": "rgba(255,214,0,0.10)"},
                            {"range": [60, 100], "color": "rgba(255,75,75,0.13)"},
                        ],
                    },
                ))
                fig_temp.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
                                      height=240, margin=dict(l=20, r=20, t=30, b=8))
                st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})
            
            # Humidity Gauge
            with g3:
                fig_humidity = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest.get('humidity', 0),
                    title={
                        "text": "Humidity<br><span style='font-size:0.75em;color:rgba(255,255,255,0.35);'>%</span>",
                        "font": {"size": 12, "color": "rgba(255,255,255,0.5)"},
                    },
                    number={"font": {"size": 36, "color": _GREEN, "family": "JetBrains Mono,monospace"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)",
                                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 8}, "nticks": 6},
                        "bar": {"color": _GREEN, "thickness": 0.2},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "borderwidth": 1, "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(255,75,75,0.13)"},
                            {"range": [40, 70], "color": "rgba(255,214,0,0.10)"},
                            {"range": [70, 100], "color": "rgba(0,255,136,0.12)"},
                        ],
                    },
                ))
                fig_humidity.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
                                          height=240, margin=dict(l=20, r=20, t=30, b=8))
                st.plotly_chart(fig_humidity, use_container_width=True, config={"displayModeBar": False})
            
            # Power Gauge
            with g4:
                power = latest.get('power_watts', 0)
                fig_power = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=power,
                    title={
                        "text": "Power<br><span style='font-size:0.75em;color:rgba(255,255,255,0.35);'>Watts</span>",
                        "font": {"size": 12, "color": "rgba(255,255,255,0.5)"},
                    },
                    number={"font": {"size": 32, "color": _YELLOW, "family": "JetBrains Mono,monospace"}},
                    gauge={
                        "axis": {"range": [0, 50000], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)",
                                "tickfont": {"color": "rgba(255,255,255,0.35)", "size": 7}, "nticks": 5},
                        "bar": {"color": _YELLOW, "thickness": 0.2},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "borderwidth": 1, "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": [
                            {"range": [0, 20000], "color": "rgba(0,255,136,0.12)"},
                            {"range": [20000, 35000], "color": "rgba(255,214,0,0.10)"},
                            {"range": [35000, 50000], "color": "rgba(255,75,75,0.13)"},
                        ],
                    },
                ))
                fig_power.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.65)", family="Inter"),
                                       height=240, margin=dict(l=20, r=20, t=30, b=8))
                st.plotly_chart(fig_power, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Could not fetch ESP32 data: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Statistics Panel ─────────────────────────────────────────────
    st.markdown('<div class="slabel">📊 Statistics (Last 60 Readings)</div>', unsafe_allow_html=True)
    
    try:
        stats_resp = requests.get(f"{esp32_url}/api/stats?window=60", timeout=3)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            
            stat_cols = st.columns(5)
            
            with stat_cols[0]:
                st.metric(
                    "Avg Temperature",
                    f"{stats.get('temperature', {}).get('avg', 0):.1f}°C",
                    f"Min: {stats.get('temperature', {}).get('min', 0):.1f}°C"
                )
            
            with stat_cols[1]:
                st.metric(
                    "Avg Humidity",
                    f"{stats.get('humidity', {}).get('avg', 0):.1f}%",
                    f"Max: {stats.get('humidity', {}).get('max', 0):.1f}%"
                )
            
            with stat_cols[2]:
                st.metric(
                    "Avg Current",
                    f"{stats.get('current', {}).get('avg', 0):.1f}A",
                    f"Peak: {stats.get('current', {}).get('max', 0):.1f}A"
                )
            
            with stat_cols[3]:
                st.metric(
                    "Avg Power",
                    f"{stats.get('power_avg_watts', 0):.0f}W",
                    f"= {stats.get('power_avg_watts', 0) / 1000:.2f}kW"
                )
            
            with stat_cols[4]:
                st.metric(
                    "Total Current",
                    f"{stats.get('current', {}).get('total', 0):.1f}A·sec",
                    "Over window"
                )
    except Exception as e:
        st.warning(f"Could not fetch statistics: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Energy Prediction ────────────────────────────────────────────
    st.markdown('<div class="slabel">🔮 Energy Consumption Forecast</div>', unsafe_allow_html=True)
    
    try:
        pred_resp = requests.get(f"{esp32_url}/api/predict?duration_hours=1", timeout=3)
        if pred_resp.status_code == 200:
            pred = pred_resp.json()
            
            pred_cols = st.columns(4)
            
            with pred_cols[0]:
                st.metric(
                    "Predicted Energy (1h)",
                    f"{pred.get('predicted_energy_kwh', 0):.2f}kWh",
                    "Next hour"
                )
            
            with pred_cols[1]:
                st.metric(
                    "Carbon Impact (1h)",
                    f"{pred.get('predicted_carbon_kg', 0):.2f}kg CO₂",
                    "Estimated"
                )
            
            with pred_cols[2]:
                st.metric(
                    "Estimated Cost (1h)",
                    f"${pred.get('predicted_cost_usd', 0):.2f}",
                    "at $0.15/kWh"
                )
            
            with pred_cols[3]:
                st.metric(
                    "Avg Current",
                    f"{pred.get('assumptions', {}).get('avg_current_a', 0):.1f}A",
                    "@ 230V AC"
                )
    except Exception as e:
        st.warning(f"Could not fetch predictions: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Live Data Table ──────────────────────────────────────────────
    st.markdown('<div class="slabel">📈 Latest Readings (Real-Time)</div>', unsafe_allow_html=True)
    
    try:
        # Create a simple data update simulation
        if st.button("🔄 Refresh Latest Data", key="esp32_refresh"):
            st.rerun()
        
        # Display latest data in a nice format
        col_l1, col_l2 = st.columns(2)
        
        with col_l1:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px;padding:16px;">
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:10px;">Current Reading</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">Temperature</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{_ORANGE};
                                        font-family:JetBrains Mono,monospace;">
                                {st.session_state.esp32_latest.get('temperature', 0):.1f}°C
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">Current</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{_CYAN};
                                        font-family:JetBrains Mono,monospace;">
                                {st.session_state.esp32_latest.get('current', 0):.1f}A
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">Humidity</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{_GREEN};
                                        font-family:JetBrains Mono,monospace;">
                                {st.session_state.esp32_latest.get('humidity', 0):.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">Power</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{_YELLOW};
                                        font-family:JetBrains Mono,monospace;">
                                {st.session_state.esp32_latest.get('power_watts', 0):.0f}W
                            </div>
                        </div>
                    </div>
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.25);margin-top:10px;">
                        Updated: {st.session_state.esp32_latest.get('timestamp', 'N/A')[:19]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_l2:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px;padding:16px;">
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:10px;">Server Status</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">Connection</div>
                            <div style="font-size:1.1rem;font-weight:700;color:{_GREEN};">
                                ✓ Connected
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);">URL</div>
                            <div style="font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.6);
                                        font-family:JetBrains Mono,monospace;word-break:break-all;">
                                {esp32_url}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error displaying latest data: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — OPTIMIZATION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown(
        '<div class="acmgs-header"><div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        '<div>'
        '<h1>⚡ Optimization Insights</h1>'
        '<p>AI-Driven Recommendations Based on Real-Time ESP32 Data</p>'
        '<div>'
        '<span class="hbadge">🔍 Analysis</span>'
        '<span class="hbadge hbadge-green">💡 Dynamic Recommendations</span>'
        '<span class="hbadge">💰 ROI</span>'
        '</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Initialize recommender
    if "recommender" not in st.session_state:
        st.session_state.recommender = DynamicRecommender()
    
    recommender = st.session_state.recommender
    
    # Get latest ESP32 data — fetch directly from server if not in session
    esp32_latest = st.session_state.get('esp32_latest', {})
    
    # If no session data, try to fetch from ESP32 server
    if not esp32_latest:
        try:
            esp32_url = "http://localhost:8001"
            resp = requests.get(f"{esp32_url}/api/latest", timeout=3)
            if resp.status_code == 200:
                esp32_latest = resp.json()
                st.session_state.esp32_latest = esp32_latest
            else:
                st.warning(f"⚠️ ESP32 server not responding. Using default values. Check Tab 7 for status.")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch ESP32 data: {e}")
    
    current_a = float(esp32_latest.get('current', 50))
    temp_c = float(esp32_latest.get('temperature', 35))
    power_w = float(esp32_latest.get('power_watts', 11500))
    humidity_pct = float(esp32_latest.get('humidity', 55))

    # Update grid carbon intensity (default: 150 gCO2/kWh)
    recommender.carbon_intensity = 150

    # ─── SECTION 1: CURRENT STATE ANALYSIS ──────────────────────────────
    st.markdown('<div class="slabel">📊 Current System State Analysis</div>', unsafe_allow_html=True)
    
    # Display data source status
    data_source = "🔴 ESP32 Real-Time Data" if esp32_latest.get('timestamp') else "🟠 Default Values (No ESP32 Connection)"
    st.info(f"**Data Source:** {data_source}")

    col_state1, col_state2, col_state3, col_state4 = st.columns(4)

    with col_state1:
        st.metric(
            label="Current Draw",
            value=f"{current_a:.1f}A",
            delta="⚠️ High" if current_a > 80 else "✓ Optimal",
            delta_color="inverse" if current_a > 80 else "normal",
        )

    with col_state2:
        st.metric(
            label="Temperature",
            value=f"{temp_c:.1f}°C",
            delta="⚠️ Elevated" if temp_c > 40 else "✓ Normal",
            delta_color="inverse" if temp_c > 40 else "normal",
        )

    with col_state3:
        carbon_per_hour = (power_w / 1000) * recommender.carbon_intensity / 1000
        st.metric(
            label="Carbon/Hour",
            value=f"{carbon_per_hour:.3f}kg CO₂",
        )

    with col_state4:
        humidity_display = float(esp32_latest.get('humidity', 55))
        st.metric(
            label="Humidity",
            value=f"{humidity_display:.0f}%",
            delta="⚠️ High" if humidity_display > 70 else "✓ Optimal",
            delta_color="inverse" if humidity_display > 70 else "normal",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── SECTION 2: GENERATE DYNAMIC RECOMMENDATIONS ─────────────────────
    st.markdown('<div class="slabel">💡 AI-Generated Optimization Opportunities</div>', unsafe_allow_html=True)

    # Generate recommendations based on current ESP32 data
    recommendations = recommender.get_recommendations(
        current_a=current_a,
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        power_w=power_w
    )

    if not recommendations:
        st.info("✓ System operating within optimal parameters. No immediate recommendations.")
    else:
        for idx, rec in enumerate(recommendations, 1):
            # Color code confidence levels
            confidence_colors = {
                "HIGH": "#00ff88",
                "MEDIUM": "#ffd600",
                "LOW": "#ff6b6b"
            }
            confidence_color = confidence_colors.get(rec.confidence, "#ffd600")
            confidence_emoji = {
                "HIGH": "✅",
                "MEDIUM": "⚠️",
                "LOW": "ℹ️"
            }.get(rec.confidence, "ℹ️")

            with st.expander(
                f"{idx}. {rec.title} — {confidence_emoji} {rec.confidence} Confidence",
                expanded=(idx == 1)
            ):
                col_rec1, col_rec2 = st.columns([2, 1])

                with col_rec1:
                    st.markdown(f"**{rec.description}**")
                    st.markdown(f"📋 **Implementation:** {rec.icon}")
                    st.markdown(f"⏱️ **Timeline:** {rec.timeline}")
                    
                    # Show triggering factors
                    st.markdown("**Identified Factors:**")
                    for factor in rec.factors:
                        st.markdown(f"  • {factor}")

                with col_rec2:
                    st.markdown("**Daily Impact:**", help="Estimated savings per day if recommendation is implemented")
                    
                    col_savings1, col_savings2, col_savings3 = st.columns(3)
                    
                    with col_savings1:
                        if rec.daily_savings["kwh"] > 0.01:
                            st.metric(
                                "Energy",
                                f"{rec.daily_savings['kwh']:.1f}",
                                "kWh",
                                label_visibility="collapsed"
                            )
                    
                    with col_savings2:
                        # Cost metric removed - industry-specific costs unknown
                        pass
                    
                    with col_savings3:
                        if rec.daily_savings["carbon"] > 0.001:
                            st.metric(
                                "Carbon",
                                f"{rec.daily_savings['carbon']:.2f}",
                                "kg",
                                label_visibility="collapsed"
                            )
                    
                    # Confidence indicator
                    st.markdown(f"<div style='background:rgba(255,255,255,0.04);border-left:3px solid {confidence_color};padding:8px 12px;border-radius:4px;margin-top:12px;font-size:0.85rem;'><strong>Confidence:</strong> {rec.confidence}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── SECTION 3: CUMULATIVE IMPACT & ROI ──────────────────────────────
    if recommendations:
        st.markdown('<div class="slabel">📈 Cumulative Impact & ROI</div>', unsafe_allow_html=True)

        cumulative = recommender.get_cumulative_impact(recommendations)
        
        col_roi1, col_roi2, col_roi3 = st.columns(3)

        with col_roi1:
            st.metric(
                label="Annual Energy Saved",
                value=f"{cumulative['annual_kwh']:.0f} kWh",
                delta=f"Daily: {cumulative['annual_kwh']/365:.1f} kWh",
            )

        with col_roi2:
            st.metric(
                label="Annual Carbon Reduced",
                value=f"{cumulative['annual_carbon']:.0f} kg CO₂",
                delta=f"Daily: {cumulative['annual_carbon']/365:.2f} kg",
            )

        with col_roi3:
            st.metric(
                label="Recommendations Count",
                value=f"{cumulative['num_recommendations']}",
                delta="Active optimizations",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ─── SECTION 4: PROJECTED PERFORMANCE ────────────────────────────
        st.markdown('<div class="slabel">🎯 Monthly Carbon Impact Projection</div>', unsafe_allow_html=True)

        months_list = []
        monthly_savings_carbon = cumulative['annual_carbon'] / 12
        monthly_current_carbon = power_w / 1000 * 730 * recommender.carbon_intensity / 1000
        
        for month in range(1, 13):
            months_list.append({
                "Month": f"M{month}",
                "Current Carbon": monthly_current_carbon,
                "Optimized Carbon": monthly_current_carbon - monthly_savings_carbon,
            })

        df_monthly = pd.DataFrame(months_list)

        fig_carbon = go.Figure()
        fig_carbon.add_trace(go.Scatter(
            x=df_monthly["Month"],
            y=df_monthly["Current Carbon"],
            mode="lines+markers",
            name="Current",
            line=dict(color="#ffa940", width=3),
        ))
        fig_carbon.add_trace(go.Scatter(
            x=df_monthly["Month"],
            y=df_monthly["Optimized Carbon"],
            mode="lines+markers",
            name="Optimized",
            line=dict(color="#177ddc", width=3),
            fill="tonexty",
        ))
        fig_carbon.update_layout(
            title="Monthly Carbon Projection (kg CO₂)",
            xaxis_title="Month",
            yaxis_title="Carbon Emissions (kg CO₂)",
            template="plotly_dark",
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_carbon, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ─── SECTION 5: EXECUTIVE SUMMARY ────────────────────────────────
        st.markdown('<div class="slabel">📋 Executive Summary</div>', unsafe_allow_html=True)

        summary_html = f"""
        <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.22);border-radius:14px;padding:20px;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
                <div style="text-align:center;">
                    <div style="font-size:1.2rem;font-weight:700;color:#00ff88;">💚 {cumulative['annual_carbon']:.0f}</div>
                    <div style="font-size:0.8rem;color:rgba(255,255,255,0.4);margin-top:4px;">kg CO₂ Annual Reduction</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.2rem;font-weight:700;color:#00d4ff;">⚡ {cumulative['annual_kwh']:.0f}</div>
                    <div style="font-size:0.8rem;color:rgba(255,255,255,0.4);margin-top:4px;">kWh Annual Savings</div>
                </div>
            </div>
            <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:16px;margin-bottom:16px;">
                <div style="font-size:0.85rem;color:rgba(255,255,255,0.7);line-height:1.6;">
                    ✓ <strong>{cumulative['num_recommendations']} actionable recommendations</strong> generated from real-time sensor data<br>
                    ✓ <strong>Reduce energy consumption by {cumulative['annual_kwh']:.0f} kWh/year</strong> through AI optimization<br>
                    ✓ <strong>Eliminate {cumulative['annual_carbon']:.0f} kg CO₂ annually</strong> with carbon-aware strategies<br>
                    ✓ <strong>Implement recommendations</strong> for maximum environmental impact
                </div>
            </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)
    else:
        st.success("✓ System is operating optimally across all parameters. Continue monitoring for changes.")


    st.markdown("<br>", unsafe_allow_html=True)

    # ─── SECTION 6: ALL OPTIMIZED BATCHES ──────────────────────────────
    st.markdown(
        '<div class="slabel">&#128225; All Optimized Batches</div>',
        unsafe_allow_html=True,
    )

    _fc1, _fc2 = st.columns([1, 9])
    with _fc1:
        if st.button("&#128260; Refresh", key="dt_feed_refresh"):
            st.cache_data.clear(); st.rerun()

    try:
        _fconn = sqlite3.connect(DB_PATH)
        # Get all recent batches with optimization data
        df_feed = pd.read_sql_query(
            """SELECT b.batch_id,
                      ROUND(b.yield,4)               AS yield,
                      ROUND(b.quality,4)             AS quality,
                      ROUND(b.energy_consumption,1)  AS energy_kwh,
                      ROUND(b.carbon_intensity,1)    AS carbon_ci,
                      ROUND(b.temperature,1)         AS temp_c,
                      ROUND(b.speed,0)               AS speed_rpm,
                      COALESCE(ee.is_anomaly,0)      AS is_anomaly,
                      COALESCE(ee.recon_error,0)     AS recon_error
               FROM batches b
               LEFT JOIN energy_embeddings ee ON b.batch_id = ee.batch_id
               ORDER BY b.batch_id DESC LIMIT 20""",
            _fconn,
        )
        _fconn.close()
    except Exception:
        df_feed = pd.DataFrame()

    if len(df_feed) > 0:
        df_feed["is_anomaly"] = (
            pd.to_numeric(df_feed["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )
        df_feed["recon_error"] = pd.to_numeric(df_feed["recon_error"], errors="coerce").round(5)
        df_feed["Zone"]   = df_feed["carbon_ci"].apply(classify_carbon_zone)
        df_feed["Status"] = df_feed["is_anomaly"].apply(
            lambda x: "ANOMALY" if x == 1 else "Normal"
        )
        df_feed["ML Verdict"] = df_feed.apply(
            lambda r: f"Err={r['recon_error']:.4f} {'[!]' if r['is_anomaly'] else '[ok]'}",
            axis=1,
        )
        _display = df_feed[[
            "batch_id", "yield", "quality", "energy_kwh",
            "carbon_ci", "temp_c", "speed_rpm", "Zone", "Status", "ML Verdict"
        ]].copy()
        _display.columns = [
            "Batch ID", "Yield", "Quality", "Energy (kWh)",
            "Carbon CI", "Temp (degC)", "Speed (rpm)", "Zone", "Status", "ML Verdict"
        ]
        st.dataframe(_display, use_container_width=True, hide_index=True, height=390)
        st.markdown(
            '<div style="font-size:0.7rem;color:rgba(255,255,255,0.25);margin-top:4px;">'
            'Last 20 batches, newest first. Click Refresh to reload from DB.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No batch feed data available.")
