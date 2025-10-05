# Fraud_fin.py — Tab-scoped, corrected
from pathlib import Path
import time
from datetime import datetime
import json
from uuid import uuid4
import streamlit.components.v1 as components

import base64
import io
import traceback
from urllib.parse import quote
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import streamlit as st
from settings_ui import apply_theme, render_settings_tab
# Optional plotting libs (used in helper charts)
import matplotlib.pyplot as plt
from auth import require_login, logout
# App modules
from database import FraudDetectionDB
# Apply styles on every run, but only rerun once

# from auth import apply_styles
# apply_styles()

# # Add this CSS to prevent flash
# st.markdown("""
# <style>
#     [data-testid="stAppViewContainer"] {
#         opacity: 1 !important;
#         transition: none !important;
#     }
#     .stApp {
#         animation: none !important;
#     }
# </style>
# """, unsafe_allow_html=True)
apply_theme()
from sqlalchemy import text
from database import engine
with engine.connect() as conn:
    dialect = conn.dialect.name
    version = conn.exec_driver_sql("SELECT VERSION()").scalar()
    print(f"DB OK → dialect={dialect} version={version}")

db = FraudDetectionDB()
user = require_login()  # ← blocks until logged in


# ---------- Page config ----------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply theme first


# ---------- Header (title + user + logout + former-hero content) ----------
# ---------- Clean, readable header ----------
def logout():
    """Handle logout functionality"""
    # Clear session state or authentication tokens
    if 'user' in st.session_state:
        del st.session_state['user']
    st.rerun()  # Or st.experimental_rerun() for older Streamlit versions
# Global Quick Actions CSS (applies to anchor button)
st.markdown("""
<style>
  .mfds-btn, .mfds-btn-link{
    display:block; width:100%; height:48px; line-height:48px;
    text-align:center; border:none; border-radius:12px;
    background:#c4171b; color:#fff !important; font-weight:700;
    box-shadow:0 8px 20px rgba(0,0,0,.10);
    cursor:pointer; text-decoration:none !important;
    transition:transform .12s ease, background .12s ease, box-shadow .12s ease;
  }
  .mfds-btn:hover, .mfds-btn-link:hover{
    transform:translateY(-1px); background:#9f1417;
    box-shadow:0 10px 24px rgba(0,0,0,.15);
  }
</style>
""", unsafe_allow_html=True)

def render_header(user):
    # 1) Inject styles
    st.markdown("""
    <style>
      :root{ --mfds-red:#c4171b; --mfds-red-dark:#9f1417; --mfds-red-verydark:#7f0f12; }
      .mfds-header-wrap{ width:100%; margin: 6px 0 18px 0; }
      .mfds-header{
        max-width: 1200px; margin: 0 auto; border-radius: 16px; padding: 20px 22px;
        background: linear-gradient(135deg,var(--mfds-red) 0%, var(--mfds-red-dark) 60%, var(--mfds-red-verydark) 100%);
        color:#fff; box-shadow: 0 12px 28px rgba(0,0,0,.18);
      }
      .mfds-top{ display:flex; align-items:center; justify-content:space-between; gap:14px; }
      .mfds-title{ margin:0; line-height:1.12; font-size:clamp(1.6rem, 2.2vw, 2.2rem); font-weight:800; letter-spacing:.2px; }
      .mfds-actions{ display:flex; align-items:center; gap:10px; }
      .mfds-badge{
        padding:6px 12px; border-radius:999px; font-weight:600;
        background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
        color:#fff; white-space:nowrap;
      }
      .mfds-logout button{
        border-radius:999px !important; padding:8px 14px !important; font-weight:700 !important;
        background:#ffffff !important; color:var(--mfds-red) !important; border:none !important;
      }
      .mfds-sub{ margin:8px 2px 10px 2px; font-size:clamp(.98rem, 1.2vw, 1.06rem); opacity:.98; }
      .mfds-chips{ display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 10px 0; }
      .mfds-chip{
        display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
        background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22); font-size:.86rem;
      }
      .mfds-chip svg{ width:16px; height:16px; }
      .mfds-howto{
        background:#fff; color:#222; border-radius:12px; padding:14px 16px; margin-top:6px;
        border:1px solid #f0d3d4; box-shadow: 0 4px 12px rgba(0,0,0,.06);
      }
      .mfds-howto h3{ margin:0 0 6px 0; color:var(--mfds-red-verydark); font-weight:800; font-size:1.05rem; }
      .mfds-howto ol{ margin:.25rem 0 0 1.1rem; padding:0; }
      .mfds-howto li{ margin:.25rem 0; }
      @media (max-width: 720px){
        .mfds-top{ flex-direction:column; align-items:flex-start; gap:8px; }
        .mfds-actions{ width:100%; justify-content:flex-start; }
      }
    </style>
    """, unsafe_allow_html=True)
    

    # 2) Render HTML
    st.markdown(f"""
    <div class="mfds-header-wrap">
      <section class="mfds-header">
        <div class="mfds-top">
          <h1 class="mfds-title">Minet Fraud Detection System</h1>
          <div class="mfds-actions">
            <div class="mfds-badge">{user.get('name','User')} ({user.get('role','user')})</div>
            <div class="mfds-logout"></div>
          </div>
        </div>

<div class="mfds-sub">Advanced AI-powered analytics for detecting suspicious claims in real time.</div>

<div class="mfds-chips" aria-hidden="true">
    <span class="mfds-chip">
        <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 3l7 3v6c0 4.418-3.134 8.418-7 9-3.866-.582-7-4.582-7-9V6l7-3z" stroke="white" stroke-opacity="0.95" stroke-width="1.3"/>
        </svg>
        Real-time flags
    </span>
    <span class="mfds-chip">
        <svg viewBox="0 0 24 24" fill="none">
            <path d="M4 19v-6m6 6V7m6 12V11m4 8H2" stroke="white" stroke-opacity="0.95" stroke-width="1.3" stroke-linecap="round"/>
        </svg>
        Risk scoring
    </span>
    <span class="mfds-chip">
        <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 3v10m0 0l-3-3m3 3l3-3M4 17v2h16v-2" stroke="white" stroke-opacity="0.95" stroke-width="1.3" stroke-linecap="round"/>
        </svg>
        Exportable
    </span>
</div>

<div class="mfds-howto">
    <h3>How to use</h3>
    <ol>
        <li>Upload a CSV or Excel file containing claims data</li>
        <li>Click <em>Process Data for Fraud Detection</em></li>
        <li>The system will automatically process and analyze your data</li>
        <li>Review the fraud detection results</li>
    </ol>
</div>

    """, unsafe_allow_html=True)

    # 3) Real logout button
    col1, col2, col3 = st.columns([1,1,1])
    with col3:
        if st.button("Logout", key="mfds_logout", use_container_width=True):
            logout()
# Add this to your render_header function's CSS section:
    st.markdown("""
<style>
    /* Improved claim drawer styling */
    .claim-drawer-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #c4171b;
    }
    
    /* Better spacing for metrics */
    .stMetric {
        margin-bottom: 10px;
    }
    
    /* Container borders for better visual separation */
    .stContainer {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)
# Usage
render_header(user)


# Add Minet logo
def add_minet_logo():
    candidates = [
        Path("assets/min1.png"), Path("assets/min1.jpg"), Path("assets/min1.jpeg"),
        Path.cwd() / "assets" / "min1.png", Path.cwd() / "assets" / "min1.jpg", Path.cwd() / "assets" / "min1.jpeg",
    ]
    logo_path = next((p for p in candidates if p.exists()), None)
    if not logo_path:
        return

    ext = logo_path.suffix.lower().lstrip(".")
    mime = "png" if ext == "png" else "jpeg"
    b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
          .minet-logo {{
            position: fixed !important;
            top: 12px !important;
            right: 20px !important;
            height: 40px !important;
            z-index: 2147483647 !important;
            pointer-events: none !important;
          }}
          @media (max-width: 640px) {{
            .minet-logo {{ right: 12px !important; top: 10px !important; height: 32px !important; }}
          }}
        </style>
        <img class="minet-logo" src="data:image/{mime};base64,{b64}" alt="Minet Logo" />
        """,
        unsafe_allow_html=True,
    )

add_minet_logo()


# Render improved header with logout button
def render_header():
    st.markdown(
        f"""
        <div class="minet-header">
            <div class="minet-header-title">Minet Fraud Detection System</div>
            <div class="minet-user-info">
                <div class="minet-user-badge">{user['name']} ({user['role']})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Logout button with custom styling
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        if st.button("Logout", key="logout_btn", use_container_width=True):
            logout()

# Render hero banner
def render_hero_banner():
    st.markdown(
        """
        <div class="minet-hero-wrap">
          <div class="minet-hero">
            <h1>Minet Fraud Detection Dashboard</h1>
            <p>Advanced AI-powered analytics for detecting suspicious claims in real time.</p>
            <div class="minet-chips" aria-hidden="true">
              <div class="minet-chip">
                <svg viewBox="0 0 24 24" fill="none"><path d="M12 3l7 3v6c0 4.418-3.134 8.418-7 9-3.866-.582-7-4.582-7-9V6l7-3z" stroke="white" stroke-opacity=".9" stroke-width="1.3"/></svg>
                Real-time flags
              </div>
              <div class="minet-chip">
                <svg viewBox="0 0 24 24" fill="none"><path d="M4 19v-6m6 6V7m6 12V11m4 8H2" stroke="white" stroke-opacity=".9" stroke-width="1.3" stroke-linecap="round"/></svg>
                Risk scoring
              </div>
              <div class="minet-chip">
                <svg viewBox="0 0 24 24" fill="none"><path d="M12 3v10m0 0l-3-3m3 3l3-3M4 17v2h16v-2" stroke="white" stroke-opacity=".9" stroke-width="1.3" stroke-linecap="round"/></svg>
                Exportable
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render the header and hero banner
#render_header()
#render_hero_banner()

# Import remaining modules
try:
    from historical_baselines import HistoricalBaselines
except Exception:
    HistoricalBaselines = None

import evaluation
from categorization import add_categorization_tab
from settings_ui import render_settings_tab

# ---------- Session defaults ----------
for key, default in [
    ('processed_data', None),
    ('scored_data', None),
    ('file_uploaded', False),
    ('current_threshold', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Model (frozen scorer) ----------
try:
    from score_runtime import StableAnomalyScorer
except ImportError:
    StableAnomalyScorer = None
    st.sidebar.error("Could not import StableAnomalyScorer. Ensure score_runtime.py is present.")

@st.cache_resource
def load_scorer():
    if StableAnomalyScorer is None:
        return None
    try:
        return StableAnomalyScorer(
            model_path="models/anomaly_v1.joblib",
            tuning_path="models/tuning_v1.json",
            ae_path="models/autoencoder_v1.pt",
        )
    except Exception as e:
        st.sidebar.error(f"Error loading scorer: {e}")
        return None

scorer = load_scorer()

_default_thr = 0.5
try:
    if scorer is not None:
        _default_thr = float(scorer.info().get("threshold", 0.5))
except Exception:
    pass

if st.session_state.get("current_threshold") is None:
    st.session_state["current_threshold"] = _default_thr

# ---------- DB / Baselines ----------
db = FraudDetectionDB() if FraudDetectionDB else None
baselines = HistoricalBaselines() if HistoricalBaselines else None

# ---------- Helpers (no Streamlit output here) ----------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    dfp = df.copy()
    for col in ['AILMENT_DATE', 'DATE_OF_BIRTH(CLAIMANT)']:
        if col in dfp.columns:
            dfp[col] = pd.to_datetime(dfp[col], errors='coerce', dayfirst=True)
    for col in ['TOTAL_PAYABLE', 'COVER_LIMIT', 'DAYS_SINCE_LAST_VISIT', 'AGE(CLAIMANT)']:
        if col in dfp.columns:
            dfp[col] = pd.to_numeric(dfp[col], errors='coerce')

    if {'TOTAL_PAYABLE', 'COVER_LIMIT'}.issubset(dfp.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            dfp['claim_ratio'] = (dfp['TOTAL_PAYABLE'] / dfp['COVER_LIMIT']).replace([np.inf, -np.inf], np.nan)
            dfp['high_claim_flag'] = (dfp['TOTAL_PAYABLE'] > 0.8 * dfp['COVER_LIMIT']).astype('Int64')

    if 'DAYS_SINCE_LAST_VISIT' in dfp.columns:
        dfp['frequent_visitor'] = (pd.to_numeric(dfp['DAYS_SINCE_LAST_VISIT'], errors='coerce') < 7).astype('Int64')

    return dfp

def _fmt_kes(x):
    try:
        return f"KES {float(x):,.0f}"
    except Exception:
        return str(x)
def _build_claim_report_text(selected_id, row_sel, C):
    """Plain-text report used for copy/email/print."""
    def _get(row, key, default='—'):
        return row.get(key, default) if key and key in row else default

    lines = [
        f"Claim Report — Visit ID: {selected_id}",
        "-" * 54,
        f"Provider:         {_get(row_sel, C['provider'])}",
        f"Service/Condition:{_get(row_sel, C['benefit'])}",
        f"Ailments:         {_get(row_sel, C['ailments'])}",
        f"Claim Amount:     {_fmt_kes(_get(row_sel, C['amount']))}",
        f"Risk Level:       {_get(row_sel, C['risk'])}",
        f"Fraud Score:      {float(_get(row_sel, C['fraud_score'], 0)):.2f}",
        "",
        "Detection Notes:",
        str(row_sel.get('Reason', '—')),
        "",
        "Recommended Actions:",
        str(row_sel.get('Action', '').replace('Action: ', '') or '—'),
        ""
    ]
    return "\n".join(lines)


# ---------- Quick Actions (uniform styling) ----------

# ---------- Quick Actions (uniform red buttons) ----------
import json
from uuid import uuid4
from urllib.parse import quote
import streamlit.components.v1 as components
import streamlit as st

_IFRAME_BTN_CSS = """
<style>
  .mfds-btn{
    display:block; width:100%; height:48px;
    border:none; border-radius:12px;
    background:#c4171b; color:#fff; font-weight:700;
    box-shadow:0 8px 20px rgba(0,0,0,.10);
    cursor:pointer;
    transition:transform .12s ease, background .12s ease, box-shadow .12s ease;
  }
  .mfds-btn:hover{
    transform:translateY(-1px); background:#9f1417;
    box-shadow:0 10px 24px rgba(0,0,0,.15);
  }
</style>
"""

def copy_button(text: str, label: str = "Copy Details"):
    btn_id = f"mfds_copy_{uuid4().hex}"
    payload = json.dumps(text)
    components.html(
        f"""
        {_IFRAME_BTN_CSS}
        <button id="{btn_id}" class="mfds-btn">{label}</button>
        <script>
          (function(){{
            const btn = document.getElementById("{btn_id}");
            const txt = {payload};
            btn.addEventListener('click', async () => {{
              try {{ await navigator.clipboard.writeText(txt); }}
              catch (e) {{
                const ta = document.createElement('textarea');
                ta.value = txt; document.body.appendChild(ta);
                ta.select(); document.execCommand('copy'); ta.remove();
              }}
              const old = btn.textContent;
              btn.textContent = "✓ Copied";
              btn.style.background = "#16a34a";
              setTimeout(() => {{ btn.textContent = old; btn.style.background = "#c4171b"; }}, 1600);
            }});
          }})();
        </script>
        """,
        height=60,
    )

def email_button(subject: str, body: str, label: str = "Email Report"):
    btn_id = f"qa_email_{uuid4().hex}"
    subject_js = json.dumps(subject)
    body_js = json.dumps(body)
    
    components.html(f"""
    <div style="width:100%">
        <style>
        .mfds-btn {{
            display: block; 
            width: 100%; 
            height: 48px;
            border: none; 
            border-radius: 12px;
            background: #c4171b; 
            color: #fff; 
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(0,0,0,.10);
            cursor: pointer;
            transition: transform .12s ease, background .12s ease, box-shadow .12s ease;
            font-family: inherit;
            font-size: 14px;
            text-decoration: none !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .mfds-btn:hover {{
            transform: translateY(-1px); 
            background: #9f1417;
            box-shadow: 0 10px 24px rgba(0,0,0,.15);
        }}
        </style>
        <button id="{btn_id}" class="mfds-btn">{label}</button>
        <script>
        (function(){{
            const subject = {subject_js};
            const body = {body_js};
            const btn = document.getElementById("{btn_id}");
            btn.addEventListener('click', () => {{
                const mailtoLink = `mailto:claims@minet.com?subject=${{encodeURIComponent(subject)}}&body=${{encodeURIComponent(body)}}`;
                window.open(mailtoLink, '_blank');
            }});
        }})();
        </script>
    </div>
    """, height=54)

def print_button(report_text: str, title: str, label: str = "Print View"):
    btn_id = f"qa_print_{uuid4().hex}"
    txt_js = json.dumps(report_text)
    title_js = json.dumps(title)
    
    components.html(f"""
    <div style="width:100%">
        <style>
        .mfds-btn {{
            display: block; 
            width: 100%; 
            height: 48px;
            border: none; 
            border-radius: 12px;
            background: #c4171b; 
            color: #fff; 
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(0,0,0,.10);
            cursor: pointer;
            transition: transform .12s ease, background .12s ease, box-shadow .12s ease;
            font-family: inherit;
            font-size: 14px;
        }}
        .mfds-btn:hover {{
            transform: translateY(-1px); 
            background: #9f1417;
            box-shadow: 0 10px 24px rgba(0,0,0,.15);
        }}
        </style>
        <button id="{btn_id}" class="mfds-btn">{label}</button>
        <script>
        (function(){{
            const txt = {txt_js};
            const ttl = {title_js};
            const btn = document.getElementById("{btn_id}");
            btn.addEventListener('click', () => {{
                const win = window.open('', '_blank');
                if(!win) {{ alert('Please allow popups for print preview.'); return; }}
                const html = `
                    <html>
                        <head><title>${{ttl}}</title></head>
                        <body>
                            <h1>${{ttl}}</h1>
                            <pre>${{txt}}</pre>
                            <script>window.print()<\/script>
                        </body>
                    </html>`;
                win.document.open();
                win.document.write(html);
                win.document.close();
            }});
        }})();
        </script>
    </div>
    """, height=54)





# ---------- Results Suite ----------
def display_results(final_results: pd.DataFrame, scored: pd.DataFrame, processing_time: float):
    st.markdown('<h2 class="sub-header">Fraud Detection Results</h2>', unsafe_allow_html=True)

    if 'selected_claim_id' not in st.session_state:
        st.session_state.selected_claim_id = None
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'filtered_view' not in st.session_state:
        st.session_state.filtered_view = None

    c1, c2, c3, c4 = st.columns(4)
    total_claims = len(final_results)
    flagged_claims = int(pd.to_numeric(final_results.get('needs_review', 0), errors='coerce').fillna(0).sum())

    with c1:
        st.metric("Total Claims", total_claims)
    with c2:
        st.metric("Needs Review", flagged_claims)
    with c3:
        rate = (flagged_claims / total_claims * 100) if total_claims else 0.0
        st.metric("Flag Rate", f"{rate:.2f}%")

    band_high = getattr(scorer, "tuning", {}).get("band_high_cut", None) if scorer else None
    thr = (st.session_state.current_threshold
           if st.session_state.current_threshold is not None
           else (scorer.info()['threshold'] if scorer else 0.5))

    def _risk(s):
        if band_high is not None and s >= band_high:
            return "Very High"
        return "High" if s >= thr else "Low"

    final_results["risk_level"] = final_results["combined_anomaly_score"].apply(_risk)
    final_results["fraud_score"] = final_results["combined_anomaly_score"]
    final_results["fraud_prediction"] = final_results.get("needs_review", 0).astype(int)

    display_columns = [c for c in [
        'VISIT_ID', 'GENDER(CLAIMANT)', 'AGE(CLAIMANT)', 'PROVIDER','AILMENTS',
        'TOTAL_PAYABLE', 'fraud_prediction', 'risk_level', 'fraud_score'
    ] if c in final_results.columns]

    st.dataframe(
        final_results[display_columns].sort_values(by='fraud_score', ascending=False).head(20),
        use_container_width=True
    )

    st.session_state.main_results = final_results.copy()

    st.markdown('<h3 class="sub-header">Results</h3>', unsafe_allow_html=True)

    def _find_col(df, candidates):
        cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().replace(" ", "").replace("_", "")
            if key in cols:
                return cols[key]
        return None

    C = {}
    C['visit_id']   = _find_col(final_results, ['VISIT_ID', 'visit_id', 'visit id'])
    C['provider']   = _find_col(final_results, ['PROVIDER', 'provider'])
    C['company']    = _find_col(final_results, ['COMPANY', 'company', 'employer'])
    C['relationship']= _find_col(final_results, ['RELATIONSHIP', 'relationship'])
    C['benefit']    = _find_col(final_results, ['BROAD_BENEFIT', 'benefit', 'AILMENTS', 'ailments'])
    C['ailments']   = _find_col(final_results, ['AILMENTS', 'ailments', 'DIAGNOSIS', 'diagnosis'])
    C['date']       = _find_col(final_results, ['AILMENT_DATE', 'SERVICE_DATE', 'VISIT_DATE', 'DATE'])
    C['amount']     = _find_col(final_results, ['TOTAL_PAYABLE', 'total_payable'])
    C['cover']      = _find_col(final_results, ['COVER_LIMIT', 'cover_limit'])
    C['days_gap']   = _find_col(final_results, ['DAYS_SINCE_LAST_VISIT'])
    C['freq']       = _find_col(final_results, ['FREQUENCY_OF_VISIT'])
    C['fraud_score']= _find_col(final_results, ['fraud_score'])
    C['fraud_pred'] = _find_col(final_results, ['fraud_prediction'])
    C['risk']       = _find_col(final_results, ['risk_level'])
    C['reason_existing'] = _find_col(final_results, ['reasons', 'reason'])

    final_results['_parsed_date'] = pd.to_datetime(final_results[C['date']], dayfirst=True, errors='coerce') if C['date'] else pd.NaT

    typ_gap = None
    if C['days_gap']:
        _g = final_results[C['days_gap']].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)
        if not _g.dropna().empty:
            typ_gap = float(_g.median())

    def _typical_cost(row_):
        if C['benefit'] and pd.notna(row_.get(C['benefit'], np.nan)):
            subset = final_results[final_results[C['benefit']] == row_[C['benefit']]]
            subcol = C['amount'] or 'TOTAL_PAYABLE'
            vals = pd.to_numeric(subset[subcol], errors='coerce')
            if not vals.dropna().empty:
                return float(vals.median())
        if C['amount']:
            return float(pd.to_numeric(final_results[C['amount']], errors='coerce').median())
        return np.nan

    def _pct_diff(a, b):
        try:
            a = float(a); b = float(b)
            return np.nan if b == 0 else (a - b) / b
        except Exception:
            return np.nan

    def compose_reason(row_, dataset=final_results):
        """Generate specific, detailed reasons like the reference script"""
        reasons = []
        
        # Define thresholds (you can adjust these based on your reference script)
        thr = {
            "provider_z_high": 3.0,
            "provider_z_very_high": 5.0,
            "recent_visit": 7.0,  # days
            "recent_same_provider": 7.0,  # days  
            "recent_ailment": 7.0,  # days
            "mismatch_many": 3.0,
            "combined_high": 0.7  # anomaly score threshold
        }
        
        def add_reason(txt):
            if txt and txt not in reasons:
                reasons.append(txt)

        # 1. Claim pattern anomaly
        if "claim_pattern_anomaly_score" in row_:
            score = pd.to_numeric(row_["claim_pattern_anomaly_score"], errors="coerce")
            if pd.notna(score) and score >= 1.0:  # High anomaly score
                add_reason("Unusual claim pattern compared to peers")

        # 2. Provider outlier detection
        if "provider_claim_zscore" in row_:
            z = pd.to_numeric(row_["provider_claim_zscore"], errors="coerce")
            if pd.notna(z):
                if z >= thr["provider_z_very_high"]:
                    add_reason("Provider appears as a strong outlier (very high claim amount)")
                elif z >= thr["provider_z_high"]:
                    add_reason("Provider appears as an outlier (high claim amount)")

        # 3. Visit frequency patterns
        if "days_since_last_visit" in row_:
            days = pd.to_numeric(row_["days_since_last_visit"], errors="coerce")
            if pd.notna(days) and days <= thr["recent_visit"]:
                if days == 0:
                    add_reason("Visited again very soon after a previous visit")
                else:
                    add_reason("Very frequent visits in a short period")

        # 4. Same provider repeat visits
        if "days_since_last_provider_visit" in row_:
            days = pd.to_numeric(row_["days_since_last_provider_visit"], errors="coerce")
            if pd.notna(days) and days <= thr["recent_same_provider"]:
                add_reason("Repeat visit to the same provider within a short time")

        # 5. New ailments soon after previous
        if "days_since_last_ailment" in row_:
            days = pd.to_numeric(row_["days_since_last_ailment"], errors="coerce")
            if pd.notna(days) and days <= thr["recent_ailment"]:
                add_reason("New ailment recorded unusually soon after a previous one")

        # 6. Data mismatches
        if "mismatch_score" in row_:
            m = pd.to_numeric(row_["mismatch_score"], errors="coerce")
            if pd.notna(m):
                if m >= thr["mismatch_many"]:
                    add_reason("Multiple data mismatches in claim details")
                elif m > 0:
                    add_reason("Some data mismatches in claim details")

        # 7. Provider risk indicators
        if "hospital_risk_score" in row_:
            h = pd.to_numeric(row_["hospital_risk_score"], errors="coerce")
            if pd.notna(h) and h >= 1.0:
                add_reason("Provider flagged with prior risk indicators")

        # 8. Location risk
        if "is_high_risk_location" in row_ and (row_["is_high_risk_location"] in [1, True, "1", "True", "true"]):
            add_reason("Location is known to be high risk")

        # 9. Company fraud history
        if "company_fraud_incident_flag" in row_ and (row_["company_fraud_incident_flag"] in [1, True, "1", "True", "true"]):
            add_reason("Company has prior fraud incidents")

        # 10. AI model flags
        if "autoencoder_anomaly_score" in row_:
            ae = pd.to_numeric(row_["autoencoder_anomaly_score"], errors="coerce")
            if pd.notna(ae) and ae >= thr["combined_high"]:
                add_reason("AI model flagged this claim as unusual")

        # 11. Overall anomaly score
        if "combined_anomaly_score" in row_:
            ca = pd.to_numeric(row_["combined_anomaly_score"], errors="coerce")
            if pd.notna(ca) and ca >= thr["combined_high"]:
                add_reason("Overall anomaly score is high")

        # 12. Fallback if no specific reasons found but claim is flagged
        if not reasons and row_.get('needs_review') == 1:
            add_reason("Ranked high by overall risk score")

        # Create actionable summary based on reasons
        actions = []
        if any("outlier" in reason.lower() for reason in reasons):
            actions.append("Review provider pricing and service justification")
        if any("frequent" in reason.lower() or "visit" in reason.lower() for reason in reasons):
            actions.append("Verify medical necessity for frequent visits")
        if any("mismatch" in reason.lower() for reason in reasons):
            actions.append("Cross-check claimant and service details")
        if any("risk" in reason.lower() for reason in reasons):
            actions.append("Enhanced verification required")
        
        if not actions:
            actions.append("Review provider documentation and patient eligibility")

        return "; ".join(reasons), "Recommended: " + "; ".join(actions)

    if st.session_state.filters_applied and st.session_state.filtered_view is not None:
        view = st.session_state.filtered_view
    else:
        view = final_results.copy()

    with st.form("filters_form", clear_on_submit=False):
        cA, cB, cC, cD, cE, cF = st.columns(6)
        flag_only = cA.toggle("Flagged only", value=True)
        risk_choices = sorted([x for x in final_results[C['risk']].dropna().unique()]) if C['risk'] else []
        risk_filter = cB.multiselect("Risk level", options=risk_choices, default=risk_choices if risk_choices else [])
        prov_choices = sorted([x for x in final_results[C['provider']].dropna().unique()]) if C['provider'] else []
        prov_filter = cC.multiselect("Provider", options=prov_choices)
        comp_choices = sorted([x for x in final_results[C['company']].dropna().unique()]) if C['company'] else []
        comp_filter = cD.multiselect("Company", options=comp_choices)
        rel_choices = sorted([x for x in final_results[C['relationship']].dropna().unique()]) if C['relationship'] else []
        rel_filter = cE.multiselect("Relationship", options=rel_choices)
        benef_choices = sorted([x for x in final_results[C['benefit']].dropna().unique()]) if C['benefit'] else []
        benef_filter = cF.multiselect("Service/Benefit", options=benef_choices)

        if C['date']:
            min_d = pd.to_datetime(final_results['_parsed_date']).min()
            max_d = pd.to_datetime(final_results['_parsed_date']).max()
            default_range = ()
            if pd.notna(min_d) and pd.notna(max_d):
                default_range = (min_d.date(), max_d.date())
            d1, d2 = st.date_input("Date range", value=default_range)
        else:
            d1, d2 = None, None

        sort_options = []
        if C['fraud_score']: sort_options.append('Fraud Score')
        if C['amount']:      sort_options.append('Claim Amount')
        if C['cover'] and C['amount']: sort_options.append('% of Coverage Limit')
        if C['days_gap']:    sort_options.append('Days since last visit')
        sort_by = st.selectbox("Sort by", options=sort_options or ['Fraud Score'])
        ascending = st.checkbox("Ascending", value=False)

        apply_filters = st.form_submit_button("Apply Filters")

        if apply_filters:
            view = final_results.copy()
            if flag_only and C['fraud_pred']:
                view = view[view[C['fraud_pred']] == 1]
            if C['risk'] and risk_filter:
                view = view[view[C['risk']].isin(risk_filter)]
            if C['provider'] and prov_filter:
                view = view[view[C['provider']].isin(prov_filter)]
            if C['company'] and comp_filter:
                view = view[view[C['company']].isin(comp_filter)]
            if C['relationship'] and rel_filter:
                view = view[view[C['relationship']].isin(rel_filter)]
            if C['benefit'] and benef_filter:
                view = view[view[C['benefit']].isin(benef_filter)]
            if C['date'] and d1 and d2:
                view = view[(view['_parsed_date'] >= pd.Timestamp(d1)) &
                            (view['_parsed_date'] <= pd.Timestamp(d2) + pd.Timedelta(days=1))]

            if C['amount'] and C['cover']:
                view['_pct_of_limit'] = pd.to_numeric(view[C['amount']], errors='coerce') / pd.to_numeric(view[C['cover']], errors='coerce')
            else:
                view['_pct_of_limit'] = np.nan
            if C['days_gap']:
                view['_days_gap_num'] = pd.to_numeric(view[C['days_gap']], errors='coerce')

            if sort_by == 'Fraud Score' and C['fraud_score']:
                view = view.sort_values(by=C['fraud_score'], ascending=ascending, na_position='last')
            elif sort_by == 'Claim Amount' and C['amount']:
                view = view.sort_values(by=C['amount'], ascending=ascending, na_position='last')
            elif sort_by == '% of Coverage Limit':
                view = view.sort_values(by='_pct_of_limit', ascending=ascending, na_position='last')
            elif sort_by == 'Days since last visit':
                view = view.sort_values(by='_days_gap_num', ascending=ascending, na_position='last')

            st.session_state.filtered_view = view.copy()
            st.session_state.filters_applied = True
            st.rerun()

    current_view = st.session_state.filtered_view if st.session_state.filters_applied else final_results

    max_rows = 500
    view_subset = current_view.head(max_rows).copy()
    reasons, actions = [], []
    for _, r in view_subset.iterrows():
        rs, ac = compose_reason(r, dataset=final_results)
        reasons.append(rs); actions.append(ac)
    view_subset['Reason'] = reasons
    view_subset['Action'] = actions

    show_cols = []
    for key in [C['visit_id'], C['provider'], C['benefit'], C['ailments'], C['amount'], C['risk'], C['fraud_score']]:
        if key: show_cols.append(key)
    show_cols += ['Reason']

    table_df = view_subset[show_cols].rename(columns={
        (C['visit_id'] or ''): 'Visit ID',
        (C['provider'] or ''): 'Provider',
        (C['benefit'] or ''): 'Service/Condition',
        (C['ailments'] or ''): 'Ailments/Diagnosis',
        (C['amount'] or ''): 'Claim Amount',
        (C['risk'] or ''): 'Risk Level',
        (C['fraud_score'] or ''): 'Fraud Score',
    }, errors='ignore')

    if 'Claim Amount' in table_df.columns:
        table_df['Claim Amount'] = table_df['Claim Amount'].apply(_fmt_kes)

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(table_df, use_container_width=True, height=520)

    with right:
    # Claim Drawer Header with better styling
      st.markdown("""
    <div style='
        background: linear-gradient(135deg, #c4171b 0%, #9f1417 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
    '>
        <h3 style='margin: 0; color: white; font-size: 1.3rem;'>Claim Details</h3>
        <p style='margin: 4px 0 0 0; opacity: 0.9; font-size: 0.9rem;'>Review and manage selected claim</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main drawer content with border
    st.markdown("""
    <div style='
        border: 2px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 20px;
        background: #fafafa;
        margin-bottom: 20px;
    '>
    """, unsafe_allow_html=True)
    
    # Use a unique key for the selectbox to maintain state
    selector_options = table_df['Visit ID'].tolist() if 'Visit ID' in table_df.columns else table_df.index.astype(str).tolist()
    
    # Initialize selected claim ID if not set
    if 'selected_claim_id' not in st.session_state or st.session_state.selected_claim_id not in selector_options:
        st.session_state.selected_claim_id = selector_options[0] if selector_options else None
    
    # Compact claim selector
    selected_id = st.selectbox(
        "**Select Claim to Review**", 
        options=selector_options,
        key="claim_selector_unique",
        index=selector_options.index(st.session_state.selected_claim_id) if st.session_state.selected_claim_id in selector_options else 0,
        help="Choose a claim to view detailed analysis"
    )
    
    # Update session state when selection changes
    if selected_id != st.session_state.selected_claim_id:
        st.session_state.selected_claim_id = selected_id
        st.rerun()

    if selected_id:
        # Find the selected row
        if 'Visit ID' in table_df.columns:
            sel_mask = view_subset[C['visit_id']] == selected_id
        else:
            sel_mask = view_subset.index.astype(str) == selected_id

        if sel_mask.any():
            row_sel = view_subset[sel_mask].iloc[0]
            
            # Summary Section with cards
            st.markdown("### Claim Summary")
            
            # Create summary cards
            col1, col2, col3 = st.columns(3)
            
            def _get_val(key, default='—'):
                return row_sel.get(key, default) if key and key in row_sel else default

            with col1:
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #c4171b;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    <div style='font-size: 0.8rem; color: #666;'>Provider</div>
                    <div style='font-weight: bold; font-size: 0.9rem;'>{_get_val(C['provider'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #c4171b;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    <div style='font-size: 0.8rem; color: #666;'>Service</div>
                    <div style='font-weight: bold; font-size: 0.9rem;'>{_get_val(C['benefit'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_val = _get_val(C['risk'])
                risk_color = "#dc3545" if risk_val == "High" else "#ffc107" if risk_val == "Medium" else "#28a745"
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid {risk_color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    <div style='font-size: 0.8rem; color: #666;'>Risk Level</div>
                    <div style='font-weight: bold; font-size: 0.9rem; color: {risk_color};'>{risk_val}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Amount and Score
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    <div style='font-size: 0.8rem; color: #666;'>Claim Amount</div>
                    <div style='font-weight: bold; font-size: 0.9rem; color: #007bff;'>{_fmt_kes(_get_val(C['amount']))}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                fraud_score = float(_get_val(C['fraud_score'], 0))
                score_color = "#dc3545" if fraud_score > 0.7 else "#ffc107" if fraud_score > 0.4 else "#28a745"
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid {score_color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                '>
                    <div style='font-size: 0.8rem; color: #666;'>Fraud Score</div>
                    <div style='font-weight: bold; font-size: 0.9rem; color: {score_color};'>{fraud_score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Reasons Section
            st.markdown("### Analysis & Reasons")
            st.markdown(f"""
            <div style='
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
            '>
                <div style='color: #856404; font-weight: bold; margin-bottom: 8px;'> Detection Notes</div>
                <div style='color: #856404;'>{row_sel['Reason']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Actionable Next Steps
            st.markdown("###  Recommended Actions")
            st.markdown(f"""
            <div style='
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
            '>
                <div style='color: #0c5460; font-weight: bold; margin-bottom: 8px;'> Next Steps</div>
                <div style='color: #0c5460;'>{row_sel['Action'].replace("Action: ", "")}</div>
            </div>
            """, unsafe_allow_html=True)

            # Assignment Section
            st.markdown("###  Assignment & Tracking")
            
            # Check existing assignment
            try:
                existing_assignment = db.get_assignment(str(selected_id)) if db else None
            except:
                existing_assignment = None
            
            if existing_assignment:
                st.markdown(f"""
                <div style='
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 16px;
                '>
                    <div style='color: #155724; font-weight: bold;'> Currently Assigned</div>
                    <div style='color: #155724;'>
                        <strong>Assignee:</strong> {existing_assignment['assignee']}<br>
                        <strong>Due Date:</strong> {existing_assignment['due_date']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(" Reassign Claim", key="reassign_btn", use_container_width=True):
                    st.session_state.reassign_mode = True
            
            # Assignment Form
            if not existing_assignment or st.session_state.get('reassign_mode', False):
                with st.form("assign_form", clear_on_submit=True):
                    st.markdown("#### Assign to Team Member")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        assignee = st.text_input(
                            "**Assignee Name**", 
                            value=existing_assignment['assignee'] if existing_assignment else "",
                            placeholder="Enter team member's name"
                        )
                    with col_b:
                        due_date = st.date_input(
                            "**Due Date**", 
                            value=pd.to_datetime(existing_assignment['due_date']) if existing_assignment else datetime.now(),
                            min_value=datetime.now()
                        )
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        assign_submit = st.form_submit_button(
                            " Assign Claim" if not existing_assignment else " Update Assignment",
                            use_container_width=True
                        )
                    with col_btn2:
                        if existing_assignment:
                            cancel_btn = st.form_submit_button("❌ Cancel", use_container_width=True)
                            if cancel_btn:
                                st.session_state.reassign_mode = False
                                st.rerun()

                    if assign_submit and assignee:
                        try:
                            if db:
                                db.add_assignment(
                                    visit_id=str(selected_id),
                                    assignee=assignee,
                                    due_date=due_date.strftime("%Y-%m-%d"),
                                    created_by_email=user["email"],
                                )
                                st.success(f"Successfully assigned to {assignee} (due {due_date})")
                                if 'reassign_mode' in st.session_state:
                                    st.session_state.reassign_mode = False
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save assignment: {e}")

            # Quick Actions
            # Quick Actions (functional)
            st.markdown("###  Quick Actions")

            report_text = _build_claim_report_text(selected_id, row_sel, C)
            subject     = f"Claim {selected_id} — Risk: {row_sel.get(C['risk'], '—')}"
            title       = f"Claim Report — {selected_id}"

            #a1, a2, a3 = st.columns([1, 1, 1], gap="large")
            a1, a2, a3 = st.columns(3)


            with a1:
              copy_button(report_text, label="Copy Details")

            with a2:
              email_button(subject, report_text, label="Email Report")

            with a3:
              print_button(report_text, title)


    st.markdown("</div>", unsafe_allow_html=True)  # Close the main drawer div

            # st.markdown("#### History")
            # try:
            #     history = db.get_claim_history(current_claim_id)
            #     if history:
            #         for h in history:
            #             st.write(f"{h['timestamp']}: {h['action']} — {h['details']}")
            #     else:
            #         st.info("No history for this claim.")
            # except Exception as e:
            #     st.error(f"Error loading history: {e}")

    st.markdown("---")
    st.markdown("### Export Results")
    export_format = st.radio("Export format", options=["CSV", "Excel"], horizontal=True)
    export_filename = st.text_input("Filename", value="fraud_results")
    if st.button("Export"):
        buffer = io.BytesIO()
        if export_format == "CSV":
            view_subset.to_csv(buffer, index=False)
            mime = "text/csv"
            ext = "csv"
        else:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                view_subset.to_excel(writer, sheet_name='Fraud Results', index=False)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ext = "xlsx"
        buffer.seek(0)
        st.download_button(
            label=f"Download {export_format}",
            data=buffer,
            file_name=f"{export_filename}.{ext}",
            mime=mime
        )

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Categorization", "Evaluation", "Settings", "About"
])

# Tab 1: Dashboard
with tab1:
    st.markdown('<h1 class="main-header">Fraud Detection Dashboard</h1>', unsafe_allow_html=True)

    # ---- Upload UI ----
    st.markdown("### Upload Claims Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Expected columns include claimant, claim, visit info (see preview)."
    )

    # ---- Format help (purely informational) ----
    with st.expander("Data Format Requirements"):
        st.markdown("""
### Required Columns:
Your data should include these columns (case-sensitive):

**Claimant Information:**
- `DATE_OF_BIRTH(CLAIMANT)`: Date of birth (format: DD/MM/YYYY)
- `AGE(CLAIMANT)`: Age of claimant (numeric)
- `GENDER(CLAIMANT)`: Gender (M/F)
- `RELATIONSHIP`: Relationship to main member (SELF, SPOUSE, CHILD)

**Claim Information:**
- `AILMENTS`: Medical condition
- `BENEFIT`: Type of benefit
- `PROVIDER`: Healthcare provider name
- `CLAIMANT_SUDDO`: Unique claimant identifier
- `BROAD_BENEFIT`: Broad benefit category
- `COMPANY`: Company name
- `COVER_LIMIT`: Coverage limit amount

**Visit Information:**
- `AILMENT_DATE`: Date of service (format: DD/MM/YYYY)
- `DAYS_SINCE_LAST_VISIT`: Days since last visit (numeric)
- `VISIT_ID`: Unique visit identifier
- `TOTAL_PAYABLE`: Claim amount (numeric)

**Additional Fields:**
- `FREQUENCY_OF_VISIT`: Frequency of visits (numeric)
- `MAIN_MEMBER_GENDER`: Gender of main member (M/F)
- `AGE(MAIN_MEMBER)`: Age of main member (numeric)
""")

        # Example CSV download
        example_columns = [
            'AILMENTS','BENEFIT','DATE_OF_BIRTH(CLAIMANT)','AGE(CLAIMANT)','GENDER(CLAIMANT)',
            'RELATIONSHIP','AILMENT_DATE','DAYS_SINCE_LAST_VISIT','DAY_OF_MONTH_VISITED',
            'MONTH_VISITED','YEAR_VISITED','PROVIDER','MEMBER_SUDDO','MAIN_MEMBER_GENDER',
            'AGE(MAIN_MEMBER)','CLAIMANT_SUDDO','BROAD_BENEFIT','COMPANY','COVER_LIMIT',
            'UNIQUE_VISIT','FREQUENCY_OF_VISIT','VISIT_ID','TOTAL_PAYABLE'
        ]
        example_df = pd.DataFrame({col: [''] for col in example_columns})

        def _download_csv(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="example_data.csv">Download Example Data (CSV)</a>'

        if st.button("See Preview", key="see_preview_template"):
            st.markdown(_download_csv(example_df), unsafe_allow_html=True)

    # ---------- EVERYTHING BELOW IS OUTSIDE THE EXPANDER ----------

    # If a new file arrives, reset cached results
    if uploaded_file is not None:
        if st.session_state.get("file_uploaded") != uploaded_file.name:
            st.session_state.processed_data = None
            st.session_state.scored_data = None
            st.session_state.final_results = None
            st.session_state.processing_time = 0.0
            st.session_state.file_uploaded = uploaded_file.name

        # small head preview (without heavy processing)
        df_head = None
        try:
            uploaded_file.seek(0)

            def _read_csv_robust(file_obj, nrows=None):
                # try common encodings used in Windows exports (fixes 0x92 smart-quote)
                for enc in ["utf-8", "utf-8-sig", "cp1252", "windows-1252", "latin-1", "ISO-8859-1"]:
                    try:
                        file_obj.seek(0)
                        return pd.read_csv(
                            file_obj, encoding=enc, on_bad_lines="skip",
                            low_memory=False, nrows=nrows
                        )
                    except Exception:
                        continue
                # last resort: ignore errors
                file_obj.seek(0)
                return pd.read_csv(
                    file_obj, encoding_errors="ignore",
                    on_bad_lines="skip", low_memory=False, nrows=nrows
                )

            if uploaded_file.name.lower().endswith(".csv"):
                df_head = _read_csv_robust(uploaded_file, nrows=5)
            else:
                df_head = pd.read_excel(uploaded_file, nrows=5, engine="openpyxl")
        except Exception:
            df_head = None

        st.markdown("### Data Preview")
        if df_head is not None:
            st.dataframe(df_head, use_container_width=True)
        else:
            st.info("Could not preview the file (format/encoding). You can still try processing it.")

        # Process button
        if st.button("Process Data for Fraud Detection", type="primary"):
            try:
                uploaded_file.seek(0)
                if uploaded_file.name.lower().endswith(".csv"):
                    df_raw = _read_csv_robust(uploaded_file)  # full read with encoding fallbacks
                else:
                    try:
                        df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
                    except Exception:
                        df_raw = pd.read_excel(uploaded_file, engine="xlrd")

                if df_raw is None or df_raw.empty:
                    st.error("Failed to read the uploaded file or the file is empty.")
                elif scorer is None:
                    st.error("Model not loaded.")
                else:
                    with st.spinner("Scoring with the frozen model..."):
                        start = time.time()
                        # Defensive normalization of incoming DF before sending to scorer
                        orig_cols = list(df_raw.columns)
                        df_raw = df_raw.copy()
                        df_raw.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('__', '_') for c in df_raw.columns]

                        # Optional debug mapping so you can verify canonicalization in the UI
                        # st.write("DEBUG: Original -> Canonical column mapping")
                        # for o, n in zip(orig_cols, df_raw.columns):
                        #     st.write(f"  '{o}' -> '{n}'")

                        # Now call the scorer using normalized df_raw
                        scored = scorer.score(df_raw)

                        
                        # DEBUG: Check what columns the scorer returns
                        # st.write(f"Scored data columns: {list(scored.columns)}")
                        st.write(f"Scored data shape: {scored.shape}")
                        st.write("First few rows of scored data:")
                        st.dataframe(scored.head(3))
                        df_aux = preprocess_data(df_raw.copy())
                        keep_aux = [c for c in df_aux.columns if c not in scored.columns]
                        final_results = pd.concat(
                            [scored.reset_index(drop=True),
                             df_aux[keep_aux].reset_index(drop=True)],
                            axis=1
                        )
                        st.session_state.processed_data = final_results
                        st.session_state.scored_data = scored
                        st.session_state.final_results = final_results
                        st.session_state.processing_time = time.time() - start
                        st.success(f"Processed {len(final_results)} rows.")
                        # Show debug output from score_runtime
                        # if 'debug_messages' in st.session_state and st.session_state.debug_messages:
                        #     with st.expander("🔍 Score Runtime Debug Output (Click to see what's happening)"):
                        #         for msg in st.session_state.debug_messages:
                        #             st.text(msg)

                        # # Also add this to check what columns we actually have in SCORED data
                        # st.write("### 🔍 Debug: Checking Engineered Features in SCORED Data")
                        # engineered_features_to_check = [
                        #     'claim_ratio', 'service_charge_pct_dev_by_ailment', 'repeat_claimant_count',
                        #     'age_rel_mismatch_flag', 'provider_claim_zscore', 'is_maternity_benefit'
                        # ]

                        # for feat in engineered_features_to_check:
                        #     if feat in scored.columns:  # Check SCORED data, not final_results
                        #         st.success(f"✅ Found {feat} in scored data")
                        #         st.write(f"Sample values: {scored[feat].head(3).tolist()}")
                        #     else:
                        #         st.error(f"❌ MISSING {feat} in scored data")

                        # # Check if explanations are working in SCORED data
                        # st.write("### 🔍 Debug: Checking Explanations in SCORED Data")
                        # if 'explanation' in scored.columns:
                        #     st.success("✅ Explanations column found in scored data")
                        #     st.write("Sample explanations:", scored['explanation'].head(3).tolist())
                        # else:
                        #     st.error("❌ No explanations column found in scored data")

                        # Clear debug messages for next run
                        st.session_state.debug_messages = []
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.code(traceback.format_exc())

    # Show results (either from this run or from cache)
    if st.session_state.get("final_results") is not None and st.session_state.get("scored_data") is not None:
        display_results(
            st.session_state.final_results,
            st.session_state.scored_data,
            st.session_state.get("processing_time", 0.0),
        )




# Tab 2: Categorization
with tab2:
    add_categorization_tab()

# Tab 3: Evaluation
with tab3:
      # Renders evaluation dashboard; it reads from session_state
     if user['role'] in ['manager', 'admin']:
        evaluation.evaluation_tab()
     else:
        st.warning("🔒 Access denied. Manager or Admin role required to view evaluation dashboard.")
        st.info("This section contains sensitive performance metrics and model evaluation data.")



# Tab 4: Settings
with tab4:
      render_settings_tab(scorer=scorer, db=db, baselines=baselines)


# Tab 5: About
with tab5:
    st.markdown("""
    ## About Minet Fraud Detection System
    
    This system uses advanced machine learning to detect potentially fraudulent insurance claims.
    
    **Features:**
    - Real-time fraud scoring
    - Historical baseline comparison
    - Provider categorization
    - Detailed claim analysis
    - Export capabilities
    
    **Version:** 1.0.0
    """)
