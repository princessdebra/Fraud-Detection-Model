# settings_ui.py
import streamlit as st
import pandas as pd
import datetime as _dt
import json
import streamlit as st
from sqlalchemy import text
from database import engine

try:
    with engine.connect() as c:
        ok = c.execute(text("SELECT 1")).scalar()
        st.sidebar.success(f"Database: {c.dialect.name} (OK)")
except Exception as e:
    st.sidebar.error(f"DB connection failed: {e}")


# ---------- THEME APPLIER (module-level) ----------
def apply_theme():
    ui_settings = st.session_state.get("settings", {}).get("ui", {})
    theme  = ui_settings.get("theme", "light")
    primary = ui_settings.get("primary_color", "#D71920")  # Minet Red

    # Base palette
    bg   = ui_settings.get("background_color_light", "#FFFFFF") if theme == "light" else ui_settings.get("background_color_dark", "#1E1E1E")
    text = ui_settings.get("text_color_light", "#000000")       if theme == "light" else ui_settings.get("text_color_dark", "#FFFFFF")

    # Common CSS (applies to both themes)
    css_common = f"""
    <style>
      body, .stApp {{
        background-color: {bg};
        color: {text};
      }}
      /* Accent the important bits with Minet red */
      h1, h2, h3 {{
        color: {primary} !important;
      }}
      .stButton>button {{
        background-color: {primary};
        color: white;
        border-radius: 8px;
      }}
      .stMetric, .stDataFrame {{
        border: 1px solid {primary}22;
        border-radius: 10px;
      }}
      /* Right-top logo holder (optional) */
      .minet-logo {{
        position: fixed; top: 16px; left: 18px; z-index: 1000;
      }}
    </style>
    """

    # Light-mode readability overrides (black text for tabs/labels/controls)
    css_light = f"""
<style>
  /* Tabs: default black; hover red; selected red with red underline */
  .stTabs [role="tab"] {{
    color: var(--settings-title, #111) !important;
    opacity: 1 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 120ms ease, border-color 120ms ease;
  }}
  /* make nested spans/p elements inherit */
  .stTabs [role="tab"] * {{
    color: inherit !important;
  }}
  .stTabs [role="tab"]:hover {{
    color: {primary} !important;
  }}
  .stTabs [role="tab"][aria-selected="true"] {{
    color: {primary} !important;
    border-bottom: 2px solid {primary} !important;
  }}

  /* Form/control labels remain black in light mode */
  label, .stMarkdown, .stCaption, .stFileUploader, .stSelectbox, 
  .stSlider, .stTextInput, .stNumberInput, .stRadio, .stCheckbox {{
    color: var(--settings-title, #111)!important;
  }}
  a, a:visited, .stMarkdown a {{
    color: var(--settings-title, #111) !important;
    text-decoration: underline;
  }}
  
  /* Labels, captions, help text */
  label,
  .stMarkdown, .stCaption,
  [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"] label,
  .stCheckbox label, .stRadio label, .stSelectbox label,
  .stNumberInput label, .stTextInput label, .stSlider label {{
    color: var(--settings-title, #111) !important;
    opacity: 1 !important;
  }}

  /* Radio/checkbox/switch option text */
  .stRadio [role="radio"] p,
  .stCheckbox p,
  .stSwitch p {{
    color: var(--settings-title, #111) !important;
    opacity: 1 !important;
  }}

  /* File uploader */
  [data-testid="stFileUploaderDropzone"] {{
    background-color: #F7F7F8 !important;
    border: 1px dashed #D0D0D0 !important;
    color: var(--settings-title, #111) !important;
  }}
  [data-testid="stFileUploaderDropzone"] * {{
    color: var(--settings-title, #111) !important;
    opacity: 1 !important;
  }}

  /* Inputs & placeholders */
  input, textarea, select {{
    color: var(--settings-title, #111) !important;
  }}
  input::placeholder, textarea::placeholder {{
    color: #666 !important;
    opacity: 1 !important;
  }}

  /* Expander headers */
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
  [data-testid="stExpander"] p,
  [data-testid="stExpander"] h4 {{
    color: var(--settings-title, #111) !important;
  }}

  /* Disabled states */
  [aria-disabled="true"], [data-disabled="true"] {{
    opacity: 0.65 !important;
    color: #666 !important;
  }}

  /* Links */
  a, a:visited, .stMarkdown a {{
    color: #000 !important;
    text-decoration: underline;
  }}
  /* 1) Labels/captions/help text -> solid black */
  label,
  .stMarkdown, .stCaption,
  [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"] label,
  .stCheckbox label, .stRadio label, .stSelectbox label,
  .stNumberInput label, .stTextInput label, .stSlider label {{
    color: #000 !important;
    opacity: 1 !important;
  }}

  /* 2) RADIO: make options readable & selected state obvious */
  .stRadio [role="radiogroup"] [role="radio"] {{
    color: #000 !important;
    border: 1px solid #D0D0D0 !important;
    border-radius: 14px !important;
    padding: 2px 10px !important;
    margin-right: 8px !important;
    background: #FFF !important;
  }}
  .stRadio [role="radiogroup"] [role="radio"] * {{
    color: inherit !important;
  }}
  .stRadio [role="radiogroup"] [role="radio"][aria-checked="true"] {{
    border-color: {primary} !important;
    color: {primary} !important;
    font-weight: 600 !important;
  }}

  /* 3) SWITCH/TOGGLE: higher contrast track & knob */
  [role="switch"] {{
    outline: 1px solid #CFCFCF !important;
    background: #F1F1F3 !important;
    border-radius: 14px !important;
  }}
  [role="switch"][aria-checked="true"] {{
    outline-color: {primary} !important;
    background: {primary}1A !important; /* light red track */
  }}
  /* Many Streamlit switches use an inner handle element; cover common cases */
  [role="switch"] * {{
    background: #808080 !important;
  }}
  [role="switch"][aria-checked="true"] * {{
    background: {primary} !important;
  }}

  /* 4) FILE UPLOADER: readable dropzone + light button */
  [data-testid="stFileUploaderDropzone"] {{
    background-color: #FAFAFB !important;
    border: 1px dashed #CFCFCF !important;
    color: #000 !important;
  }}
  [data-testid="stFileUploaderDropzone"] * {{
    color: #000 !important;
    opacity: 1 !important;
  }}
  /* “Browse files” button inside uploader */
  [data-testid="stFileUploaderDropzone"] button {{
    background: #FFFFFF !important;
    color: #000 !important;
    border: 1px solid #D0D0D0 !important;
    border-radius: 8px !important;
  }}
  [data-testid="stFileUploaderDropzone"] button:hover {{
    border-color: {primary} !important;
    color: {primary} !important;
  }}

  /* 5) DOWNLOAD BUTTON (st.download_button) – light variant */
  .stDownloadButton > button {{
    background: #FFFFFF !important;
    color: #000 !important;
    border: 1px solid #D0D0D0 !important;
    border-radius: 8px !important;
  }}
  .stDownloadButton > button:hover {{
    border-color: {primary} !important;
    color: {primary} !important;
  }}

  /* 6) Inputs & placeholders */
  input, textarea, select {{
    color: #000 !important;
  }}
  input::placeholder, textarea::placeholder {{
    color: #666 !important;
    opacity: 1 !important;
  }}

  /* 7) Expanders & subtle titles */
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
  [data-testid="stExpander"] p,
  [data-testid="stExpander"] h4 {{
    color: #000 !important;
  }}

  /* 8) Disabled states – readable */
  [aria-disabled="true"], [data-disabled="true"] {{
    opacity: 0.7 !important;
    color: #666 !important;
  }}

  /* 9) Links */
  a, a:visited, .stMarkdown a {{
    color: #000 !important;
    text-decoration: underline;
  }}
  /* === FIX: Radio labels (light/dark) unreadable in light mode === */
  /* Make all text inside the radio widget fully black and opaque */
  .stRadio, .stRadio * {{
    color: #000 !important;
    opacity: 1 !important;
  }}
  /* If Streamlit renders native inputs, ensure accent shows clearly */
  .stRadio input[type="radio"] {{
    accent-color: {primary};
  }}
  /* Give each option a subtle pill so it’s obvious on white */
  .stRadio [role="radiogroup"] [role="radio"] {{
    background: #FFFFFF !important;
    border: 1px solid #D0D0D0 !important;
    border-radius: 14px !important;
    padding: 2px 10px !important;
    margin-right: 8px !important;
  }}
  .stRadio [role="radiogroup"] [role="radio"][aria-checked="true"] {{
    border-color: {primary} !important;
    color: {primary} !important;
    font-weight: 600 !important;
  }}

  /* === FIX: Shadow Mode switch low contrast in light mode === */
  /* Cover both Streamlit’s current and old switch DOMs */
  [data-testid="stSwitch"] [role="switch"],
  .stSwitch [role="switch"],
  [role="switch"] {{
    background: #EDEFF3 !important;           /* light track */
    border: 1px solid #C9CDD3 !important;      /* define edges */
    border-radius: 14px !important;
    box-shadow: none !important;
  }}
  /* inner handle/children – make the knob visible on white */
  [data-testid="stSwitch"] [role="switch"] *,
  .stSwitch [role="switch"] *,
  [role="switch"] * {{
    background: #FFFFFF !important;            /* white knob */
    box-shadow: 0 0 0 1px #C9CDD3 inset !important;
  }}
  /* Checked state gets Minet red emphasis */
  [data-testid="stSwitch"] [role="switch"][aria-checked="true"],
  .stSwitch [role="switch"][aria-checked="true"],
  [role="switch"][aria-checked="true"] {{
    background: {primary}1A !important;        /* light red track */
    border-color: {primary} !important;
  }}
  [data-testid="stSwitch"] [role="switch"][aria-checked="true"] *,
  .stSwitch [role="switch"][aria-checked="true"] *,
  [role="switch"][aria-checked="true"] * {{
    background: {primary} !important;          /* red knob */
    box-shadow: none !important;
  }}
</style>
"""


    # Dark-mode small tweaks to keep contrast high
    css_dark = f"""
<style>
  /* Tabs: default white; hover red; selected red with red underline */
  .stTabs [role="tab"] {{
    color: #FFFFFF !important;
    opacity: 0.95 !important;
    border-bottom: 2px solid transparent !important;
    transition: color 120ms ease, border-color 120ms ease;
  }}
  .stTabs [role="tab"] * {{
    color: inherit !important;
  }}
  .stTabs [role="tab"]:hover {{
    color: {primary} !important;
  }}
  .stTabs [role="tab"][aria-selected="true"] {{
    color: {primary} !important;
    border-bottom: 2px solid {primary} !important;
  }}

  /* Form/control labels white in dark mode */
  label, .stMarkdown, .stCaption, .stFileUploader, .stSelectbox, 
  .stSlider, .stTextInput, .stNumberInput, .stRadio, .stCheckbox {{
    color: #FFFFFF !important;
  }}
  a, a:visited, .stMarkdown a {{
    color: #FFFFFF !important;
    text-decoration: underline;
  }}
</style>
"""


    # Inject CSS
    st.markdown(css_common, unsafe_allow_html=True)
    st.markdown(css_light if theme == "light" else css_dark, unsafe_allow_html=True)


# ---------- SETTINGS TAB ----------
def render_settings_tab(scorer=None, db=None, baselines=None):
    st.header("System Settings")
    st.caption("Configure model behavior, risk bands, rules, notifications, data retention, integrations, and UI.")

    # ---------- Initialize settings state ----------
    _default_settings = {
        "model": {
            "threshold": float(getattr(scorer, "info", lambda: {"threshold": 0.5})().get("threshold", 0.5)) if scorer else float(st.session_state.get("current_threshold", 0.5)),
            "target_flag_rate": 0.10,
            "very_high_band": 0.90,
            "high_band": 0.70,
            "explainability": True,
            "calibration_mode": "stable",
            "shadow_mode": False,
        },
        "rules": {
            "provider_rules": [],
            "watchlists": {"providers": [], "members": [], "diagnoses": []},
            "hard_stops": {"duplicate_claim_window_days": 2, "max_per_day_visits": 3},
        },
        # >>> UI must be top-level, not under "rules"
        "ui": {
            "theme": "light",  # light | dark
            "primary_color": "#D71920",
            "background_color_light": "#FFFFFF",
            "background_color_dark": "#1E1E1E",
            "text_color_light": "#000000",
            "text_color_dark": "#FFFFFF",
        },
        "notifications": {
            "email_enabled": True,
            "email_recipients": [],
            "slack_enabled": False,
            "slack_channel": "",
            "digest_frequency": "daily",
            "min_severity_to_notify": "High",
        },
        "data_privacy": {
            "data_retention_days": 365,
            "pii_masking": True,
            "export_redaction": True,
            "audit_logging": True
        },
        "integrations": {
            "webhook_enabled": False,
            "webhook_url": "",
            "siem_enabled": False,
            "siem_index": "fraud-events",
        },
        "advanced": {
            "sampling_rate": 1.0,
            "baseline_refresh_days": 30,
            "auto_retrain": False,
            "feature_drift_alert": True,
        },
        "_meta": {"updated_at": _dt.datetime.utcnow().isoformat(timespec="seconds")}
    }

    if "settings" not in st.session_state:
        st.session_state["settings"] = _default_settings

    # Make sure "ui" exists even if older JSON was imported
    st.session_state["settings"].setdefault("ui", _default_settings["ui"])
    settings = st.session_state["settings"]

    # ---------- Top-level actions ----------
    t1, t2, t3 = st.columns([1.5, 1, 1])
    with t1:
        st.subheader("Environment")
        st.toggle("Shadow Mode (safe trial)", key="shadow_mode_ui",
                  value=settings["model"]["shadow_mode"],
                  help="Run new settings in parallel without affecting reviewers.")
        settings["model"]["shadow_mode"] = st.session_state["shadow_mode_ui"]

    with t2:
        st.subheader("Import Settings")
        json_file = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed")
        if json_file is not None:
            try:
                imported = json.loads(json_file.read().decode("utf-8"))
                if isinstance(imported, dict) and "model" in imported and "rules" in imported:
                    st.session_state["settings"] = imported
                    st.session_state["settings"].setdefault("ui", _default_settings["ui"])
                    settings = st.session_state["settings"]
                    st.success("Settings imported.")
                else:
                    st.error("Invalid settings file: missing required sections.")
            except Exception as e:
                st.error(f"Failed to import: {e}")

    with t3:
        st.subheader("Export Settings")
        export_bytes = json.dumps(settings, indent=2).encode("utf-8")
        st.download_button("Download JSON", data=export_bytes, file_name="fraud_settings.json", mime="application/json")

    # ---------- Tabs ----------
    s1, s2 = st.tabs([
        "Model & Thresholds",
        "Theme & Branding",
    ])
       # ---------- Tabs ----------
   # s1, s2, s3, s4, s5, s6, s7, s8 = st.tabs([
    #    "Model & Thresholds",
     #   "Risk Rules & Watchlists",
      #  "Notifications",
       # "Data & Privacy",
        #"Integrations",
        #"Audit & Versioning",
        #"Advanced",
        #"Theme & Branding",
    #])

    # === Model & Thresholds ===
    with s1:
        st.subheader("Model Tuning")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            thr = st.slider("Detection Threshold", 0.0, 1.0, float(settings["model"]["threshold"]),
                            help="Higher = fewer flags (more conservative).")
            settings["model"]["threshold"] = float(thr)
        with c2:
            target_rate = st.number_input("Target Flag Rate", min_value=0.0, max_value=1.0,
                                          value=float(settings["model"]["target_flag_rate"]), step=0.01)
            settings["model"]["target_flag_rate"] = float(target_rate)
        with c3:
            mode = st.selectbox("Calibration Mode", ["stable", "adaptive"],
                                index=(["stable","adaptive"].index(settings["model"]["calibration_mode"])
                                       if settings["model"]["calibration_mode"] in ["stable","adaptive"] else 0))
            settings["model"]["calibration_mode"] = mode

        st.markdown("---")
        st.subheader("Risk Banding")
        b1, b2, _ = st.columns(3)
        with b1:
            vh = st.slider("Very High Band ≥", 0.5, 1.0, float(settings["model"]["very_high_band"]), 0.01)
        with b2:
            hi = st.slider("High Band ≥", 0.3, 0.99, float(settings["model"]["high_band"]), 0.01)
        if vh <= hi:
            st.warning("Very High band should be greater than High band. Adjusted automatically.")
            vh = max(vh, hi + 0.01)
        settings["model"]["very_high_band"] = float(round(vh, 3))
        settings["model"]["high_band"] = float(round(hi, 3))

        st.markdown("---")
        st.subheader("Explainability & Preview")
        df = st.session_state.get("processed_data")
        if df is not None and "combined_anomaly_score" in df.columns:
            thr_use = settings["model"]["threshold"]
            flagged = int((df["combined_anomaly_score"] >= thr_use).sum())
            total = len(df)
            rate = (flagged/total*100) if total else 0.0
            st.info(f"**Preview** at threshold {thr_use:.3f}: {flagged:,} of {total:,} claims flagged ({rate:.2f}%).")
        else:
            st.caption("Upload and process data in the Detection tab to preview impact here.")

    # === UI & Branding ===
    with s2:
        st.subheader("Theme & Branding")
        theme_choice = st.radio("Select Theme", ["light", "dark"],
                                index=(["light","dark"].index(settings["ui"].get("theme","light"))
                                       if settings["ui"].get("theme") in ["light","dark"] else 0),
                                horizontal=True)
        settings["ui"]["theme"] = theme_choice

        color = st.color_picker("Primary Color (Minet Red)", settings["ui"].get("primary_color", "#D71920"), key="ui_primary_color")
        settings["ui"]["primary_color"] = color
        st.caption("Primary color is used for buttons, headers, borders and highlights.")

        st.markdown("—")

    # ---------- Save / Apply ----------
    left, right = st.columns([1,1])
    with left:
        apply_live = st.button("Apply Settings", type="primary")
    with right:
        reset = st.button("Reset to Defaults")

    if reset:
        st.session_state["settings"] = _default_settings
        st.success("Settings reset to defaults.")

    if apply_live:
        settings["_meta"]["updated_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds")
        st.session_state["settings"] = settings
        # Sync live threshold
        if scorer and hasattr(scorer, "update_threshold"):
            try:
                scorer.update_threshold(settings["model"]["threshold"])
                st.session_state["current_threshold"] = settings["model"]["threshold"]
            except Exception as e:
                st.warning(f"Could not update model threshold: {e}")
        if db and hasattr(db, "save_settings"):
            try:
                db.save_settings(settings)
                st.success("Settings saved.")
            except Exception as e:
                st.error(f"Failed to save settings: {e}")
        else:
            st.info("Settings applied for this session. Connect DB to persist.")
