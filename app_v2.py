# app_v2.py — MIRA v2 UI
# Run with: streamlit run app_v2.py

import sys
import json
import time
import subprocess
import uuid as _uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent.escalation_rules import evaluate_escalation_rules
from evals.eval_runner import EvalRunner

SETTINGS_FILE   = ROOT / "data" / "settings.json"
BP_HISTORY_FILE = ROOT / "data" / "bp_history.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRA",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0a18; }
[data-testid="stSidebar"]          { background: #0f0f1f; border-right: 1px solid #1e1b4b; }
section.main > div                 { padding-top: 1.5rem; }
p, li                              { color: #94a3b8; font-size: 0.97rem; }

/* ── Buttons — consistent font, visible white text ── */
div[data-testid="stButton"] > button {
    font-family: inherit;
    font-size: 0.92rem;
    font-weight: 600;
    color: #ffffff !important;
    border-radius: 10px;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #a855f7, #6366f1, #06b6d4);
    border: none;
    color: #ffffff !important;
    font-weight: 700;
    padding: 0.6rem 1.8rem;
    box-shadow: 0 4px 14px rgba(99,102,241,.35);
}
div[data-testid="stButton"] > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] span,
div[data-testid="stButton"] > button[kind="primary"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.92;
    box-shadow: 0 6px 20px rgba(99,102,241,.5);
}
/* Form submit buttons — same gradient and white bold text */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #a855f7, #6366f1, #06b6d4);
    border: none;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 0.92rem;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    box-shadow: 0 4px 14px rgba(99,102,241,.35);
}
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stFormSubmitButton"] > button span,
div[data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
    font-weight: 700 !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.92;
    box-shadow: 0 6px 20px rgba(99,102,241,.5);
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #13132a;
    border: 1px solid #2d2b5a;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
div[data-testid="stButton"] > button[kind="secondary"] p,
div[data-testid="stButton"] > button[kind="secondary"] span,
div[data-testid="stButton"] > button[kind="secondary"] * {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

.hero {
    text-align: center;
    padding: 4rem 1rem 1rem;
}
.hero-brand {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-greeting {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.15rem;
    font-weight: 500;
    color: #6b7280;
    margin-bottom: 1.4rem;
    letter-spacing: 0.02em;
}
.hero-tagline {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.8;
}
.phase-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    max-width: 860px;
    margin: 1.5rem auto 2.5rem;
}
.phase-card {
    background: #13132a;
    border: 1px solid #2d2b5a;
    border-radius: 14px;
    padding: 1.6rem 1.4rem;
    text-align: center;
}
.phase-num {
    display: inline-block;
    background: #2e1065;
    color: #a855f7;
    font-weight: 800;
    font-size: 0.72rem;
    border-radius: 6px;
    padding: 2px 10px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.phase-title { color: #e2e8f0; font-weight: 700; font-size: 1rem; margin: 0.6rem 0 0.5rem; }
.phase-desc  { color: #6b7280; font-size: 0.88rem; line-height: 1.7; }

.step-bar { display:flex; align-items:center; gap:.5rem; margin:1.2rem 0 1.8rem; flex-wrap:wrap; }
.step-dot  { width:10px; height:10px; border-radius:50%; background:#1e1b4b; flex-shrink:0; }
.step-dot.done   { background:#06b6d4; }
.step-dot.active { background:#06b6d4; box-shadow:0 0 10px #06b6d4; }
.step-label      { font-size:.92rem; color:#6b7280; white-space:nowrap; }
.step-label.active { color:#06b6d4; font-weight:700; }
.step-sep  { flex:1; height:1px; background:#1e1b4b; min-width:10px; }

.hitl-zone2 { border-left:4px solid #f59e0b; background:#1c1a0f; border-radius:0 10px 10px 0; padding:1.2rem 1.5rem; margin:.8rem 0; }
.hitl-zone3 { border-left:4px solid #ef4444; background:#1c0f0f; border-radius:0 10px 10px 0; padding:1.2rem 1.5rem; margin:.8rem 0; }
.zone-badge  { display:inline-block; padding:4px 14px; border-radius:20px; font-size:.8rem; font-weight:700; margin-bottom:.6rem; }
.zone1-badge { background:#064e3b; color:#34d399; }
.zone2-badge { background:#451a03; color:#fbbf24; }
.zone3-badge { background:#450a0a; color:#f87171; }

.rec-block {
    background: #0f0f25;
    border: 1px solid #2d2b5a;
    border-radius: 14px;
    padding: 2rem 2.2rem;
    margin-top: 1rem;
}
/* Alert boxes — bold white text on all st.error / st.warning / st.success / st.info */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

/* run-id badge used in page headers */
.run-id-tag {
    color: #e2e8f0;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    font-size: 0.78em;
    background: #13132a;
    border: 1px solid #2d2b5a;
    padding: 2px 12px;
    border-radius: 6px;
    vertical-align: middle;
}
/* sidebar expander header text — bold white */
[data-testid="stSidebar"] details summary span,
[data-testid="stSidebar"] details summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

.rec-model  { color: #a855f7; font-size: 1.8rem; font-weight: 900; margin-bottom: 0.2rem; }
.rec-label  { color: #a855f7; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 1rem 0 0.2rem; }
.rec-value  { color: #e2e8f0; font-size: 0.92rem; line-height: 1.7; }
.verdict-yes { display:inline-block; background:#064e3b; color:#34d399; font-weight:700; padding:4px 18px; border-radius:20px; font-size:0.9rem; margin-bottom:1rem; }
.verdict-no  { display:inline-block; background:#450a0a; color:#f87171;  font-weight:700; padding:4px 18px; border-radius:20px; font-size:0.9rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}

def processed(run_id: str, suffix: str) -> Path:
    return ROOT / "processed" / f"{run_id}_{suffix}.json"

def load_run_history() -> list:
    runs = []
    run_files = sorted(
        (ROOT / "logs" / "runs").glob("*_run.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in run_files[:20]:
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return runs

def load_settings() -> dict:
    defaults = {"llm_model": "gpt-4o-mini", "theme": "dark"}
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            defaults.update(data)
    except Exception:
        pass
    return defaults

def save_settings(data: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_bp_history() -> list:
    try:
        if BP_HISTORY_FILE.exists():
            return json.loads(BP_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def save_bp(problem: str):
    history = load_bp_history()
    if problem and problem not in history:
        history.insert(0, problem)
        history = history[:20]
        BP_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        BP_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

def delete_single_run(run_id: str):
    for p in (ROOT / "logs" / "runs").glob(f"{run_id}_run.json"):
        p.unlink(missing_ok=True)
    for p in (ROOT / "logs" / "overrides").glob(f"{run_id}_*.json"):
        p.unlink(missing_ok=True)
    processed_dir = ROOT / "processed"
    if processed_dir.exists():
        for p in processed_dir.glob(f"{run_id}_*"):
            p.unlink(missing_ok=True)

def delete_run_history():
    runs_dir = ROOT / "logs" / "runs"
    if runs_dir.exists():
        for p in runs_dir.glob("*_run.json"):
            p.unlink(missing_ok=True)
    processed_dir = ROOT / "processed"
    if processed_dir.exists():
        for p in processed_dir.glob("*.json"):
            p.unlink(missing_ok=True)
        for p in processed_dir.glob("*.csv"):
            p.unlink(missing_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.console.append(f"[{ts}] {msg}")

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "page":             "home",
    "run_id":           None,
    "dataset_path":     None,
    "target_col":       None,
    "business_problem": None,
    "data_card":        None,
    "model_selection":  None,
    "recommendation":   None,
    "eval_report":      None,
    "hitl_approved":    None,
    "hitl_start_time":  None,
    "error_message":    None,
    "console":          [],
    "sidebar_json_view": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    current_page = st.session_state.page

    if current_page == "home":
        # Minimal sidebar on home — only Settings
        st.markdown(
            "<div style='color:#a855f7;font-weight:800;font-size:0.8rem;"
            "text-transform:uppercase;letter-spacing:0.1em;padding:0.5rem 0 1rem;'>"
            "MIRA</div>",
            unsafe_allow_html=True,
        )
        if st.button("Settings", use_container_width=True):
            st.session_state.page = "settings"
            st.rerun()
    else:
        # Brand header
        st.markdown(
            "<div style='text-align:center;padding:0.6rem 0 0.8rem;'>"
            "<span style='color:#a855f7;font-weight:800;font-size:1rem;letter-spacing:0.06em;'>MIRA</span>"
            "<span style='color:#4b5563;font-size:1rem;'> &times; </span>"
            "<span style='color:#06b6d4;font-weight:800;font-size:1rem;letter-spacing:0.06em;'>OpenHands</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Home button — same gradient as New Run
        if st.button("Home", use_container_width=True, type="primary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "home"
            st.rerun()

        st.markdown("---")

        st.markdown(
            "<div style='color:#a855f7;font-weight:800;font-size:0.8rem;"
            "text-transform:uppercase;letter-spacing:0.1em;padding:0.4rem 0 0.6rem;'>"
            "Run History</div>",
            unsafe_allow_html=True,
        )

        if st.button("New Run", use_container_width=True, type="primary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "run"
            st.rerun()

        st.markdown("")

        history = load_run_history()
        if not history:
            st.caption("No runs yet.")
        else:
            for run in history:
                rid = run.get("run_id", "?")
                ds  = Path(run.get("dataset_path", "unknown")).name

                with st.expander(rid[:12], expanded=False):
                    st.caption(f"Dataset: {ds}")

                    dc  = load_json(processed(rid, "data_card"))
                    ms  = load_json(processed(rid, "model_selection"))
                    rec = load_json(processed(rid, "recommendation"))
                    ev  = load_json(processed(rid, "eval_report"))

                    if dc:
                        rows     = dc.get("rows", "—")
                        features = dc.get("features", "—")
                        metric   = dc.get("priority_metric", "—")
                        imbal    = dc.get("class_imbalance_detected", False)
                        st.markdown(
                            f"**Phase 1 — EDA**  \n"
                            f"{rows} rows · {features} features  \n"
                            f"Metric: `{metric}`"
                            + (f"  \nImbalance detected" if imbal else ""),
                            unsafe_allow_html=False,
                        )

                    if ms:
                        winner  = ms.get("selected_model", "—")
                        verdict = ms.get("test_verdict", "—")
                        n_models = len(ms.get("models_trained", []))
                        st.markdown(
                            f"**Phase 2 — Models**  \n"
                            f"{n_models} models trained  \n"
                            f"Winner: {winner}  \n"
                            f"Stress test: {verdict}"
                        )

                    if rec:
                        model = (
                            rec.get("recommended_model")
                            or rec.get("selected_model")
                            or "—"
                        )
                        conf = rec.get("confidence_score")
                        zone = rec.get("routing_zone", "—")
                        conf_str = f"{float(conf):.2f}" if conf else "—"
                        st.markdown(
                            f"**Phase 3 — Recommendation**  \n"
                            f"Model: {model}  \n"
                            f"Confidence: {conf_str}  \n"
                            f"Zone: {zone}"
                        )

                    st.markdown("")
                    col_view, col_del = st.columns(2)
                    with col_view:
                        if st.button("View Report", key=f"view_{rid}", use_container_width=True, type="primary"):
                            st.session_state.run_id          = rid
                            st.session_state.recommendation  = rec
                            st.session_state.data_card       = dc
                            st.session_state.model_selection = ms
                            st.session_state.eval_report     = ev
                            st.session_state.hitl_approved   = True
                            st.session_state.page            = "results"
                            st.session_state.sidebar_json_view = None
                            st.rerun()
                    with col_del:
                        if st.button("Delete", key=f"del_{rid}", use_container_width=True):
                            delete_single_run(rid)
                            st.rerun()

            st.markdown("---")
            if st.button("Delete All Runs", use_container_width=True):
                delete_run_history()
                st.rerun()

        st.markdown("---")

        if st.button("Settings", use_container_width=True, type="primary"):
            st.session_state.page = "settings"
            st.rerun()

# ── Sidebar JSON viewer ───────────────────────────────────────────────────────
if st.session_state.sidebar_json_view:
    jv = st.session_state.sidebar_json_view
    st.markdown(f"### {jv['label']}")
    if st.button("Back"):
        st.session_state.sidebar_json_view = None
        st.rerun()
    st.json(jv["data"])
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <div class="hero-brand">
            <span style="color:#a855f7;">MIRA</span>
            <span style="color:#4b5563;"> &times; </span>
            <span style="color:#06b6d4;">OpenHands</span>
        </div>
        <div class="hero-greeting">Hi, I'm MIRA</div>
        <div class="hero-subtitle">Model Intelligence &amp; Recommendation Agent</div>
        <div class="hero-tagline">
            Upload your data. Describe your business goal in plain English.<br>
            MIRA runs the full ML pipeline and tells you exactly what to deploy and&nbsp;why.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="phase-grid">
        <div class="phase-card">
            <span class="phase-num">Phase 1</span>
            <div class="phase-title">Understand Your Data</div>
            <div class="phase-desc">
                MIRA cleans and profiles your dataset, then uses AI to infer
                the right success metric directly from your business problem.
                No dropdowns, no manual selection.
            </div>
        </div>
        <div class="phase-card">
            <span class="phase-num">Phase 2</span>
            <div class="phase-title">Test Multiple Classification Models</div>
            <div class="phase-desc">
                Five models are trained with 5 fold cross validation and
                stress tested for overfitting, data leakage, and stability
                before any recommendation is made.
            </div>
        </div>
        <div class="phase-card">
            <span class="phase-num">Phase 3</span>
            <div class="phase-title">AI Writes the Recommendation</div>
            <div class="phase-desc">
                The agent reasons over all results and delivers a plain English
                deployment recommendation with a confidence score, risk flags,
                and an explicit YES / NO verdict.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([2, 1, 2])[1]
    with col:
        if st.button("Start MIRA", type="primary", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "run"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "settings":

    st.markdown("## Settings")
    st.markdown("Configure MIRA behaviour and appearance.")

    current_settings = load_settings()

    with st.form("settings_form"):
        st.markdown("#### LLM Model")
        st.caption("Model used by the MIRA agent for Phase 3 recommendation generation.")
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        current_model = current_settings.get("llm_model", "gpt-4o-mini")
        llm_model = st.selectbox(
            "LLM Model",
            options=model_options,
            index=model_options.index(current_model) if current_model in model_options else 0,
            label_visibility="collapsed",
        )

        st.markdown("#### UI Theme")
        st.caption("Controls the application colour scheme.")
        theme_options = ["dark", "light"]
        current_theme = current_settings.get("theme", "dark")
        theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            label_visibility="collapsed",
        )

        saved = st.form_submit_button("Save Settings", type="primary")

    if saved:
        save_settings({"llm_model": llm_model, "theme": theme})
        st.success("Settings saved.")

    st.markdown("")
    st.markdown("#### Business Problem History")
    st.caption("Previously entered business problems saved for re-use.")
    bp_hist = load_bp_history()
    if bp_hist:
        for i, bp in enumerate(bp_hist):
            st.markdown(f"**{i+1}.** {bp}")
        if st.button("Clear Business Problem History"):
            BP_HISTORY_FILE.write_text("[]", encoding="utf-8")
            st.success("Business problem history cleared.")
            st.rerun()
    else:
        st.caption("No history yet. Business problems are saved automatically when you run MIRA.")


# ═════════════════════════════════════════════════════════════════════════════
# RUN PAGE — upload + configure
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "run":

    st.markdown("## New Run")
    st.markdown("Upload your dataset, select the target column, and describe your business goal.")
    st.markdown("")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("**Dataset**")
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xls", "xlsx"],
            label_visibility="collapsed",
        )

        dataset_path = None
        columns = []

        if uploaded:
            save_dir = ROOT / "data" / "raw"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / uploaded.name
            save_path.write_bytes(uploaded.getvalue())
            dataset_path = str(save_path)
            try:
                df = pd.read_excel(save_path) if uploaded.name.endswith((".xls", ".xlsx")) else pd.read_csv(save_path)
                columns = list(df.columns)
                st.success(f"**{uploaded.name}** · {len(df):,} rows · {len(columns)} columns")
                with st.expander("Preview"):
                    st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            existing = sorted((ROOT / "data" / "raw").glob("*.csv")) + \
                       sorted((ROOT / "data" / "raw").glob("*.xls*"))
            if existing:
                st.markdown("**Or choose an existing dataset**")
                choice = st.selectbox(
                    "Existing dataset",
                    ["— select —"] + [p.name for p in existing],
                    label_visibility="collapsed",
                )
                if choice != "— select —":
                    dataset_path = str(ROOT / "data" / "raw" / choice)
                    try:
                        df = pd.read_csv(dataset_path) if choice.endswith(".csv") else pd.read_excel(dataset_path)
                        columns = list(df.columns)
                        st.info(f"**{choice}** · {len(df):,} rows · {len(columns)} columns")
                    except Exception:
                        pass

        st.markdown("**Target Column**")
        target_col = st.selectbox(
            "Target column",
            options=["— select —"] + columns,
            disabled=not columns,
            label_visibility="collapsed",
        )

        st.markdown("**Business Problem**")
        st.caption("Describe what you want to predict and why it matters. MIRA infers the right metric from this.")

        bp_history = load_bp_history()
        business_problem = ""
        if bp_history:
            bp_choice = st.selectbox(
                "Previous business problems",
                ["— enter new —"] + bp_history,
                label_visibility="collapsed",
            )
            if bp_choice != "— enter new —":
                business_problem = bp_choice

        business_problem = st.text_area(
            "Business problem",
            value=business_problem,
            placeholder="E.g. A retail bank wants to identify customers likely to churn in the next 90 days so the retention team can intervene early — missing a churner costs far more than a false alarm.",
            height=130,
            label_visibility="collapsed",
        )

        st.markdown("")
        can_run = (
            dataset_path is not None
            and target_col not in (None, "— select —")
            and len((business_problem or "").strip()) >= 20
        )

        if st.button("Run MIRA", type="primary", use_container_width=True, disabled=not can_run):
            bp_clean = business_problem.strip()
            save_bp(bp_clean)
            st.session_state.run_id           = f"run_{_uuid.uuid4().hex[:8]}"
            st.session_state.dataset_path     = dataset_path
            st.session_state.target_col       = target_col
            st.session_state.business_problem = bp_clean
            st.session_state.data_card        = None
            st.session_state.model_selection  = None
            st.session_state.recommendation   = None
            st.session_state.eval_report      = None
            st.session_state.hitl_approved    = None
            st.session_state.hitl_start_time  = None
            st.session_state.console          = []
            st.session_state.page             = "phase1"
            st.rerun()

        if not can_run:
            st.caption("Upload a dataset, select a target column, and describe the business problem to continue.")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "phase1":

    st.markdown(f"## Running &nbsp; <span class='run-id-tag'>{st.session_state.run_id}</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot active"></div><div class="step-label active">Phase 1 — Understanding your data</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Phase 2 — Training &amp; Testing Models</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Phase 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id = st.session_state.run_id
    out    = ROOT / "processed"
    out.mkdir(exist_ok=True)
    dc_out  = out / f"{run_id}_data_card.json"
    cleaned = out / f"{run_id}_cleaned.csv"

    cmd = [
        sys.executable, "scripts/EDA.py",
        "--dataset",          st.session_state.dataset_path,
        "--target",           st.session_state.target_col,
        "--output",           str(dc_out),
        "--cleaned-output",   str(cleaned),
        "--business-problem", st.session_state.business_problem,
    ]
    with st.spinner("Profiling your data and inferring the success metric..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        stderr = result.stderr or ""
        if "INVALID_BUSINESS_PROBLEM" in stderr:
            reason = ""
            for line in stderr.splitlines():
                if "INVALID_BUSINESS_PROBLEM" in line:
                    reason = line.split("INVALID_BUSINESS_PROBLEM:")[-1].strip()
                    break
            st.warning(
                f"**Business problem could not be validated.** {reason}\n\n"
                "Please describe a real prediction objective, for example:\n"
                "- Predict which customers are likely to cancel their subscription in the next 30 days.\n"
                "- Identify loan applicants at high risk of default.\n"
                "- Flag insurance claims likely to be fraudulent before payout."
            )
            if st.button("Go Back"):
                st.session_state.page = "run"
                st.rerun()
        else:
            st.session_state.error_message = stderr
            st.session_state.page = "error"
            st.rerun()
    else:
        log("Phase 1 complete — data profiled, metric inferred")
        st.session_state.data_card = load_json(dc_out)
        st.session_state.page = "phase2"
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "phase2":

    st.markdown(f"## Running &nbsp; <span class='run-id-tag'>{st.session_state.run_id}</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Phase 2 — Training &amp; Testing Models</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Phase 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id  = st.session_state.run_id
    out     = ROOT / "processed"
    dc_out  = out / f"{run_id}_data_card.json"
    ms_out  = out / f"{run_id}_model_selection.json"
    cleaned = out / f"{run_id}_cleaned.csv"

    cmd = [
        sys.executable, "scripts/Modeltrain.py",
        "--cleaned-data", str(cleaned),
        "--data-card",    str(dc_out),
        "--target",       st.session_state.target_col,
        "--output",       str(ms_out),
    ]
    with st.spinner("Training and stress testing 5 models — this takes 1 to 3 minutes..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        st.session_state.error_message = result.stderr
        st.session_state.page = "error"
    else:
        log("Phase 2 complete — models trained, stress tests done")
        st.session_state.model_selection = load_json(ms_out)
        st.session_state.page = "phase3"
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "phase3":

    st.markdown(f"## Running &nbsp; <span class='run-id-tag'>{st.session_state.run_id}</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Phase 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("The agent is reasoning over all phase outputs and writing your deployment recommendation.")

    run_id = st.session_state.run_id
    out    = ROOT / "processed"

    with st.spinner("Writing deployment recommendation — typically 1 to 5 minutes..."):
        try:
            from agent.mira_agent import MIRAAgent
            from schemas.input_schema import AgentInput
            agent_input = AgentInput(
                dataset_path=st.session_state.dataset_path,
                target_column=st.session_state.target_col,
                business_problem=st.session_state.business_problem,
                task_type="auto", max_models=5, max_iterations=40,
                run_id=run_id, output_path=str(out) + "/",
            )
            MIRAAgent(agent_input).run()
            rec_path = out / f"{run_id}_recommendation.json"
            st.session_state.recommendation = load_json(rec_path)
            log("Phase 3 complete — recommendation written")
            st.session_state.page = "hitl"
        except Exception as e:
            log(f"Phase 3 error: {e}")
            st.session_state.error_message = str(e)
            st.session_state.page = "error"
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# HITL GATE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "hitl":

    run_id       = st.session_state.run_id
    rec          = st.session_state.recommendation or {}
    dc           = st.session_state.data_card or {}
    ms           = st.session_state.model_selection or {}
    confidence   = float(rec.get("confidence_score", 0) or 0)
    flags        = rec.get("flags", []) or []
    routing_zone = rec.get("routing_zone", "zone_2")
    model        = (
        rec.get("recommended_model")
        or rec.get("selected_model")
        or (rec.get("recommendations") or {}).get("selected_model")
        or ms.get("selected_model")
        or "Unknown"
    )
    auc          = rec.get("primary_metric_value")
    exec_summary = rec.get("executive_summary", "")
    review_reason = rec.get("human_review_reason", "")

    escalation = evaluate_escalation_rules(dc, ms)
    esc_rules  = escalation.get("rules_triggered", [])

    if escalation.get("hard_escalation"):
        routing_zone = "zone_3"
    elif not esc_rules and routing_zone == "zone_2":
        routing_zone = "zone_1"

    if not confidence:
        winner_data = next((m for m in ms.get("models_trained", []) if m.get("name") == model), {})
        _pm    = dc.get("priority_metric", "roc_auc")
        _pmcol = {"roc_auc": "cv_roc_auc_mean", "recall": "cv_recall_mean",
                  "f1_score": "cv_f1_mean", "precision": "cv_precision_mean"}.get(_pm, "cv_roc_auc_mean")
        _score = winner_data.get(_pmcol, 0) or 0
        n_issues = len(esc_rules)
        confidence = max(0.0, min(1.0, float(_score) - 0.08 * n_issues))

    if not flags and esc_rules:
        flags = [r["rule_name"] for r in esc_rules]

    if st.session_state.hitl_start_time is None:
        st.session_state.hitl_start_time = time.time()

    st.markdown(f"## Human Review &nbsp; <span class='run-id-tag'>{run_id}</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 3 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    if routing_zone == "zone_1":
        st.markdown('<span class="zone-badge zone1-badge">Zone 1 — Auto-Approve Eligible</span>', unsafe_allow_html=True)
        st.success(f"Confidence {confidence:.3f} — no flags detected. You can proceed automatically.")
        if st.button("Proceed to Report", type="primary"):
            st.session_state.hitl_approved = True
            st.session_state.page = "eval"
            st.rerun()
    else:
        badge = "zone3-badge" if routing_zone == "zone_3" else "zone2-badge"
        label = "Zone 3 — Priority Review" if routing_zone == "zone_3" else "Zone 2 — Review Required"
        box   = "hitl-zone3" if routing_zone == "zone_3" else "hitl-zone2"
        st.markdown(f'<div class="{box}"><span class="zone-badge {badge}">{label}</span></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Model", model)
        c2.metric("Confidence Score",  f"{confidence:.3f}")
        c3.metric("Primary Metric",    f"{auc:.4f}" if isinstance(auc, float) else "—")

        if flags:
            st.markdown("**Flags triggered**")
            fc = st.columns(min(len(flags), 3))
            for i, f in enumerate(flags):
                fc[i % len(fc)].error(f)

        if review_reason:
            st.warning(f"**Why review is needed:** {review_reason}")

        if esc_rules:
            with st.expander(f"Issues Detected ({len(esc_rules)})", expanded=True):
                for rule in esc_rules:
                    sev   = rule.get("severity", "")
                    color = "#ef4444" if sev == "CRITICAL" else "#f59e0b"
                    st.markdown(
                        f"**{rule.get('rule_name')}** "
                        f"<span style='color:{color};font-size:.8rem'>[{sev}]</span>  \n{rule.get('detail','')}",
                        unsafe_allow_html=True,
                    )

        if exec_summary:
            st.markdown("**Executive Summary**")
            st.info(exec_summary)

        st.divider()
        st.markdown("### Your Decision")
        st.caption("Your rationale is logged for audit purposes. A rationale is required.")

        CATS = {
            "PERFORMANCE_ACCEPTABLE": "Performance is acceptable for this business context",
            "BUSINESS_PRIORITY":      "Chosen based on cost, interpretability, or team preference",
            "DATA_QUALITY_RESOLVED":  "Flagged data issue confirmed as not deployment-blocking",
            "DOMAIN_KNOWLEDGE":       "Applied domain expertise the system could not infer",
            "METRIC_MISMATCH":        "Override the inferred success metric",
            "RISK_ACCEPTED":          "Explicitly acknowledge and accept the flagged risk",
            "AGENT_ERROR":            "Recommendation was factually wrong — flag for review",
        }

        with st.form("hitl_form"):
            decision  = st.radio("Decision", ["Approve — proceed to report", "Reject — do not proceed"])
            category  = st.selectbox("Reason Category", list(CATS.keys()), format_func=lambda k: f"{k}  —  {CATS[k]}")
            rationale = st.text_area("Rationale", placeholder="Briefly explain your decision...", height=90)
            submitted = st.form_submit_button("Submit Decision", type="primary")

        if submitted:
            if not rationale.strip():
                st.error("Please write a rationale before submitting.")
            else:
                duration = round(time.time() - st.session_state.hitl_start_time, 1)
                approved = "Approve" in decision
                try:
                    from agent.main import write_override_log
                    write_override_log(
                        run_id=run_id, routing_zone=routing_zone,
                        escalation_rules=esc_rules, recommendation=rec,
                        human_decision="APPROVED" if approved else "REJECTED",
                        override_category=category, human_rationale=rationale.strip(),
                        review_duration_seconds=duration,
                    )
                except Exception:
                    pass
                log(f"Review: {'APPROVED' if approved else 'REJECTED'} · {duration}s")
                st.session_state.hitl_approved = approved
                st.session_state.page = "eval" if approved else "results"
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# EVAL
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "eval":

    st.markdown(f"## Running &nbsp; <span class='run-id-tag'>{st.session_state.run_id}</span>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 3 — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Review — Done</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Generating Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id   = st.session_state.run_id
    out_path = str(ROOT / "processed") + "/"
    priority = (st.session_state.data_card or {}).get("priority_metric", "roc_auc")

    with st.spinner("Running 7 layer evaluation and generating final report..."):
        try:
            EvalRunner(run_id=run_id, output_path=out_path, priority_metric=priority).run()
            ep = ROOT / "processed" / f"{run_id}_eval_report.json"
            st.session_state.eval_report = load_json(ep)
            log("Evaluation complete — report ready")
        except Exception as e:
            log(f"Eval error: {e}")

    st.session_state.page = "results"
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    run_id   = st.session_state.run_id or "?"
    rec      = st.session_state.recommendation or {}
    dc       = st.session_state.data_card or {}
    ms       = st.session_state.model_selection or {}
    ev       = st.session_state.eval_report or {}
    rejected = st.session_state.hitl_approved is False

    st.markdown(f"## Recommendation &nbsp; <span class='run-id-tag'>{run_id}</span>", unsafe_allow_html=True)

    if rejected:
        st.error("This run was rejected at the review step. No deployment recommendation was finalized. Audit log saved.")
        st.markdown("")
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("Start a New Run", type="primary", use_container_width=True):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.page = "run"
                st.rerun()
        st.stop()

    # ── Shared derivations ──
    model = (
        rec.get("recommended_model")
        or rec.get("selected_model")
        or (rec.get("recommendations") or {}).get("selected_model")
        or ms.get("selected_model")
        or "—"
    )
    _pm    = dc.get("priority_metric", "roc_auc")
    _pmcol = {"roc_auc": "cv_roc_auc_mean", "recall": "cv_recall_mean",
               "f1_score": "cv_f1_mean", "precision": "cv_precision_mean"}.get(_pm, "cv_roc_auc_mean")
    winner_data = next((m for m in ms.get("models_trained", []) if m.get("name") == model), {})
    _score = winner_data.get(_pmcol)
    _gap   = winner_data.get("overfitting_gap", 0)
    _rows  = dc.get("rows", 0)

    confidence = float(rec.get("confidence_score", 0) or 0)
    if not confidence and _score:
        _esc       = evaluate_escalation_rules(dc, ms)
        n_issues   = len(_esc.get("rules_triggered", []))
        confidence = round(max(0.0, min(1.0, float(_score) - 0.08 * n_issues)), 3)

    zone        = rec.get("routing_zone", "")
    summary     = rec.get("executive_summary", "")
    flags       = rec.get("flags", []) or []
    metric      = dc.get("priority_metric", "—")
    metric_rsn  = dc.get("metric_reason", "")

    if _score is not None:
        _metric_label = {"roc_auc": "AUC", "recall": "recall", "f1_score": "F1",
                         "precision": "precision"}.get(_pm, _pm)
        _overfit_line = (
            "It shows no signs of memorising the training data — performance holds up on unseen records."
            if abs(_gap) <= 0.05 else
            f"There is a small generalisation gap ({_gap:.3f}) which is within acceptable limits."
        )
        selection_reason = (
            f"{model} scored the highest {_metric_label} of {_score:.3f} "
            f"across five rounds of testing on {_rows:,} records. {_overfit_line}"
        )
    else:
        selection_reason = (
            rec.get("selection_reason") or rec.get("selection_reasoning")
            or ms.get("selection_reasoning")
            or (rec.get("recommendations") or {}).get("reason") or ""
        )

    if zone == "zone_3":
        verdict_yes = False
    elif zone in ("zone_1", "zone_2"):
        verdict_yes = True
    elif confidence >= 0.70:
        verdict_yes = True
    elif summary:
        verdict_yes = any(w in summary.upper() for w in ("DEPLOY", "RECOMMEND", "YES"))
    else:
        verdict_yes = True

    verdict_cls  = "verdict-yes" if verdict_yes else "verdict-no"
    verdict_txt  = "DEPLOY" if verdict_yes else "DO NOT DEPLOY"
    zone_display = zone.replace("_", " ").title() if zone else "—"
    bp           = st.session_state.business_problem or ""

    tabs = st.tabs(["Overview", "Data Profile", "Model Rankings", "Recommendation", "Eval Report"])

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(
            f'<div class="rec-block">'
            f'<div class="rec-model">{model}</div>'
            f'<div class="{verdict_cls}">{verdict_txt}</div>'
            f'<div class="rec-label">Confidence Score</div>'
            f'<div class="rec-value">{confidence:.2f} / 1.00 &nbsp;&middot;&nbsp; {zone_display}</div>'
            f'<div class="rec-label">Why This Model</div>'
            f'<div class="rec-value">{selection_reason or "—"}</div>'
            f'<div class="rec-label">Success Metric</div>'
            f'<div class="rec-value"><strong>{metric}</strong> &mdash; {metric_rsn or "—"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if bp:
            st.markdown("")
            st.markdown("**Business Problem**")
            st.info(bp)

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Flags")
            if flags:
                for f in flags:
                    st.warning(f)
            else:
                st.success("No flags — clean run")
            st.markdown("#### Stress Tests")
            findings = ms.get("test_findings") or rec.get("test_findings") or []
            for finding in findings:
                (st.success if "PASSED" in finding.upper() else st.error)(finding)
        with col_b:
            if summary:
                st.markdown("#### Executive Summary")
                st.info(summary)
            next_steps = rec.get("next_steps", []) or []
            if next_steps:
                st.markdown("#### Next Steps")
                for s in next_steps:
                    st.markdown(f"- {s}")

        st.markdown("")
        if st.button("Run MIRA Again", type="primary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "run"
            st.rerun()

    # ── Tab 2: Data Profile (Phase 1) ────────────────────────────────────────
    with tabs[1]:
        if not dc:
            st.caption("No data profile available.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",           f"{dc.get('rows', 0):,}")
            c2.metric("Features",       dc.get("features", "—"))
            c3.metric("Success Metric", dc.get("priority_metric", "—"))
            c4.metric("Class Imbalance","Yes" if dc.get("class_imbalance_detected") else "No")

            st.divider()
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Outcome Distribution**")
                cd = dc.get("class_distribution", {})
                if cd:
                    chart_df = pd.DataFrame(
                        {"Proportion": list(cd.values())},
                        index=[f"Class {k}" for k in cd],
                    )
                    st.bar_chart(chart_df)
                    st.caption(f"Minority class: {dc.get('minority_class_ratio', 0):.1%}")
                else:
                    st.caption("No distribution data.")
            with cb:
                st.markdown("**Strongest Predictors**")
                hcf = dc.get("high_correlation_features", [])
                if hcf:
                    hcf_df = pd.DataFrame(hcf).rename(
                        columns={"feature": "Feature", "correlation": "Correlation"}
                    )
                    st.bar_chart(hcf_df.set_index("Feature")["Correlation"])
                else:
                    st.caption("No correlation data.")

            st.divider()
            st.markdown("**Why this success metric?**")
            st.info(f"**{dc.get('priority_metric','?')}** — {dc.get('metric_reason','—')}")

            cl = dc.get("cleaning_log", [])
            if cl:
                with st.expander("Data Cleaning Log"):
                    for e in cl:
                        st.markdown(f"- {e}")

    # ── Tab 3: Model Rankings (Phase 2) ──────────────────────────────────────
    with tabs[2]:
        if not ms:
            st.caption("No model data available.")
        else:
            models = ms.get("models_trained", [])
            winner = ms.get("selected_model", "")
            mcol   = _pmcol

            table_rows = []
            for m in sorted(models, key=lambda x: x.get(mcol, 0), reverse=True):
                gap = m.get("overfitting_gap", 0)
                table_rows.append({
                    "Model Ranking": ("Winner: " if m["name"] == winner else "") + m["name"],
                    "AUC":         round(m.get("cv_roc_auc_mean", 0), 4),
                    "Recall":      round(m.get("cv_recall_mean", 0), 4),
                    "F1":          round(m.get("cv_f1_mean", 0), 4),
                    "Precision":   round(m.get("cv_precision_mean", 0), 4),
                    "Overfit Gap": round(gap, 4),
                    "Stability":   "High risk" if gap > 0.10 else ("Watch" if gap > 0.07 else "Good"),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            st.caption(f"Ranked by **{_pm}** — inferred from your business problem description.")

            st.markdown("**Generalisation Risk by Model**")
            gap_df = pd.DataFrame([
                {"Model": m["name"], "Overfitting Gap": round(m.get("overfitting_gap", 0), 4)}
                for m in models
            ]).set_index("Model")
            st.bar_chart(gap_df)
            st.caption("Gap = train score minus validation score. Above 0.10 is a warning sign.")

            fi = ms.get("feature_importance", {})
            if fi:
                st.markdown("**Feature Importance**")
                fi_df = pd.DataFrame(
                    sorted(fi.items(), key=lambda x: x[1], reverse=True),
                    columns=["Feature", "Importance"],
                ).set_index("Feature")
                st.bar_chart(fi_df)

    # ── Tab 4: Recommendation (Phase 3) ──────────────────────────────────────
    with tabs[3]:
        if not rec or model == "—":
            st.caption("No recommendation produced for this run.")
        else:
            if bp:
                st.markdown("**Business Problem**")
                st.info(bp)
                st.markdown("")

            st.markdown(
                f'<div class="rec-block">'
                f'<div class="rec-model">{model}</div>'
                f'<div class="{verdict_cls}">{verdict_txt}</div>'
                f'<div class="rec-label">Why This Model</div>'
                f'<div class="rec-value">{selection_reason or "—"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence Score",  f"{confidence:.3f}")
            c2.metric("Success Metric",    metric)
            c3.metric("Alternative Model", (
                rec.get("alternative_model") or rec.get("runner_up_model")
                or ms.get("runner_up_model") or "—"
            ))

            st.divider()
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Next Steps**")
                ns = rec.get("next_steps", []) or []
                for s in ns:
                    st.markdown(f"- {s}")
                if not ns:
                    st.caption("—")
                st.markdown("**Deployment Considerations**")
                for d in rec.get("deployment_considerations", []):
                    st.markdown(f"- {d}")
            with cb:
                st.markdown("**Risks**")
                risks = rec.get("risks", []) or []
                for r in risks:
                    st.warning(r)
                if not risks:
                    st.caption("—")
                st.markdown("**Tradeoffs**")
                for t in rec.get("tradeoffs", []):
                    st.markdown(f"- {t}")

            bi = rec.get("business_impact")
            if bi:
                st.divider()
                st.markdown("**Business Impact**")
                if isinstance(bi, dict):
                    for k, v in bi.items():
                        st.markdown(f"**{k.replace('_',' ').title()}:** {v}")
                else:
                    st.markdown(str(bi))

            with st.expander("Full Recommendation JSON"):
                st.json(rec)

    # ── Tab 5: Eval Report ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
        <style>
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] { color: white !important; font-weight: bold !important; }
        </style>
        """, unsafe_allow_html=True)
        if not ev:
            st.markdown("<b style='color:white'>Evaluation report not available for this run.</b>", unsafe_allow_html=True)
        else:
            overall    = ev.get("overall_score")
            unit       = ev.get("unit_tests", {})
            prod       = ev.get("production_checklist", {})
            summary_ev = ev.get("summary", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Score",    f"{overall:.1f}%" if isinstance(overall, (int, float)) else "—")
            c2.metric("Unit Tests",       f"{unit.get('passed','?')}/{unit.get('tests_run','?')}")
            c3.metric("Production Ready", "Yes" if prod.get("production_ready") else "No")
            c4.metric("Quality Score",    f"{summary_ev.get('quality_pct', 0):.1f}%" if summary_ev else "—")

            st.divider()
            if prod.get("checklist"):
                st.markdown("<b style='color:white;font-size:1rem'>Production Checklist</b>", unsafe_allow_html=True)
                rows_html = ""
                for item in prod["checklist"]:
                    icon      = "PASS" if item["passed"] else "FAIL"
                    icon_col  = "#22c55e" if item["passed"] else "#ef4444"
                    crit_lbl  = "Yes" if item.get("critical") else "No"
                    name      = item["item"].replace("_", " ").title()
                    detail    = item.get("detail", "")
                    rows_html += (
                        f"<tr>"
                        f"<td style='color:white;font-weight:bold;padding:9px 14px;border-bottom:1px solid #2d3748'>{name}</td>"
                        f"<td style='color:white;font-weight:bold;padding:9px 14px;border-bottom:1px solid #2d3748;text-align:center'>{crit_lbl}</td>"
                        f"<td style='color:{icon_col};font-weight:bold;padding:9px 14px;border-bottom:1px solid #2d3748;text-align:center'>{icon}</td>"
                        f"<td style='color:white;font-weight:bold;padding:9px 14px;border-bottom:1px solid #2d3748;font-size:.85rem'>{detail}</td>"
                        f"</tr>"
                    )
                th_style = "color:white;font-weight:bold;padding:9px 14px;border-bottom:2px solid #4a5568;text-align:left;background:#1a202c"
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;margin-top:8px'>"
                    f"<thead><tr>"
                    f"<th style='{th_style}'>Check</th>"
                    f"<th style='{th_style};text-align:center'>Critical</th>"
                    f"<th style='{th_style};text-align:center'>Result</th>"
                    f"<th style='{th_style}'>Detail</th>"
                    f"</tr></thead>"
                    f"<tbody>{rows_html}</tbody>"
                    f"</table>",
                    unsafe_allow_html=True,
                )

            hitl_ev = ev.get("hitl_gate", {})
            if hitl_ev:
                st.divider()
                st.markdown("<b style='color:white;font-size:1rem'>Review Gate Risk Score</b>", unsafe_allow_html=True)
                rs = hitl_ev.get("total_risk_score", 0)
                th = hitl_ev.get("risk_threshold", 5)
                st.metric("Risk Score", f"{rs}  (limit: {th})",
                          delta=f"{'Below' if rs < th else 'Above'} threshold")
                for risk in hitl_ev.get("risks_identified", []):
                    fn = st.error if risk.get("severity") == "HIGH" else st.warning
                    fn(f"**{risk['risk']}** — {risk['detail']}")

            with st.expander("Full Eval Report JSON"):
                st.json(ev)


# ═════════════════════════════════════════════════════════════════════════════
# ERROR PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "error":
    st.error("MIRA encountered an error during this run.")
    if st.session_state.error_message:
        with st.expander("Error details"):
            st.code(st.session_state.error_message)
    if st.button("Start Over"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.page = "run"
        st.rerun()