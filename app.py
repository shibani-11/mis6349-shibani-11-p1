# app.py — MIRA Web UI v3
# Run with: streamlit run app.py

import sys
import json
import subprocess
import time
import uuid as _uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent.escalation_rules import evaluate_escalation_rules
from evals.eval_runner import EvalRunner

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MIRA — ML Recommendation Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0a0a18; }
[data-testid="stSidebar"]          { background: #0f0f1f; border-right: 1px solid #1e1b4b; }
[data-testid="stSidebar"] > div    { padding-top: 1rem; }
section.main > div                 { padding-top: 1.5rem; }

/* ── Hero ── */
.hero { text-align:center; padding:4rem 1rem 1.5rem; }
.hero h1 {
    font-size:3.8rem; font-weight:900; letter-spacing:-.04em;
    background:linear-gradient(135deg,#a855f7 0%,#6366f1 45%,#06b6d4 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 .8rem;
}
.hero .tagline {
    color:#94a3b8; font-size:1.15rem; max-width:600px;
    margin:0 auto 2rem; line-height:1.75;
}

/* ── Step cards (homepage) ── */
.phase-row {
    display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; margin:2rem 0;
}
.phase-card {
    background:#13132a; border:1px solid #2d2b5a; border-radius:14px; padding:1.5rem;
    transition:border-color .2s, box-shadow .2s, transform .2s;
}
.phase-card:hover {
    border-color:#a855f7; transform:translateY(-3px);
    box-shadow:0 8px 24px rgba(168,85,247,.15);
}
.phase-card .step-num {
    display:inline-block; background:#2e1065; color:#a855f7;
    font-weight:800; font-size:.72rem; border-radius:6px; padding:2px 8px; margin-bottom:.6rem;
}
.phase-card .phase-icon { font-size:1.8rem; display:block; margin-bottom:.4rem; }
.phase-card h3 { color:#e2e8f0; font-size:.97rem; font-weight:700; margin:0 0 .4rem; }
.phase-card p  { color:#6b7280; font-size:.84rem; line-height:1.65; margin:0; }

/* ── Form card (new run) ── */
.form-card {
    background:#0f0f25; border:1px solid #1e1b4b; border-radius:16px;
    padding:2rem 2.5rem; max-width:720px; margin:0 auto;
}
.form-section-label {
    color:#a855f7; font-size:.74rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.09em; margin:.9rem 0 .3rem;
}

/* ── Step progress bar ── */
.step-bar { display:flex; align-items:center; gap:.5rem; margin:1.5rem 0; }
.step-dot  { width:10px; height:10px; border-radius:50%; background:#1e1b4b; flex-shrink:0; }
.step-dot.done   { background:#06b6d4; }
.step-dot.active { background:#06b6d4; box-shadow:0 0 10px #06b6d4; }
.step-label { font-size:.8rem; color:#6b7280; white-space:nowrap; }
.step-label.active { color:#06b6d4; font-weight:600; }
.step-sep { flex:1; height:1px; background:#1e1b4b; }

/* ── HITL banners ── */
.hitl-zone2 {
    border-left:4px solid #f59e0b; background:#1c1a0f;
    border-radius:0 10px 10px 0; padding:1.2rem 1.5rem; margin:.8rem 0;
}
.hitl-zone3 {
    border-left:4px solid #ef4444; background:#1c0f0f;
    border-radius:0 10px 10px 0; padding:1.2rem 1.5rem; margin:.8rem 0;
}
.zone-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:.8rem; font-weight:700; margin-bottom:.6rem;
}
.zone1-badge { background:#064e3b; color:#34d399; }
.zone2-badge { background:#451a03; color:#fbbf24; }
.zone3-badge { background:#450a0a; color:#f87171; }

/* ── Recommendation card ── */
.rec-card {
    background:linear-gradient(135deg,#2e1065 0%,#0f0f25 100%);
    border:1px solid #7c3aed; border-radius:14px; padding:1.8rem; margin:1rem 0;
}
.rec-card h2 { color:#c4b5fd; font-size:1.5rem; margin:0 0 .5rem; }
.rec-card p  { color:#cbd5e1; margin:0; line-height:1.65; }

/* ── Console output ── */
.console {
    background:#05050f; border:1px solid #1e1b4b; border-radius:8px;
    padding:1rem; font-family:monospace; font-size:.77rem;
    color:#a78bfa; max-height:220px; overflow-y:auto; line-height:1.7;
}

/* ── Settings section card ── */
.settings-section {
    background:#0f0f25; border:1px solid #1e1b4b; border-radius:12px;
    padding:1.4rem 1.6rem; margin:.8rem 0;
}
.settings-section h3 { color:#e2e8f0; font-size:.97rem; font-weight:700; margin:0 0 1rem; }

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#f59e0b,#ea580c);
    border:none; color:white; font-weight:700;
    padding:.7rem 2rem; border-radius:10px;
    box-shadow:0 4px 14px rgba(245,158,11,.3);
    transition:opacity .2s, transform .2s, box-shadow .2s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity:.92; transform:translateY(-1px);
    box-shadow:0 6px 20px rgba(245,158,11,.4);
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color:#c4b5fd !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color:#6b7280 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}

def log(msg: str):
    st.session_state.console.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def processed(run_id: str, suffix: str) -> Path:
    return ROOT / "processed" / f"{run_id}_{suffix}.json"

def load_run_history() -> list:
    runs = []
    for p in sorted((ROOT / "logs" / "runs").glob("*_run.json"), reverse=True)[:20]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            runs.append(d)
        except Exception:
            pass
    return runs

def load_run_data(run_id: str) -> dict:
    return {
        "data_card":       load_json(processed(run_id, "data_card")),
        "model_selection": load_json(processed(run_id, "model_selection")),
        "recommendation":  load_json(processed(run_id, "recommendation")),
        "eval_report":     load_json(processed(run_id, "eval_report")),
    }

def delete_run(run_id: str):
    for suffix in ["data_card", "model_selection", "recommendation", "eval_report"]:
        p = ROOT / "processed" / f"{run_id}_{suffix}.json"
        if p.exists():
            p.unlink()
    cleaned = ROOT / "processed" / f"{run_id}_cleaned.csv"
    if cleaned.exists():
        cleaned.unlink()
    run_log = ROOT / "logs" / "runs" / f"{run_id}_run.json"
    if run_log.exists():
        run_log.unlink()

# ─── Session state ─────────────────────────────────────────────────────────────

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
    "console":          [],
    "hitl_start_time":  None,
    "hitl_approved":    None,
    "error_message":    None,
    "view_run_id":      None,
    "settings": {
        "max_models":       5,
        "cv_folds":         5,
        "zone1_threshold":  0.85,
        "llm_model":        "gpt-4o-mini",
        "show_console":     True,
        "keep_run_history": 15,
        "enable_judge":     False,
    },
    "confirm_delete": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("")

    if st.button("🏠  Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    if st.button("✨  New Recommendation", use_container_width=True, type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.page = "new_run"
        st.rerun()

    if st.button("⚙️  Settings", use_container_width=True):
        st.session_state.page = "settings"
        st.rerun()

    st.divider()

    history = load_run_history()
    limit   = st.session_state.settings.get("keep_run_history", 15)

    with st.expander(f"📁  Recent Runs  ({len(history)})", expanded=True):
        if not history:
            st.caption("No runs yet — start a new recommendation.")
        else:
            for run in history[:limit]:
                rid   = run.get("run_id", "?")
                model = run.get("recommended_model") or "—"
                ds    = Path(run.get("dataset_path", "unknown")).name
                ok    = run.get("success", False)
                dot   = "🟢" if ok else "🔴"

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(
                        f"{dot} **{rid[:12]}**  \n{ds} · {model}",
                        key=f"run_{rid}",
                        use_container_width=True,
                    ):
                        data = load_run_data(rid)
                        st.session_state.view_run_id     = rid
                        st.session_state.data_card       = data["data_card"]
                        st.session_state.model_selection = data["model_selection"]
                        st.session_state.recommendation  = data["recommendation"]
                        st.session_state.eval_report     = data["eval_report"]
                        st.session_state.run_id          = rid
                        st.session_state.hitl_approved   = True
                        st.session_state.page            = "results"
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{rid}", help=f"Delete {rid}"):
                        st.session_state.confirm_delete = rid
                        st.rerun()

                if st.session_state.confirm_delete == rid:
                    st.warning(f"Delete **{rid[:12]}**? This cannot be undone.")
                    cc1, cc2 = st.columns(2)
                    if cc1.button("Yes, delete", key=f"yes_{rid}", type="primary"):
                        delete_run(rid)
                        st.session_state.confirm_delete = None
                        st.rerun()
                    if cc2.button("Cancel", key=f"no_{rid}"):
                        st.session_state.confirm_delete = None
                        st.rerun()


# ─── HOME page ────────────────────────────────────────────────────────────────

if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>MIRA</h1>
        <div class="tagline">
            Upload your business data, describe what you want to predict,
            and MIRA automatically finds and recommends the best AI model —
            with a built-in human review step before any deployment decision.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cl, cc, cr = st.columns([2, 1.2, 2])
    with cc:
        if st.button("✨  Get My Recommendation", type="primary", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "new_run"
            st.rerun()

    st.markdown("---")
    st.markdown("#### How it works — six steps from data to decision")

    st.markdown("""
    <div class="phase-row">
        <div class="phase-card">
            <span class="step-num">Step 1</span>
            <span class="phase-icon">📤</span>
            <h3>Tell us your goal</h3>
            <p>Upload your data file (CSV or Excel) and describe your business question
            in plain English. No technical knowledge required.</p>
        </div>
        <div class="phase-card">
            <span class="step-num">Step 2</span>
            <span class="phase-icon">🔎</span>
            <h3>We understand your data</h3>
            <p>MIRA automatically checks data quality, identifies patterns, and determines
            which success metric matters most for your specific goal.</p>
        </div>
        <div class="phase-card">
            <span class="step-num">Step 3</span>
            <span class="phase-icon">🏁</span>
            <h3>We test multiple approaches</h3>
            <p>Five different AI approaches are trained and rigorously stress-tested —
            checking for overfitting, data leakage, and stability — before any recommendation is made.</p>
        </div>
        <div class="phase-card">
            <span class="step-num">Step 4</span>
            <span class="phase-icon">🤖</span>
            <h3>AI writes the recommendation</h3>
            <p>MIRA reasons over all results and produces a plain-English recommendation —
            which approach to deploy, why, what risks to expect, and what to do next.</p>
        </div>
        <div class="phase-card">
            <span class="step-num">Step 5</span>
            <span class="phase-icon">👁️</span>
            <h3>You review and decide</h3>
            <p>High-confidence results can proceed automatically. For uncertain cases, you get
            a review screen where you approve or reject before anything is finalized.</p>
        </div>
        <div class="phase-card">
            <span class="step-num">Step 6</span>
            <span class="phase-icon">📊</span>
            <h3>Full report delivered</h3>
            <p>Every run produces a complete report — approach rankings, data profile,
            recommendation rationale, and an explicit production-readiness verdict.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── SETTINGS page ────────────────────────────────────────────────────────────

elif st.session_state.page == "settings":

    st.markdown("## Settings")
    st.markdown("Adjust MIRA's behaviour for future runs. Changes apply to new runs only.")

    cfg = st.session_state.settings

    st.markdown('<div class="settings-section"><h3>🏋️ Model Training</h3>', unsafe_allow_html=True)
    max_models = st.slider("Maximum approaches to train", 2, 7, cfg["max_models"],
        help="More models = more thorough search, longer runtime.")
    cv_folds   = st.slider("Cross-validation folds", 3, 10, cfg["cv_folds"],
        help="More folds = more reliable estimates, longer runtime.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="settings-section"><h3>👁️ Human Review Gate</h3>', unsafe_allow_html=True)
    zone1_threshold = st.slider(
        "Auto-proceed confidence threshold", 0.50, 1.0, cfg["zone1_threshold"], step=0.05,
        help="Runs with confidence at or above this value skip the manual review screen.",
    )
    st.caption(f"Currently: confidence ≥ **{zone1_threshold:.2f}** auto-proceeds · below requires your review.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="settings-section"><h3>🤖 AI (LLM) Settings</h3>', unsafe_allow_html=True)
    llm_opts    = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    llm_model   = st.selectbox("LLM for metric inference", llm_opts,
        index=llm_opts.index(cfg["llm_model"]) if cfg["llm_model"] in llm_opts else 0,
        help="Model used to infer the evaluation metric from your business problem description.")
    enable_judge = st.toggle("Enable LLM judge evaluation (slower, costs more)", value=cfg["enable_judge"],
        help="Runs an additional LLM-as-judge pass on the final recommendation. Adds 30–60 seconds per run.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="settings-section"><h3>🗂️ Display & History</h3>', unsafe_allow_html=True)
    show_console    = st.toggle("Show run console in results", value=cfg["show_console"])
    keep_run_history = st.slider("Recent runs to show in sidebar", 5, 30, cfg["keep_run_history"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("Save Settings", type="primary"):
        st.session_state.settings = {
            "max_models":        max_models,
            "cv_folds":          cv_folds,
            "zone1_threshold":   zone1_threshold,
            "llm_model":         llm_model,
            "enable_judge":      enable_judge,
            "show_console":      show_console,
            "keep_run_history":  keep_run_history,
        }
        st.success("Settings saved for this session.")


# ─── NEW RUN page ─────────────────────────────────────────────────────────────

elif st.session_state.page == "new_run":

    st.markdown("## New Recommendation")
    st.markdown("Fill in the details below and click **Run MIRA** when ready.")
    st.markdown("")

    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:

        st.markdown('<div class="form-section-label">Dataset</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a file", type=["csv", "xls", "xlsx"], label_visibility="collapsed")

        dataset_path = None
        columns = []

        if uploaded:
            save_dir = ROOT / "data" / "raw"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / uploaded.name
            save_path.write_bytes(uploaded.getvalue())
            dataset_path = str(save_path)
            try:
                df_prev = pd.read_excel(save_path) if uploaded.name.endswith((".xls", ".xlsx")) else pd.read_csv(save_path)
                columns = list(df_prev.columns)
                st.success(f"**{uploaded.name}** · {len(df_prev):,} rows · {len(columns)} columns")
                with st.expander("Preview (first 5 rows)"):
                    st.dataframe(df_prev.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            existing = sorted((ROOT / "data" / "raw").glob("*.csv")) + \
                       sorted((ROOT / "data" / "raw").glob("*.xls*"))
            if existing:
                st.markdown('<div class="form-section-label">Or choose an existing dataset</div>', unsafe_allow_html=True)
                choice = st.selectbox("Existing dataset", ["— select —"] + [p.name for p in existing], label_visibility="collapsed")
                if choice != "— select —":
                    dataset_path = str(ROOT / "data" / "raw" / choice)
                    try:
                        df_prev = pd.read_csv(dataset_path) if choice.endswith(".csv") else pd.read_excel(dataset_path)
                        columns = list(df_prev.columns)
                        st.info(f"**{choice}** · {len(df_prev):,} rows · {len(columns)} columns")
                        with st.expander("Preview (first 5 rows)"):
                            st.dataframe(df_prev.head(), use_container_width=True)
                    except Exception:
                        pass

        st.markdown('<div class="form-section-label">Target Column</div>', unsafe_allow_html=True)
        target_col = st.selectbox(
            "Target column",
            options=["— select —"] + columns,
            disabled=not columns,
            label_visibility="collapsed",
        )

        st.markdown('<div class="form-section-label">Business Problem</div>', unsafe_allow_html=True)
        st.caption("Describe what you are trying to predict and why it matters. MIRA will infer the right success metric from this text.")
        business_problem = st.text_area(
            "Business problem",
            placeholder="Example: A retail bank wants to identify customers at risk of churning so the retention team can intervene early — reducing churn and protecting revenue.",
            height=120,
            label_visibility="collapsed",
        )

        st.markdown("")
        can_run = (
            dataset_path is not None
            and target_col not in (None, "— select —")
            and len((business_problem or "").strip()) >= 20
        )

        if st.button("▶  Run MIRA", type="primary", use_container_width=True, disabled=not can_run):
            st.session_state.run_id           = f"run_{_uuid.uuid4().hex[:8]}"
            st.session_state.dataset_path     = dataset_path
            st.session_state.target_col       = target_col
            st.session_state.business_problem = business_problem.strip()
            st.session_state.page             = "phase1"
            st.session_state.console          = []
            st.session_state.data_card        = None
            st.session_state.model_selection  = None
            st.session_state.recommendation   = None
            st.session_state.eval_report      = None
            st.session_state.hitl_approved    = None
            st.session_state.hitl_start_time  = None
            st.rerun()

        if not can_run:
            st.caption("Upload a dataset, select a target column, and describe the business problem to continue.")


# ─── PHASE 1 ─────────────────────────────────────────────────────────────────

elif st.session_state.page == "phase1":
    st.markdown(f"## Run `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot active"></div>
        <div class="step-label active">Step 1 — Understanding your data</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Step 2 — Testing approaches</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Step 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id  = st.session_state.run_id
    out     = ROOT / "processed"
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
    with st.spinner("Profiling your data and identifying the right success metric..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        log(f"Phase 1 FAILED: {result.stderr[:300]}")
        st.session_state.error_message = result.stderr
        st.session_state.page = "error"
    else:
        log("Data profiling complete")
        st.session_state.data_card = load_json(dc_out)
        st.session_state.page = "phase2"
    st.rerun()


# ─── PHASE 2 ─────────────────────────────────────────────────────────────────

elif st.session_state.page == "phase2":
    st.markdown(f"## Run `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div>
        <div class="step-label">Step 1 — Data profiled</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div>
        <div class="step-label active">Step 2 — Testing approaches</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Step 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div>
        <div class="step-label">Report</div>
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
    with st.spinner("Training and stress-testing multiple approaches — this takes 1–3 minutes..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.returncode != 0:
        log(f"Phase 2 FAILED: {result.stderr[:300]}")
        st.session_state.error_message = result.stderr
        st.session_state.page = "error"
    else:
        log("Approach testing complete")
        st.session_state.model_selection = load_json(ms_out)
        st.session_state.page = "phase3"
    st.rerun()


# ─── PHASE 3 ─────────────────────────────────────────────────────────────────

elif st.session_state.page == "phase3":
    st.markdown(f"## Run `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Step 1</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Step 2</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Step 3 — Writing recommendation</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("MIRA is reasoning over the results and writing your recommendation. This typically takes 1–5 minutes.")

    run_id = st.session_state.run_id
    out    = ROOT / "processed"

    with st.spinner("Writing deployment recommendation..."):
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
            log("Recommendation written")
            st.session_state.page = "hitl"
        except Exception as e:
            log(f"Phase 3 error: {e}")
            st.session_state.error_message = str(e)
            st.session_state.page = "error"
    st.rerun()


# ─── HITL GATE page ───────────────────────────────────────────────────────────

elif st.session_state.page == "hitl":
    run_id = st.session_state.run_id
    rec    = st.session_state.recommendation or {}
    dc     = st.session_state.data_card or {}
    ms     = st.session_state.model_selection or {}

    confidence   = rec.get("confidence_score", 0.0) or 0.0
    flags        = rec.get("flags", [])
    routing_zone = rec.get("routing_zone", "zone_2")
    model        = rec.get("recommended_model", "Unknown")
    auc          = rec.get("primary_metric_value")
    exec_summary = rec.get("executive_summary", "")
    review_reason = rec.get("human_review_reason", "")

    escalation = evaluate_escalation_rules(dc, ms)
    esc_rules  = escalation.get("rules_triggered", [])
    if escalation.get("hard_escalation"):
        routing_zone = "zone_3"

    if st.session_state.hitl_start_time is None:
        st.session_state.hitl_start_time = time.time()

    st.markdown(f"## Review Required · `{run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Step 1</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Step 2</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Step 3</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    if routing_zone == "zone_1":
        st.markdown('<span class="zone-badge zone1-badge">Auto-Approve Eligible</span>', unsafe_allow_html=True)
        st.success(f"Confidence {confidence:.3f} — high enough to proceed automatically. No flags detected.")
        if st.button("✅  Proceed to Report", type="primary"):
            st.session_state.hitl_approved = True
            st.session_state.page = "eval"
            st.rerun()
    else:
        box_cls   = "hitl-zone3" if routing_zone == "zone_3" else "hitl-zone2"
        badge_cls = "zone3-badge" if routing_zone == "zone_3" else "zone2-badge"
        zone_label = "Priority Review Required" if routing_zone == "zone_3" else "Review Recommended"

        st.markdown(f'<div class="{box_cls}"><span class="zone-badge {badge_cls}">{zone_label}</span></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Recommended Approach", model)
        col2.metric("Confidence Score", f"{confidence:.3f}")
        col3.metric("Primary Metric", f"{auc:.4f}" if isinstance(auc, float) else "—")

        if flags:
            st.markdown("**Flags**")
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
            st.markdown("**Summary**")
            st.info(exec_summary)

        fds = rec.get("feature_drivers", [])
        if fds:
            with st.expander("Top Predictors"):
                fd_df = pd.DataFrame(fds[:5])
                if "feature" in fd_df.columns and "importance" in fd_df.columns:
                    st.bar_chart(fd_df.set_index("feature")["importance"])

        st.divider()
        st.markdown("### Your Decision")
        st.caption("Your rationale is logged for audit purposes.")

        CATS = {
            "PERFORMANCE_ACCEPTABLE":  "Performance is acceptable for this specific business context",
            "BUSINESS_PRIORITY":       "Approach chosen based on cost, interpretability, or team familiarity",
            "DATA_QUALITY_RESOLVED":   "Flagged data issue confirmed as not deployment-blocking",
            "DOMAIN_KNOWLEDGE":        "Applied domain expertise the system could not infer",
            "METRIC_MISMATCH":         "Override the inferred success metric",
            "RISK_ACCEPTED":           "Explicitly acknowledge and accept the flagged risk",
            "AGENT_ERROR":             "Recommendation was factually wrong — flag for review",
        }

        with st.form("hitl_form"):
            decision  = st.radio("Decision", ["Approve — proceed to report", "Reject — do not proceed"])
            category  = st.selectbox("Reason Category", list(CATS.keys()), format_func=lambda k: f"{k}  —  {CATS[k]}")
            rationale = st.text_area("Rationale (required)", placeholder="Briefly explain your decision...", height=90)
            submitted = st.form_submit_button("Submit Decision", type="primary")

        if submitted:
            if not rationale.strip():
                st.error("Please provide a rationale before submitting.")
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
                log(f"Review: {'APPROVED' if approved else 'REJECTED'} · zone={routing_zone} · {duration}s")
                st.session_state.hitl_approved = approved
                st.session_state.page = "eval" if approved else "results"
                st.rerun()


# ─── EVAL page ────────────────────────────────────────────────────────────────

elif st.session_state.page == "eval":
    st.markdown(f"## Run `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Step 1</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Step 2</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Step 3</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Generating Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id   = st.session_state.run_id
    out_path = str(ROOT / "processed") + "/"
    priority = (st.session_state.data_card or {}).get("priority_metric", "roc_auc")

    with st.spinner("Running quality checks and generating your report..."):
        try:
            EvalRunner(run_id=run_id, output_path=out_path, priority_metric=priority).run()
            ep = ROOT / "processed" / f"{run_id}_eval_report.json"
            st.session_state.eval_report = load_json(ep)
            log("Report complete")
        except Exception as e:
            log(f"Eval error: {e}")

    st.session_state.page = "results"
    st.rerun()


# ─── RESULTS page ─────────────────────────────────────────────────────────────

elif st.session_state.page in ("results", "view_run"):
    run_id   = st.session_state.run_id or st.session_state.view_run_id or "?"
    dc       = st.session_state.data_card or {}
    ms       = st.session_state.model_selection or {}
    rec      = st.session_state.recommendation or {}
    ev       = st.session_state.eval_report or {}
    rejected = st.session_state.hitl_approved is False

    st.markdown(f"## Report · `{run_id}`")
    if rejected:
        st.error("Run rejected at the review step. Audit log saved. Full report was not generated.")

    tabs = st.tabs(["📋 Summary", "🔍 Data Profile", "🏆 Approach Rankings", "🎯 Recommendation", "📊 Eval Report"])

    # ── Summary ──
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended Approach", rec.get("recommended_model", "—"))
        c2.metric("Confidence Score",     f"{rec.get('confidence_score', 0):.3f}" if isinstance(rec.get("confidence_score"), float) else "—")
        c3.metric("Review Zone",          rec.get("routing_zone", "—"))
        c4.metric("Stress Test Verdict",  ms.get("test_verdict", "—"))

        st.markdown("")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Stress Test Results**")
            for f in ms.get("test_findings", []):
                (st.success if "PASSED" in f.upper() else st.error)(f)
        with cb:
            flags = rec.get("flags", [])
            st.markdown("**Flags**")
            if flags:
                for flag in flags:
                    st.warning(flag)
            else:
                st.success("No flags — clean run")

        if rec.get("executive_summary"):
            st.divider()
            st.markdown("**Executive Summary**")
            st.markdown(
                f'<div class="rec-card"><p>{rec["executive_summary"]}</p></div>',
                unsafe_allow_html=True,
            )

        if st.session_state.settings.get("show_console", True) and st.session_state.console:
            with st.expander("Run Console"):
                st.markdown(
                    '<div class="console">' + "<br>".join(st.session_state.console) + '</div>',
                    unsafe_allow_html=True,
                )

    # ── Data Profile ──
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
                    st.bar_chart(pd.DataFrame({"Proportion": list(cd.values())}, index=[f"Class {k}" for k in cd]))
                    st.caption(f"Minority class: {dc.get('minority_class_ratio', 0):.1%}")
            with cb:
                st.markdown("**Strongest Predictors**")
                hcf = dc.get("high_correlation_features", [])
                if hcf:
                    hcf_df = pd.DataFrame(hcf).rename(columns={"feature": "Feature", "correlation": "Correlation"})
                    st.bar_chart(hcf_df.set_index("Feature")["Correlation"])

            st.markdown("**Why this success metric?**")
            st.info(f"**Metric:** `{dc.get('priority_metric','?')}`  \n**Reason:** {dc.get('metric_reason','—')}")

            cl = dc.get("cleaning_log", [])
            if cl:
                with st.expander("Data Cleaning Log"):
                    for e in cl:
                        st.markdown(f"- {e}")

    # ── Approach Rankings ──
    with tabs[2]:
        if not ms:
            st.caption("No approach data available.")
        else:
            models  = ms.get("models_trained", [])
            winner  = ms.get("selected_model", "")
            pm      = dc.get("priority_metric", "roc_auc")
            mcol    = {"roc_auc": "cv_roc_auc_mean", "recall": "cv_recall_mean", "f1_score": "cv_f1_mean", "precision": "cv_precision_mean"}.get(pm, "cv_roc_auc_mean")

            rows = []
            for m in sorted(models, key=lambda x: x.get(mcol, 0), reverse=True):
                gap = m.get("overfitting_gap", 0)
                rows.append({
                    "Approach":      ("🏆 " if m["name"] == winner else "") + m["name"],
                    "AUC":           round(m.get("cv_roc_auc_mean", 0), 4),
                    "Recall":        round(m.get("cv_recall_mean", 0), 4),
                    "F1":            round(m.get("cv_f1_mean", 0), 4),
                    "Precision":     round(m.get("cv_precision_mean", 0), 4),
                    "Overfit Gap":   round(gap, 4),
                    "Stability":     "⚠️ High risk" if gap > 0.10 else ("⚡ Watch" if gap > 0.07 else "✅ Good"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Ranked by **{pm}** — inferred from your business problem description.")

            st.markdown("**Generalization Risk by Approach**")
            gap_data = pd.DataFrame([{"Approach": m["name"], "Gap": round(m.get("overfitting_gap", 0), 4)} for m in models]).set_index("Approach")
            st.bar_chart(gap_data)
            st.caption("Gap = performance on training data minus performance on unseen data. Above 0.10 is a warning sign.")

            fi = ms.get("feature_importance", {})
            if fi:
                with st.expander("What drives the predictions?"):
                    fi_df = pd.DataFrame(sorted(fi.items(), key=lambda x: x[1], reverse=True), columns=["Feature", "Importance"]).set_index("Feature")
                    st.bar_chart(fi_df)

    # ── Recommendation ──
    with tabs[3]:
        if not rec or not rec.get("recommended_model"):
            st.caption("No recommendation produced for this run.")
        else:
            st.markdown(
                f'<div class="rec-card">'
                f'<h2>Deploy: {rec.get("recommended_model")}</h2>'
                f'<p>{rec.get("selection_reason", "")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Primary Metric",   f"{rec.get('primary_metric_value', 0):.4f}" if isinstance(rec.get("primary_metric_value"), float) else "—")
            c2.metric("Confidence Score", f"{rec.get('confidence_score', 0):.3f}"     if isinstance(rec.get("confidence_score"), float) else "—")
            c3.metric("Alternative",      rec.get("alternative_model", "—"))

            st.divider()
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Next Steps**")
                for s in rec.get("next_steps", []):
                    st.markdown(f"- {s}")
                st.markdown("**Deployment Considerations**")
                for d in rec.get("deployment_considerations", []):
                    st.markdown(f"- {d}")
            with cb:
                st.markdown("**Risks**")
                for r in rec.get("risks", []):
                    st.warning(r)
                st.markdown("**Tradeoffs**")
                for t in rec.get("tradeoffs", []):
                    st.markdown(f"- {t}")

            bi = rec.get("business_impact", {})
            if bi:
                st.divider()
                st.markdown("**Business Impact**")
                for k, v in bi.items():
                    st.markdown(f"**{k.replace('_',' ').title()}:** {v}")

            with st.expander("Full Recommendation (technical detail)"):
                st.json(rec)

    # ── Eval Report ──
    with tabs[4]:
        if not ev:
            st.caption("No report." + (" Run was rejected at the review step." if rejected else ""))
        else:
            summary = ev.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Behaviour Score", f"{summary.get('behavior_avg_pct', 0):.1f}%")
            c2.metric("Unit Tests",      f"{summary.get('unit_tests_pct', 0):.1f}%")
            c3.metric("Quality Score",   f"{summary.get('quality_pct', 0):.1f}%")
            c4.metric("Production Ready","✅ Yes" if ev.get("production_checklist", {}).get("production_ready") else "❌ No")

            st.divider()
            prod = ev.get("production_checklist", {})
            if prod.get("checklist"):
                st.markdown("**Production Checklist**")
                for item in prod["checklist"]:
                    icon = "✅" if item["passed"] else "❌"
                    crit = " *(critical)*" if item.get("critical") else ""
                    st.markdown(
                        f"{icon} **{item['item'].replace('_',' ').title()}**{crit}  \n"
                        f"<span style='color:#64748b;font-size:.8rem'>{item.get('detail','')}</span>",
                        unsafe_allow_html=True,
                    )

            hitl_ev = ev.get("hitl_gate", {})
            if hitl_ev:
                st.divider()
                st.markdown("**Review Gate Risk Score**")
                rs = hitl_ev.get("total_risk_score", 0)
                th = hitl_ev.get("risk_threshold", 5)
                st.metric("Risk Score", f"{rs} / {th}", delta=f"{'Below' if rs < th else 'Above'} threshold")
                for risk in hitl_ev.get("risks_identified", []):
                    fn = st.error if risk.get("severity") == "HIGH" else st.warning
                    fn(f"**{risk['risk']}** — {risk['detail']}")

            with st.expander("Full Report (technical detail)"):
                st.json(ev)


# ─── ERROR page ───────────────────────────────────────────────────────────────

elif st.session_state.page == "error":
    st.error("## Something went wrong")
    if st.session_state.error_message:
        with st.expander("Error details"):
            st.code(st.session_state.error_message)
    st.info("Click **New Recommendation** in the sidebar to start again.")
