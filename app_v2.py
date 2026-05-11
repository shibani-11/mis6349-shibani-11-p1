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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0a18; }
[data-testid="stSidebar"]          { background: #0f0f1f; border-right: 1px solid #1e1b4b; }
section.main > div                 { padding-top: 1.5rem; }
p, li                              { color: #94a3b8; }

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
.hero-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 1rem;
}
.hero-tagline {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.75;
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
    font-size: 0.7rem;
    border-radius: 6px;
    padding: 2px 10px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.phase-icon  { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.phase-title { color: #e2e8f0; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; }
.phase-desc  { color: #6b7280; font-size: 0.82rem; line-height: 1.65; }

.step-bar { display:flex; align-items:center; gap:.5rem; margin:1.2rem 0 1.8rem; flex-wrap:wrap; }
.step-dot  { width:10px; height:10px; border-radius:50%; background:#1e1b4b; flex-shrink:0; }
.step-dot.done   { background:#06b6d4; }
.step-dot.active { background:#06b6d4; box-shadow:0 0 10px #06b6d4; }
.step-label      { font-size:.8rem; color:#6b7280; white-space:nowrap; }
.step-label.active { color:#06b6d4; font-weight:600; }
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
.rec-model  { color: #a855f7; font-size: 1.8rem; font-weight: 900; margin-bottom: 0.2rem; }
.rec-label  { color: #a855f7; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 1rem 0 0.2rem; }
.rec-value  { color: #e2e8f0; font-size: 0.92rem; line-height: 1.7; }
.verdict-yes { display:inline-block; background:#064e3b; color:#34d399; font-weight:700; padding:4px 18px; border-radius:20px; font-size:0.9rem; margin-bottom:1rem; }
.verdict-no  { display:inline-block; background:#450a0a; color:#f87171;  font-weight:700; padding:4px 18px; border-radius:20px; font-size:0.9rem; margin-bottom:1rem; }

.json-file-btn {
    background: #13132a;
    border: 1px solid #1e1b4b;
    border-radius: 6px;
    padding: 4px 10px;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-bottom: 4px;
    width: 100%;
    text-align: left;
    cursor: pointer;
}
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
    for p in sorted((ROOT / "logs" / "runs").glob("*_run.json"), reverse=True)[:20]:
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return runs

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

# ── Sidebar — Run History as folders ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='color:#a855f7;font-weight:800;font-size:0.8rem;"
        "text-transform:uppercase;letter-spacing:0.1em;padding:0.5rem 0;'>"
        "📁 Run History</div>",
        unsafe_allow_html=True,
    )

    if st.button("＋  New Run", use_container_width=True, type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.page = "run"
        st.rerun()

    st.markdown("---")

    history = load_run_history()
    if not history:
        st.caption("No runs yet.")
    else:
        for run in history:
            rid   = run.get("run_id", "?")
            model = run.get("recommended_model") or "—"
            ds    = Path(run.get("dataset_path", "unknown")).name
            ok    = run.get("success", False)
            dot   = "🟢" if ok else "🔴"

            with st.expander(f"{dot}  `{rid[:12]}`", expanded=False):
                st.caption(f"{ds}  ·  {model}")
                st.markdown("**Output files:**")

                for suffix, label, icon in [
                    ("data_card",       "data_card.json",       "📊"),
                    ("model_selection",  "model_selection.json", "🏁"),
                    ("recommendation",   "recommendation.json",  "🤖"),
                    ("eval_report",      "eval_report.json",     "📋"),
                ]:
                    p = processed(rid, suffix)
                    if p.exists():
                        if st.button(
                            f"{icon}  {label}",
                            key=f"json_{rid}_{suffix}",
                            use_container_width=True,
                        ):
                            st.session_state.sidebar_json_view = {
                                "label": label,
                                "data": load_json(p),
                            }
                            st.rerun()
                    else:
                        st.caption(f"~~{icon} {label}~~ (not yet)")

                if st.button(
                    "View Recommendation",
                    key=f"view_{rid}",
                    use_container_width=True,
                ):
                    st.session_state.run_id          = rid
                    st.session_state.recommendation  = load_json(processed(rid, "recommendation"))
                    st.session_state.data_card       = load_json(processed(rid, "data_card"))
                    st.session_state.model_selection = load_json(processed(rid, "model_selection"))
                    st.session_state.eval_report     = load_json(processed(rid, "eval_report"))
                    st.session_state.hitl_approved   = True
                    st.session_state.page            = "results"
                    st.session_state.sidebar_json_view = None
                    st.rerun()

# ── Sidebar JSON viewer ───────────────────────────────────────────────────────
if st.session_state.sidebar_json_view:
    jv = st.session_state.sidebar_json_view
    st.markdown(f"### 📄 {jv['label']}")
    if st.button("← Back"):
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
        <div class="hero-title">Model Intelligence<br>&amp; Recommendation Agent</div>
        <div class="hero-tagline">
            Upload your data. Describe your business goal in plain English.<br>
            MIRA runs the full ML pipeline and tells you exactly what to deploy — and why.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="phase-grid">
        <div class="phase-card">
            <span class="phase-num">Phase 1</span>
            <span class="phase-icon">🔎</span>
            <div class="phase-title">Understand Your Data</div>
            <div class="phase-desc">
                MIRA cleans and profiles your dataset, then uses AI to infer
                the right success metric directly from your business problem —
                no dropdowns, no manual selection.
            </div>
        </div>
        <div class="phase-card">
            <span class="phase-num">Phase 2</span>
            <span class="phase-icon">🏁</span>
            <div class="phase-title">Test Multiple Approaches</div>
            <div class="phase-desc">
                Five models are trained with 5-fold cross-validation and
                stress-tested for overfitting, data leakage, and stability
                before any recommendation is made.
            </div>
        </div>
        <div class="phase-card">
            <span class="phase-num">Phase 3</span>
            <span class="phase-icon">🤖</span>
            <div class="phase-title">AI Writes the Recommendation</div>
            <div class="phase-desc">
                The agent reasons over all results and delivers a plain-English
                deployment recommendation with a confidence score, risk flags,
                and an explicit YES / NO verdict.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([2, 1, 2])[1]
    with col:
        if st.button("🚀  Start MIRA", type="primary", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "run"
            st.rerun()


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
        business_problem = st.text_area(
            "Business problem",
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

        if st.button("▶  Run MIRA", type="primary", use_container_width=True, disabled=not can_run):
            st.session_state.run_id           = f"run_{_uuid.uuid4().hex[:8]}"
            st.session_state.dataset_path     = dataset_path
            st.session_state.target_col       = target_col
            st.session_state.business_problem = business_problem.strip()
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

    st.markdown(f"## Running · `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot active"></div><div class="step-label active">Phase 1 — Understanding your data</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Phase 2 — Testing approaches</div>
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
        st.session_state.error_message = result.stderr
        st.session_state.page = "error"
    else:
        log("Phase 1 complete — data profiled, metric inferred")
        st.session_state.data_card = load_json(dc_out)
        st.session_state.page = "phase2"
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "phase2":

    st.markdown(f"## Running · `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Phase 2 — Testing 5 models</div>
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
    with st.spinner("Training and stress-testing 5 models — this takes 1–3 minutes..."):
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

    st.markdown(f"## Running · `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 ✓</div>
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

    with st.spinner("Writing deployment recommendation — typically 1–5 minutes..."):
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

    st.markdown(f"## Human Review · `{run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 3 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Review</div>
        <div class="step-sep"></div>
        <div class="step-dot"></div><div class="step-label">Report</div>
    </div>
    """, unsafe_allow_html=True)

    if routing_zone == "zone_1":
        st.markdown('<span class="zone-badge zone1-badge">Zone 1 — Auto-Approve Eligible</span>', unsafe_allow_html=True)
        st.success(f"Confidence {confidence:.3f} — no flags detected. You can proceed automatically.")
        if st.button("✅  Proceed to Report", type="primary"):
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

    st.markdown(f"## Running · `{st.session_state.run_id}`")
    st.markdown("""
    <div class="step-bar">
        <div class="step-dot done"></div><div class="step-label">Phase 1 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 2 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Phase 3 ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot done"></div><div class="step-label">Review ✓</div>
        <div class="step-sep"></div>
        <div class="step-dot active"></div><div class="step-label active">Generating Report</div>
    </div>
    """, unsafe_allow_html=True)

    run_id   = st.session_state.run_id
    out_path = str(ROOT / "processed") + "/"
    priority = (st.session_state.data_card or {}).get("priority_metric", "roc_auc")

    with st.spinner("Running 7-layer evaluation and generating final report..."):
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
# RESULTS — plain English recommendation
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    run_id   = st.session_state.run_id or "?"
    rec      = st.session_state.recommendation or {}
    dc       = st.session_state.data_card or {}
    ms       = st.session_state.model_selection or {}
    ev       = st.session_state.eval_report or {}
    rejected = st.session_state.hitl_approved is False

    st.markdown(f"## Recommendation · `{run_id}`")

    if rejected:
        st.error("This run was rejected at the review step. No deployment recommendation was finalized. Audit log saved.")
        if st.button("Start a New Run"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.page = "run"
            st.rerun()
        st.stop()

    # ── Verdict banner ──
    # Fallback chain for model name across different schema versions
    model = (
        rec.get("recommended_model")
        or rec.get("selected_model")
        or (rec.get("recommendations") or {}).get("selected_model")
        or ms.get("selected_model")
        or "—"
    )
    confidence  = float(rec.get("confidence_score", 0) or 0)
    zone        = rec.get("routing_zone", "")
    summary     = rec.get("executive_summary", "")
    flags       = rec.get("flags", []) or []
    metric      = dc.get("priority_metric", "—")
    metric_rsn  = dc.get("metric_reason", "")
    business_impact  = rec.get("business_impact", "")
    selection_reason = (
        rec.get("selection_reason")
        or rec.get("selection_reasoning")
        or ms.get("selection_reasoning")
        or (rec.get("recommendations") or {}).get("reason")
        or ""
    )
    next_steps  = rec.get("next_steps", []) or []
    risks       = rec.get("risks", []) or []

    # Verdict: zone_3 = hard stop; otherwise approved runs are a deploy
    if zone == "zone_3":
        verdict_yes = False
    elif zone in ("zone_1", "zone_2"):
        verdict_yes = True
    elif confidence >= 0.70:
        verdict_yes = True
    elif summary:
        verdict_yes = any(w in summary.upper() for w in ("DEPLOY", "RECOMMEND", "YES"))
    else:
        verdict_yes = True  # approved & no signal = optimistic default

    verdict_cls = "verdict-yes" if verdict_yes else "verdict-no"
    verdict_txt = "✅  DEPLOY" if verdict_yes else "❌  DO NOT DEPLOY"
    zone_display = zone.replace("_", " ").title() if zone else "—"

    st.markdown(
        f"""
        <div class="rec-block">
            <div class="rec-model">{model}</div>
            <div class="{verdict_cls}">{verdict_txt}</div>

            <div class="rec-label">Confidence Score</div>
            <div class="rec-value">{confidence:.2f} / 1.00 &nbsp;·&nbsp; {zone_display}</div>

            <div class="rec-label">Why This Model</div>
            <div class="rec-value">{selection_reason or '—'}</div>

            <div class="rec-label">Success Metric</div>
            <div class="rec-value"><strong>{metric}</strong> — {metric_rsn or '—'}</div>

            <div class="rec-label">Business Impact</div>
            <div class="rec-value">{business_impact or '—'}</div>

            <div class="rec-label">Executive Summary</div>
            <div class="rec-value">{summary or '—'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Flags & next steps ──
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
        st.markdown("#### Next Steps")
        if next_steps:
            for step in next_steps:
                st.markdown(f"- {step}")
        else:
            st.caption("—")

        st.markdown("#### Risks")
        if risks:
            for risk in risks:
                st.markdown(f"- {risk}")
        else:
            st.caption("No risks documented.")

    # ── Eval summary ──
    if ev:
        st.markdown("---")
        st.markdown("#### Evaluation Report")
        overall = ev.get("overall_score")
        unit    = ev.get("unit_tests", {})
        prod    = ev.get("production_checklist", {})

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Overall Score",   f"{overall:.1%}" if isinstance(overall, float) else "—")
        ec2.metric("Unit Tests",      f"{unit.get('passed', '?')}/{unit.get('total', '?')}")
        ec3.metric("Production Ready", "✅ Yes" if prod.get("deployment_approved") else "❌ No")

    # ── New run CTA ──
    st.markdown("")
    if st.button("Run MIRA Again", type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.page = "run"
        st.rerun()


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