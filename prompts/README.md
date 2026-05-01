# Prompt Version Changelog
## MIRA — Model Intelligence & Recommendation Agent

> One prompt file per version. Never edit a committed file — create a new one.
> Commit message format: `prompt: vX.Y.Z reason`

| Version | File | Date | What Changed | Why |
|---------|------|------|-------------|-----|
| v0.1.0 | `mira_agent_v0_1_0/` (multi-file) | 2026-04-01 | Initial multi-persona structure (Data Analyst, ML Engineer, ML Test Engineer, Data Scientist) | Baseline — separate prompts per phase |
| v0.2.0 | `mira_agent_v0_2_0.md` | 2026-04-05 | Consolidated to single-agent prompt; added CoT gates between phases | Multi-persona approach produced inconsistent phase handoffs |
| v0.3.0 | `mira_agent_v0_3_0.md` | 2026-04-10 | Added explicit CoT reasoning blocks (cite specific numbers before phase transition); expanded model_selection and recommendation schemas | Agent was skipping phases without verifying prior output |
| v0.4.0 | `mira_agent_v0_4_0.md` | 2026-04-15 | Added full Python script templates per phase — retired | Too long for gpt-4o-mini context window; agent ignored later instructions |
| v0.4.1 | `mira_agent_v0_4_1.md` | 2026-04-18 | JSON schema templates (not Python); short `SCHEMA OK` assertion per phase; paths read from Run Context; LightGBM enforced | Replaced script templates with JSON templates to cut prompt length |
| v0.5.0 | `mira_agent_v0_5_0.md` | 2026-04-28 | **DECIDING mode** — added confidence_score (reasoning certainty, not AUC), routing_zone, flags[], three-zone HITL gate; mira-recommend AgentSkill integrated; rubber-stamp prevention | Session 6 HITL gate design; confidence score was measuring output quality, not reasoning certainty |
