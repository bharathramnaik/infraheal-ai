# 🏆 TCS & AMD AI Hackathon — Strategic Idea & Implementation Plan

## Hackathon Context

| Parameter | Detail |
|---|---|
| **Team** | team-790 (bharathram.naiktv) |
| **Track** | Track 1 – Agents |
| **GPU Time** | 4 hours per 24-hour window |
| **Storage** | 28 GB on GPU cloud, save to `/workspace/shared` |
| **Environment** | ROCm + vLLM on AMD Instinct GPU |
| **Submission Deadline** | 14th June 2026, 11:59 PM IST |
| **Current Date** | 11th June 2026 (~3 days remaining) |

---

## Strategic Use Case Analysis

I evaluated all **50+ use cases** across 3 tracks against these constraints:

| Constraint | Impact |
|---|---|
| **4 hrs GPU time** | Must be efficient — no long fine-tuning jobs |
| **ROCm + vLLM** | Optimized for LLM inference, not training |
| **Public datasets only** | No TCS-proprietary data |
| **Solo developer** | Must be completable by 1 person |
| **Scoring: 40% Technical** | Working demo is critical |
| **Scoring: 15% Innovation** | Novel architecture matters |

### Why Track 1 (Agents) is the Best Fit

- **Track 3 (Fine-Tuning)**: ❌ Requires significant GPU training time — risky with 4-hour slots
- **Track 2 (Multimodal)**: ⚠️ Needs video/image datasets + multiple model loading — memory-heavy
- **Track 1 (Agents)**: ✅ Uses vLLM for fast inference, multi-agent orchestration shows technical depth, public data available

---

## 🎯 Recommended Use Case: AGENTS_026 — Autonomous Incident Diagnosis & Resolution Agent

> *"Build an agentic AI system that continuously monitors infrastructure (logs, metrics, events), identifies anomalies, performs root cause analysis, and executes remediation actions (restart services, scale resources) with minimal human intervention."*

### Why This Is the Winning Choice

| Factor | Score | Reasoning |
|---|---|---|
| **Feasibility (4hrs GPU)** | ⭐⭐⭐⭐⭐ | Pure inference with vLLM — no training needed |
| **Technical Depth** | ⭐⭐⭐⭐⭐ | Multi-agent orchestration, RAG, tool-calling, anomaly detection |
| **Innovation** | ⭐⭐⭐⭐⭐ | Combines ML anomaly detection + LLM reasoning + autonomous remediation |
| **Public Data** | ⭐⭐⭐⭐⭐ | Synthetic logs + open monitoring datasets abundant |
| **Demo Impact** | ⭐⭐⭐⭐⭐ | Real-time dashboard with live anomaly detection is visually stunning |
| **Business Relevance** | ⭐⭐⭐⭐⭐ | Every enterprise runs IT infra — universal applicability |
| **AMD Alignment** | ⭐⭐⭐⭐⭐ | Showcases vLLM on ROCm for real-time agentic inference |

---

## Solution Architecture: **"InfraHeal AI"**

> An autonomous, multi-agent IT operations intelligence platform that detects, diagnoses, and resolves infrastructure incidents in real-time.

```
┌─────────────────────────────────────────────────────────────────┐
│                     InfraHeal AI Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 DATA INGESTION LAYER                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ System   │  │ Metrics  │  │ Network  │  │ App      │       │
│  │ Logs     │  │ (CPU/Mem)│  │ Events   │  │ Traces   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
│       └──────────────┴──────┬───────┴──────────────┘             │
│                             ▼                                    │
│  🔍 ANOMALY DETECTION ENGINE (ML-based)                         │
│  ┌─────────────────────────────────────────────┐                │
│  │ • Statistical Anomaly Detection (Z-score)   │                │
│  │ • Pattern Matching (Regex + Severity)       │                │
│  │ • Time-Series Analysis (Rolling Windows)    │                │
│  └────────────────────┬────────────────────────┘                │
│                       ▼                                          │
│  🤖 MULTI-AGENT ORCHESTRATOR (vLLM on AMD GPU)                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │                                                    │         │
│  │  Agent 1: 🔎 TRIAGE AGENT                         │         │
│  │  → Classifies severity, categorizes incident      │         │
│  │                                                    │         │
│  │  Agent 2: 🧠 ROOT CAUSE ANALYSIS AGENT            │         │
│  │  → Correlates logs/metrics, identifies root cause  │         │
│  │  → Uses RAG over runbooks & past incidents         │         │
│  │                                                    │         │
│  │  Agent 3: 🔧 REMEDIATION AGENT                    │         │
│  │  → Generates & executes fix actions                │         │
│  │  → Tool-calling: restart, scale, rollback          │         │
│  │                                                    │         │
│  │  Agent 4: 📋 REPORTING AGENT                      │         │
│  │  → Creates incident report with timeline           │         │
│  │  → Generates postmortem with recommendations       │         │
│  │                                                    │         │
│  └────────────────────────────────────────────────────┘         │
│                       ▼                                          │
│  📺 INTERACTIVE DASHBOARD (Gradio/Streamlit)                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Real-time log stream with anomaly highlighting   │         │
│  │ • Agent reasoning chain visualization              │         │
│  │ • Remediation action log with approval workflow    │         │
│  │ • Performance metrics (latency, tokens, GPU usage) │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## User Review Required

> [!IMPORTANT]
> **Use Case Selection**: I recommend **AGENTS_026 (Autonomous Incident Diagnosis & Resolution Agent)** as the optimal choice. However, here are strong runner-up options if you have a different preference:
> - **AGENTS_007** — Telecom NOC Agentic Copilot (very similar, but telecom-specific)
> - **AGENTS_032** — Unified Observability & RCA Agent (cross-tower correlation)
> - **AGENTS_003** — Policy & Document Comparison Assistant (simpler but lower innovation score)
> - **MULTIMODAL_007** — Mutation to Mechanism to Therapy (if you prefer biomedical/science track)

> [!IMPORTANT]
> **LLM Model Choice**: The vLLM environment likely has these models available:
> - **Qwen2.5-7B-Instruct** (recommended — great tool-calling, efficient on single GPU)
> - **Llama-3.1-8B-Instruct** (good alternative)
> - **DeepSeek-R1-Distill-Qwen-7B** (strong reasoning)
> 
> Which model would you prefer, or should I design for Qwen2.5-7B?

> [!WARNING]
> **GPU Time Strategy**: You have ~3 days with 4hrs/day = ~12 hours max GPU time. I recommend:
> - **Session 1 (Tonight)**: Setup environment, test vLLM, run basic pipeline
> - **Session 2 (June 12)**: Full agent pipeline development & testing
> - **Session 3 (June 13)**: Polish demo, record video, final testing
> - All code prep, data generation, and non-GPU work done LOCALLY first

---

## Open Questions

> [!IMPORTANT]
> 1. **Are you working solo or do you have team members?** This affects how we split the work.
> 2. **Do you have any domain preference?** (IT Ops is universally applicable, but if you have expertise in telecom, finance, etc., we can tailor)
> 3. **Preferred UI framework?** I recommend **Gradio** (easiest in Jupyter) or **Streamlit** (prettier dashboards)

---

## Proposed Implementation

### Component 1: Synthetic Data Generator (LOCAL — No GPU needed)

#### [NEW] `data_generator.py`
- Generate realistic synthetic infrastructure logs (syslog, application, network)
- Create metrics time-series data (CPU, memory, disk, network)
- Inject known anomaly patterns (memory leaks, disk full, service crashes, DDoS patterns)
- Generate a runbook knowledge base for RAG

#### [NEW] `sample_data/` directory
- `system_logs.jsonl` — 10,000 synthetic log entries
- `metrics.csv` — Time-series infrastructure metrics
- `runbooks.json` — Incident response procedures for RAG
- `past_incidents.json` — Historical incidents for context

---

### Component 2: Anomaly Detection Engine (LOCAL + GPU)

#### [NEW] `anomaly_detector.py`
- Statistical anomaly detection (Z-score, IQR on metrics)
- Log pattern matching (error/critical detection with severity scoring)
- Correlation engine (temporal correlation of events across sources)
- Alert generation with context packaging for agents

---

### Component 3: Multi-Agent Orchestrator (GPU — vLLM)

#### [NEW] `agents/orchestrator.py`
- Main agent orchestration loop using vLLM's OpenAI-compatible API
- Agent communication protocol (structured JSON message passing)
- Tool registry for remediation actions

#### [NEW] `agents/triage_agent.py`
- Severity classification (P1-P4)
- Incident categorization (infrastructure, application, network, security)
- Initial impact assessment

#### [NEW] `agents/rca_agent.py`
- Root Cause Analysis using log correlation + LLM reasoning
- RAG-powered runbook lookup (using embeddings or BM25)
- Generates structured RCA report with confidence scores

#### [NEW] `agents/remediation_agent.py`
- Generates remediation actions (restart, scale, rollback, config change)
- Tool-calling interface for executing actions
- Safety checks and human-in-the-loop approval for critical actions

#### [NEW] `agents/reporting_agent.py`
- Generates incident timeline
- Creates postmortem document
- Produces executive summary with business impact

---

### Component 4: RAG Pipeline (GPU)

#### [NEW] `rag/knowledge_base.py`
- BM25-based document retrieval (no embedding model needed — saves GPU memory)
- Runbook indexing and search
- Past incident retrieval for pattern matching

---

### Component 5: Interactive Dashboard (GPU — Gradio)

#### [NEW] `dashboard.py`
- Real-time log stream viewer with anomaly highlighting
- Agent reasoning chain visualization (step-by-step thinking)
- Remediation action panel with approve/reject workflow
- Performance metrics panel (latency, tokens used, GPU memory)
- Incident history and postmortem viewer

---

### Component 6: Jupyter Notebook (GPU — Main Entry Point)

#### [NEW] `InfraHeal_AI_Demo.ipynb` (converted from .py for Jupyter)
- Cell 1: Environment setup & vLLM initialization
- Cell 2: Data loading & anomaly detection
- Cell 3: Agent pipeline execution
- Cell 4: Dashboard launch
- Cell 5: Performance benchmarking (for Slide 4 metrics)

---

### Component 7: Presentation Materials (LOCAL)

#### [NEW] `presentation/` directory
- Architecture diagram
- Demo flow script
- Performance metrics collection

---

## Execution Timeline

### Phase 1: Pre-GPU Prep (NOW — Local Machine)
**Estimated time: 2-3 hours**
- [x] Read all hackathon documents
- [ ] Write complete codebase locally
- [ ] Generate synthetic datasets
- [ ] Test non-GPU components locally
- [ ] Prepare all code for copy-paste into Jupyter

### Phase 2: GPU Session 1 (When you request notebook)
**Target: 4 hours**
- [ ] Upload code to `/workspace/shared`
- [ ] Test vLLM server connectivity
- [ ] Verify model availability (check which models are pre-loaded)
- [ ] Run full agent pipeline end-to-end
- [ ] Debug and fix any ROCm/vLLM compatibility issues
- [ ] Save working state to `/workspace/shared`

### Phase 3: GPU Session 2 (June 12-13)
**Target: 4 hours**
- [ ] Polish UI and demo flow
- [ ] Collect performance metrics (latency, tokens, GPU memory)
- [ ] Record demo video
- [ ] Run stress test scenarios
- [ ] Final code cleanup

### Phase 4: Submission Prep (June 13-14 — Local)
- [ ] Create 5-slide presentation
- [ ] Edit demo video
- [ ] Package code for submission
- [ ] Submit via Ultimatix Prime Events

---

## Verification Plan

### Automated Tests
```bash
# Test data generation
python data_generator.py --verify

# Test anomaly detection on synthetic data
python anomaly_detector.py --test

# Test agent pipeline (requires vLLM)
python -c "from agents.orchestrator import InfraHealOrchestrator; o = InfraHealOrchestrator(); o.run_test()"
```

### Manual Verification
- End-to-end demo walkthrough: inject anomaly → detect → diagnose → remediate → report
- GPU memory usage monitoring during inference
- Latency measurement for each agent step
- Token count tracking for cost efficiency metrics

### Metrics to Capture for Presentation (Slide 4)
- End-to-end latency per incident (target: <30 seconds)
- GPU memory usage (target: <16 GB for 7B model)
- Tokens per incident diagnosis (target: <2000 tokens)
- Anomaly detection accuracy on synthetic data (target: >95%)
