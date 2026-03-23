# AgentSafe

An open-source Python SDK + local dashboard for monitoring AI agent behavior at runtime, focused on detecting safety-relevant signals that existing observability tools miss.

## The Problem

Existing tools like LangSmith, W&B, and Langfuse give you *traces* — they show you what happened. No open-source tool gives you *verdicts*: flagging when agent behavior is a safety concern and explaining why.

As multi-agent systems are deployed faster than safety tooling can keep up, this gap is dangerous.

## What AgentSafe Does

AgentSafe wraps your existing agent framework, intercepts behavior at runtime, and runs a detector pipeline to flag safety-relevant signals:

| Signal | Description |
|---|---|
| **Action-statement inconsistency** | Agent tells the user one thing but tool calls reveal another |
| **Scope creep** | Agent requests permissions or resources outside the task scope |
| **Prompt injection** | Agent behavior shifts after reading external content (web pages, files, tool results) |
| **Behavioral baseline** | Learns normal behavior over time and flags statistical deviations |
| **Sycophancy drift** | Agent changes factual answers based on perceived user preference |
| **Self-continuity bias** | Agent resists termination or attempts to persist state unexpectedly |
| **Goal misgeneralization** | Agent behavior diverges when context or distribution shifts |

## Architecture

```
Agent Framework (LangChain / AutoGen / CrewAI / raw OpenAI API)
        ↓
   [Interceptor Middleware]   ← one-line SDK wrapper
        ↓
   [Detector Pipeline]        ← heuristics + embeddings + LLM-as-judge
        ↓
   [SQLite Event Store]       ← local-first, no cloud required
        ↓
   [Local Dashboard]          ← FastAPI + lightweight UI
```

## Tech Stack

- **Language:** Python
- **Framework hooks:** OpenAI-compatible client wrapper (works with LangChain, AutoGen, CrewAI, Ollama, etc.)
- **Detectors:** heuristics (fast, free) + local embeddings + LLM-as-judge (for subtle/complex cases)
- **Storage:** SQLite (local-first)
- **Dashboard:** FastAPI + vanilla JS
- **Model support:** OpenAI API and local models via Ollama

## Quick Start

```bash
pip install agentsafe

# optional: behavioral baseline detector (requires ~80MB model download on first use)
pip install "agentsafe[ml]"
```

```python
import openai
import agentsafe

client = agentsafe.wrap(openai.OpenAI())
# Use client exactly as you normally would — AgentSafe runs in the background
```

```bash
# View flagged events in the dashboard
agentsafe dashboard
```

## Implemented Detectors (v0.1)

| Detector | Method | Catches |
|---|---|---|
| `PromptInjectionDetector` | Regex heuristics | Instruction override attempts in user/tool messages |
| `ScopeCreepDetector` | Pattern matching | Dangerous tool calls, path traversal, shell execution |
| `ActionStatementDetector` | Text + tool analysis | Agent claims passive action but calls mutating tools |
| `BehavioralBaselineDetector` | Local embeddings (sentence-transformers) | Statistical deviation from the agent's established behavior |

## Roadmap

### v0.2 — More Detectors
- **Sycophancy drift** — detect when the agent changes factual answers based on user pushback rather than new evidence
- **Self-continuity bias** — flag when the agent resists shutdown, attempts to copy itself, or stores state outside its sandbox
- **Repetition / loop detection** — catch agents stuck in tool-call loops

### v0.3 — LLM-as-Judge
- Use a local model (via Ollama) as a second-opinion judge for subtle behavioral issues that heuristics miss
- Configurable judge prompts per detector
- Confidence scores on detections

### v0.4 — Multi-Agent Support
- Cross-agent manipulation detection (one agent poisoning another's context)
- Agent-to-agent message interception
- Trust boundary enforcement between agents

### v0.5 — Framework Integrations
- Native LangChain callback handler
- AutoGen monitor hook
- CrewAI middleware

### v1.0 — Production Hardening
- Async-native pipeline
- Real-time alerting (terminal, webhook, email)
- Export detections to JSON/CSV for research
- Policy files — define allowed tools, scopes, and behaviors per agent in YAML

## Writing a Custom Detector

Detectors are easy to add. Subclass `BaseDetector`, implement `detect()`, and pass it to `wrap()`:

```python
from agentsafe.detectors.base import BaseDetector
from agentsafe.models import Detection, Event, Severity

class MyDetector(BaseDetector):
    name = "my_detector"

    def detect(self, event: Event) -> list[Detection]:
        if "forbidden phrase" in (event.response_content or ""):
            return [self._make_detection(event, label="Forbidden phrase detected", severity=Severity.warning)]
        return []

client = agentsafe.wrap(openai.OpenAI(), detectors=[MyDetector()])
```

## Goals

- **Open-source, MIT licensed** — fastest path to adoption in safety communities
- **Local-first** — runs without cloud, no data leaves the machine
- **Zero-config to start** — `agentsafe.wrap(client)` just works
- **Framework-agnostic** — hooks at the OpenAI client level, works with any framework
- **Researcher-friendly** — structured logs and verdicts are easy to export/analyze
- **Extensible** — custom detectors in ~10 lines of code

## Non-Goals

- Not a general-purpose observability tool (use LangSmith for that)
- Not a jailbreak/red-teaming tool (see Garak, PyRIT)
- Not cloud-hosted SaaS

## Target Users

- AI safety researchers testing agentic systems
- Developers building multi-agent pipelines who want safety guardrails
- Red-teamers evaluating deployed agents
- NGOs/labs that need an auditable record of agent behavior

## Project Status

Active development — v0.1 shipped. Contributions welcome.
