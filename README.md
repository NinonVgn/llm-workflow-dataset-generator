# LLM Workflow Cost & Latency Dataset Generator

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)
![Workflows](https://img.shields.io/badge/workflows-8%20finance%20types-orange.svg)
![Models](https://img.shields.io/badge/models-12%20LLMs-purple.svg)

> Synthetic dataset generator for enterprise LLM workflow cost & latency attribution — 8 real finance workflows, 12 models, 100k+ runs out of the box.

---

## Why this exists

Most LLM cost observability tools show you a total bill. They don't show you **which component of which workflow is responsible** — and 40-66% of enterprise LLM costs are invisible by default, landing on separate vendor invoices (embedding providers, compliance SaaS, chat frameworks).

This generator produces labeled training data for models that predict and attribute LLM costs **per component, per workflow run** — before you have enough production traces to train on.

---

## What's in the dataset

Each row is one **cost component** inside a workflow run. Every run has 5 components:

| Component | What it captures |
|---|---|
| `model` | Core LLM inference — the call you see on your dashboard |
| `prompt` | System prompt and few-shot prefix tokens, rebilled on every call |
| `orchestration` | Agent framework hops, tool calls, classifier calls |
| `retrieval` | Vector search and embedding refresh |
| `history` | Conversation history or context re-injected on every turn |

### Sample rows

```
run_id                               workflow_slug                    component  model_slug        tokens_input  tokens_output  cost_total_usd  is_invisible  latency_total_ms
f94d639c-...  earnings-call-analysis-rag       model      claude-sonnet-4   43821         1823           0.1489          False         2341
f94d639c-...  earnings-call-analysis-rag       prompt     claude-sonnet-4   52034         0              0.1561          False         661
f94d639c-...  earnings-call-analysis-rag       history    claude-sonnet-4   237329        0              0.5767          True          661
f94d639c-...  earnings-call-analysis-rag       retrieval  text-embedding-3  9812          0              0.0013          True          143
f94d639c-...  earnings-call-analysis-rag       orchestr.  claude-sonnet-4   6103          412            0.0246          True          1820
```

**Key fields:**

| Field | Description |
|---|---|
| `run_id` | UUID linking all components of one run |
| `workflow_slug` | Workflow type |
| `component` | `model` / `prompt` / `orchestration` / `retrieval` / `history` |
| `model_slug` | LLM used for this component |
| `tokens_input` / `tokens_output` | Token counts |
| `cached_tokens_input` | Tokens served from cache |
| `cost_input_usd` / `cost_cached_input_usd` / `cost_output_usd` | Cost breakdown |
| `cost_total_usd` | Total component cost in USD |
| `cost_share_pct` | Component's % share of the total run cost |
| `is_invisible` | `True` if this cost doesn't appear in standard dashboards |
| `invisible_reason` | Why it's invisible (e.g. *"costs land on compliance SaaS invoice"*) |
| `latency_ttft_ms` | Time to first token |
| `latency_total_ms` | Total component latency |
| `retried` / `retry_count` | Retry behaviour |
| `status` | `success` / `error` / `timeout` |
| `timestamp_start` / `timestamp_end` | Execution window |

---

## Workflows covered

All 8 workflows are modeled on real enterprise finance and investment management use cases, with token counts, model choices, and cost proportions grounded in observed production patterns.

| Workflow | Who runs it | Avg cost/run | Invisible share |
|---|---|---|---|
| `earnings-call-analysis-rag` | Hedge funds, research desks | $1.09 | 63% |
| `market-research-synthesis-agentic` | Multi-strat, macro funds | $4.89 | 57% |
| `regulatory-filing-review` | Compliance teams | $0.78 | 55% |
| `portfolio-monitoring-signal-generation` | Quant and systematic funds | $0.60 | 55% |
| `deal-investment-memo-generation` | PE / VC investment teams | $4.79 | 78% |
| `news-filing-alert-triage` | News desks, event-driven funds | $0.018 | 53% |
| `counterparty-kyc-document-review` | Risk and compliance | $0.75 | 66% |
| `internal-research-chatbot-rag` | Research analysts, PMs | $0.63 | 63% |

---

## Models simulated

Pricing sourced from public API pricing pages (2025/2026). Latency profiles calibrated from published benchmarks.

| Model | Vendor | Input $/1M | Output $/1M |
|---|---|---|---|
| GPT-5 | OpenAI | $1.25 | $10.00 |
| GPT-5 Mini | OpenAI | $0.25 | $2.00 |
| GPT-4.1 | OpenAI | $2.00 | $8.00 |
| o3 | OpenAI | $15.00 | $60.00 |
| Claude Opus 4 | Anthropic | $15.00 | $75.00 |
| Claude Sonnet 4 | Anthropic | $3.00 | $15.00 |
| Claude Haiku 4 | Anthropic | $1.00 | $5.00 |
| Gemini 2.5 Pro | Google | $1.25 | $10.00 |
| Gemini 2.5 Flash | Google | $0.30 | $2.50 |
| Llama 3.3 70B (self-hosted) | Meta | $0.12 | $0.12 |
| DeepSeek V3 | DeepSeek | $0.27 | $1.10 |
| text-embedding-3-large | OpenAI | $0.13 | — |

---

## Quickstart

No dependencies beyond Python stdlib.

```bash
git clone https://github.com/NinonVgn/llm-workflow-dataset-generator
cd llm-workflow-dataset-generator

# Generate 100k runs (default)
python src/generator.py --workflows workflows.json

# Custom parameters
python src/generator.py \
  --workflows workflows.json \
  --n-runs 100000 \
  --output-dir output \
  --formats csv jsonl \
  --start-date 2024-01-01 \
  --seed 42
```

Output written to `output/`:

```
output/
├── workflow_traces.csv     # ~100 MB at 100k runs
├── workflow_traces.jsonl   # same data, one JSON object per line
└── summary.json            # aggregate stats
```

---

## Realism design

| Feature | Implementation |
|---|---|
| Token distribution | Log-normal around real template means — not uniform |
| Caching | Realistic cache-hit fractions by component type (prompt/history cache more) |
| Latency | Model-specific TTFT + generation time profiles |
| Retries | ~2.5% of components retry; latency compounds per retry |
| Errors | ~0.2% of components fail with `error` or `timeout` |
| Invisible costs | `is_invisible` flag and `invisible_reason` text from real source data |
| Temporal spread | Runs distributed across 12 months with realistic intra-day clustering |

---

## Bring your own workflows

The generator reads from `workflows.json`. Add a workflow following the existing schema and it will be included immediately — no code changes needed.

```json
{
  "slug": "my-custom-workflow",
  "name": "My Custom Workflow",
  "default_model_slug": "claude-sonnet-4",
  "scenario": { "active_users": 20, "runs_per_day": 5, ... },
  "cost_breakdown": [
    { "component": "model", "tokens_in_per_run": 30000, ... },
    ...
  ]
}
```

---

## Use cases

- Training cost-attribution models for LLM observability platforms
- Benchmarking workflow cost optimizers (prompt caching strategies, model routing)
- Generating realistic load profiles for infrastructure cost forecasting
- Research on hidden/invisible cost patterns in production AI systems

---

## Cite this work

```bibtex
@software{llm_workflow_cost_dataset,
  title  = {LLM Workflow Cost \& Latency Dataset Generator},
  year   = {2026},
  url    = {https://github.com/NinonVgn/llm-workflow-dataset-generator},
  license = {MIT}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome for new workflow types, additional models, and latency profile updates.

---

## License

MIT — free to use, modify, and distribute.
