"""
Enterprise Finance/Accounting LLM Workflow Dataset Generator
Based on real workflow definitions from workflows.json.
Generates realistic traces with cost and latency per component step.
"""

import uuid
import random
import json
import csv
import math
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Latency profiles per model (ms) - not in source JSON, calibrated from
# public benchmark data (2025/2026)
# ---------------------------------------------------------------------------

MODEL_LATENCY = {
    "gpt-5":                    {"ttft_mean": 900,  "ttft_std": 220, "tok_per_sec": 50},
    "gpt-5-mini":               {"ttft_mean": 400,  "ttft_std": 100, "tok_per_sec": 130},
    "gpt-4-1":                  {"ttft_mean": 700,  "ttft_std": 160, "tok_per_sec": 70},
    "o3":                       {"ttft_mean": 2500, "ttft_std": 600, "tok_per_sec": 30},
    "claude-opus-4":            {"ttft_mean": 1100, "ttft_std": 280, "tok_per_sec": 45},
    "claude-sonnet-4":          {"ttft_mean": 700,  "ttft_std": 170, "tok_per_sec": 75},
    "claude-haiku-4":           {"ttft_mean": 280,  "ttft_std": 70,  "tok_per_sec": 180},
    "gemini-2-5-pro":           {"ttft_mean": 950,  "ttft_std": 240, "tok_per_sec": 55},
    "gemini-2-5-flash":         {"ttft_mean": 320,  "ttft_std": 80,  "tok_per_sec": 210},
    "llama-3-3-70b-self-host":  {"ttft_mean": 500,  "ttft_std": 180, "tok_per_sec": 90},
    "deepseek-v3":              {"ttft_mean": 600,  "ttft_std": 150, "tok_per_sec": 80},
    "text-embedding-3-large":   {"ttft_mean": 120,  "ttft_std": 30,  "tok_per_sec": 800},
}

# Components that use the embedding model for retrieval
EMBEDDING_COMPONENTS = {"retrieval"}


# ---------------------------------------------------------------------------
# Data class for one output row
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    run_id: str
    workflow_slug: str
    workflow_name: str
    component: str
    component_label: str
    model_slug: str
    model_name: str
    vendor: str
    calls_in_run: int
    tokens_input: int
    tokens_output: int
    cached_tokens_input: int
    cost_input_usd: float
    cost_cached_input_usd: float
    cost_output_usd: float
    cost_total_usd: float
    cost_share_pct: float
    is_invisible: bool
    invisible_reason: str
    latency_ttft_ms: int
    latency_generation_ms: int
    latency_total_ms: int
    retried: bool
    retry_count: int
    status: str
    timestamp_start: str
    timestamp_end: str
    # Scenario context
    active_users: int
    runs_per_day: int


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WorkflowGenerator:

    def __init__(self, workflows_path: str, seed: Optional[int] = 42):
        with open(workflows_path) as f:
            raw = json.load(f)

        # Index models by slug
        self.models: Dict[str, Any] = {m["slug"]: m for m in raw["models_reference"]}
        self.workflows: List[Dict] = raw["workflows"]

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Token sampling - log-normal around the template mean with ~25% std
    # ------------------------------------------------------------------

    def _sample_around(self, mean: int, spread: float = 0.25) -> int:
        if mean == 0:
            return 0
        sigma = spread
        mu = math.log(mean)
        val = int(random.lognormvariate(mu, sigma))
        # cap at 3x mean to avoid extreme outliers
        return max(1, min(mean * 3, val))

    # ------------------------------------------------------------------
    # Cached token fraction (realistic: 10-40% of input may be cached)
    # ------------------------------------------------------------------

    def _cached_fraction(self, component: str, model_slug: str) -> float:
        model = self.models.get(model_slug, {})
        if model.get("cached_input_price_per_1m_usd") is None:
            return 0.0
        # prompt and history components benefit most from caching
        if component in ("prompt", "history"):
            return random.uniform(0.15, 0.45)
        elif component == "retrieval":
            return random.uniform(0.05, 0.20)
        return random.uniform(0.0, 0.10)

    # ------------------------------------------------------------------
    # Cost calculation using real pricing from JSON
    # ------------------------------------------------------------------

    def _compute_cost(
        self,
        model_slug: str,
        tokens_in: int,
        tokens_out: int,
        cached_tokens: int,
    ) -> tuple:
        model = self.models.get(model_slug)
        if not model:
            return 0.0, 0.0, 0.0

        price_in    = model["input_price_per_1m_usd"] / 1_000_000
        price_out   = model["output_price_per_1m_usd"] / 1_000_000
        cache_price = (model.get("cached_input_price_per_1m_usd") or model["input_price_per_1m_usd"]) / 1_000_000

        non_cached = max(0, tokens_in - cached_tokens)
        cost_in        = non_cached * price_in
        cost_cached_in = cached_tokens * cache_price
        cost_out       = tokens_out * price_out

        return round(cost_in, 8), round(cost_cached_in, 8), round(cost_out, 8)

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def _compute_latency(self, model_slug: str, tokens_out: int) -> tuple:
        profile = MODEL_LATENCY.get(model_slug, {"ttft_mean": 600, "ttft_std": 150, "tok_per_sec": 80})
        ttft = max(30, int(random.gauss(profile["ttft_mean"], profile["ttft_std"])))
        gen_ms = max(50, int((tokens_out / profile["tok_per_sec"]) * 1000 * random.uniform(0.85, 1.20)))
        return ttft, gen_ms

    # ------------------------------------------------------------------
    # Retry / error simulation
    # ------------------------------------------------------------------

    def _retry_and_status(self) -> tuple:
        retried = random.random() < 0.025
        retry_count = random.randint(1, 3) if retried else 0
        if retried and retry_count >= 3 and random.random() < 0.12:
            status = random.choice(["error", "timeout"])
        else:
            status = "success"
        return retried, retry_count, status

    # ------------------------------------------------------------------
    # Resolve which model to use for a component
    # ------------------------------------------------------------------

    def _model_for_component(self, workflow: Dict, component: str) -> str:
        # Retrieval always uses the embedding model
        if component == "retrieval":
            return "text-embedding-3-large"
        return workflow["default_model_slug"]

    # ------------------------------------------------------------------
    # Generate one run of a workflow
    # ------------------------------------------------------------------

    def generate_run(self, workflow: Dict, base_ts: datetime) -> List[StepRecord]:
        run_id = str(uuid.uuid4())
        scenario = workflow["scenario"]
        records: List[StepRecord] = []
        current_ts = base_ts

        for comp in workflow["cost_breakdown"]:
            component = comp["component"]
            model_slug = self._model_for_component(workflow, component)
            model_info = self.models.get(model_slug, {})

            # Sample tokens around the template mean
            tokens_in  = self._sample_around(comp["tokens_in_per_run"])
            tokens_out = self._sample_around(comp["tokens_out_per_run"]) if comp["tokens_out_per_run"] else 0
            calls      = max(1, self._sample_around(comp["calls_per_run"], spread=0.15))

            # Caching
            cached_frac   = self._cached_fraction(component, model_slug)
            cached_tokens = int(tokens_in * cached_frac)

            # Cost
            cost_in, cost_cached, cost_out = self._compute_cost(
                model_slug, tokens_in, tokens_out, cached_tokens
            )
            cost_total = round(cost_in + cost_cached + cost_out, 8)

            # Latency
            ttft_ms, gen_ms = self._compute_latency(model_slug, tokens_out)
            retried, retry_count, status = self._retry_and_status()
            extra_latency = int(ttft_ms * retry_count * random.uniform(0.7, 1.1)) if retried else 0
            total_ms = ttft_ms + gen_ms + extra_latency

            end_ts = current_ts + timedelta(milliseconds=total_ms)

            records.append(StepRecord(
                run_id=run_id,
                workflow_slug=workflow["slug"],
                workflow_name=workflow["name"],
                component=component,
                component_label=comp["label"],
                model_slug=model_slug,
                model_name=model_info.get("name", model_slug),
                vendor=model_info.get("vendor", ""),
                calls_in_run=calls,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cached_tokens_input=cached_tokens,
                cost_input_usd=cost_in,
                cost_cached_input_usd=cost_cached,
                cost_output_usd=cost_out,
                cost_total_usd=cost_total,
                cost_share_pct=comp["share_of_run_pct"],
                is_invisible=bool(comp.get("invisible_reason")),
                invisible_reason=comp.get("invisible_reason") or "",
                latency_ttft_ms=ttft_ms,
                latency_generation_ms=gen_ms,
                latency_total_ms=total_ms,
                retried=retried,
                retry_count=retry_count,
                status=status,
                timestamp_start=current_ts.isoformat(),
                timestamp_end=end_ts.isoformat(),
                active_users=scenario["active_users"],
                runs_per_day=scenario["runs_per_day"],
            ))

            # Inter-component gap
            current_ts = end_ts + timedelta(milliseconds=random.randint(10, 150))

        return records

    # ------------------------------------------------------------------
    # Generate full dataset
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_runs: int = 100_000,
        output_dir: str = "output",
        formats: List[str] = ("csv", "jsonl"),
        start_date: str = "2024-01-01",
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_dt = datetime.fromisoformat(start_date)
        total_seconds = 365 * 24 * 3600

        all_records: List[StepRecord] = []
        print(f"Generating {n_runs:,} workflow runs across {len(self.workflows)} workflow types...")

        for i in range(n_runs):
            workflow = random.choice(self.workflows)
            offset = random.uniform(0, total_seconds)
            ts = base_dt + timedelta(seconds=offset)
            records = self.generate_run(workflow, ts)
            all_records.extend(records)

            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} / {n_runs:,} runs ({len(all_records):,} component records)")

        print(f"\nTotal component records: {len(all_records):,}")

        if "csv" in formats:
            path = output_path / "workflow_traces.csv"
            print(f"Writing CSV -> {path}")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
                writer.writeheader()
                for r in all_records:
                    writer.writerow(asdict(r))
            print(f"  {path.stat().st_size / 1_048_576:.1f} MB")

        if "jsonl" in formats:
            path = output_path / "workflow_traces.jsonl"
            print(f"Writing JSONL -> {path}")
            with open(path, "w", encoding="utf-8") as f:
                for r in all_records:
                    f.write(json.dumps(asdict(r)) + "\n")
            print(f"  {path.stat().st_size / 1_048_576:.1f} MB")

        self._write_summary(all_records, output_path)
        print("\nDone.")

    def _write_summary(self, records: List[StepRecord], output_path: Path):
        run_ids = set(r.run_id for r in records)
        total_cost = sum(r.cost_total_usd for r in records)
        invisible_cost = sum(r.cost_total_usd for r in records if r.is_invisible)

        wf_counts: Dict[str, int] = {}
        comp_costs: Dict[str, float] = {}
        for r in records:
            wf_counts[r.workflow_slug] = wf_counts.get(r.workflow_slug, 0) + 1
            comp_costs[r.component] = comp_costs.get(r.component, 0.0) + r.cost_total_usd

        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(run_ids),
            "total_component_records": len(records),
            "avg_components_per_run": round(len(records) / len(run_ids), 2),
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_run_usd": round(total_cost / len(run_ids), 6),
            "invisible_cost_usd": round(invisible_cost, 4),
            "invisible_share_pct": round(invisible_cost / total_cost * 100, 1) if total_cost else 0,
            "error_rate": round(sum(1 for r in records if r.status != "success") / len(records), 4),
            "retry_rate": round(sum(1 for r in records if r.retried) / len(records), 4),
            "workflow_distribution": wf_counts,
            "cost_by_component": {k: round(v, 4) for k, v in comp_costs.items()},
        }

        path = output_path / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate enterprise finance LLM workflow traces from real workflow definitions"
    )
    parser.add_argument("--workflows", type=str, default="workflows.json",
                        help="Path to workflows.json")
    parser.add_argument("--n-runs", type=int, default=100_000,
                        help="Number of workflow runs to generate")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--formats", nargs="+", default=["csv", "jsonl"],
                        choices=["csv", "jsonl"])
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen = WorkflowGenerator(workflows_path=args.workflows, seed=args.seed)
    gen.generate_dataset(
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        formats=args.formats,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
