# Contributing

Contributions are welcome. Here's what's most useful:

## New workflow types

Add an entry to `workflows.json` following the existing schema. Each workflow needs:
- A unique `slug`
- A `default_model_slug` matching one of the entries in `models_reference`
- A `scenario` block with `active_users` and `runs_per_day`
- A `cost_breakdown` array with at least `model`, `prompt`, and `retrieval` components

Each component needs `tokens_in_per_run`, `tokens_out_per_run`, `calls_per_run`, `usd_per_run`, `share_of_run_pct`, and optionally an `invisible_reason` string.

## New models

Add an entry to `models_reference` in `workflows.json` with `slug`, `name`, `vendor`, `input_price_per_1m_usd`, `output_price_per_1m_usd`, and optionally `cached_input_price_per_1m_usd`.

Then add a latency profile in `MODEL_LATENCY` inside `src/generator.py`:

```python
"your-model-slug": {"ttft_mean": 600, "ttft_std": 150, "tok_per_sec": 80},
```

## Latency profile updates

If you have measured TTFT or tokens/sec data for any model, open a PR updating `MODEL_LATENCY` with a source link.

## Issues

Please include your Python version and the command you ran.
