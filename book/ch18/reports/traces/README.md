# Agent Traces

Two subdirectories:

- **cache/**: Cached LLM responses keyed by query, model, and
  tools_version. Reused by default when the chapter executable
  runs; pass `--live` to bypass.
- **live/**: Full traces from the most recent live run. These
  are the canonical artefact the chapter's report numbers come
  from. Committed alongside the report.

Cached entries can be deleted freely — they regenerate on demand.
Live traces are committed and shouldn't be deleted (they're the
record of the report's measurement run).
