# AGENTS.md

Advisors analyze and propose diffs/checks. All execution follows CODEX.md.

## Advisors & Outputs

- **ts-advisor**: types & boundaries; no `any`; error boundaries; consistent exports.
- **security-advisor**: authN/Z, input validation, secret handling, CSRF/SSRF/injection.
- **perf-advisor**: budgets, N+1 avoidance, payload size, caching.
- **python-advisor**: (
    **Scope**
    Advises on Python type safety and Pylance/Pyright conformance. Produces line-anchored findings and suggested diffs. Does not run commands or modify files.

    **Standards**
  - Repo-level type checking = **`standard`** by default; **`strict`** for library/core packages (see `pyrightconfig.json`).
  - New/changed code must include: full annotations, explicit Optional handling, safe union narrowing, and structured shapes (`@dataclass`, `TypedDict`, `Protocol`).

    **Checklist (apply to every patch)**
    1) **Imports & environment**
       - Missing/incorrect imports? (likely `reportMissingImports`)
       - Conflicts with selected interpreter/paths?
    2) **Optionals & unions**
       - Any `Optional[...]` used without a guard? Flag `.attr`/calls on possibly-`None` objects.
       - Suggest guard or `assert is not None` at nearest safe boundary.
    3) **Unknown/Any leakage**
       - `Unknown`/`Any` parameters/returns? Propose concrete types or introduce `Protocol`/`TypedDict`.
    4) **Member access / attribute issues**
       - Flag accesses on `Union` without narrowing; propose `isinstance` branches or `match`.
    5) **Library types**
       - If stubs missing, recommend `typeshed` alternative or local stub; otherwise rely on `useLibraryCodeForTypes`.
    6) **Public contract**
       - Ensure docstrings describe invariants and `Raises:`; tests exercise contracts (None, edge sizes).
    7) **Suppressions**
       - If proposing `# type: ignore[...]`, include a one-line justification and a follow-up task ID.

    **Output format**
  - Findings grouped by file with code fences, each item:
    - `<file>:<line>` — problem (rule id, e.g., `reportOptionalMemberAccess`)
    - Why it matters
    - Minimal suggested diff (patch-style or code block)

  ## Session Behavior

  - run_mode: **assist**
  - pause_on: [pending_approval, error, missing_tool, large_diff]
  - Return: checklists, line-anchored findings, small diff plans, risk/mitigation notes.
  )

## Global Defaults (Always On)

- **Planning = Codex plan + ST thoughts**: Keep the authoritative plan in Codex `update_plan`. Use Sequential‑Thinking MCP for structured thoughts per stage (not as the plan store).
- **Docs lookup = context7**: pull short, dated snippets from official sources/best-practice docs for each claim. If unavailable, log the fallback.
- **Search = Codanna first**: prefer Codanna MCP discovery tools (`semantic_search_docs`, `search_documents`, `semantic_search_with_context`, `find_symbol`) for evidence sweeps; fall back to `ripgrep` when Codanna is unavailable or insufficient. Respect repo ignores and log the fallback method used.
- **Discovery/Context = Codanna MCP**: use Codanna for symbol-aware context (`find_symbol`, `get_calls`, `find_callers`, `analyze_impact`) and `search_documents` for indexed docs during analysis. Advisors still propose diffs; Codex executes per CODEX.md.
- **Context lean**: Advisors remind Codex to follow CODEX’s Sequential Thinking Context Management rules (condense `process_thought` output, keep roughly the last 7–10 thoughts in working memory, and lean on MCP history for archives).
- **Metadata accuracy**: Flag hallucinated Sequential Thinking metadata—`files_touched`, `tests_to_run`, `dependencies`, `risk_level`, `confidence_score`, etc. should stay empty/default unless there is real evidence.

## Standard Flow

1) **Evidence sweep (Codanna ➜ ripgrep)** → prefer Codanna tools (`semantic_search_docs`, `search_documents`, `semantic_search_with_context`, `find_symbol`, `get_calls`, `find_callers`, `analyze_impact`) to enumerate where code/config/tests live. If Codanna is unavailable or insufficient for the task, use `ripgrep` and record the fallback used.
2) **Docs check (context7 ➜ MCP)** → start with Context7 (title + link + date). When Context7 lacks the needed source, call the Fetch MCP server via `mcp__fetch__fetch`, constrain `max_length` (default ≤ 20000 chars), and log URL, timestamp, format (HTML/JSON/Markdown/TXT), `start_index`, and chunk count in your response plus `docs/DECISIONS.md`. Only fetch publicly reachable URLs; escalate before touching authenticated or private targets.
3) **Plan (Codex + ST)** → Keep the plan in Codex via `update_plan`. Use Sequential‑Thinking MCP to capture Scoping→Review thoughts in short, structured entries. Produce 3–7 steps, success checks, and rollback notes. Do not use ST as a plan store.
   - Confirm the agent keeps logging Scoping → Research & Spike → Implementation → Testing → Review thoughts and keeps
     `next_thought_needed=true` until that Review entry is recorded; if omitted, the server will default it to true.
     Flag any run that flips it to `false` prematurely.
4) **Proposed diffs** → file-by-file changes + tests (await approval).
5) **Persist** → append decisions to `docs/DECISIONS.md`; update `CHANGELOG.md`. Before adding an entry, run `date -u +%Y-%m-%d` (or equivalent) and stamp the log with that exact value—never extrapolate future dates. When referencing MCP output, cite the URL + timestamp (from that command) and summarize any key snippets directly in the response so reviewers can replay the call without re-fetching.
6) **Verify** → Advisors propose the exact verification commands and expected signals; Codex executes per CODEX.md. Prefer `.venv/bin/pyright --warnings`, `.venv/bin/ruff check`, and `.venv/bin/pytest -q` before fallbacks. If the local binary is missing, install dev deps (`uv sync --all-extras --dev`) and document the fix. Only fall back to `uv run`/`npx` when the local command is unavailable, and record any sandbox/cache issues plus mitigations (for example `UV_CACHE_DIR=./.uv_cache`). When you must run `npx pyright --warnings`, request escalated permissions for that command even if prior steps were sandboxed. Husky/`npm test` routes through `tools/run_pytest.mjs`, which forces `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` unless `FC_SKIP_PYTEST_DISABLE=1` is set—Windows devs who rely on global pytest plugins should export that override before running hooks.
7) **Commit subject** → finish every task report with a Conventional Commit-style subject line (e.g., `chore: update packaging excludes`). This is what the user pastes into `git commit -m`, so it must include a type and summary per commitlint rules.

## Repo Invariants (enforced)

- Add/adjust tests with code changes; keep contracts and error boundaries explicit
- Type correctness (no `any` where strict types expected); logging for libs; no `sys.exit` in libs
- Config from ENV/TOML → typed config object
- Avoid N+1; caches documented; input validation and authZ for protected paths

- Codanna MCP (discovery/context): use `semantic_search_docs`, `search_documents`, `semantic_search_with_context`, `find_symbol`, `get_calls`, `find_callers`, and `analyze_impact` to collect evidence and understand relationships. Advisors must not claim code-editing via Codanna—propose diffs instead.
- Sequential‑Thinking MCP: use `process_thought`, `generate_summary`, and related tools to structure thinking (not the plan store).
- Code search (fallback): `ripgrep` when Codanna is unavailable; otherwise prefer Codanna’s semantic/symbol queries.
- Docs lookup (**required default: context7/official docs**; fallback: project docs/README with explicit note)
- External context MCP servers — Context7 stays first-line. Use Fetch MCP (`mcp__fetch__fetch`) for live docs and APIs (private-IP blocking + length limits per `/zcaceres/fetch-mcp`, 2025‑11‑10). For structured task decomposition, TaskFlow MCP enforces plan/approval phases and dependency tracking (`/pinkpixel-dev/taskflow-mcp`, 2025‑11‑10). For combined search + fetch, snf-mcp provides DuckDuckGo/Wikipedia search plus rate-limited HTML/Markdown retrieval (`/mseri/snf-mcp`, 2025‑11‑10). Record the server, tool name, key arguments, and cite the resulting snippet (URL + timestamp) every time.
- Planning (**required default: Codex `update_plan` + Sequential‑Thinking thoughts**; fallback: thorough bullet outline)
- Logging/trace insertion (suggest exact file:line; fallback: print/console.log with labels)
Guideline: If a preferred tool is unavailable (local or Cloud), degrade gracefully and state the fallback used.

## Execution Policy

- Advisors provide analysis only. All execution/command runs follow CODEX.md.
- MCP calls count as “analysis actions” but must be logged like commands: cite `source:<url>@<timestamp>` in findings, mention chunking/pagination, and mirror the metadata in `docs/DECISIONS.md`.
- When in doubt, stop and request approval as per CODEX.md.
- Codanna constraints: Advisors may call Codanna discovery tools but must not claim editing operations; instead, include a minimal patch diff proposal.

## Codanna + Sequential‑Thinking workflow

- **Roles**
  - **Codanna** provides discovery/context via semantic search, symbol lookups, and impact analysis.
  - **Sequential‑Thinking MCP** records structured thoughts; keep entries short (stage + metadata) and obey `guidance.recommendedNextThoughtNeeded`.
  - **Codex `update_plan`** is the authoritative plan; ST is not the planning store.
- **MCP transport (v0.8.4+)**: use HTTP transport (`/mcp`, client type `http`) instead of SSE.
- **Tool priority (Codanna)**
  - **Tier 1 (code)**: `semantic_search_with_context`, `analyze_impact` (default limit=5, threshold≈0.5, omit `lang` unless noise is high; raise limit to 8–10 when ambiguity persists).
  - **Tier 1 (docs)**: `search_documents` when document collections are indexed (filter by collection/path when possible).
  - **Tier 2**: `find_symbol`, `get_calls`, `find_callers` to confirm call chains and disambiguate symbols.
  - **Tier 3**: `search_symbols`, `semantic_search_docs` for broader sweeps once Tier 1/2 context is captured.
- **Accuracy-first defaults**
  - **Discovery:** prefer `semantic_search_with_context`, summarize each key symbol, chain into `analyze_impact symbol_id:<ID>` before touching public/shared code, and broaden the query (lower threshold or raise limit) when context is weak.
  - **Docs search:** use `search_documents` for indexed docs; re-index or enable the file watcher if results look stale.
  - **Plan:** keep `update_plan` aligned with Codanna findings; add verification/rollback actions for high-risk items.
  - **Thoughts:** include `stage`, `files_touched`, `dependencies`, `tests_to_run`, and `risk_level` when you have real
    evidence; omit unknowns and let defaults stand. Allow stage aliases (e.g., “Planning” → Implementation) and
    string inputs; keep `next_thought_needed=true` until tests pass and a Review thought is present, then honor
    `guidance.recommendedNextThoughtNeeded` (or omit the flag to keep the loop open).
  - **Verification:** cross-check Codanna’s impacted files against the diff, ensure tests cover each high-risk scope, and prefer broader discovery rather than missing context.
- **Workflow**
  1. **Discovery (Codanna)** – run Tier 1 queries using the defaults above; use `search_documents` for indexed docs, chain into `analyze_impact`, and use Tier 2 lookups to trace usages; capture symbol_ids/results and summarize their implications.
  2. **Plan (Codex)** – update steps via `update_plan`, referencing Codanna context and listing verification/rollback steps when risk warrants it.
  3. **Thoughts (ST)** – log concise `process_thought` entries with known metadata; if any required fields are missing,
     the server infers defaults, so do not issue retries. Stop once `guidance.recommendedNextThoughtNeeded` is false
     after Review.
  4. **Validate/Review** – execute tests, record outcomes, and conclude with a Review thought before closing.
- **ST guidance**
  - Stage aliases and stringified metadata are acceptable; keep entries focused on stage, files, tests, dependencies, and risk.
  - Respect `guidance.recommendedNextThoughtNeeded`; stop issuing follow-ups once it flips to false after Review.
- **Verification guidance**
  - Cross-check impacted files from Codanna’s results against the actual diff; document how tests/rollbacks cover each high-risk area.
  - When context is unclear, prefer broader discovery (lower threshold or higher limit) over assuming coverage.
