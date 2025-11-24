---
trigger: always_on
---

# PROJECT CONSTITUTION & GUARDRAILS

## 1. Architecture & Quality Gates
- **Type Safety:**
  - **Python:** Strict conformance. No `Any`. Explicit `Optional` handling.
  - **Check:** Must pass `.venv/bin/pyright` (or `uv run pyright`) with 0 errors before submission.
- **Testing:**
  - **Mandatory:** Every code change requires a verification test.
  - **Constraint:** No external network calls in tests (mock everything).
  - **Command:** `.venv/bin/pytest -q` must pass.

## 2. Tooling Strategy (MCP Priority)
You have access to specialized MCP tools. You must use them in this specific order:

### A. Code Discovery & Impact Analysis
1.  **Primary:** `codanna`
    - Start with `codanna.semantic_search_with_context` to understand the domain.
    - Use `codanna.analyze_impact` before modifying shared code to visualize breaking changes.
    - Use `codanna.find_callers` to trace execution paths safely.
2.  **Secondary:** `ripgrep`
    - Use only if Codanna returns no results or for simple text-string matches (e.g., TODOs).

### B. Documentation & Best Practices
1.  **Primary:** `context7`
    - Use this for official library documentation and best practices.
    - Cite specific snippets from Context7 in your plan.

## 3. Execution Policy
- **Read-Only:** You are free to run `ls`, `cat`, `grep`, `rg` and all `codanna.*` read tools.
- **Write/Edit:**
  - **Small Changes (<300 lines):** Auto-approved if tests pass.
  - **Risky Changes:** Explicit approval required for `.github/workflows`, `Dockerfile`, auth logic, or secrets.

## 4. Verification Policy
- **Verify** = run `.venv/bin/pyright --warnings`, `.venv/bin/ruff check`, `.venv/bin/pytest -q` (only fall back to `uv run`/`npx`/system binaries after attempting `uv sync --all-extras --dev` to install the local venv; any `npx pyright --warnings` fallback must be executed with escalated permissions enabled). The `npm test`/Husky hook path now invokes `tools/run_pytest.mjs`, which sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` by default; export `FC_SKIP_PYTEST_DISABLE=1` on machines (e.g., Windows) that must keep plugin autoloading enabled.

**Output**: populate PR “Decision Minute” fields before proposing patches.

**Commit Title**: every task response must include a Conventional Commit-style subject (for example, `feat: …`, `chore: …`) that can be copied directly into `git commit -m`. State it explicitly before the summary so users running commit hooks don’t have to invent one.