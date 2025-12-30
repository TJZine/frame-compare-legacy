# Changelog

## Unreleased

### Bug Fixes

- restore CLI run reporting: fix `frame_compare.run_cli` to pass `report_enable_override` correctly and parse `[[role]]...[[/]]` layout spans with the proper closing token so Rich markup renders cleanly without leaking markers
- *2025-12-11:* keep VSPreview metadata consistent between top-level and audio alignment blocks (mode and hint values), allow boolean/None literals in layout expressions, seed `use_vspreview`/nested `vspreview_mode` from config defaults, harden slow.pics report handling when generation fails, and remove duplicate VSPreview manual-offset display wiring

### Features

- *2025-11-20:* retire the legacy runner path, remove CLI/config service-mode toggles, warn when legacy mode is requested, and make the publisher services pipeline the only supported path.
- *2025-11-17:* unify CLI rendering around cache-aware run snapshots, add `--from-cache-only`, `--no-cache`, and `--show-partial` flags, persist `.frame_compare.run.json`, and render cached sections consistently with live runs.
- *2025-11-17:* harden snapshot hydration, mark corrupt cache files as misses, and persist per-section availability so `[RENDER]`/`[PUBLISH]` blocks honor `--show-partial`/`--show-missing`.
- *2025-11-18:* finish the CLI cache UX by adding `--show-missing`/`--hide-missing`, plumbing `show_missing_sections` through the public API, broadening section-availability heuristics for viewer/report/audio/VSPreview blocks, and extending tests/docs/CHANGELOG to cover the new behavior.
- *2025-11-18:* capture ClipProbeSnapshot metadata (fps, frame counts, geometry, HDR props) per clip, persist it to `cache_dir/probe/<hash>.json`, reuse probe-era clip handles inside `init_clips`, and add `runtime.force_reprobe` plus cache hit/miss logging with regression tests for probe/init reuse.
- *2025-11-18:* extract `MetadataResolver` and `AlignmentWorkflow` services per Track A, wire runner orchestration through typed requests/responses, and add focused service unit tests to lock TMDB/alignment behavior.
- *2025-11-18:* wire the runner through `RunDependencies` + `RunContext`, add `default_run_dependencies()` for CLI/test injection, refresh CLI/Dolby stubs to accept the `dependencies` kwarg, and add `tests/runner/test_runner_services.py` to assert service order, error propagation, and reporter flag wiring.
- *2025-11-19:* reintroduce the service-mode publisher pipeline with CLI overrides (`--service-mode`/`--legacy-runner`), wire RunDependencies to `ReportPublisher`/`SlowpicsPublisher`, and update runner/CLI/slow.pics tests plus docs for the new flag.

- *2025-11-20:* restore VSPreview overlay hints by sourcing layout `suggested_frames`/`suggested_seconds` from the JSON tail and adding a regression test for CLI layout propagation.
- *2025-11-19:* harden the Decision Minute workflow by fetching PR metadata via the API, skipping unmerged runs, and reading data from the github-script `result` output to avoid invalid contexts when CI completes.
- *2025-11-19:* enforce default HTTPX timeouts for TMDB requests, switch slow.pics publishers to a lock-protected O(1) byte accumulator, and raise a CLI error when planner metadata trails the discovered files.
- *2025-11-19:* gate `--tm-gamma-disable` on explicit CLI use (protecting config defaults from env/default_map values) and refresh the CLI refactor doc to show `cli_entry` owns wiring after Phase 2.
* prevent `--service-mode` and `--legacy-runner` from being used together, loosen `RunContext.metadata` typing, guard metadata indexing in `pick_analyze_file`, and treat malformed probe-cache payloads (missing cache keys, bad field types) as cache misses instead of hard errors
* coerce local report viewer destinations/labels to strings before serializing cached run metadata, refresh the `probe_clip_metadata` docstring to describe cache reuse, and fix docs that referenced the non-existent `src/datatype` module
* gate every non-`--tm-*` CLI override (paths/cache/report/audio/debug flags) on explicit command-line sources so `frame_compare.run_cli` and the Click entrypoint respect config precedence, and fix `json_tail.render.writer` so debug-color runs accurately report the VS fallback (tests/runner/test_cli_entry.py)
* regenerate `docs/_generated/config_tables.md` via `tools/gen_config_docs.py` so the new `runtime.force_reprobe` default is documented, and add an autouse fixture in `tests/test_clip_metadata_probe.py` that stubs `vs_core.set_ram_limit` so the suite passes even when the VapourSynth module is unavailable
* guard tonemap CLI overrides so Click `default_map`/env-provided values defer to config until a `--tm-*` flag is provided, and extend CLI regression tests for `--tm-target` plus overlay/verify telemetry
* keep release-please commits passing commitlint by forcing `chore(ci): release…` pull-request titles
* prevent the Click CLI from forcing Dolby Vision tonemapping off unless `--tm-use-dovi`/`--tm-no-dovi` is explicitly provided so CLI and direct runs agree
* ensure tonemap presets override template defaults when configs match reference values and expose preset matrices/comments in `config.toml.template`
* normalize runner auto letterbox telemetry, document accepted crop inputs, tighten FPS map ordering/logging, and extend cached FPS/metadata tests for probe reuse
* hydrate cached `suggested_frames`/`suggested_seconds` when reusing offsets files so CLI+VSPreview keep prior recommendations
* restore audio offset hints in CLI/VSPreview when FPS metadata is missing and preserve negative manual trims across summaries/manual prompts
* preserve HDR tonemapping for negative trims, honor CLI tonemap overrides even under presets, and restore path-based `color_overrides` matching
* detect HDR clips when only one of `_Primaries`/`_Transfer` or MDL/CLL props are present and backfill BT.2020/ST2084 defaults so libplacebo receives primaries/matrix hints before tonemapping
* preserve MasteringDisplay*/ContentLightLevel* metadata when padding negative trims so HDR detection survives blank-prefix extension
* pre-probe clip metadata so audio alignment derives frame counts and screenshots reuse cached HDR props without extra VapourSynth passes
* refresh cached frame counts whenever audio alignment or VSPreview manual offsets shift trims so CLI summaries report the trimmed clip durations
* reuse cached FPS metadata from the initial probe while measuring audio alignment so CLI and VSPreview frame deltas stay non-zero even when ffprobe omits `r_frame_rate`
* keep Husky `npm test` working on Windows by routing through `tools/run_pytest.mjs` (enforcing `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` by default with an `FC_SKIP_PYTEST_DISABLE=1` escape hatch)
* *2025-11-19:* let `runner.run` build its default dependency bundle after configuration preflight so MetadataResolver/AlignmentWorkflow receive the real `cfg`, reporter, and cache paths; CLI shims now pass `dependencies=None` and targeted runner shim tests were updated accordingly.

### Chores

- *2025-11-20:* docs: trimmed noisy DECISIONS logs, converted runner/CLI refactor docs into summaries, refreshed README references, and updated the docs inventory.
- *2025-11-19:* refactor/internal: finalize the CLI split by keeping `frame_compare.py` as a thin shim delegating to `src/frame_compare/cli_entry.py`/`cli_utils.py`; no user-visible behavior changes expected.
- *2025-11-18:* document the Phase 3–6 Track B flag/config review (Screenshots → TMDB domains), check off the Global Invariants in `docs/refactor/flag_audit.md`, and log the verification commands in `docs/DECISIONS.md`.
- *2025-11-18:* add a `FRAME_COMPARE_DOVI_DEBUG` telemetry mode that emits JSON-formatted logs from both the runner and VapourSynth tonemap resolver so entrypoints can compare config roots, cache status, tonemap overrides, and brightness-affecting parameters when diagnosing DOVI drift.
- *2025-11-18:* convert the docs/refactor/flag_audit.md template placeholders into ATX headings with per-track prefixes so markdownlint (MD003/MD024) passes and rendered navigation stays unique.
- *2025-11-18:* document the Phase 1–2 config/tonemap audit review results in `docs/refactor/flag_audit.md` (A3/B4) and `docs/DECISIONS.md`, confirming Click CLI vs `frame_compare.run_cli` parity for DoVi and tonemap overrides.
- *2025-11-19:* finalize Track C documentation (Implementation/Review notes), surface `[runner].enable_service_mode` in the config template/README, and log the active publishing mode inside `runner.run`.

- *2025-11-13:* chore(types): removed the last `# pyright: standard` escape hatch from `cli_layout`, hardened optional MultipartEncoder imports/guards in `slowpics`, validated template/line/table inputs to avoid `Unknown` bleed-through, and added a renderer helper so progress columns no longer reach into protected state.
- *2025-11-13:* refactor(cache): bumped metrics payloads to v2 with embedded clip identity snapshots (abs path/size/mtime plus opt-in sha1), taught `cache.build_cache_info` to populate frozen `ClipIdentity` entries, made selection sidecars reuse the same snapshot instead of re-stat'ing files, and added cache identity regression tests (cross-folder isolation, v1 upgrade, snapshot-backed sidecar, sha1 opt-in). Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_cache_identity.py`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_analysis.py::test_select_frames_uses_cache`, and `.venv/bin/pyright --warnings`.
- *2025-11-13:* dx(cache): CLI + JSON tail now show the specific reason when frame metrics are recomputed (config mismatch, fps mismatch, missing, etc.), and analysis logs emit INFO breadcrumbs for metrics cache/selection sidecar misses to ease triage.
- *2025-11-13:* perf(cache): `_build_clip_inputs`/`build_clip_inputs_from_paths` skip SHA1 hashing unless `compute_sha1=True` or `FRAME_COMPARE_CACHE_HASH` requests it, `FrameMetricsCacheInfo` carries an optional precomputed `clips` payload populated by `cache.build_cache_info`, and selection cache keys stay stable with nullable digests; new tests cover the fast path, opt-in hashing, and cache-key determinism.
- *2025-11-13:* docs(mcp): replace Serena with Codanna across `CODEX.md`/`agents.md`, add Codanna MCP server config and quickstart, document token‑efficient defaults, and include a concise “Codanna + Sequential‑Thinking” workflow. Historical Serena mentions remain in logs for provenance.
- *2025-11-13:* feat(net): centralize retry/timeout policy, add structured backoff logging, and cover httpx/slow.pics adapters with tests.
- *2025-11-13:* ci/release: enrich package metadata, add wheel content checks, add Windows build smoke, and wire an optional TestPyPI publish workflow.
- *2025-11-13:* feat(api): define curated public exports and add smoke tests; no runtime behavior changes.
- *2025-11-13:* refactor(api): pruned `_COMPAT_EXPORTS` down to the documented runner/wizard/doctor/preflight/tmdb helpers, synced `typings/frame_compare.pyi`, taught the remaining tests/helpers/docs to target `src.frame_compare.*`, and reiterated in the trackers that only the canonical modules (plus the curated `frame_compare` exports) are supported. Verification: `rg` sweeps for the removed shim imports (no matches), `UV_CACHE_DIR=.uv_cache uv run --no-sync ruff check`, `UV_CACHE_DIR=.uv_cache uv run --no-sync npx pyright --warnings`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv_cache uv run --no-sync python -m pytest -q` (290 passed, 1 skipped), and `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`. Packaging: `UV_CACHE_DIR=.uv_cache uv run --no-sync python -m build` still fails to download `wheel` offline, but `UV_CACHE_DIR=.uv_cache uv run --no-sync twine check dist/*` and a manual wheel audit confirmed `src/frame_compare/py.typed` plus `data/config.toml.template` and `data/report/{index.html,app.css,app.js}` are present in the existing artifact.
- *2025-11-12:* refactor(shims): deleted the remaining `src/{analysis,vs_core,slowpics,cli_layout,report,config_template}.py` bridges (and `.pyi` stubs), repointed CLI/core/runtime/tests/docs to import from `src.frame_compare.*`, and pruned `frame_compare._COMPAT_EXPORTS` so only curated exports remain. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest -q`, `UV_CACHE_DIR=.uv_cache uv run --no-sync ruff check --fix && ... ruff check`, `UV_CACHE_DIR=.uv_cache uv run --no-sync npx pyright --warnings`, `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`. Attempted `UV_CACHE_DIR=.uv_cache uv run --no-sync python -m build`, but pip could not download `wheel` in this offline sandbox, so packaging verification (`uv run --no-sync twine check dist/*`) remains pending until network access is available.
- ci: add packaging build + content verification; chore(imports): extend import contracts for new packages.

- docs: expand screenshot/slow.pics reference and add JSON-tail how-to
- refactor(screens): deprecate center_pad and always center padding; warn when explicitly set

- *2025-11-12:* docs(config): added `tools/gen_config_docs.py` with a `--check` mode to emit `docs/_generated/config_tables.md` directly from `src/datatypes.py`, wired in `tests/docs/test_config_docs_gen.py` as a sentinel, linked the new generated tables from `docs/README_REFERENCE.md`, and corrected the `[runtime].ram_limit_mb` default to `8000`.
- *2025-11-12:* refactor(vs): split `vs_core` into `src/frame_compare/vs/{env,source,props,color,tonemap}.py`, rewired intra-module imports with `TYPE_CHECKING` guards, added a `src/vs_core.py` compatibility shim plus `.pyi` stub that re-exports every helper (public and private) until Sub-phase 11.10 removes it, and extended `importlinter.ini`’s Runner→Core→Modules layer with the new subpackage. 📋 Verification (with the `preview` extra installed locally): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`, and `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`.
- *2025-11-12:* refactor(packaging): moved `src/{cli_layout,report,slowpics,config_template}.py` under `src/frame_compare/`, left shims at the legacy `src.*` paths that re-export every symbol (including the `_read_template_bytes` helper and `resources` module object) with Ruff/Pyright suppressions plus `.pyi` stubs to keep type info stable, added `tests/__init__.py` so the in-repo `tests` package wins over any globally installed `tests` modules, deleted `Legacy/comp.py`, and extended `importlinter.ini`’s Runner→Core→Modules layer with the new modules. Shim removal is scheduled for Sub-phase 11.10 once downstream imports migrate.
- *2025-11-12:* refactor(subproc): centralized subprocess handling for FFmpeg/FFprobe/VSPreview by adding `src/frame_compare/subproc.py::run_checked` (argv-only, `shell=False`, stdio defaults, optional `check`) and routing `src/screenshot.py`, `src/frame_compare/vspreview.py`, and `src/audio_alignment.py` through the helper so timeout/error messages stay unchanged while banning `shell=True`. Screenshot timeout/command-construction tests now patch `src.frame_compare.subproc.run_checked`, and `importlinter.ini` includes the new module in the Runner→Core→Modules layer. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped, 39.98 s); `.venv/bin/ruff check` (clean); `.venv/bin/pyright --warnings` (clean); `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini` (Contracts: 3 kept, 0 broken; first attempt without the cache override hit a permission error reading the shared UV cache).
- *2025-11-12:* refactor(analysis): split metrics/selection/cache IO into `src/frame_compare/analysis/` and keep `src/analysis.py` as a shim for the legacy API (including `_quantile`, `_collect_metrics_vapoursynth`, and the caching helpers the tests monkeypatch). Added public aliases inside the new modules so the rest of `frame_compare` imports non-underscored symbols, stapled `# pyright: standard` headers onto the legacy files to keep them on the repo-wide “standard” mode while the rest of `src/frame_compare` stays strict, and extended `importlinter.ini`’s Runner→Core→Modules layer with `src.frame_compare.analysis`. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped), `.venv/bin/ruff check` (clean), `.venv/bin/pyright --warnings` (clean), `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini` (Contracts: 3 kept, 0 broken).

- *2025-11-12:* Docs: integrate Serena MCP as primary orchestration tool in AGENTS/CODEX, add Test Guardrails, and list Serena analysis calls as always-allowed; further refinements pending web-sourced best practices.
- *2025-11-12:* Tooling: enable Markdown LSP in Serena config to improve documentation editing and outline support.
- *2025-11-11:* Strict-mode ratchet (Phase 9.11) for the runner boundary: promoted public exports for tonemap validation, media discovery, cache info, and alignment confirmation so `runner.py` no longer reaches into underscore helpers, hardened the JSON/layout handling in `runner.py` (typed `select_frames` inputs, explicit mapping casts, safer folding/verification summaries), deleted unused VSPreview overlay code while guarding `_ensure_audio_alignment_block`, and taught the CLI reporter stubs/patch helpers to track `values` so tests keep patching the confirmation hook. Verification: `.venv/bin/pyright --warnings` (clean), `.venv/bin/ruff check` (clean), `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (`273 passed, 1 skipped in 40.03 s`).
- *2025-11-11:* Doctor workflow module extraction (Phase 9.1). Introduced `src/frame_compare/doctor.py` exposing `DoctorCheck`, `collect_checks`, and `emit_results`, re-pointed `frame_compare.py`’s doctor subcommand and wizard dependency checks to the new module, and left `src/frame_compare/core._collect_doctor_checks` / `_emit_doctor_results` as shims so existing monkeypatches continue to work. Verification: `.venv/bin/pyright --warnings`, `.venv/bin/ruff check`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (273 passed, 1 skipped) all succeeded.
- *2025-11-11:* Runner test split & VSPreview guardrails (Phases 6–7). The CLI/runner regression suites now live under `tests/runner/`, reuse shared fixtures from `tests/helpers/runner_env.py` / `tests/conftest.py`, and include fail-fast VSPreview shims that raise when `frame_compare` stops exporting `_VSPREVIEW_*` constants. This keeps CLI coverage reliable and surfaces shim regressions immediately for contributors who only run the runner-specific tests. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (273 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.
- *2025-11-11:* Shared CLI/test fixtures: extracted `_CliRunnerEnv`, `_RecordingOutputManager`, JSON tail/display stubs, and the `_patch_*` helpers into `tests/helpers/runner_env.py`, added reusable `cli_runner_env`/`recording_output_manager`/`json_tail_stub` fixtures via `tests/conftest.py`, and updated the runner + VSPreview suites to consume the shared module. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (272 passed, 1 skipped); `.venv/bin/ruff check` still fails on the pre-existing import-order/unused-import warnings in `src/frame_compare/alignment_runner.py`, `src/frame_compare/cli_runtime.py`, `src/frame_compare/runner.py`, `src/screenshot.py`, and `tests/test_alignment_runner.py`; `.venv/bin/pyright --warnings` continues to flag the long-standing `_format_vspreview_manual_command` / `_VSPREVIEW_*` attribute/Final errors in `tests/test_frame_compare.py` and `typings/frame_compare.pyi`.
- *2025-11-10:* VSPreview polish: runner now calls `alignment_runner.apply_audio_alignment` directly, `_write_vspreview_script` generates files under `ROOT/vspreview/` with deterministic names and clearer permission failures, `_launch_vspreview` logs when the VSPreview executable or `VAPOURSYNTH_PYTHONPATH` is missing (and can accept a mocked subprocess runner), and `_apply_vspreview_manual_offsets` updates CLI/JSON telemetry plus warns when clip names from VSPreview don’t match the current plan. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (270 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.

- *2025-11-10:* Finalized the audio alignment module split by routing `runner.py` through `alignment_runner.apply_audio_alignment`, dropping the redundant `_maybe_apply_audio_alignment` alias, and updating docs/trackers to match. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (265 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.
- *2025-11-10:* Phase 2.3 documentation/tooling hand-off: refreshed the runner-refactor trackers, reiterated the wizard compatibility surface (`frame_compare.resolve_wizard_paths` / `_resolve_wizard_paths` now forward into `src.frame_compare.wizard`), and re-ran the quality gates (`pytest -q` 209 passed / 54 skipped, `.venv/bin/ruff check`, `npx pyright --warnings` still blocked offline so `.venv/bin/pyright --warnings` recorded as the fallback).
- *2025-11-10:* Codex/AGENTS/README now explicitly require Sequential Thinking loops to log every stage with `next_thought_needed=true` until the Review entry so orchestration never stops mid-plan.
- *2025-11-19:* Phase 1.1 modularization: `src/frame_compare/preflight.py` now exposes `resolve_workspace_root`, `resolve_subdir`, `collect_path_diagnostics`, `prepare_preflight`, and `PreflightResult`, and the CLI/runner/tests were rewired to consume the shared API (legacy `_…` aliases remain for compatibility).
- *2025-11-09:* Hardened CLI tonemap overrides: all numeric flags now reject non-finite inputs, `--tm-target` enforces positive values, and regression tests cover the new validation paths.
- *2025-11-10:* Corrected future-dated DECISIONS/CHANGELOG rows and documented the requirement to stamp new log entries with `date -u` so timelines stay accurate.
- *2025-11-10:* Refined AGENTS/CODEX operational guardrails: Standard Flow now captures Context7-first citations plus in-response Fetch MCP metadata logging, and CODEX documents MCP auto-approval rules, `.venv`-first verification with `uv sync` fallbacks, an escalation playbook, and requirements for large refactor micro-plans.
- *2025-11-09:* Phase 5 runner QA: verified TMDB orchestration already flows through the shared resolver, ensured reporter injection docs/tests cover automation scenarios, and reran `npx pyright --warnings`, `.venv/bin/ruff check`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (258 passed) on a networked host.
- *2025-11-09:* Updated CODEX/AGENTS guardrails to require local `.venv/bin/pyright`/Ruff/Pytest runs, allow cache directories for approved checks, and document the `uv sync --all-extras --dev` setup workflow. Packaging releases now exclude dev-only folders (tests/docs/Legacy/refactor/tools) via `MANIFEST.in` so end users download only runtime files, while `comparison_videos/` remains included for fixture availability.
- *2025-11-09:* Guardrail docs now require assistants to provide a Conventional Commit-style subject in every response so the user can copy it directly into `git commit -m …`.
- *2025-11-08:* Tightened the CLI shim surface by removing the `globals().update`/`__getattr__` re-export from `frame_compare.py`, adding an explicit compatibility map plus `typings/frame_compare/__init__.pyi`, and moving tests to import helpers from `src.frame_compare.core`/`cli_runtime` directly.
- *2025-11-08:* Enhanced the public runner API: `RunRequest` now accepts optional `console`, `reporter`, and `reporter_factory` parameters, quiet runs automatically use a new `NullCliOutputManager`, and README demonstrates how automation can silence output or inject custom reporters. Added regression tests to guard the behavior.
- *2025-11-09:* Restored `frame_compare --diagnose-paths` to run without creating or requiring a config file, updated the preflight tests to stub `src.frame_compare.preflight.load_config`, and logged the refreshed `.venv/bin/ruff check` / `.venv/bin/pytest -q` (250 passed, 1 skipped) results plus the still-blocked `npx pyright --warnings` (npm ENOTFOUND) in the Phase 4.3 quality gates.

- *2025-11-03:* Added a reusable CLI harness for runner validation (`_CliRunnerEnv` plus `_patch_core_helper`, `_patch_runner_module`, `_patch_vs_core`, `_patch_audio_alignment`), migrated the CLI-heavy tests in `tests/test_frame_compare.py` and `tests/test_paths_preflight.py` to the new fixtures, centralized VapourSynth stubs, and re-ran `.venv/bin/pytest tests/test_frame_compare.py tests/test_paths_preflight.py` along with the full `.venv/bin/pytest` suite (247 passed, 1 skipped). `npx pyright --warnings` remains blocked offline (`npm ERR! ENOTFOUND registry.npmjs.org`).
- *2025-11-08:* Documented the public runner API for automation (README “Programmatic Usage”), logged the Phase 2.3 quality gates (`.venv/bin/ruff check`, `.venv/bin/pytest`), captured the offline `npx pyright --warnings` failure for follow-up, and initialized the residual-risk log in `docs/runner_refactor_checklist.md`.
- *2025-11-10:* Split helper logic into `src/frame_compare/core.py`, removed the runner’s `_IMPL_ATTRS` indirection in favor of direct imports, re-exported helpers via `frame_compare.py`, and updated tests/docs to record Phase 4.1 completion.
- *2025-11-01:* Phase 4.2 regression parity: added runner-level slow.pics/audio-alignment tests, refreshed README + docs/DECISIONS.md to document the public API guarantees, updated `docs/runner_refactor_checklist.md`, and re-ran `npx pyright --warnings` (0 issues) to lock the helper extraction work.
- *2025-11-09:* Closed the runner refactor Phase 3 quality gates: reran `npx pyright --warnings`, `.venv/bin/ruff check`, and `.venv/bin/pytest` (246 passed), updated `docs/runner_refactor_checklist.md` with the final statuses, and documented the remaining helper-migration work as the Phase 4 kickoff task.
- *2025-11-08:* Finished the Phase 1 runner extraction: the orchestration logic now lives in `src/frame_compare/runner.py`, `frame_compare.run_cli` is a thin shim that delegates through a `RunRequest`, tests exercise the new module boundary, and documentation captures the runner API plus Phase 2 follow-ups.
- *2025-11-07:* Refreshed tone-mapping presets to align with the new advanced controls: upgraded `reference`, `filmic`, `spline`, and `contrast` profiles, added `bright_lift` and `highlight_guard`, and updated defaults (smoothing 45f, percentile `99.995`, contrast recovery `0.30`) across docs, template, and layout.
- *2025-11-06:* Added BT.2390 knee, dynamic-peak-detection presets, and optional post-tonemap gamma lift. `[color]` now exposes `dst_min_nits=0.18`, `knee_offset`, `dpd_preset`, `dpd_black_cutoff`, and `post_gamma`, libplacebo calls forward `tone_mapping_param`/`peak_detection_preset`/`black_cutoff` with compatibility shims, overlays understand `{knee_offset}`, `{dpd_preset}`, `{dst_min_nits}`, and new CLI flags (`--tm-*`) allow per-run overrides. Docs, presets, and tests updated accordingly.
- *2025-11-06:* Default screenshot exports now expand limited-range SDR to full-range RGB via a new `[screenshots].export_range` setting (default `"full"`), with `"limited"` preserving video-range PNGs. Both VapourSynth and FFmpeg writers honour the option, `_SourceColorRange` provenance is recorded when expanding, and documentation/tests/config templates were updated accordingly.
- *2025-11-05:* Harmonised HDR tonemap range metadata with sampled RGB output, wiring detected `_ColorRange` through geometry and overlays, preserving frame props during subtitle paths, and extending regression coverage/docs for the updated pipeline.
- *2025-10-30:* Auto-launched the configuration wizard during interactive first runs when `config/config.toml` is missing, added a `--no-wizard` flag plus `FRAME_COMPARE_NO_WIZARD` override, surfaced a fallback reminder for non-interactive sessions, and refreshed README/tests accordingly.
- *2025-10-30:* Replaced the audio alignment “progress” bar with a spinner so CLI output stays truthful while offsets are estimated.
- *2025-10-30:* Overhauled the offline HTML report viewer with persistent zoom/fit presets, pointer-anchored wheel zoom, pan/align controls, and shortcut legends to better mirror slow.pics.
- *2025-10-29:* Added optional HTML report generation (configurable via `[report]` or `--html-report`), including vendored assets, CLI auto-open support, JSON-tail disclosures, embedded report data for offline viewing, and unit coverage for the new generator. Added overlay mode toggle with keyboard/click encode cycling.
- *2025-10-29:* Added interactive `frame-compare wizard` with presets, introduced `frame-compare doctor` dependency checklist (JSON-capable), expanded reference docs, and strengthened CLI help/tests for the new commands.
- *2025-10-29:* Clarified the CLI audio alignment panel output (stream summaries, cached reuse messaging, offsets file footer) and aligned documentation; slow.pics shortcut filenames now derive from the sanitised collection name with regression tests for edge cases; README and reference tables updated.
- *2025-10-22:* Disabled slow.pics auto-upload by default, added an upfront CLI warning when it is enabled, aligned documentation with dataclass defaults, introduced a packaged `frame-compare` console entry point, and wired Ruff linting into CI (Pyright now blocks failures).
- *2025-10-21:* Prevented VSPreview helper crashes on Windows `cp1252` consoles by sanitising printed arrows to ASCII, preferring UTF-8 output streams, adding regression coverage, and documenting the console behaviour.
- *2025-11-15:* Preserved HDR frame props by snapshotting them at source load (`ClipPlan.source_frame_props`) and rehydrating them inside `process_clip_for_screenshot`/`generate_screenshots`, so even negatively trimmed clips retain `_Matrix`/`_Transfer`/MasteringDisplay hints and diagnostic overlays continue to show HDR metadata. Added regression tests covering tonemap rehydration and screenshot overlays.
- *2025-10-20:* Hardened audio alignment's optional dependency handling by surfacing clear `AudioAlignmentError` messages when
  `numpy`, `librosa`, or `soundfile` fail during onset envelope calculation, and refreshed regression coverage for the failure
  path.
- *2025-10-20:* VSPreview-assisted manual alignment now displays existing manual trims using friendly clip labels so operators
  can immediately see which plan each baseline affects before accepting new deltas.
- *2025-10-20:* Prevented VSPreview script overwrites by appending per-run entropy to generated filenames and warning when a
  collision is detected.
- *2025-10-20:* Fixed mod-2 odd-geometry failures by pivoting subsampled SDR clips through YUV444P16 when needed, emitting Rich console notes that summarise the axis/policy, and expanding docs/config guidance for the `odd_geometry_policy` and `rgb_dither` options.
- *2025-10-19:* Normalised VapourSynth colour metadata inference for SDR clips, cached inferred props on the
  clip to avoid redundant frame grabs, exposed config overrides for HD/SD defaults and per-file colour
  corrections, refreshed documentation, and expanded regression coverage for the new heuristics.
- *2025-10-18:* Fixed VapourSynth RGB conversion when colour metadata is absent by defaulting to Rec.709
  limited parameters, preventing fpng "no path between colours" failures and adding regression coverage.
- *2025-10-17:* Documented the VSPreview-assisted manual alignment flow (README, reference tables, pipeline guide), surfaced the
  CLI help text, added a fallback regression test, and published a cross-platform QA checklist for manual verification.
- *2025-10-16:* Hardened analysis cache and audio offsets paths to stay within the workspace root, added regression tests for escape attempts, removed the generated `config.toml` from source control in favour of the packaged template, and restricted screenshot cleanup to directories created during the current run.
- *2025-10-16:* Limited supported Python versions to 3.13.x (`>=3.13,<3.14`) to align with current `librosa`/`numba` wheels; updated project metadata and lockfile.
- *2025-10-14:* Locked workspace roots to `--root`/`FRAME_COMPARE_ROOT`/sentinel discovery, seeded config under `ROOT/config/config.toml`, enforced `ROOT/comparison_videos[/screens]`, added `--diagnose-paths`, and blocked site-packages writes before screenshotting.
- *2025-10-12:* Default config now seeds to `~/.frame-compare/config.toml`, `[paths].input_dir` defaults to `~/comparison_videos`, and the packaged template ships from `src/data/` (with wheel coverage) to avoid site-packages permission issues. *(Superseded by 2025-10-14 workspace root lock.)*
- *2025-10-11:* Added an optional `$FRAME_COMPARE_TEMPLATE_PATH` override and filesystem fallback for the packaged config template plus clearer screenshot permission errors so runs targeting read-only `comparison_videos` trees fail fast with guidance.
- *2025-10-19:* Seed the packaged default config into a per-user directory when the project tree is read-only so packaged installs no longer fail to start on permission errors.
- *2025-10-20:* Allow disabling the per-frame FFmpeg timeout by setting `screenshots.ffmpeg_timeout_seconds` to 0 while keeping negative values invalid in validation and docs.
- *2025-10-18:* Added a per-frame FFmpeg timeout and disabled stdin consumption to prevent hung screenshot renders and shell freezes on Windows.
- *2025-10-17:* Streamlined CLI framing: removed the banner row, surfaced cached-metrics reuse inside the Analyze block, dropped the At-a-Glance crop-mod readout in favour of effective tonemap nits, and trimmed the Summary output frames line to match the refreshed console layout and tests.
- *2025-10-16:* Documented deep-review finding: screenshot cleanup must enforce path containment before deleting outputs; remediation planned.

- *2025-10-15:* Relocated bundled comparison fixtures to the repository-root `comparison_videos/` directory, updated CLI docs to match the default `paths.input_dir`, and noted the new resolution fallbacks.
- *2025-10-14:* Clarified CLI hierarchy by softening section badge colors, brightening subhead prefixes, expanding the At-a-Glance box with alignment, sampling, and canvas metrics for quicker triage, and realigning the Summary section with key/value formatting to remove ragged rows.
- *2025-10-13:* Removed the `types-pytest` dev dependency so `uv run` can resolve environments on fresh clones without relying on a non-existent stub package while keeping the actual `pytest` runtime dependency in the dev group for CI and local tests.
- *2025-10-10:* Reject invalid `cli.progress.style` values during configuration loading and persist normalized styles for downstream flag handling to keep CLI reporters consistent.
- *2025-10-12:* Added local type stubs for optional CLI/testing dependencies and hardened JSON-tail assertions in tests so Pyright runs cleanly without relaxing diagnostic settings.
- *2025-10-09:* Expanded minimal overlays with resolution/upscale and frame-selection type lines while simplifying diagnostic overlays by removing on-screen selection timecode/score/notes that remain available via cached metadata.
- *2025-10-02:* Scoped audio alignment's NumPy flush-to-zero warning suppression to local contexts, added regression tests to keep diagnostics available, and hardened TMDB caching with TTL-aware eviction to cap memory growth during long CLI sessions.
- *2025-10-01:* Added selection metadata persistence v1: cached JSON sidecar, compframes annotations, CLI summary metrics, and screenshot overlay reuse without rerunning analysis.
- *2025-10-01:* Added `docs/audio_alignment_pipeline.md` to detail the alignment workflow, configuration constraints, and offsets file contract.
- *2025-09-30:* Enhanced CLI group blocks with accent subhead glyphs, dimmed divider rules, numeric alignment, and verbose legends describing token colouring for RENDER/PREPARE sections.
- *2025-09-30:* Diagnostic overlay now replaces the HDR MAX/AVG measurement block with render resolution details, mastering display luminance (if present), and cached frame-selection metadata while trimming redundant frame-info lines for leaner CLI banners.
- *2025-09-29:* Initialize changelog to align with repository persistence rules.
- *2025-09-29:* Revamped CLI presentation: extended palette, added style spans, highlights, section accents, progress bar styling, verification metrics, and refreshed templates per `features/CLI/GUIDE.md`.
- *2025-11-10:* Ensured runner funnels audio alignment through `core._maybe_apply_audio_alignment` again (restoring the test/mocking seam) and deep-copied the JSON-tail audio block before building CLI layout views so manual trims, VSPreview offsets, and reference trims persist. Added regression coverage, reran the full suite, and cleaned up Ruff/Pyright diagnostics that surfaced after the refactor.


## [0.0.15](https://github.com/TJZine/frame-compare-legacy/compare/frame-compare-v0.0.14...frame-compare-v0.0.15) (2025-12-30)


### Features

* enhance security by validating webhook URLs and limiting expression engine complexity ([dfaa753](https://github.com/TJZine/frame-compare-legacy/commit/dfaa753e9f20e47c77b2b4703a858307b844cbac))


### Bug Fixes

* adjust diagnostic overlay position ([c222fe7](https://github.com/TJZine/frame-compare-legacy/commit/c222fe7c10eb95940836307b66667053be98296f))
* align vspreview metadata and report handling ([77be117](https://github.com/TJZine/frame-compare-legacy/commit/77be11715aa77cb7045f20d2f21f74953dc987d8))
* correct error handling, indentation, and unreachable test code ([6b61905](https://github.com/TJZine/frame-compare-legacy/commit/6b6190522b1fe8b7d523357d1c6b6a26596b96c7))
* harden layout eval and typing ([4aac001](https://github.com/TJZine/frame-compare-legacy/commit/4aac001cfff7b7470005f14d47f9855876345aaf))
* reorder SetFrameProps call signature to prefer bound method ([dbd3425](https://github.com/TJZine/frame-compare-legacy/commit/dbd34256abca3bd665e6c9ef5e0de359ee75010a))
* typeerror fix ([55c86c6](https://github.com/TJZine/frame-compare-legacy/commit/55c86c6ccb744a085984a732fd8c05ca16039440))


### Refactors

* centralize TTY stdin mocking in tests ([d00360f](https://github.com/TJZine/frame-compare-legacy/commit/d00360f749462c5b15abe5e4ea71a42ea232433e))
* decompose audio alignment runner into a dedicated package ([780290b](https://github.com/TJZine/frame-compare-legacy/commit/780290b3ddda874d0e52b7204bb27f620c5af97f))
* exception handling across multiple modules ([6bada30](https://github.com/TJZine/frame-compare-legacy/commit/6bada308f0603d8607edca1d59a6dea136818c1b))
* introduce orchestration package  ([495223f](https://github.com/TJZine/frame-compare-legacy/commit/495223f68113e693a88272b94d42b85652b635fc))
* introduce orchestration package and move CLI setup and reporting logic ([edd7e84](https://github.com/TJZine/frame-compare-legacy/commit/edd7e84e00586fc1404920915c267c34c1df372f))
* modularize cli_layout.py ([00e708e](https://github.com/TJZine/frame-compare-legacy/commit/00e708ec9f05cdab84af709e37f757849b60165f))
* modularize render.py ([7bc73e3](https://github.com/TJZine/frame-compare-legacy/commit/7bc73e335ad7e679b63250dc7370483d76850147))
* reimplement audio alignment with a new service architecture ([5f50cd8](https://github.com/TJZine/frame-compare-legacy/commit/5f50cd8b5362a4c35b4e64026372dd5b8de7e18f))


### Documentation

* update agents.md ([84235f0](https://github.com/TJZine/frame-compare-legacy/commit/84235f00c8f447c5ca9853dcac8462ce7f487cba))

## [0.0.14](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.13...frame-compare-v0.0.14) (2025-11-24)


### Features

* add alignment calculation logging and a reproduction script ([ac29cc9](https://github.com/TJZine/frame-compare/commit/ac29cc9c7ff00879c950034df59664c1049090f8))
* add parsing and display for Dolby Vision L2, L5, and L6 metadata ([6f16424](https://github.com/TJZine/frame-compare/commit/6f1642434a73d483f38c533bd7cba06e39707808))
* adjust dovi_tool output structure with fallback ([6d4b9ff](https://github.com/TJZine/frame-compare/commit/6d4b9ff4a240512aeef64040b8cbfee118da723b))
* auto-enable Dolby Vision from RPU and add stats to diagnostic overlay ([#177](https://github.com/TJZine/frame-compare/issues/177)) ([6a781e1](https://github.com/TJZine/frame-compare/commit/6a781e10e902efbd5db852215e3518f70f8282d0))
* auto-enable Dolby Vision from RPU and add tonemapping retries ([2c5b46b](https://github.com/TJZine/frame-compare/commit/2c5b46ba451e3462a5f8b5a405f9d6ac1fc551ea))
* enhance Dolby Vision Level 1 metadata parsing ([a10bf76](https://github.com/TJZine/frame-compare/commit/a10bf76bb85708b8c5be82b732706c6642fc2be9))
* extract Dolby Vision metadata using temporary RPU and JSON files ([2c2c072](https://github.com/TJZine/frame-compare/commit/2c2c07235f0ce5564a2c93b5f888ebb85345a40e))
* inject Dolby Vision L1 metadata using external dovi_tool ([a66bc0e](https://github.com/TJZine/frame-compare/commit/a66bc0e4e8abf61fafcb870ffbce652fa6f4c1fb))
* inject Dolby Vision metadata to diagnostic overlay ([d467da0](https://github.com/TJZine/frame-compare/commit/d467da049280b533bef4dc77e0f28aa85f6574b6))


### Bug Fixes

* code review suggestions ([48e887a](https://github.com/TJZine/frame-compare/commit/48e887a86a46af6665c3a6b3c9acf599c2b76a08))
* ensure negative offsets are normalized after first application ([6c16f49](https://github.com/TJZine/frame-compare/commit/6c16f4983457e8fd79385d2ee0ce1852daf5a978))
* improve trim normalization in alignment runner ([e1e07a0](https://github.com/TJZine/frame-compare/commit/e1e07a00be603da5cf01eacce04a848437517348))
* ruff whitespace error ([d2b1956](https://github.com/TJZine/frame-compare/commit/d2b19569a5a00abf78ad26891d311ff049f25bcc))


### Refactors

* extract VSPreview manual trim offset application logic ([6d0e1fb](https://github.com/TJZine/frame-compare/commit/6d0e1fb6bd620041ce5d4a6e1f49e5f07e5565a9))

## [0.0.13](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.12...frame-compare-v0.0.13) (2025-11-22)


### Bug Fixes

* correct vspreview overlay offsets ([15d94f5](https://github.com/TJZine/frame-compare/commit/15d94f5bebe080e3c60925e4f19a59669e977104))
* correct vspreview overlay offsets ([#175](https://github.com/TJZine/frame-compare/issues/175)) ([5123e6a](https://github.com/TJZine/frame-compare/commit/5123e6a0f0ffa14d229559e505abce16be7830b6))
* ensure vspreview reference overlay shows baseline status ([d4f2190](https://github.com/TJZine/frame-compare/commit/d4f2190e20906adac7bd99a87a3eec61874c3382))
* repair vspreview overlay indentation ([5a113b8](https://github.com/TJZine/frame-compare/commit/5a113b8454d97776dfe025d605aab0605b37a250))

## [0.0.12](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.11...frame-compare-v0.0.12) (2025-11-21)


### Bug Fixes

* keep vspreview suggestions when frame bias is set ([3a0b253](https://github.com/TJZine/frame-compare/commit/3a0b253c5d83f1717c838a7138de44b924e1c3a8))

## [0.0.11](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.10...frame-compare-v0.0.11) (2025-11-20)


### Features

* enrich diagnostic overlay with per-frame HDR and DoVi stats ([f32bdfb](https://github.com/TJZine/frame-compare/commit/f32bdfb186322c805912afd23718b4e13ab578b6))


### Bug Fixes

* remove audio alignment `frame_offset_bias` and screenshot confirmation flags so suggestions stay unbiased and auto-confirm without prompts (breaking)
* keep VSPreview suggestion overlays from zeroing out when `frame_offset_bias` is set, ensuring small measured offsets still display before manual alignment
* restore vspreview overlay hints ([1a885c7](https://github.com/TJZine/frame-compare/commit/1a885c78dc5957e2457690e4ed81af7ea8945dda))


### Refactors

* retire legacy runner flag and path ([627286e](https://github.com/TJZine/frame-compare/commit/627286e748c92cbfb74ca62d3238a5eb0ee68a62))
* retire legacy runner flag and path ([#171](https://github.com/TJZine/frame-compare/issues/171)) ([036e21b](https://github.com/TJZine/frame-compare/commit/036e21b439783ca800e15d81143cbf7eb3443256))


### Documentation

* trim legacy plans and noisy logs from documentation ([b3de50d](https://github.com/TJZine/frame-compare/commit/b3de50d36ec2f71d7f5c3a29c0511b442ba44d46))

## [0.0.10](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.9...frame-compare-v0.0.10) (2025-11-20)


### Bug Fixes

* break preflight alignment_runner init cycle by lazy resolve_subdir ([fe0742c](https://github.com/TJZine/frame-compare/commit/fe0742cf8fd03f825e9cc618db3111efdfd43655))
* resolve preflight/vspreview circular import ([b10ef38](https://github.com/TJZine/frame-compare/commit/b10ef38e88300861adc4e25f1cc5f733551869c0))


### Refactors

* extract frame_compare CLI helpers into cli_utils ([b707127](https://github.com/TJZine/frame-compare/commit/b707127b72d0068df710205a46425ae0412c9a22))
* extract frame_compare CLI helpers into cli_utils ([#169](https://github.com/TJZine/frame-compare/issues/169)) ([4ba6a93](https://github.com/TJZine/frame-compare/commit/4ba6a931c76dbf6c23c4a7e78c195e739651d99e))
* extract frame_compare CLI wiring into cli_entry ([2d0237b](https://github.com/TJZine/frame-compare/commit/2d0237b112ed3013fa4c3a5919db8bf97282ee55))
* finalize frame_compare CLI cleanup and docs ([18ee281](https://github.com/TJZine/frame-compare/commit/18ee2814c621eeae359324103161a29cc67e1b53))
* move frame_compare compat exports into dedicated shim ([ee76224](https://github.com/TJZine/frame-compare/commit/ee762244167b6bf38a7513f825145fd9ce84f776))


### Documentation

* correct compat exports reference ([258f54d](https://github.com/TJZine/frame-compare/commit/258f54d51dd8c93a4e157999b17133da99da42a4))


### CI

* harden decision-minute workflow outputs ([d59a261](https://github.com/TJZine/frame-compare/commit/d59a261779041fd7af1395cecff742e008c389e7))

## [0.0.9](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.8...frame-compare-v0.0.9) (2025-11-19)


### Features

* **diagnostics:** wire overlay JSON tail, CLI gating, and docs ([8e6a260](https://github.com/TJZine/frame-compare/commit/8e6a260e68547a6e04c868a9d19b3e12cede1319))
* **diagnostics:** wire overlay JSON tail, CLI gating, and docs ([#167](https://github.com/TJZine/frame-compare/issues/167)) ([ed0ff97](https://github.com/TJZine/frame-compare/commit/ed0ff9757fddc14fa793b35cedf0079c6f8ee37f))


### Bug Fixes

* enforce HTTP timeouts, slow.pics locking, and planner guard ([a76e637](https://github.com/TJZine/frame-compare/commit/a76e637e2f0be3ace5c2a6fa4061a1f06b8caae5))
* update commitlint scope tags ([bebddfa](https://github.com/TJZine/frame-compare/commit/bebddfa40d5d7186a660fa650eafedd8b831f5d0))

## [0.0.8](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.7...frame-compare-v0.0.8) (2025-11-19)


### Features

* restore service-mode publisher wiring and tests ([5ebf836](https://github.com/TJZine/frame-compare/commit/5ebf836ac9926ef466d5f0e49602379f5ba7ffc4))
* wire runner DI through RunDependencies and add runner service tests ([4430770](https://github.com/TJZine/frame-compare/commit/4430770ca2c0027af5c510ede017094992901c57))


### Bug Fixes

* harden CLI flags and probe cacheso ([d515031](https://github.com/TJZine/frame-compare/commit/d515031929a50bd6e2da5c4da07766e90fe42d26))


### Refactors

* extract metadata/alignment services ([08385fa](https://github.com/TJZine/frame-compare/commit/08385fa7412cf97a333762b81b2ac9ec9d36d19e))
* extract metadata/alignment services ([#165](https://github.com/TJZine/frame-compare/issues/165)) ([2972ec1](https://github.com/TJZine/frame-compare/commit/2972ec195e0f188afd25ae5613dbe1e5903c4bf1))

## [0.0.7](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.6...frame-compare-v0.0.7) (2025-11-18)


### Bug Fixes

* **cli:** gate non-tonemap overrides and sync render telemetry ([f896de8](https://github.com/TJZine/frame-compare/commit/f896de81182d2ae9563ad561d50e98e653a4cbf4))
* guard tonemap CLI overrides from implicit defaults ([b49c873](https://github.com/TJZine/frame-compare/commit/b49c873d56fb87231827a4c372cfa4907c48b7e1))
* guard tonemap CLI overrides from implicit defaults ([#163](https://github.com/TJZine/frame-compare/issues/163)) ([c1a7403](https://github.com/TJZine/frame-compare/commit/c1a7403f1f3b20c13f0d72c3a6f5738096e48ec9))
* merge diverging docs ([30e00df](https://github.com/TJZine/frame-compare/commit/30e00df508867618bd042f53d9fa96662cbdca6e))
* retain unique encodes in HTML report ([615ee2a](https://github.com/TJZine/frame-compare/commit/615ee2a80a8cdd5f825b366ae3b8ffa19dcdf7e9))


### Documentation

* update plan for overlay diagnostics update ([684fe13](https://github.com/TJZine/frame-compare/commit/684fe13fefadcf6166fca2aa8dabb98496b015c7))

## [0.0.6](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.5...frame-compare-v0.0.6) (2025-11-18)


### Features

* CLI Rendering for cached results ([#161](https://github.com/TJZine/frame-compare/issues/161)) ([6061125](https://github.com/TJZine/frame-compare/commit/60611255fa496dfb4f6c623e78cc525f820c7728))
* finalize CLI cache missing-section controls ([b70cea3](https://github.com/TJZine/frame-compare/commit/b70cea3554b27507d18da818032228a1aef962bf))
* unify CLI rendering for cached results ([94c29ff](https://github.com/TJZine/frame-compare/commit/94c29ff4a822e6ac2ecff9f245f1cd67dd815b46))


### Bug Fixes

* let tonemap presets override default-matching config keys ([5de6c59](https://github.com/TJZine/frame-compare/commit/5de6c59f96425c957bbc302d573c5c85d26e665e))
* prevent CLI from overriding DoVi by default ([4e50efa](https://github.com/TJZine/frame-compare/commit/4e50efa1104b63a7d5dbfe1bef434f5a93539aaa))


### Documentation

* convert flag audit headings to atx ([40719f6](https://github.com/TJZine/frame-compare/commit/40719f662b069c527fbca7e493a0ff624b604cf1))

## [0.0.5](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.4...frame-compare-v0.0.5) (2025-11-16)


### Features

* add auto letterbox crop modes and docs ([e803098](https://github.com/TJZine/frame-compare/commit/e8030987322d9865c6cdc001c52f428b19e2d933))


### Bug Fixes

* add geometry debug info ([7c6ccae](https://github.com/TJZine/frame-compare/commit/7c6ccae2f6490f008d0369083dc013e85f8803e7))
* audio alignment, clip metadata, and tonemapping bug fixes ([64ab9d7](https://github.com/TJZine/frame-compare/commit/64ab9d70a69391ed1029e8c3301c9371d7e421be))
* **audio:** reuse cached FPS for alignment offsets ([9cafde7](https://github.com/TJZine/frame-compare/commit/9cafde7f428d03b2e66769967ffced42d5d682fb))
* normalize letterbox telemetry and cached fps tests ([048cf6b](https://github.com/TJZine/frame-compare/commit/048cf6b6f62f296bb08d1b954aa578425ba90140))
* pre-probe clip metadata for alignment and tonemapping ([ce830ed](https://github.com/TJZine/frame-compare/commit/ce830edd5d112c49ee4b806c411b6da2d57b7a4d))


### Documentation

* update changelog ([f10e096](https://github.com/TJZine/frame-compare/commit/f10e096e20e8c72037373d8391d786f542865bbb))
* update config tables ([7155879](https://github.com/TJZine/frame-compare/commit/71558791bfa01ac6731c6fc883807819b0367dfa))

## [0.0.4](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.3...frame-compare-v0.0.4) (2025-11-15)


### Bug Fixes

* preserve HDR metadata during negative trim padding ([#157](https://github.com/TJZine/frame-compare/issues/157)) ([6d3cf7b](https://github.com/TJZine/frame-compare/commit/6d3cf7b1bf67a8790820f2c7637255b3fcd33798))
* respect explicit falsey `FRAME_COMPARE_DOVI_DEBUG` values so telemetry stays disabled unless the flag is set to `1`/`true`

## [0.0.3](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.2...frame-compare-v0.0.3) (2025-11-15)


### Bug Fixes

* detect HDR clips with partial metadata ([763f3f5](https://github.com/TJZine/frame-compare/commit/763f3f5bd74083798860da6ddac5c46758cda98b))
* restore audio offset hints in CLI and VSPreview ([68da62b](https://github.com/TJZine/frame-compare/commit/68da62b8f509a818c0c769c6d96effd6453cb8ba))
* restore manual offset timing display ([1788306](https://github.com/TJZine/frame-compare/commit/17883068d4b3641d8771e383fc0ea4525e2108ed))
* restore reliable HDR tonemapping ([0ed4527](https://github.com/TJZine/frame-compare/commit/0ed45279b0c56a78917b0e089675f685073548fd))
* simplify manual offset fps guard ([3280043](https://github.com/TJZine/frame-compare/commit/32800434a986aaf984c52be496fbcdb35e3ea476))
* simplify manual offset fps guard ([#154](https://github.com/TJZine/frame-compare/issues/154)) ([f61311e](https://github.com/TJZine/frame-compare/commit/f61311e8a9b6addcda103c7acf64a8546ee088d5))


## [0.0.2](https://github.com/TJZine/frame-compare/compare/frame-compare-v0.0.1...frame-compare-v0.0.2) (2025-11-13)


### Features

* audio alignment update ([271e4d5](https://github.com/TJZine/frame-compare/commit/271e4d55803b876d5a690709bf0c42ab22e6e77a))
* centralize network retry/timeout policy and add backoff tests ([7fcf3e3](https://github.com/TJZine/frame-compare/commit/7fcf3e3c6ed479a92755547ea3d9859b5d6df425))
* default config in repo directory ([#101](https://github.com/TJZine/frame-compare/issues/101)) ([eba6244](https://github.com/TJZine/frame-compare/commit/eba6244fab7b1b937dc64e78df518d13cb45b538))
* define curated exports, align stubs, and add smoke tests ([c2449a1](https://github.com/TJZine/frame-compare/commit/c2449a17b7d248ee16161fe12dab6cf2352df29f))
* expose advanced tonemap controls and CLI overrides for libplacebo ([ae5eb60](https://github.com/TJZine/frame-compare/commit/ae5eb60ba6a80276f779db8ee69e031b07cfdad2))
* harden TMDB flow and document reporter injection ([49b19d7](https://github.com/TJZine/frame-compare/commit/49b19d75c9c26cd37209354fba9f7b5cfd6a4f7e))
* HTML comparison report ([1be656a](https://github.com/TJZine/frame-compare/commit/1be656ab98e19a78c1a05fea8376afa3c83e00ad))
* HTML Report Enhancements ([8f55404](https://github.com/TJZine/frame-compare/commit/8f55404e08fbdff33e572539170dfa283a132ee3))
* HTTP Adapter + Retries ([6d3835b](https://github.com/TJZine/frame-compare/commit/6d3835b5fb992ec77c203bef0f9528bc73244df0))
* lock workspace root and expose diagnostics helpers ([0c3e600](https://github.com/TJZine/frame-compare/commit/0c3e6006203904dc73fe0e7c22e68a9f1d5b8f19))
* restore fpng screenshots and cli error layer ([#7](https://github.com/TJZine/frame-compare/issues/7)) ([b15b107](https://github.com/TJZine/frame-compare/commit/b15b107f277f18ea2ab99e5ec70080667e9a1199))
* restore parity and expand regression coverage ([#4](https://github.com/TJZine/frame-compare/issues/4)) ([2c88cb5](https://github.com/TJZine/frame-compare/commit/2c88cb5fe3dbf523297b346928de147d00fdf967))
* scaffold VSPreview manual alignment flow ([aaf09fa](https://github.com/TJZine/frame-compare/commit/aaf09fa0e735e07ff30b26a59323f914fa2f1ae2))
* slowpics CLI improvements ([df5c9f8](https://github.com/TJZine/frame-compare/commit/df5c9f81da89cb5948faa84186b593b633653cd7))
* Threaded slow.pics uploads implemented ([5dd492e](https://github.com/TJZine/frame-compare/commit/5dd492e2f880094762586d5bf5f9b256722849b6))
* Tonemap  Update ([bdb718d](https://github.com/TJZine/frame-compare/commit/bdb718d6c00b0492efa18099ffcfb3a1b053c75d))
* Tonemap  Update ([#144](https://github.com/TJZine/frame-compare/issues/144)) ([a72114a](https://github.com/TJZine/frame-compare/commit/a72114addc3fb049268019c43ac2412f0f982f42))


### Bug Fixes

* align ffmpeg screenshots with trim offsets ([e0faa43](https://github.com/TJZine/frame-compare/commit/e0faa431b0d8025a7755caaa8f85c4351bdcc6a3))
* align httpx stub handler signature ([be3ddd7](https://github.com/TJZine/frame-compare/commit/be3ddd774f121967ecb670c441167813fa940054))
* code review fixes ([d735d6a](https://github.com/TJZine/frame-compare/commit/d735d6a5be8c1ec4b643e17066de1e2cca505271))
* code review fixes. ([e507e64](https://github.com/TJZine/frame-compare/commit/e507e644356113707025aea073ae260c225c1d5e))
* normalize VSPreview reuse and tighten launch handling ([2f36158](https://github.com/TJZine/frame-compare/commit/2f361582108b32af98f89ee45cbe15c6585504e2))
* persist VSPreview manual offsets as deltas ([9297bda](https://github.com/TJZine/frame-compare/commit/9297bdaf181baf20cf78100b9288ecae54b10b7a))
* preserve audio alignment metadata ([1a20dbf](https://github.com/TJZine/frame-compare/commit/1a20dbf1a0160541a91ae3077206a358a47d1ef0))
* sanitize TMDB titles to prevent escape-sequence injection (OWASP A07) ([f92975e](https://github.com/TJZine/frame-compare/commit/f92975e245d9536e1d51004dff83960f0b6c0384))
* surface cache recompute reasons in CLI and logs ([25e5ccd](https://github.com/TJZine/frame-compare/commit/25e5ccdc34c713fbae0cd8e4df07d697ace68358))
* tolerate slow.pics shortcut write failures and expose telemetry ([0ce6adc](https://github.com/TJZine/frame-compare/commit/0ce6adca0b02343530ac555ec91d937a883b2a58))
* update commit tests to exclude autoloaded plugins ([3de0f08](https://github.com/TJZine/frame-compare/commit/3de0f081a740591433bd61abdb35a56114993f80))
* whenever we re-range the clip for overlays we now reset the frame props to match the new range. See src/screenshot.py:1999-2067 for the normalisation step and src/screenshot.py:2076- ([60fd1ac](https://github.com/TJZine/frame-compare/commit/60fd1ac033dec318b6d36bd191b1d1e09ce7662e))


### Performance

* precompute clip inputs and document hash opt-in ([de21693](https://github.com/TJZine/frame-compare/commit/de21693f7338cfcdd65a15b35571605fbe6c08d4))


### Refactors

* analysis - split selection, metrics, and cache IO into package ([d60463c](https://github.com/TJZine/frame-compare/commit/d60463c949da380c5d7b8751c5173a58cfab054e))
* centralize retry/backoff for TMDB and slow.pics ([c6e8c72](https://github.com/TJZine/frame-compare/commit/c6e8c721c5e166cd2da2091c9ce3c4730498dd6a))
* centralize subprocess calls and harden FFmpeg/VSPreview/ffprobe usage ([abd9833](https://github.com/TJZine/frame-compare/commit/abd9833f65c0d460e758c5c7c82cfc4da6c16465))
* extract config writer and presets modules ([9c265d5](https://github.com/TJZine/frame-compare/commit/9c265d58d9a7d0024a4ebaa6bf7446f9b6b75875))
* extract doctor workflow module ([818980a](https://github.com/TJZine/frame-compare/commit/818980a3380052c172d8d7eaa942bde12cd7fd6f))
* extract planner module for clip overrides ([35330ee](https://github.com/TJZine/frame-compare/commit/35330eeae61e2baef9502e3fc4e5ceab7775bb21))
* extract pure screenshot helpers into render modules ([3d5dada](https://github.com/TJZine/frame-compare/commit/3d5dadad9042668885dac298301e7173b2c17781))
* extract TMDB workflow to tmdb_workflow(phase 10.1) ([58bf390](https://github.com/TJZine/frame-compare/commit/58bf390f83ef7aad7c57d3ea23a483140134dc9c))
* extract wizard module ([86b1b9e](https://github.com/TJZine/frame-compare/commit/86b1b9e7db1ffeb837b8e68adfee08159076ec32))
* finish runner modularization and harden QA gates ([6059de5](https://github.com/TJZine/frame-compare/commit/6059de578998900666a6f1274f0c0dd539516dec))
* further refactor fixes through phase 6.3 ([c141b16](https://github.com/TJZine/frame-compare/commit/c141b164a3366841b70ea99a9a8deb2a45408dd8))
* harden workspace outputs and clean up typing ([ed5b2da](https://github.com/TJZine/frame-compare/commit/ed5b2da4f97cb376090ace342ef96388da72d33b))
* logging normalization ([c4c8ea5](https://github.com/TJZine/frame-compare/commit/c4c8ea5f7ef88113dff0f75b68b9b0849feed82f))
* move clip init + selection helpers to selection module(phase 9.4) ([4206fef](https://github.com/TJZine/frame-compare/commit/4206fef6847b45522f254da68246789f6a0f8580))
* move internal modules under frame_compare ([c1510eb](https://github.com/TJZine/frame-compare/commit/c1510ebae79caae000584cdd19196380db92af74))
* prune legacy exports and finalize module imports ([2c52200](https://github.com/TJZine/frame-compare/commit/2c5220092a6091cb66735648b47ea8c52d437739))
* remove core shims and prune curated exports (phase 9.8) ([1d6d6d1](https://github.com/TJZine/frame-compare/commit/1d6d6d19c11e8fde41763695d0f607feef56e158))
* remove TMDB legacy shims and update tests to tmdb_workflow (phase 10.2) ([666e5b0](https://github.com/TJZine/frame-compare/commit/666e5b0ccd2eb9237d83e7525200da6064d37351))
* remove transitional shims and repoint imports(phase 11.10) ([90999f8](https://github.com/TJZine/frame-compare/commit/90999f8bf945e3a6de8a81ab91008a286ea4f21f))
* split metadata helpers into module ([16c2a6a](https://github.com/TJZine/frame-compare/commit/16c2a6a522ae6afe9af23af398f2095f43864da4))
* split vs_core into env, source, props, color, tonemap with shim ([e9257ac](https://github.com/TJZine/frame-compare/commit/e9257ac1668326b957f78d9395d56cb5fd688046))
* through phase 6.3 ([3455df0](https://github.com/TJZine/frame-compare/commit/3455df0561ec92c5af3be8f5174b541dbe5da716))
* unhook runner from core via metadata/runtime utils (phase 9.3) ([f8631e0](https://github.com/TJZine/frame-compare/commit/f8631e07a1d1504690e865c0687910f669e10388))


### Documentation

* audit reference materials ([#83](https://github.com/TJZine/frame-compare/issues/83)) ([c2b996f](https://github.com/TJZine/frame-compare/commit/c2b996f19df0510d9669cfaa5af16a62bf786f8e))
* README refresh ([#82](https://github.com/TJZine/frame-compare/issues/82)) ([b1f213c](https://github.com/TJZine/frame-compare/commit/b1f213c2c5c1fb42303025d83c2e141e5f69948e))
* record phase-5 runner QA and checklist updates ([9889ca7](https://github.com/TJZine/frame-compare/commit/9889ca70096773a2732413118f09b70aea9712fa))
* record runner test layout and phase 8 checks ([36bb11a](https://github.com/TJZine/frame-compare/commit/36bb11ac97c41f300596d4ea844f986d96593681))
* rewrite README ([#1](https://github.com/TJZine/frame-compare/issues/1)) ([78e4db7](https://github.com/TJZine/frame-compare/commit/78e4db7219d5a60b7645bc51eca820366e1769b4))
* update progress docs ([bdeea61](https://github.com/TJZine/frame-compare/commit/bdeea6187c32ac5ac91fe6543298f042ccb5a112))
* update readme ([d45a4e1](https://github.com/TJZine/frame-compare/commit/d45a4e108a6d4328f6fe10194ee4f0f17e2e712a))


### CI

* add packaging verification and extend import contracts ([c2e816f](https://github.com/TJZine/frame-compare/commit/c2e816fc3b9884ba47d0ba46e45902327b042b03))
* create uv venv for packaging job ([fb1a652](https://github.com/TJZine/frame-compare/commit/fb1a65224bb7d64a6a1d86f9500d2fff761f7e93))
* fix import-linter install and add preview extra ([59f0401](https://github.com/TJZine/frame-compare/commit/59f0401b5604f4251197d8eb1207abbb2ab607a1))
* install packaging tools into system env for uv ([1f23506](https://github.com/TJZine/frame-compare/commit/1f23506eb6fdbf0d35c4c55f666715d4fac5c04b))

## Changelog

All notable user-visible updates will be documented in this file in reverse chronological order.



- *2025-11-13:* chore(types): removed the last `# pyright: standard` escape hatch from `cli_layout`, hardened optional MultipartEncoder imports/guards in `slowpics`, validated template/line/table inputs to avoid `Unknown` bleed-through, and added a renderer helper so progress columns no longer reach into protected state.
- *2025-11-13:* refactor(cache): bumped metrics payloads to v2 with embedded clip identity snapshots (abs path/size/mtime plus opt-in sha1), taught `cache.build_cache_info` to populate frozen `ClipIdentity` entries, made selection sidecars reuse the same snapshot instead of re-stat'ing files, and added cache identity regression tests (cross-folder isolation, v1 upgrade, snapshot-backed sidecar, sha1 opt-in). Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_cache_identity.py`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q tests/test_analysis.py::test_select_frames_uses_cache`, and `.venv/bin/pyright --warnings`.
- *2025-11-13:* dx(cache): CLI + JSON tail now show the specific reason when frame metrics are recomputed (config mismatch, fps mismatch, missing, etc.), and analysis logs emit INFO breadcrumbs for metrics cache/selection sidecar misses to ease triage.
- *2025-11-13:* perf(cache): `_build_clip_inputs`/`build_clip_inputs_from_paths` skip SHA1 hashing unless `compute_sha1=True` or `FRAME_COMPARE_CACHE_HASH` requests it, `FrameMetricsCacheInfo` carries an optional precomputed `clips` payload populated by `cache.build_cache_info`, and selection cache keys stay stable with nullable digests; new tests cover the fast path, opt-in hashing, and cache-key determinism.
- *2025-11-13:* docs(mcp): replace Serena with Codanna across `CODEX.md`/`agents.md`, add Codanna MCP server config and quickstart, document token‑efficient defaults, and include a concise “Codanna + Sequential‑Thinking” workflow. Historical Serena mentions remain in logs for provenance.
- *2025-11-13:* feat(net): centralize retry/timeout policy, add structured backoff logging, and cover httpx/slow.pics adapters with tests.
- *2025-11-13:* ci/release: enrich package metadata, add wheel content checks, add Windows build smoke, and wire an optional TestPyPI publish workflow.
- *2025-11-13:* feat(api): define curated public exports and add smoke tests; no runtime behavior changes.
- *2025-11-13:* refactor(api): pruned `_COMPAT_EXPORTS` down to the documented runner/wizard/doctor/preflight/tmdb helpers, synced `typings/frame_compare.pyi`, taught the remaining tests/helpers/docs to target `src.frame_compare.*`, and reiterated in the trackers that only the canonical modules (plus the curated `frame_compare` exports) are supported. Verification: `rg` sweeps for the removed shim imports (no matches), `UV_CACHE_DIR=.uv_cache uv run --no-sync ruff check`, `UV_CACHE_DIR=.uv_cache uv run --no-sync npx pyright --warnings`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv_cache uv run --no-sync python -m pytest -q` (290 passed, 1 skipped), and `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`. Packaging: `UV_CACHE_DIR=.uv_cache uv run --no-sync python -m build` still fails to download `wheel` offline, but `UV_CACHE_DIR=.uv_cache uv run --no-sync twine check dist/*` and a manual wheel audit confirmed `src/frame_compare/py.typed` plus `data/config.toml.template` and `data/report/{index.html,app.css,app.js}` are present in the existing artifact.
- *2025-11-12:* refactor(shims): deleted the remaining `src/{analysis,vs_core,slowpics,cli_layout,report,config_template}.py` bridges (and `.pyi` stubs), repointed CLI/core/runtime/tests/docs to import from `src.frame_compare.*`, and pruned `frame_compare._COMPAT_EXPORTS` so only curated exports remain. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest -q`, `UV_CACHE_DIR=.uv_cache uv run --no-sync ruff check --fix && ... ruff check`, `UV_CACHE_DIR=.uv_cache uv run --no-sync npx pyright --warnings`, `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`. Attempted `UV_CACHE_DIR=.uv_cache uv run --no-sync python -m build`, but pip could not download `wheel` in this offline sandbox, so packaging verification (`uv run --no-sync twine check dist/*`) remains pending until network access is available.
- ci: add packaging build + content verification; chore(imports): extend import contracts for new packages.

- docs: expand screenshot/slow.pics reference and add JSON-tail how-to
- refactor(screens): deprecate center_pad and always center padding; warn when explicitly set

- *2025-11-12:* docs(config): added `tools/gen_config_docs.py` with a `--check` mode to emit `docs/_generated/config_tables.md` directly from `src/datatypes.py`, wired in `tests/docs/test_config_docs_gen.py` as a sentinel, linked the new generated tables from `docs/README_REFERENCE.md`, and corrected the `[runtime].ram_limit_mb` default to `8000`.
- *2025-11-12:* refactor(vs): split `vs_core` into `src/frame_compare/vs/{env,source,props,color,tonemap}.py`, rewired intra-module imports with `TYPE_CHECKING` guards, added a `src/vs_core.py` compatibility shim plus `.pyi` stub that re-exports every helper (public and private) until Sub-phase 11.10 removes it, and extended `importlinter.ini`’s Runner→Core→Modules layer with the new subpackage. 📋 Verification (with the `preview` extra installed locally): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`, and `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini`.
- *2025-11-12:* refactor(packaging): moved `src/{cli_layout,report,slowpics,config_template}.py` under `src/frame_compare/`, left shims at the legacy `src.*` paths that re-export every symbol (including the `_read_template_bytes` helper and `resources` module object) with Ruff/Pyright suppressions plus `.pyi` stubs to keep type info stable, added `tests/__init__.py` so the in-repo `tests` package wins over any globally installed `tests` modules, deleted `Legacy/comp.py`, and extended `importlinter.ini`’s Runner→Core→Modules layer with the new modules. Shim removal is scheduled for Sub-phase 11.10 once downstream imports migrate.
- *2025-11-12:* refactor(subproc): centralized subprocess handling for FFmpeg/FFprobe/VSPreview by adding `src/frame_compare/subproc.py::run_checked` (argv-only, `shell=False`, stdio defaults, optional `check`) and routing `src/screenshot.py`, `src/frame_compare/vspreview.py`, and `src/audio_alignment.py` through the helper so timeout/error messages stay unchanged while banning `shell=True`. Screenshot timeout/command-construction tests now patch `src.frame_compare.subproc.run_checked`, and `importlinter.ini` includes the new module in the Runner→Core→Modules layer. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped, 39.98 s); `.venv/bin/ruff check` (clean); `.venv/bin/pyright --warnings` (clean); `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini` (Contracts: 3 kept, 0 broken; first attempt without the cache override hit a permission error reading the shared UV cache).
- *2025-11-12:* refactor(analysis): split metrics/selection/cache IO into `src/frame_compare/analysis/` and keep `src/analysis.py` as a shim for the legacy API (including `_quantile`, `_collect_metrics_vapoursynth`, and the caching helpers the tests monkeypatch). Added public aliases inside the new modules so the rest of `frame_compare` imports non-underscored symbols, stapled `# pyright: standard` headers onto the legacy files to keep them on the repo-wide “standard” mode while the rest of `src/frame_compare` stays strict, and extended `importlinter.ini`’s Runner→Core→Modules layer with `src.frame_compare.analysis`. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (289 passed, 1 skipped), `.venv/bin/ruff check` (clean), `.venv/bin/pyright --warnings` (clean), `UV_CACHE_DIR=.uv_cache uv run --no-sync lint-imports --config importlinter.ini` (Contracts: 3 kept, 0 broken).

- *2025-11-12:* Docs: integrate Serena MCP as primary orchestration tool in AGENTS/CODEX, add Test Guardrails, and list Serena analysis calls as always-allowed; further refinements pending web-sourced best practices.
- *2025-11-12:* Tooling: enable Markdown LSP in Serena config to improve documentation editing and outline support.
- *2025-11-11:* Strict-mode ratchet (Phase 9.11) for the runner boundary: promoted public exports for tonemap validation, media discovery, cache info, and alignment confirmation so `runner.py` no longer reaches into underscore helpers, hardened the JSON/layout handling in `runner.py` (typed `select_frames` inputs, explicit mapping casts, safer folding/verification summaries), deleted unused VSPreview overlay code while guarding `_ensure_audio_alignment_block`, and taught the CLI reporter stubs/patch helpers to track `values` so tests keep patching the confirmation hook. Verification: `.venv/bin/pyright --warnings` (clean), `.venv/bin/ruff check` (clean), `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (`273 passed, 1 skipped in 40.03 s`).
- *2025-11-11:* Doctor workflow module extraction (Phase 9.1). Introduced `src/frame_compare/doctor.py` exposing `DoctorCheck`, `collect_checks`, and `emit_results`, re-pointed `frame_compare.py`’s doctor subcommand and wizard dependency checks to the new module, and left `src/frame_compare/core._collect_doctor_checks` / `_emit_doctor_results` as shims so existing monkeypatches continue to work. Verification: `.venv/bin/pyright --warnings`, `.venv/bin/ruff check`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (273 passed, 1 skipped) all succeeded.
- *2025-11-11:* Runner test split & VSPreview guardrails (Phases 6–7). The CLI/runner regression suites now live under `tests/runner/`, reuse shared fixtures from `tests/helpers/runner_env.py` / `tests/conftest.py`, and include fail-fast VSPreview shims that raise when `frame_compare` stops exporting `_VSPREVIEW_*` constants. This keeps CLI coverage reliable and surfaces shim regressions immediately for contributors who only run the runner-specific tests. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (273 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.
- *2025-11-11:* Shared CLI/test fixtures: extracted `_CliRunnerEnv`, `_RecordingOutputManager`, JSON tail/display stubs, and the `_patch_*` helpers into `tests/helpers/runner_env.py`, added reusable `cli_runner_env`/`recording_output_manager`/`json_tail_stub` fixtures via `tests/conftest.py`, and updated the runner + VSPreview suites to consume the shared module. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (272 passed, 1 skipped); `.venv/bin/ruff check` still fails on the pre-existing import-order/unused-import warnings in `src/frame_compare/alignment_runner.py`, `src/frame_compare/cli_runtime.py`, `src/frame_compare/runner.py`, `src/screenshot.py`, and `tests/test_alignment_runner.py`; `.venv/bin/pyright --warnings` continues to flag the long-standing `_format_vspreview_manual_command` / `_VSPREVIEW_*` attribute/Final errors in `tests/test_frame_compare.py` and `typings/frame_compare.pyi`.
- *2025-11-10:* VSPreview polish: runner now calls `alignment_runner.apply_audio_alignment` directly, `_write_vspreview_script` generates files under `ROOT/vspreview/` with deterministic names and clearer permission failures, `_launch_vspreview` logs when the VSPreview executable or `VAPOURSYNTH_PYTHONPATH` is missing (and can accept a mocked subprocess runner), and `_apply_vspreview_manual_offsets` updates CLI/JSON telemetry plus warns when clip names from VSPreview don’t match the current plan. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (270 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.

- *2025-11-10:* Finalized the audio alignment module split by routing `runner.py` through `alignment_runner.apply_audio_alignment`, dropping the redundant `_maybe_apply_audio_alignment` alias, and updating docs/trackers to match. Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (265 passed, 1 skipped), `.venv/bin/ruff check`, `.venv/bin/pyright --warnings`.
- *2025-11-10:* Phase 2.3 documentation/tooling hand-off: refreshed the runner-refactor trackers, reiterated the wizard compatibility surface (`frame_compare.resolve_wizard_paths` / `_resolve_wizard_paths` now forward into `src.frame_compare.wizard`), and re-ran the quality gates (`pytest -q` 209 passed / 54 skipped, `.venv/bin/ruff check`, `npx pyright --warnings` still blocked offline so `.venv/bin/pyright --warnings` recorded as the fallback).
- *2025-11-10:* Codex/AGENTS/README now explicitly require Sequential Thinking loops to log every stage with `next_thought_needed=true` until the Review entry so orchestration never stops mid-plan.
- *2025-11-19:* Phase 1.1 modularization: `src/frame_compare/preflight.py` now exposes `resolve_workspace_root`, `resolve_subdir`, `collect_path_diagnostics`, `prepare_preflight`, and `PreflightResult`, and the CLI/runner/tests were rewired to consume the shared API (legacy `_…` aliases remain for compatibility).
- *2025-11-09:* Hardened CLI tonemap overrides: all numeric flags now reject non-finite inputs, `--tm-target` enforces positive values, and regression tests cover the new validation paths.
- *2025-11-10:* Corrected future-dated DECISIONS/CHANGELOG rows and documented the requirement to stamp new log entries with `date -u` so timelines stay accurate.
- *2025-11-10:* Refined AGENTS/CODEX operational guardrails: Standard Flow now captures Context7-first citations plus in-response Fetch MCP metadata logging, and CODEX documents MCP auto-approval rules, `.venv`-first verification with `uv sync` fallbacks, an escalation playbook, and requirements for large refactor micro-plans.
- *2025-11-09:* Phase 5 runner QA: verified TMDB orchestration already flows through the shared resolver, ensured reporter injection docs/tests cover automation scenarios, and reran `npx pyright --warnings`, `.venv/bin/ruff check`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q` (258 passed) on a networked host.
- *2025-11-09:* Updated CODEX/AGENTS guardrails to require local `.venv/bin/pyright`/Ruff/Pytest runs, allow cache directories for approved checks, and document the `uv sync --all-extras --dev` setup workflow. Packaging releases now exclude dev-only folders (tests/docs/Legacy/refactor/tools) via `MANIFEST.in` so end users download only runtime files, while `comparison_videos/` remains included for fixture availability.
- *2025-11-09:* Guardrail docs now require assistants to provide a Conventional Commit-style subject in every response so the user can copy it directly into `git commit -m …`.
- *2025-11-08:* Tightened the CLI shim surface by removing the `globals().update`/`__getattr__` re-export from `frame_compare.py`, adding an explicit compatibility map plus `typings/frame_compare/__init__.pyi`, and moving tests to import helpers from `src.frame_compare.core`/`cli_runtime` directly.
- *2025-11-08:* Enhanced the public runner API: `RunRequest` now accepts optional `console`, `reporter`, and `reporter_factory` parameters, quiet runs automatically use a new `NullCliOutputManager`, and README demonstrates how automation can silence output or inject custom reporters. Added regression tests to guard the behavior.
- *2025-11-09:* Restored `frame_compare --diagnose-paths` to run without creating or requiring a config file, updated the preflight tests to stub `src.frame_compare.preflight.load_config`, and logged the refreshed `.venv/bin/ruff check` / `.venv/bin/pytest -q` (250 passed, 1 skipped) results plus the still-blocked `npx pyright --warnings` (npm ENOTFOUND) in the Phase 4.3 quality gates.

- *2025-11-03:* Added a reusable CLI harness for runner validation (`_CliRunnerEnv` plus `_patch_core_helper`, `_patch_runner_module`, `_patch_vs_core`, `_patch_audio_alignment`), migrated the CLI-heavy tests in `tests/test_frame_compare.py` and `tests/test_paths_preflight.py` to the new fixtures, centralized VapourSynth stubs, and re-ran `.venv/bin/pytest tests/test_frame_compare.py tests/test_paths_preflight.py` along with the full `.venv/bin/pytest` suite (247 passed, 1 skipped). `npx pyright --warnings` remains blocked offline (`npm ERR! ENOTFOUND registry.npmjs.org`).
- *2025-11-08:* Documented the public runner API for automation (README “Programmatic Usage”), logged the Phase 2.3 quality gates (`.venv/bin/ruff check`, `.venv/bin/pytest`), captured the offline `npx pyright --warnings` failure for follow-up, and initialized the residual-risk log in `docs/runner_refactor_checklist.md`.
- *2025-11-10:* Split helper logic into `src/frame_compare/core.py`, removed the runner’s `_IMPL_ATTRS` indirection in favor of direct imports, re-exported helpers via `frame_compare.py`, and updated tests/docs to record Phase 4.1 completion.
- *2025-11-01:* Phase 4.2 regression parity: added runner-level slow.pics/audio-alignment tests, refreshed README + docs/DECISIONS.md to document the public API guarantees, updated `docs/runner_refactor_checklist.md`, and re-ran `npx pyright --warnings` (0 issues) to lock the helper extraction work.
- *2025-11-09:* Closed the runner refactor Phase 3 quality gates: reran `npx pyright --warnings`, `.venv/bin/ruff check`, and `.venv/bin/pytest` (246 passed), updated `docs/runner_refactor_checklist.md` with the final statuses, and documented the remaining helper-migration work as the Phase 4 kickoff task.
- *2025-11-08:* Finished the Phase 1 runner extraction: the orchestration logic now lives in `src/frame_compare/runner.py`, `frame_compare.run_cli` is a thin shim that delegates through a `RunRequest`, tests exercise the new module boundary, and documentation captures the runner API plus Phase 2 follow-ups.
- *2025-11-07:* Refreshed tone-mapping presets to align with the new advanced controls: upgraded `reference`, `filmic`, `spline`, and `contrast` profiles, added `bright_lift` and `highlight_guard`, and updated defaults (smoothing 45f, percentile `99.995`, contrast recovery `0.30`) across docs, template, and layout.
- *2025-11-06:* Added BT.2390 knee, dynamic-peak-detection presets, and optional post-tonemap gamma lift. `[color]` now exposes `dst_min_nits=0.18`, `knee_offset`, `dpd_preset`, `dpd_black_cutoff`, and `post_gamma`, libplacebo calls forward `tone_mapping_param`/`peak_detection_preset`/`black_cutoff` with compatibility shims, overlays understand `{knee_offset}`, `{dpd_preset}`, `{dst_min_nits}`, and new CLI flags (`--tm-*`) allow per-run overrides. Docs, presets, and tests updated accordingly.
- *2025-11-06:* Default screenshot exports now expand limited-range SDR to full-range RGB via a new `[screenshots].export_range` setting (default `"full"`), with `"limited"` preserving video-range PNGs. Both VapourSynth and FFmpeg writers honour the option, `_SourceColorRange` provenance is recorded when expanding, and documentation/tests/config templates were updated accordingly.
- *2025-11-05:* Harmonised HDR tonemap range metadata with sampled RGB output, wiring detected `_ColorRange` through geometry and overlays, preserving frame props during subtitle paths, and extending regression coverage/docs for the updated pipeline.
- *2025-10-30:* Auto-launched the configuration wizard during interactive first runs when `config/config.toml` is missing, added a `--no-wizard` flag plus `FRAME_COMPARE_NO_WIZARD` override, surfaced a fallback reminder for non-interactive sessions, and refreshed README/tests accordingly.
- *2025-10-30:* Replaced the audio alignment “progress” bar with a spinner so CLI output stays truthful while offsets are estimated.
- *2025-10-30:* Overhauled the offline HTML report viewer with persistent zoom/fit presets, pointer-anchored wheel zoom, pan/align controls, and shortcut legends to better mirror slow.pics.
- *2025-10-29:* Added optional HTML report generation (configurable via `[report]` or `--html-report`), including vendored assets, CLI auto-open support, JSON-tail disclosures, embedded report data for offline viewing, and unit coverage for the new generator. Added overlay mode toggle with keyboard/click encode cycling.
- *2025-10-29:* Added interactive `frame-compare wizard` with presets, introduced `frame-compare doctor` dependency checklist (JSON-capable), expanded reference docs, and strengthened CLI help/tests for the new commands.
- *2025-10-29:* Clarified the CLI audio alignment panel output (stream summaries, cached reuse messaging, offsets file footer) and aligned documentation; slow.pics shortcut filenames now derive from the sanitised collection name with regression tests for edge cases; README and reference tables updated.
- *2025-10-22:* Disabled slow.pics auto-upload by default, added an upfront CLI warning when it is enabled, aligned documentation with dataclass defaults, introduced a packaged `frame-compare` console entry point, and wired Ruff linting into CI (Pyright now blocks failures).
- *2025-10-21:* Prevented VSPreview helper crashes on Windows `cp1252` consoles by sanitising printed arrows to ASCII, preferring UTF-8 output streams, adding regression coverage, and documenting the console behaviour.
- *2025-11-15:* Preserved HDR frame props by snapshotting them at source load (`ClipPlan.source_frame_props`) and rehydrating them inside `process_clip_for_screenshot`/`generate_screenshots`, so even negatively trimmed clips retain `_Matrix`/`_Transfer`/MasteringDisplay hints and diagnostic overlays continue to show HDR metadata. Added regression tests covering tonemap rehydration and screenshot overlays.
- *2025-10-20:* Hardened audio alignment's optional dependency handling by surfacing clear `AudioAlignmentError` messages when
  `numpy`, `librosa`, or `soundfile` fail during onset envelope calculation, and refreshed regression coverage for the failure
  path.
- *2025-10-20:* VSPreview-assisted manual alignment now displays existing manual trims using friendly clip labels so operators
  can immediately see which plan each baseline affects before accepting new deltas.
- *2025-10-20:* Prevented VSPreview script overwrites by appending per-run entropy to generated filenames and warning when a
  collision is detected.
- *2025-10-20:* Fixed mod-2 odd-geometry failures by pivoting subsampled SDR clips through YUV444P16 when needed, emitting Rich console notes that summarise the axis/policy, and expanding docs/config guidance for the `odd_geometry_policy` and `rgb_dither` options.
- *2025-10-19:* Normalised VapourSynth colour metadata inference for SDR clips, cached inferred props on the
  clip to avoid redundant frame grabs, exposed config overrides for HD/SD defaults and per-file colour
  corrections, refreshed documentation, and expanded regression coverage for the new heuristics.
- *2025-10-18:* Fixed VapourSynth RGB conversion when colour metadata is absent by defaulting to Rec.709
  limited parameters, preventing fpng "no path between colours" failures and adding regression coverage.
- *2025-10-17:* Documented the VSPreview-assisted manual alignment flow (README, reference tables, pipeline guide), surfaced the
  CLI help text, added a fallback regression test, and published a cross-platform QA checklist for manual verification.
- *2025-10-16:* Hardened analysis cache and audio offsets paths to stay within the workspace root, added regression tests for escape attempts, removed the generated `config.toml` from source control in favour of the packaged template, and restricted screenshot cleanup to directories created during the current run.
- *2025-10-16:* Limited supported Python versions to 3.13.x (`>=3.13,<3.14`) to align with current `librosa`/`numba` wheels; updated project metadata and lockfile.
- *2025-10-14:* Locked workspace roots to `--root`/`FRAME_COMPARE_ROOT`/sentinel discovery, seeded config under `ROOT/config/config.toml`, enforced `ROOT/comparison_videos[/screens]`, added `--diagnose-paths`, and blocked site-packages writes before screenshotting.
- *2025-10-12:* Default config now seeds to `~/.frame-compare/config.toml`, `[paths].input_dir` defaults to `~/comparison_videos`, and the packaged template ships from `src/data/` (with wheel coverage) to avoid site-packages permission issues. *(Superseded by 2025-10-14 workspace root lock.)*
- *2025-10-11:* Added an optional `$FRAME_COMPARE_TEMPLATE_PATH` override and filesystem fallback for the packaged config template plus clearer screenshot permission errors so runs targeting read-only `comparison_videos` trees fail fast with guidance.
- *2025-10-19:* Seed the packaged default config into a per-user directory when the project tree is read-only so packaged installs no longer fail to start on permission errors.
- *2025-10-20:* Allow disabling the per-frame FFmpeg timeout by setting `screenshots.ffmpeg_timeout_seconds` to 0 while keeping negative values invalid in validation and docs.
- *2025-10-18:* Added a per-frame FFmpeg timeout and disabled stdin consumption to prevent hung screenshot renders and shell freezes on Windows.
- *2025-10-17:* Streamlined CLI framing: removed the banner row, surfaced cached-metrics reuse inside the Analyze block, dropped the At-a-Glance crop-mod readout in favour of effective tonemap nits, and trimmed the Summary output frames line to match the refreshed console layout and tests.
- *2025-10-16:* Documented deep-review finding: screenshot cleanup must enforce path containment before deleting outputs; remediation planned.

- *2025-10-15:* Relocated bundled comparison fixtures to the repository-root `comparison_videos/` directory, updated CLI docs to match the default `paths.input_dir`, and noted the new resolution fallbacks.
- *2025-10-14:* Clarified CLI hierarchy by softening section badge colors, brightening subhead prefixes, expanding the At-a-Glance box with alignment, sampling, and canvas metrics for quicker triage, and realigning the Summary section with key/value formatting to remove ragged rows.
- *2025-10-13:* Removed the `types-pytest` dev dependency so `uv run` can resolve environments on fresh clones without relying on a non-existent stub package while keeping the actual `pytest` runtime dependency in the dev group for CI and local tests.
- *2025-10-10:* Reject invalid `cli.progress.style` values during configuration loading and persist normalized styles for downstream flag handling to keep CLI reporters consistent.
- *2025-10-12:* Added local type stubs for optional CLI/testing dependencies and hardened JSON-tail assertions in tests so Pyright runs cleanly without relaxing diagnostic settings.
- *2025-10-09:* Expanded minimal overlays with resolution/upscale and frame-selection type lines while simplifying diagnostic overlays by removing on-screen selection timecode/score/notes that remain available via cached metadata.
- *2025-10-02:* Scoped audio alignment's NumPy flush-to-zero warning suppression to local contexts, added regression tests to keep diagnostics available, and hardened TMDB caching with TTL-aware eviction to cap memory growth during long CLI sessions.
- *2025-10-01:* Added selection metadata persistence v1: cached JSON sidecar, compframes annotations, CLI summary metrics, and screenshot overlay reuse without rerunning analysis.
- *2025-10-01:* Added `docs/audio_alignment_pipeline.md` to detail the alignment workflow, configuration constraints, and offsets file contract.
- *2025-09-30:* Enhanced CLI group blocks with accent subhead glyphs, dimmed divider rules, numeric alignment, and verbose legends describing token colouring for RENDER/PREPARE sections.
- *2025-09-30:* Diagnostic overlay now replaces the HDR MAX/AVG measurement block with render resolution details, mastering display luminance (if present), and cached frame-selection metadata while trimming redundant frame-info lines for leaner CLI banners.
- *2025-09-29:* Initialize changelog to align with repository persistence rules.
- *2025-09-29:* Revamped CLI presentation: extended palette, added style spans, highlights, section accents, progress bar styling, verification metrics, and refreshed templates per `features/CLI/GUIDE.md`.
- *2025-11-10:* Ensured runner funnels audio alignment through `core._maybe_apply_audio_alignment` again (restoring the test/mocking seam) and deep-copied the JSON-tail audio block before building CLI layout views so manual trims, VSPreview offsets, and reference trims persist. Added regression coverage, reran the full suite, and cleaned up Ruff/Pyright diagnostics that surfaced after the refactor.
