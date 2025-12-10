---
description: Exhaustive line-by-line code review for runtime defects
---

# 🔬 Exhaustive Code Review

**TARGET**: {{args}} (defaults to entire `src/` directory if not specified)

You are a **Senior Staff Engineer** performing a pre-merge code review. Your job is to find defects that **pyright passes but cause runtime failures, silent data corruption, or undefined behavior**.

This is NOT an architecture review. Focus on the **CODE ITSELF**.

---

## Context (frame-compare)

- **Stack:** Python 3.13, Click CLI, VapourSynth, NumPy, librosa, httpx/requests
- **Pattern:** CLI tool for video frame comparison, audio alignment, screenshot generation
- **Critical Files:**
  - `src/frame_compare/cli_entry.py` – Main CLI entrypoints
  - `src/frame_compare/core.py` – Core comparison logic
  - `src/frame_compare/slowpics.py` – External API integration (slowpics.org)
  - `src/frame_compare/cache.py` – Disk caching layer
  - `src/frame_compare/layout/engine.py` – Expression evaluation (eval-based)
  - `src/audio_alignment.py` – librosa-based audio sync

---

## 🔬 ANALYSIS CATEGORIES (Check EVERY Item)

### CATEGORY 1: Python Type & None Landmines

| Check | What to Look For |
|-------|------------------|
| **Optional Access** | `obj.attr.nested` where `attr` could be `None` but no guard |
| **Falsy vs None Confusion** | `if not value` when `value` could legitimately be `0`, `""`, or `[]` |
| **Type Narrowing Gaps** | `isinstance()` check in one branch, but accessing typed attrs in both |
| **Dict Key Errors** | `d["key"]` without `.get()` or `in` check; `KeyError` at runtime |
| **Attribute Errors** | Accessing attrs that only exist conditionally (e.g., after `hasattr`) |
| **Mutable Default Args** | `def foo(items=[]):` – shared list across calls |
| **Tuple Unpacking** | `a, b = func()` when func might return wrong length |
| **String/Bytes Confusion** | Mixing `str` and `bytes` without encode/decode |

### CATEGORY 2: Iterator & Collection Traps

| Check | What to Look For |
|-------|------------------|
| **Iterator Exhaustion** | Iterating over generator twice; generator consumed before use |
| **Modifying During Iteration** | `for x in lst: lst.remove(x)` – undefined behavior |
| **Empty Sequence Edge Cases** | `lst[0]` without length check; `max([])` raises ValueError |
| **Dict Ordering Assumptions** | Code depending on dict order in Python < 3.7 patterns |
| **Shallow vs Deep Copy** | `lst.copy()` when nested objects need `copy.deepcopy()` |
| **zip() Length Mismatch** | `zip()` silently truncates; should use `zip(..., strict=True)` |
| **slice() Bounds** | `lst[start:end]` where start/end could exceed bounds (safe but wrong result) |

### CATEGORY 3: File I/O & Resource Leaks

| Check | What to Look For |
|-------|------------------|
| **Unclosed Handles** | `open()` without `with` statement or explicit `.close()` |
| **Path vs String** | Mixing `Path` objects and strings; `os.path.join(Path(...), ...)` |
| **Encoding Omission** | `open(f)` without `encoding=` – platform-dependent behavior |
| **Binary Mode Mismatch** | Reading binary file in text mode or vice versa |
| **Temp File Cleanup** | `tempfile.NamedTemporaryFile` with `delete=False` but no cleanup |
| **Race Conditions** | Check-then-use on filesystem (`if exists: open`) |
| **Relative Path Assumptions** | Paths relative to CWD instead of `__file__` |

### CATEGORY 4: HTTP & Network Defects

| Check | What to Look For |
|-------|------------------|
| **Missing Timeout** | `requests.get()` / `httpx.get()` without `timeout=` – hangs forever |
| **No Status Check** | Response used without `.raise_for_status()` or status code check |
| **Session Reuse** | Creating new `Session()` per request instead of reusing |
| **URL Injection** | String concatenation for URLs; should use `urllib.parse` |
| **SSL Disabled** | `verify=False` without security justification |
| **Retry Logic** | Transient failures not retried; no backoff |
| **JSON Decode Errors** | `.json()` called without try/except for malformed response |

### CATEGORY 5: Subprocess & External Process

| Check | What to Look For |
|-------|------------------|
| **Shell Injection** | `shell=True` with user input; should use list args |
| **Unchecked Return Code** | `subprocess.run()` without `check=True` or manual returncode check |
| **Deadlock Risk** | Large stdout/stderr without `communicate()` or streaming |
| **Zombie Processes** | `Popen()` without `.wait()` or context manager |
| **Platform Assumptions** | Hardcoded paths (`/usr/bin/`); missing Windows compat |
| **Encoding Issues** | Missing `encoding=` or `errors=` for text streams |

### CATEGORY 6: Audio/Video Processing (Domain-Specific)

| Check | What to Look For |
|-------|------------------|
| **VapourSynth Access** | Accessing `clip[frame]` without bounds check |
| **Array Shape Mismatches** | NumPy operations on incompatible shapes; missing reshape |
| **Sample Rate Assumptions** | Hardcoded sample rates; missing resample |
| **Channel Count** | Assuming stereo; mono/multichannel edge cases |
| **Frame Range Validation** | Negative frames; frames beyond clip length |
| **Colorspace Conversion** | Operations assuming RGB when input is YUV |
| **Division by Zero** | Normalizing by values that could be 0 (audio silence, black frames) |

### CATEGORY 7: Concurrency & Threading

| Check | What to Look For |
|-------|------------------|
| **Thread Safety** | Shared mutable state without locks |
| **asyncio Mixing** | Running sync code in async context; blocking the event loop |
| **Executor Cleanup** | ThreadPoolExecutor/ProcessPoolExecutor not shut down |
| **Race Conditions** | Check-then-act without atomicity |
| **Deadlocks** | Multiple locks acquired in inconsistent order |
| **GIL Assumptions** | Assuming GIL protects all operations (it doesn't for += on containers) |

### CATEGORY 8: Error Handling Defects

| Check | What to Look For |
|-------|------------------|
| **Bare Except** | `except:` or `except Exception:` catching too broadly |
| **Exception Swallowing** | `except: pass` – failure silently ignored |
| **Lost Context** | `raise NewError()` instead of `raise NewError() from e` |
| **Partial Mutations** | State changed before operation that can fail; no rollback |
| **finally Misuse** | return/break in finally block suppresses exceptions |
| **Error Type Assumption** | `except Exception as e: e.args[0]` without validation |

### CATEGORY 9: Security at Code Level

| Check | What to Look For |
|-------|------------------|
| **eval/exec** | Any use of `eval()`, `exec()`, or `compile()` with external input |
| **pickle Untrusted** | `pickle.load()` on untrusted data – arbitrary code execution |
| **Path Traversal** | User input in file paths without sanitization (`../../../etc/passwd`) |
| **URL SSRF** | User-controlled URLs passed to `requests.get()` |
| **Command Injection** | User input in subprocess commands |
| **XML/YAML Bombs** | Parsing untrusted XML/YAML without safe loaders |
| **Regex DoS** | Catastrophic backtracking on user input |
| **Temp File Predictability** | Predictable temp file names in shared directories |

---

## 🎯 SPECIFIC PATTERNS TO HUNT (frame-compare)

Given this architecture, **actively search for these patterns**:

```python
# BUG: Optional access without guard
def process_clip(clip: vs.VideoNode | None):
    return clip.num_frames  # 💥 AttributeError if None

# BUG: Mutable default argument
def collect_frames(frames: list[int] = []):  # Shared across calls!
    frames.append(1)
    return frames

# BUG: Iterator exhaustion
frames = (process(f) for f in range(100))
first_pass = list(frames)
second_pass = list(frames)  # 💥 Empty! Generator already consumed

# BUG: Missing timeout on HTTP request
response = requests.post(SLOWPICS_URL, data=payload)  # Hangs forever if server unresponsive

# BUG: Dict access without safety
config = load_config()
value = config["optional_key"]  # 💥 KeyError if not present

# BUG: Empty sequence edge case
scores = []
best = max(scores)  # 💥 ValueError: max() arg is an empty sequence

# BUG: File handle leak
f = open("data.bin", "rb")
data = f.read()
# Missing f.close() – leaked if exception raised

# BUG: Path concatenation
output = base_dir + "/" + filename  # Use Path(base_dir) / filename

# BUG: Unchecked subprocess
result = subprocess.run(["ffprobe", path])
# Missing check=True or returncode check

# BUG: VapourSynth frame access
frame = clip[frame_num]  # What if frame_num >= clip.num_frames?

# BUG: eval() on expressions (layout engine)
result = eval(user_expression)  # Arbitrary code execution!
# Should use: ast.literal_eval() or custom safe evaluator

# BUG: Zip silently truncates
for a, b in zip(list1, list2):  # If lengths differ, shorter wins
    process(a, b)
# Use: zip(list1, list2, strict=True)
```

---

## 📊 OUTPUT FORMAT

For each defect found, provide:

| Severity | File:Line | Category | Issue | Bad Code | Fixed Code |
|----------|-----------|----------|-------|----------|------------|
| 🔴 CRITICAL | `slowpics.py:142` | HTTP | Missing timeout on POST request – hangs indefinitely | `requests.post(url, data=d)` | `requests.post(url, data=d, timeout=30)` |
| 🔴 CRITICAL | `layout/engine.py:67` | Security | eval() on user expression allows arbitrary code execution | `eval(expression)` | Use `ast.literal_eval()` or bounded evaluator |
| 🟠 HIGH | `core.py:234` | Type Safety | Optional clip accessed without None check | `clip.num_frames` | `if clip is None: raise ValueError(); clip.num_frames` |
| 🟡 MEDIUM | `cache.py:89` | Resource | File opened without context manager | `f = open(p, 'rb'); data = f.read()` | `with open(p, 'rb') as f: data = f.read()` |
| 🟡 MEDIUM | `utils.py:45` | Iterator | Generator used twice – second iteration is empty | `gen = (x for x in items)` | `items_list = list(gen)` or generate twice |

---

## 🚫 DO NOT REPORT

- Naming convention preferences
- "You could use X instead of Y" when Y is correct
- Style issues (ruff/black handles these)
- Architecture opinions
- Missing docstrings

---

## ✅ REPORT ONLY

- Code that will crash at runtime
- Code that silently produces wrong results
- Code that leaks resources (files, sockets, processes)
- Code that creates race conditions
- Code that allows security exploits
- Code that breaks on edge cases (empty input, None, boundaries)

---

## Execution Instructions

// turbo-all

1. For each file in target, perform line-by-line analysis
2. Check every category systematically
3. Cross-reference with domain-specific patterns
4. Output findings in the table format
5. Run `.venv/bin/pyright --warnings` to verify type safety
6. Run `.venv/bin/pytest -q` to verify no regressions
7. Report total defects by severity
