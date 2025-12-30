#!/usr/bin/env node
/**
 * Cross-platform pytest runner that enforces PYTEST_DISABLE_PLUGIN_AUTOLOAD
 * unless FC_SKIP_PYTEST_DISABLE=1 is set in the environment.
 *
 * WHY THIS EXISTS:
 * ================
 * VapourSynth and VSPreview are Windows/Linux-native C extensions that cannot
 * be installed on macOS. However, VSPreview ships a pytest plugin (vsengine)
 * that auto-loads when present in the Python environment.
 *
 * On macOS development environments:
 *   - The vsengine plugin attempts to import vapoursynth at pytest startup
 *   - This fails because vapoursynth isn't available on macOS
 *   - All tests abort before any test code runs
 *
 * On Windows/Linux (where frame-compare actually runs):
 *   - VapourSynth is available, so the plugin loads successfully
 *   - Tests run normally with full VS integration
 *
 * SOLUTION:
 * =========
 * Setting PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 prevents pytest from loading any
 * plugins not explicitly listed in conftest.py or pyproject.toml. This allows
 * tests to run on macOS by blocking the vsengine plugin's auto-discovery.
 *
 * The pyproject.toml also includes `addopts = "-p no:vsengine"` as a belt-and-
 * suspenders approach to ensure the vsengine plugin is never loaded.
 *
 * CROSS-PLATFORM BEHAVIOR:
 * ========================
 *   macOS (dev):     PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 → tests run with VS stubs
 *   Windows (prod):  FC_SKIP_PYTEST_DISABLE=1 → tests run with real VapourSynth
 *   Linux (CI/CD):   Can use either mode depending on whether VS is installed
 *
 * Set FC_SKIP_PYTEST_DISABLE=1 on machines with full VapourSynth installations
 * where you want plugin autoloading to work normally.
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import process from 'node:process';

const env = { ...process.env };
const skipDisable = env.FC_SKIP_PYTEST_DISABLE === '1';
if (!skipDisable) {
  env.PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1';
}

const platformScriptsDir = process.platform === 'win32' ? 'Scripts' : 'bin';
const pytestExecutable = process.platform === 'win32' ? 'pytest.exe' : 'pytest';

const candidateCommands = [];

const appendCandidate = (base) => {
  if (!base) {
    return;
  }
  candidateCommands.push(join(base, platformScriptsDir, pytestExecutable));
};

appendCandidate(env.VIRTUAL_ENV);
appendCandidate(join(process.cwd(), '.venv'));

const resolvedCommand =
  candidateCommands.find((cmd) => existsSync(cmd)) ??
  null;

const command = resolvedCommand ?? (process.platform === 'win32' ? 'python' : 'python3');
const args = resolvedCommand ? ['-q'] : ['-m', 'pytest', '-q'];

const child = spawn(command, args, {
  stdio: 'inherit',
  env,
  shell: false,
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

child.on('error', (error) => {
  console.error('[run_pytest] Failed to execute pytest:', error);
  process.exit(1);
});
