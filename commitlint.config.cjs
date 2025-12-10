// commitlint.config.cjs
/** @type {import('@commitlint/types').UserConfig} */
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2, 'always',
      ['feat', 'fix', 'docs', 'chore', 'refactor', 'perf', 'test', 'ci', 'build', 'revert', 'style']
    ],
    // Expanded scopes - level 1 = warn (not error) for flexibility
    'scope-enum': [
      1, 'always',
      [
        // Core domains
        'hdr', 'sdr', 'vs', 'cli', 'report', 'html', 'analysis', 'audio',
        'tonemap', 'overlay', 'tmdb', 'geometry', 'color', 'diagnostics',
        // Infrastructure
        'ci', 'docs', 'deps', 'build', 'config', 'lint', 'types',
        // Features/components
        'layout', 'render', 'cache', 'slowpics', 'webhook', 'preview',
        'screenshots', 'planner', 'runner', 'wizard', 'preflight',
        // Cross-cutting
        'security', 'tests', 'perf', 'logging', 'errors', 'utils', 'net',
        // Broad catch-alls
        'core', 'api', 'ui', 'db', 'misc'
      ]
    ],
    // Allow empty scope for simple commits
    'scope-empty': [1, 'never'],
    // Relaxed length limits (practical maximums for GitHub UI)
    'header-max-length': [2, 'always', 150],
    'subject-max-length': [1, 'always', 120],  // Warn at 120, not error
    'body-max-line-length': [0, 'always', 300],  // Disabled - let markdown/URLs be long
    'subject-case': [2, 'never', ['sentence-case', 'start-case', 'pascal-case', 'upper-case']]
  },
  defaultIgnores: true,
  ignores: [
    (msg) => /^Bump\s.+\sto\s.+$/.test(msg),
    (msg) => /^chore\(deps\):/i.test(msg)
  ],
  helpUrl: 'https://commitlint.js.org/'
};

