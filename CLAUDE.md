# CLAUDE.md

All project rules live in AGENTS.md (shared with other coding agents):

@AGENTS.md

Claude-specific notes:
- The owner works unattended. Make sensible choices, work autonomously, and report what was
  verified. Never reboot the host, and run anything risky in the Hyper-V VM.
- Long-running jobs (VM tests, soaks, WACK, packaging) should run in the background, and the
  host should stay idle while audio suites run.
- Keep `docs/STATUS.md` current at the end of each piece of work.
