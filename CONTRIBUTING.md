# Contributing to ToneSphere

Thank you for considering it. Two things to know before you open a pull request.

## Your contribution is licensed under the GPL

ToneSphere is free software under the GNU General Public License, version 3 or later
(`LICENSE`). By submitting a contribution you license it under that same licence, to the project
and to everyone who receives it, and you confirm that you have the right to do so: the
[Developer Certificate of Origin](https://developercertificate.org/), which you state by signing
off each commit (`git commit -s`). No copyright assignment, no separate agreement.

The virtual audio driver in `drivers/win/vad/` is derived from Microsoft's samples under the
MS-PL and keeps that licence file in its directory. The name "ToneSphere" and the publisher name
"Neural Nexus Studios" are not licensed by any of this.

## The engineering rules

[AGENTS.md](AGENTS.md) is the contract for this repository, for people and coding agents alike.
The short version: a feature does not exist until a test proves it moves audio, an unmeasured
value is `--` and never `0`, and nothing locks, allocates or does I/O on an audio thread.
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) sets up a machine, and [docs/STATUS.md](docs/STATUS.md)
is the record of what is actually verified.

Before a pull request: `.\scripts\build.ps1 -Test`, `npm run typecheck` and `npx playwright test`
in `ui/`, and for an audio-path change `.\tests\run-all.ps1` on a machine with VB-CABLE (say which
suites you ran). CI runs the rest.
