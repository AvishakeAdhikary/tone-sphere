# Contributing to ToneSphere

Thank you for considering it. Two things to know before you open a pull request.

## Your contribution is licensed under MIT

ToneSphere's own source is under the MIT Licence (`LICENSE`). By submitting a contribution
you license it under that same licence, to the project and to everyone who receives it, and
you confirm that you have the right to do so — the
[Developer Certificate of Origin](https://developercertificate.org/), which you state by
signing off each commit (`git commit -s`).

That matters for a concrete reason: the publisher may one day offer paid editions of
ToneSphere (Terms and Conditions, section 3), and MIT-licensed contributions can go into any
edition, paid or free, exactly as the publisher's own code can. Nothing else is asked of you:
no copyright assignment, no separate agreement.

The exception is `native/asio/`, which is GPLv3 because it is built on Steinberg's ASIO SDK
under that licence; a contribution to that directory is licensed under GPLv3. And
`driver/windows_virtual_audio/` is derived from Microsoft's samples under MS-PL. Each keeps
its licence file in its directory.

The name "ToneSphere" and the publisher name "Neural Nexus Studios" are not licensed by any
of this.

## The engineering rules

[AGENTS.md](AGENTS.md) is the contract for this repository, for people and coding agents
alike. The short version: a feature does not exist until a test proves it moves audio, an
unmeasured value is `--` and never `0`, and nothing runs Python, allocates, locks or logs on
the audio thread. `docs/TESTING.md` explains how tests are written, and
`docs/IMPLEMENTATION_STATUS.md` is the record of what is actually done.

Before a pull request: `uv run ruff check .` and `uv run pytest -m "not hardware"`; for an
audio-path change, `uv run pytest -m hardware` on a machine with a device, and say which.
