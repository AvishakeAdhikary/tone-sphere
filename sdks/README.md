# sdks/

Third-party SDKs the native build compiles against. **Nothing here except this file is
committed** — `scripts/fetch_sdks.py` downloads each SDK at a pinned version:

```
uv run python scripts/fetch_sdks.py
```

| Directory | SDK | Pinned by | Licence | Used by |
|---|---|---|---|---|
| `vst3sdk/` | Steinberg VST3 SDK 3.8.1 | git tag `v3.8.1_build_84` of `github.com/steinbergmedia/vst3sdk` | MIT (from 3.8.0 on) | `native/vst3/` (the host), `native/test_plugin/` |
| `opus/` | libopus 1.5.2 (Xiph) | SHA-256 of `opus-1.5.2.tar.gz` from `downloads.xiph.org` (`65c1d2f7…9a7ce1`) | BSD 3-clause | `opus.dll` (Windows), loaded by `tonesphere/network/opus.py`; its `COPYING` ships in `licenses/opus` |
| `asiosdk/` | Steinberg ASIO SDK 2.3.4 | SHA-256 of `ASIO-SDK_2.3.4_2025-10-15.zip` from `download.steinberg.net` (the target of `steinberg.net/asiosdk`) | Dual: proprietary Steinberg ASIO licence **or** GPL version 3 | `native/asio/` only |

## Why they are fetched, not committed

**VST3.** MIT since 3.8.0, so committing it would be allowed; it is fetched anyway because
it is large, has its own submodules, and pinning a tag keeps upgrades a one-line change.
Distributions of ToneSphere keep the SDK's MIT notice.

**ASIO.** ToneSphere uses the ASIO SDK under its **GPLv3 option**. The SDK's own
`LICENSE.txt` says "GPL Version 3" without "or later", so it is GPLv3 only. Consequences,
which `AGENTS.md` makes binding:

- The code that includes ASIO SDK headers lives in `native/asio/`, is GPLv3, and builds to
  its own DLL (`tonesphere_asio.dll`). The rest of ToneSphere's source stays MIT.
- Any binary distribution that includes `tonesphere_asio.dll` is distributed under GPLv3
  as a whole, with source available.
- The proprietary option would require a licence agreement signed by Steinberg before
  publishing; ToneSphere does not use it.
- "ASIO" is a Steinberg trademark. It may not appear in ToneSphere's product name. If the
  ASIO-compatible logo is ever used, it must follow *Steinberg ASIO Usage Guidelines.pdf*
  in the SDK.

`fetch_sdks.py` refuses a zip whose checksum differs from the pin, and refuses to extract
one whose licence no longer offers GPLv3.

**libopus.** BSD 3-clause. Built by `native/CMakeLists.txt` into its own `opus.dll` with
Xiph's defaults (no deep-learning PLC or DRED models); on Linux and macOS the system's
libopus is used instead. Distributions that include `opus.dll` carry its notice.

## Supplying an SDK by hand

With no network, place the SDKs at the paths above yourself: a clone of the VST3 SDK at
the pinned tag (with `--recurse-submodules`), and the contents of the ASIO zip's
`ASIOSDK/` folder in `asiosdk/`. The build checks for the headers it needs, not for the
fetch script's stamp files.
