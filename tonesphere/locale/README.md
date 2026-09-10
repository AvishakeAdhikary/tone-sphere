# Translation catalogs

Two languages ship: `en.json` (English, the base — every other catalog is checked against
it for key set and placeholders by `tests/test_i18n.py`) and `hi.json` (Hindi).

Hindi is not an arbitrary second choice. The publisher, Neural Nexus Studios, is based in
Kolkata, West Bengal, India, and can actually read and vouch for it — which is the whole
reason the set is two languages instead of a dozen nobody here has read. `hi.json` is a
machine translation the same way any addition here would start as one; it is marked
`"review_status": "unreviewed"` in its own metadata, and nothing in the interface should
imply otherwise.

## Format

```json
{
  "meta": {
    "locale": "hi",
    "name": "Hindi",
    "native_name": "हिन्दी",
    "rtl": false,
    "review_status": "unreviewed",
    "note": "..."
  },
  "strings": {
    "transport.start_engine": "इंजन शुरू करें",
    "transport.buffer_frames": "{frames} फ़्रेम"
  }
}
```

- `meta.locale` must match the filename stem (`hi.json` → `"hi"`).
- `strings` keys are dotted, grouped by the panel they appear in (`transport.*`,
  `mixer.*`, `patch.*`, `menu.*`, `about.*`, …) — see `tonesphere/i18n.py`'s callers for
  the full set.
- `{placeholder}` tokens must match the English string exactly, same names, same count.
  A renamed placeholder is a runtime error the moment that string is shown, not a build
  error, so `tests/test_i18n.py` checks every catalog for this on every run.
- `rtl: true` flips the whole interface's layout direction (`tonesphere/ui/app.py`). No
  catalog here needs it yet, but the code reads this flag rather than hardcoding a list of
  right-to-left locale codes, so adding one later is a catalog change, not a code change.

## Contributing a correction or a new language

Open an issue or a pull request against the relevant `.json` file. A native speaker's
correction to `hi.json` is exactly the kind of contribution this project wants — flag which
keys you're fixing and why the existing translation reads wrong to a working audio
engineer, not just a dictionary. A new language needs every key `en.json` has, no more and
no fewer; `tests/test_i18n.py` will fail loudly on either.
