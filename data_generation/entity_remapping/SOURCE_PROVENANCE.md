# Current release note

[FIXES.md](../../docs/FIXES.md) documents the current I/O, validation, resume, and fail-closed changes. The following earlier provenance record describes the preceding release and must not override the current commands.

---

# Original source and code identity

- Original archive: `16-12-2025_final paper(1).zip` (provided by the author)
- Original notebook: `16-12-2025_final paper/dataset_creation.ipynb`
- Original code location: **cell 0** (the first cell), 149 lines
- Derived source file: `remap_entities.py`
- Derived notebook: `remap_entities.ipynb` (single code cell, execution outputs cleared)
- SHA-256 of the extracted source text: `088791cb231c0cd938817c1f17629fda99dba4d64a71672e6874c242fb7ec3d0`

## Modifications

**No source-code lines were changed.** The first cell was extracted verbatim to a `.py` file, and an independent notebook was created using the same source code. All other original notebook cells and all saved execution outputs were omitted because they belong to other processing stages and/or may contain sensitive information. No local datasets, API credentials, logs, or generated outputs were packaged.

An independent README, requirements list, and `.gitignore` were added outside the original Python code.

## Verification performed

1. Python AST syntax parsing: passed.
2. `remap_entities.py` UTF-8 source hash matches the exact original first-cell source: passed.
3. `remap_entities.ipynb` source also matches the original first-cell source: passed.
4. Static check of the extracted code for API credentials and network/API client usage: no matches.
5. Full runtime check: **not run** (source workbook and spaCy/WordNet language data not available in packaging environment).
