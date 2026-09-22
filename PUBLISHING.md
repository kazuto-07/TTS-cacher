# Publishing to PyPI

How to get `pip install tts-cache` working. Nothing here has been run yet — the package
builds and passes `twine check`, but it has never been uploaded.

Two things are permanent once you upload: **the project name** and **every version
number**. A version can never be reused, even after deleting the release. So rehearse on
TestPyPI first.

## Before the first upload

- [ ] `pyproject.toml` points Homepage and Issues at `github.com/kazuto-07/tts-cache`.
      Confirm that is the right account — the links are baked into the PyPI page.
- [ ] Commit the source and push it. Publishing a version whose code is not in a repo
      means you cannot reconstruct later what `0.1.0` was.
- [ ] Name check: `tts-cache` was free on PyPI as of 2026-09-20 (`/pypi/tts-cache/json`
      returned 404). Re-check before uploading — someone else can take it at any time.

## 1. Accounts

Register on [pypi.org](https://pypi.org/account/register/) and, separately, on
[test.pypi.org](https://test.pypi.org/account/register/) — they do not share accounts.
2FA is mandatory on both.

## 2. API token

PyPI → Account settings → API tokens → *Add API token*. Scope it to "Entire account" for
the first upload (a project-scoped token cannot exist before the project does). The token
starts with `pypi-`.

Put it in `~/.pypirc`:

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
  username = __token__
  password = pypi-AgENdGVzdC5weXBp...
```

Or pass it per-command as `TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...`.

`~/.pypirc` is a plaintext credential — it belongs in your home directory, never in this
repo.

## 3. Build

```bash
pip install build twine
rm -rf dist/            # always build fresh; stale artifacts get uploaded by accident
python -m build         # -> dist/tts_cache-0.1.0.tar.gz and ...-py3-none-any.whl
python -m twine check dist/*
```

`twine check` validates the metadata and that the README renders on PyPI. Both artifacts
should say PASSED.

Worth eyeballing once: `unzip -l dist/*.whl` should show `tts_cache/` plus a `dist-info`
with the LICENSE and the `tts-cache` entry point — no tests, no `.tts-cache/`.

## 4. Rehearse on TestPyPI

```bash
python -m twine upload --repository testpypi dist/*

# in a throwaway venv:
pip install --index-url https://test.pypi.org/simple/ tts-cache
python -c "from tts_cache import TTSCache, LocalStorage, SqliteIndex; print(TTSCache)"
tts-cache stats --help
```

TestPyPI has no copy of the optional dependencies, so installing an extra like
`tts-cache[supabase]` from there will fail to resolve. That is expected — it says nothing
about the real index.

Check the rendered page at `https://test.pypi.org/project/tts-cache/`: README formatting,
the classifiers, and the project links.

## 5. Publish

```bash
rm -rf dist/ && python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

`pip install tts-cache` works within about a minute.

## 6. After the first release

- Replace the account-wide token with a **project-scoped** one, or drop tokens entirely
  and set up [trusted publishing](https://docs.pypi.org/trusted-publishers/) — GitHub
  Actions uploads on a version tag with no secret stored anywhere.
- Tag the commit you released: `git tag v0.1.0 && git push --tags`.

## Releasing a later version

1. Bump `__version__` in `src/tts_cache/__init__.py`. That is the only place it lives —
   `pyproject.toml` reads it from there, so the wheel and the import cannot disagree.
2. `pytest` and `ruff check .` clean. CI runs both on every push.
3. Rebuild from a clean `dist/`, `twine check`, upload.
4. Tag it.

A mistake in a published version is fixed by publishing the next version. `pip install
tts-cache==0.1.0` can be *yanked* (PyPI → Manage → Yank) so resolvers skip it, but the
files stay downloadable and the number stays burned.
