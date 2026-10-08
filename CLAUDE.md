# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Miop.IA is a Chrome extension (Manifest V3) plus a FastAPI backend. Together they flag stylistic signs of misinformation in Portuguese-language news. Code, comments, commit messages and identifiers are in Portuguese, so match that.

## Product constraints (from `prd/PRD_MiopIA.md`, read it before UI/API changes)

- The product is **not a fact-checker**. Never present output as a verdict ("this is false") or a probability as certainty ("98% falsa"). Scores must be qualified ("sinais fortes de", "possível").
- Displayed confidence numbers must reflect the honest, leakage-free validated performance. Don't use metadata (author, domain, date) or untruncated text as features, because those leak the label.
- Never commit raw third-party training data. Only trained weights ship with the product.
- LGPD: don't store personally identifiable info with user feedback (feedback is keyed by an opaque `client_id`).
- Inspect the existing API contract before proposing new endpoints or schema changes.

## Commands

All backend commands run from `backend/` (imports are absolute from there, e.g. `api.main`, `train.train`). Python 3.13.

```bash
pip install -r requirements-dev.txt     # also installs the spaCy model pt_core_news_lg from a wheel URL
alembic upgrade head                    # or: python scripts/init_db.py (same migrations)
uvicorn api.main:app --reload --port 8000

pytest -v                                         # full suite
pytest tests/test_api.py -v                       # one file
pytest tests/test_api.py::nome_do_teste -v        # one test
python train/train.py                             # retrain; needs api/data/dataset_11.csv + fake_br_master.csv

# Extension extraction benchmark (Node 18+, jsdom)
cd miopia/benchmark && npm install && npm run benchmark
```

Config comes from env vars or `backend/.env` (see `.env.example`): `DATABASE_URL`, `CORS_ORIGINS`, `LIMITE_DIARIO_POR_CLIENTE`, `MODEL_PATH`, `ENVIRONMENT`. Both SQLite (`sqlite:///...`) and Postgres URLs work. `postgres://` and `postgresql://` are rewritten to `postgresql+psycopg2://`.

**Model path gotcha:** the default `MODEL_PATH` in `api/inferencia.py` is `models/stacking_miopia_0961.joblib`, but the committed artifact (and what `train.py` writes) is `models/stacking_miopia_v1.joblib`. Set `MODEL_PATH=models/stacking_miopia_v1.joblib` when running the API locally.

CI (`.github/workflows/ci.yml`, Postgres 15 service) runs these steps in order: `alembic upgrade head`, then `pytest -v`, then a bundle-load check (`carregar_bundle_stacking()`), then a live E2E run that starts uvicorn and curls `POST /analisar` followed by `POST /avaliar`.

## Architecture

**Request flow:** the extension's content script extracts article text in three layers (site-specific selectors, then Mozilla Readability, then a semantic fallback) and counts links in the DOM. The popup sends `{texto, url, num_links}` to `POST /analisar`. Community feedback goes to `POST /avaliar` (`noticia_id`, `client_id`, `avaliacao`). `/health` and `/ready` are monitoring endpoints.

**`/analisar` pipeline (`api/main.py`):**
1. `filtro.validar_viabilidade_analise` rejects texts that are too short (400).
2. A SHA-256 hash of the normalized text plus the current `Modelo` row acts as a cache key. On a hit, the stored `Noticia` is returned without re-running inference.
3. `feature_extraction.extrair_pacote_analise` runs spaCy (`pt_core_news_lg`) and pyspellchecker. It returns the 15 stylometric features and several text variants (`texto_cru` = truncated text, lemmatized text, etc.).
4. `inferencia.predizer_risco_stacking` scores the text with the stacking ensemble:
   - char TF-IDF + LinearSVC
   - word TF-IDF + LinearSVC
   - XGBoost over 103 dense features (15 stylometric features + LDA/NMF topic blocks with k=8 and k=30, each block contributing k+3 features)
   - a logistic-regression meta-model over those 3 outputs.

   The result is mapped to a `faixa` (Confiável / Atenção / Suspeita) plus `orientacao` text using the bundle's `limiar`.
5. The result is persisted to `Noticia` with its features as columns, linked to a `Modelo` row (version, F1, threshold).

**Train/serve parity is the central invariant.**
- `api/bundle_spec.py` is the single source of truth for feature order and count (`FEATURE_ORDER`, 103) and for the required bundle keys. Both `train/train.py` and the API import it.
- `api/feature_contract.py` documents each stylometric formula.
- `train.py` reuses the API's feature extraction, so any change to feature extraction must keep the training and production paths identical (`test_parity*.py`, `atol=1e-5`).
- Training uses `GroupShuffleSplit`/`GroupKFold` on `id_noticia` and out-of-fold predictions for the meta-model (`test_oof_protocol.py` guards against leakage).

**Fail-closed model loading:** `carregar_bundle_stacking` validates keys, dimensions (103 dense features, 3 meta inputs, topic counts, vocabulary alignment) and consistency with `models/model_manifest.json` (`feature_count`, `threshold`, `F1`). Any mismatch raises, and the endpoints return 503. If you retrain, update the bundle and the manifest together.

**Tests never use the production artifact.** `tests/conftest.py` trains a small synthetic bundle via `train.train.treinar_stacking` into a temp dir and points `MODEL_PATH` at it before importing the API. Importing `api.*` before conftest runs therefore picks up the wrong model.

**DB:** SQLAlchemy 2 models are in `api/models.py` (`Modelo`, `Noticia`, `Avaliacao`) and Alembic migrations are in `backend/alembic/versions/`. Migrations must support downgrade with existing data (e.g. give new non-null columns a `server_default`). Production runs on Neon Postgres, deployed on Render.

## Two extension directories

- `miopia/` is the extension the README tells you to load unpacked. It has `scripts/content.js`, an options page, and the `benchmark/` suite with fixtures for G1, UOL and others. Each fixture `.json` declares `expected_method`, word count ± tolerance, and `must_include`/`must_not_include` snippets.
- `frontend/` is the newer v2 UI. It has a service worker, `src/content.js`, the design reference in `src/assets/` (`design.png`, `DESIGN_TOKENS_MiopIA.md`) and the modal described in the PRD.

Both call the API at `http://localhost:8000`, which is hardcoded in the popup JS and in `host_permissions`. Check which directory a task targets before editing.
