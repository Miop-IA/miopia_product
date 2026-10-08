"""
Falhas de modelo em tempo de requisição: com o bundle inválido, /analisar falha
fechado (HTTP 503) e nada é gravado. O caminho do modelo é trocado via MODEL_PATH,
o mesmo mecanismo usado pelo conftest — sem monkeypatch em os.path.
"""
import json
import os
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from api.inferencia import resetar_cache_bundle
from api.main import app
from tests.conftest import BUNDLE_TESTE_PATH
from tests.test_api import setup_database  # noqa: F401  (fixture: banco SQLite de teste)

TEXTO = "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente. " * 5


@pytest.fixture
def modelo_em(monkeypatch, tmp_path, setup_database):
    """Copia o bundle de teste + manifesto para tmp_path, aplica uma mutação e aponta a API para lá."""

    def _preparar(mutar_bundle=None, mutar_manifesto=None, sem_manifesto=False, conteudo_bruto=None):
        caminho = tmp_path / "modelo.joblib"
        manifesto_origem = os.path.join(os.path.dirname(BUNDLE_TESTE_PATH), "model_manifest.json")
        if conteudo_bruto is not None:
            caminho.write_bytes(conteudo_bruto)
        else:
            bundle = joblib.load(BUNDLE_TESTE_PATH)
            if mutar_bundle:
                mutar_bundle(bundle)
            joblib.dump(bundle, caminho)
        if not sem_manifesto:
            with open(manifesto_origem, encoding="utf-8") as f:
                manifesto = json.load(f)
            if mutar_manifesto:
                mutar_manifesto(manifesto)
            (tmp_path / "model_manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")
        monkeypatch.setenv("MODEL_PATH", str(caminho))
        get_settings.cache_clear()
        resetar_cache_bundle()

    yield _preparar
    monkeypatch.setenv("MODEL_PATH", BUNDLE_TESTE_PATH)
    get_settings.cache_clear()
    resetar_cache_bundle()


def _analisar_sem_lifespan():
    # Sem o "with": o lifespan (que já recusaria o startup) não roda; testamos a rota isolada.
    return TestClient(app).post("/analisar", json={"texto": TEXTO})


def _assert_503(resp):
    assert resp.status_code == 503
    assert "não está carregado" in resp.json()["detail"]


def test_falha_arquivo_inexistente(monkeypatch, tmp_path, setup_database):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nao_existe.joblib"))
    get_settings.cache_clear()
    resetar_cache_bundle()
    try:
        _assert_503(_analisar_sem_lifespan())
    finally:
        monkeypatch.setenv("MODEL_PATH", BUNDLE_TESTE_PATH)
        get_settings.cache_clear()
        resetar_cache_bundle()


def test_falha_arquivo_corrompido(modelo_em):
    modelo_em(conteudo_bruto=b"Este nao e um arquivo joblib valido")
    _assert_503(_analisar_sem_lifespan())


def test_falha_bundle_incompleto(modelo_em):
    modelo_em(mutar_bundle=lambda b: b.pop("svm_palavras"))
    _assert_503(_analisar_sem_lifespan())


def test_falha_dimensao_errada(modelo_em):
    def trocar_lda(b):
        b["lda_30"] = b["lda_8"]
    modelo_em(mutar_bundle=trocar_lda)
    _assert_503(_analisar_sem_lifespan())


def test_falha_manifesto_ausente(modelo_em):
    modelo_em(sem_manifesto=True)
    _assert_503(_analisar_sem_lifespan())


def test_falha_manifesto_inconsistente(modelo_em):
    modelo_em(mutar_manifesto=lambda m: m.update({"F1": 0.50}))
    _assert_503(_analisar_sem_lifespan())


def test_startup_recusa_bundle_invalido(modelo_em):
    from api.bundle_spec import ModeloInvalidoError

    modelo_em(mutar_bundle=lambda b: b.pop("scaler_estilo"))
    with pytest.raises(ModeloInvalidoError):
        with TestClient(app):
            pass
