"""
Falha fechada da API: com o modelo ausente, corrompido, incompleto, com dimensão errada
ou com manifesto ausente/inconsistente, /analisar responde 503 e o startup é recusado.

Cada cenário copia o bundle sintético (e seu manifesto) para uma pasta temporária,
estraga um dos artefatos e aponta MODEL_PATH para ela.
"""
import json
import os
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from api.bundle_spec import ModeloInvalidoError
from api.inferencia import NOME_MANIFESTO
from api.main import app
from tests.conftest import BUNDLE_TESTE_PATH
from tests.test_api import TEXTO_VALIDO_LONGO, client

MANIFESTO_TESTE_PATH = os.path.join(os.path.dirname(BUNDLE_TESTE_PATH), NOME_MANIFESTO)


@pytest.fixture
def copia_modelo(tmp_path):
    """Copia bundle + manifesto de teste para tmp_path e devolve os caminhos das cópias."""
    bundle_path = tmp_path / "stacking.joblib"
    manifest_path = tmp_path / NOME_MANIFESTO
    shutil.copy(BUNDLE_TESTE_PATH, bundle_path)
    shutil.copy(MANIFESTO_TESTE_PATH, manifest_path)
    return bundle_path, manifest_path


def _assert_falha_fechada(caminho_modelo, apontar_modelo):
    apontar_modelo(caminho_modelo)

    response = client.post("/analisar", json={"texto": TEXTO_VALIDO_LONGO})
    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo de classificação indisponível."

    with pytest.raises(ModeloInvalidoError):
        with TestClient(app):
            pass


def test_falha_arquivo_inexistente(apontar_modelo, tmp_path):
    _assert_falha_fechada(tmp_path / "nao_existe.joblib", apontar_modelo)


def test_falha_arquivo_corrompido(apontar_modelo, copia_modelo):
    bundle_path, _ = copia_modelo
    bundle_path.write_bytes(b"Este nao e um arquivo joblib valido")
    _assert_falha_fechada(bundle_path, apontar_modelo)


def test_falha_bundle_incompleto(apontar_modelo, copia_modelo):
    bundle_path, _ = copia_modelo
    bundle = joblib.load(bundle_path)
    del bundle["xgb_denso"]
    joblib.dump(bundle, bundle_path)
    _assert_falha_fechada(bundle_path, apontar_modelo)


class MockXGB:
    n_features_in_ = 999


def test_falha_dimensao_errada(apontar_modelo, copia_modelo):
    bundle_path, _ = copia_modelo
    bundle = joblib.load(bundle_path)
    bundle["xgb_denso"] = MockXGB()
    joblib.dump(bundle, bundle_path)
    _assert_falha_fechada(bundle_path, apontar_modelo)


def test_falha_manifesto_ausente(apontar_modelo, copia_modelo):
    bundle_path, manifest_path = copia_modelo
    manifest_path.unlink()
    _assert_falha_fechada(bundle_path, apontar_modelo)


def test_falha_manifesto_inconsistente(apontar_modelo, copia_modelo):
    bundle_path, manifest_path = copia_modelo
    manifesto = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifesto["F1"] = manifesto["F1"] - 0.5
    manifest_path.write_text(json.dumps(manifesto), encoding="utf-8")
    _assert_falha_fechada(bundle_path, apontar_modelo)
