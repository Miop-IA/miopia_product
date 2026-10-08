import os
import json
import joblib
import pytest
import tempfile
import importlib
from fastapi.testclient import TestClient
from api.main import app

from tests.test_api import setup_database

import api.inferencia

@pytest.fixture
def clean_model_cache(setup_database):
    """Limpa o cache em memória do modelo para forçar recarregamento em cada teste."""
    api.inferencia._stacking_bundle = None
    yield
    api.inferencia._stacking_bundle = None

def get_base_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "stacking_miopia_v1.joblib")
    manifest_path = os.path.join(base_dir, "models", "model_manifest.json")
    return model_path, manifest_path

def test_falha_arquivo_inexistente(monkeypatch, clean_model_cache):
    original_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if "stacking_miopia_v1.joblib" in str(p) else original_exists(p))
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        # A API deve falhar fechada (HTTP 503)
        assert response.status_code == 503
        assert "não está carregado" in response.json()["detail"]

def test_falha_arquivo_corrompido(monkeypatch, clean_model_cache):
    # Cria um arquivo falso que não é um joblib válido
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"Este nao e um arquivo joblib valido")
        fake_path = tmp.name

    original_join = os.path.join
    def mock_join(*args):
        if "stacking_miopia_v1.joblib" in args:
            return fake_path
        return original_join(*args)

    monkeypatch.setattr(os.path, "join", mock_join)
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        assert response.status_code == 503
        assert "não está carregado" in response.json()["detail"]
        
    os.remove(fake_path)

def test_falha_bundle_incompleto(monkeypatch, clean_model_cache):
    # Carrega o modelo real e remove uma chave
    model_path, manifest_path = get_base_paths()
    real_bundle = joblib.load(model_path)
    del real_bundle["xgb_denso"]
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        joblib.dump(real_bundle, tmp.name)
        fake_path = tmp.name

    original_join = os.path.join
    def mock_join(*args):
        if "stacking_miopia_v1.joblib" in args:
            return fake_path
        return original_join(*args)

    monkeypatch.setattr(os.path, "join", mock_join)
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        assert response.status_code == 503
        
    os.remove(fake_path)

class MockXGB:
    n_features_in_ = 999

def test_falha_dimensao_errada(monkeypatch, clean_model_cache):
    model_path, manifest_path = get_base_paths()
    real_bundle = joblib.load(model_path)
    
    # Substitui por um mock com dimensão errada
    real_bundle["xgb_denso"] = MockXGB()
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        joblib.dump(real_bundle, tmp.name)
        fake_path = tmp.name

    original_join = os.path.join
    def mock_join(*args):
        if "stacking_miopia_v1.joblib" in args:
            return fake_path
        return original_join(*args)

    monkeypatch.setattr(os.path, "join", mock_join)
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        assert response.status_code == 503
        
    os.remove(fake_path)

def test_falha_manifesto_ausente(monkeypatch, clean_model_cache):
    original_exists = os.path.exists
    def mock_exists(p):
        if "model_manifest.json" in str(p):
            return False
        return original_exists(p)
        
    monkeypatch.setattr(os.path, "exists", mock_exists)
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        assert response.status_code == 503

def test_falha_manifesto_inconsistente(monkeypatch, clean_model_cache):
    model_path, manifest_path = get_base_paths()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        real_manifest = json.load(f)
        
    # Invalida o F1
    real_manifest["F1"] = 0.50
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp:
        json.dump(real_manifest, tmp)
        fake_manifest_path = tmp.name

    original_join = os.path.join
    def mock_join(*args):
        if "model_manifest.json" in args:
            return fake_manifest_path
        return original_join(*args)

    monkeypatch.setattr(os.path, "join", mock_join)
    
    with TestClient(app) as client:
        response = client.post("/analisar", json={"texto": "Um texto válido que tenha mais de trinta palavras para não cair no filtro inicial de volume insuficiente." * 5})
        assert response.status_code == 503
        
    os.remove(fake_manifest_path)
