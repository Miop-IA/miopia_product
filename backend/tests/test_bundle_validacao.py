"""
MI-16 — Sem fallback artificial: modelo ausente/corrompido/incompatível sempre é erro.
MI-17 — Validação de estrutura e dimensionalidade do bundle no startup (XGBoost = 103 features).
"""
import copy
import inspect

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api import inferencia
from api.bundle_spec import (
    CHAVES_OBRIGATORIAS,
    FEATURE_COUNT,
    FEATURE_ORDER,
    BundleIncompativelError,
    ModeloAusenteError,
    ModeloCorrompidoError,
    ModeloInvalidoError,
    validar_bundle,
)
from api.inferencia import carregar_bundle_stacking, extrair_features_topicos, resetar_cache_bundle
from tests.conftest import BUNDLE_TESTE_PATH


@pytest.fixture(scope="module")
def bundle_valido():
    return joblib.load(BUNDLE_TESTE_PATH)


@pytest.fixture
def salvar(tmp_path):
    def _salvar(bundle, nome="bundle.joblib"):
        caminho = tmp_path / nome
        joblib.dump(bundle, caminho)
        return str(caminho)
    return _salvar


@pytest.fixture
def apontar_modelo(monkeypatch):
    """Aponta a API para outro caminho de modelo e restaura o bundle de teste ao final."""
    from api.config import get_settings

    def _apontar(caminho):
        monkeypatch.setenv("MODEL_PATH", caminho)
        get_settings.cache_clear()
        resetar_cache_bundle()

    yield _apontar
    monkeypatch.setenv("MODEL_PATH", BUNDLE_TESTE_PATH)
    get_settings.cache_clear()
    resetar_cache_bundle()


def _sem_erro_de_validacao(bundle):
    validar_bundle(bundle)


# --------------------------------------------------------------------------- #
# Contrato de 103 features
# --------------------------------------------------------------------------- #

def test_contrato_103_features():
    assert FEATURE_COUNT == 103 == 15 + 11 + 11 + 33 + 33
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)) == 103
    assert FEATURE_ORDER[15] == "lda8_tema_0"
    assert FEATURE_ORDER[26] == "nmf8_tema_0"
    assert FEATURE_ORDER[37] == "lda30_tema_0"
    assert FEATURE_ORDER[70] == "nmf30_tema_0"
    assert FEATURE_ORDER[-1] == "nmf30_ajuste"


def test_bundle_valido_carrega(bundle_valido):
    _sem_erro_de_validacao(bundle_valido)
    assert bundle_valido["xgb_denso"].n_features_in_ == 103
    assert carregar_bundle_stacking(BUNDLE_TESTE_PATH)["version"] == "sintetico-teste"


# --------------------------------------------------------------------------- #
# MI-16 — falha explícita
# --------------------------------------------------------------------------- #

def test_modelo_ausente_gera_erro(tmp_path):
    with pytest.raises(ModeloAusenteError):
        carregar_bundle_stacking(str(tmp_path / "nao_existe.joblib"))


def test_modelo_corrompido_gera_erro(tmp_path):
    caminho = tmp_path / "corrompido.joblib"
    caminho.write_bytes(b"isto nao e um joblib valido \x00\xff" * 10)
    with pytest.raises(ModeloCorrompidoError):
        carregar_bundle_stacking(str(caminho))


def test_modelo_truncado_gera_erro(tmp_path):
    original = open(BUNDLE_TESTE_PATH, "rb").read()
    caminho = tmp_path / "truncado.joblib"
    caminho.write_bytes(original[: len(original) // 2])
    with pytest.raises(ModeloCorrompidoError):
        carregar_bundle_stacking(str(caminho))


@pytest.mark.parametrize("chave", CHAVES_OBRIGATORIAS)
def test_chave_ausente_gera_erro(bundle_valido, salvar, chave):
    b = dict(bundle_valido)
    del b[chave]
    with pytest.raises(BundleIncompativelError, match=chave):
        carregar_bundle_stacking(salvar(b))


def test_bundle_nao_dict_gera_erro(salvar):
    with pytest.raises(BundleIncompativelError):
        carregar_bundle_stacking(salvar(["nao", "e", "dict"]))


def test_xgb_com_feature_count_incompativel_gera_erro(bundle_valido, salvar):
    """Ex.: o antigo modelo dummy treinava o XGBoost com 15 + 40 = 55 features."""
    b = dict(bundle_valido)
    X = np.random.rand(10, 55)
    b["xgb_denso"] = XGBClassifier(n_estimators=2, max_depth=1).fit(X, np.arange(10) % 2)
    with pytest.raises(BundleIncompativelError, match="xgb_denso espera 55"):
        carregar_bundle_stacking(salvar(b))


def test_nao_existe_gerador_de_modelo_dummy():
    assert not hasattr(inferencia, "_criar_baseline_stacking")
    fonte = inspect.getsource(inferencia)
    assert "np.zeros" not in fonte
    assert "fallback" not in fonte.lower().replace("sem fallback", "")


def test_temas_ausentes_nao_retornam_vetor_zerado(bundle_valido):
    b = dict(bundle_valido)
    del b["lda_30"]
    with pytest.raises(BundleIncompativelError, match="lda_30"):
        extrair_features_topicos("governo economia", b)


def test_predicao_recusa_modelo_ausente(apontar_modelo, tmp_path):
    from api.feature_extraction import extrair_pacote_analise

    apontar_modelo(str(tmp_path / "nao_existe.joblib"))
    features, textos = extrair_pacote_analise("texto qualquer para teste de recusa " * 5)
    with pytest.raises(ModeloAusenteError):
        inferencia.predizer_risco_stacking(features, textos)


def test_api_nao_classifica_sem_modelo(apontar_modelo, tmp_path):
    from tests.test_api import TEXTO_VALIDO_LONGO, TestingSessionLocal, client, engine
    from api.database import Base
    from api.models import Noticia

    Base.metadata.create_all(bind=engine)
    apontar_modelo(str(tmp_path / "nao_existe.joblib"))
    resp = client.post("/analisar", json={"texto": TEXTO_VALIDO_LONGO})
    assert resp.status_code == 503
    assert client.get("/health").status_code == 503
    with TestingSessionLocal() as db:
        assert db.query(Noticia).count() == 0


# --------------------------------------------------------------------------- #
# MI-17 — estrutura e dimensionalidade
# --------------------------------------------------------------------------- #

def _mutar(bundle_valido, **mudancas):
    b = dict(bundle_valido)
    b.update(mudancas)
    return b


@pytest.mark.parametrize(
    "mudancas, trecho",
    [
        ({"feature_count": 102}, "feature_count deve ser 103"),
        ({"feature_count": 103.0}, "feature_count deve ser 103"),
        ({"feature_order": FEATURE_ORDER[:-1]}, "feature_order tem 102"),
        ({"feature_order": FEATURE_ORDER[:15] + FEATURE_ORDER[26:37] + FEATURE_ORDER[15:26] + FEATURE_ORDER[37:]},
         "feature_order diverge na posição 15"),
        ({"feature_order": "nao-e-lista"}, "feature_order deve ser lista"),
        ({"limiar": 1.5}, "limiar"),
        ({"limiar": float("nan")}, "limiar"),
        ({"limiar": "0.46"}, "limiar"),
        ({"f1_score": 0.0}, "f1_score"),
        ({"f1_score": None}, "f1_score"),
        ({"version": ""}, "version"),
        ({"version": 1}, "version"),
    ],
)
def test_metadados_invalidos(bundle_valido, mudancas, trecho):
    with pytest.raises(BundleIncompativelError, match=trecho):
        validar_bundle(_mutar(bundle_valido, **mudancas))


def test_feature_names_estilo_divergente(bundle_valido):
    nomes = list(bundle_valido["feature_names_estilo"])
    nomes[0], nomes[1] = nomes[1], nomes[0]
    with pytest.raises(BundleIncompativelError, match="feature_names_estilo"):
        validar_bundle(_mutar(bundle_valido, feature_names_estilo=nomes))


@pytest.mark.parametrize("chave", ["svm_caracteres", "svm_palavras", "xgb_denso", "meta_modelo"])
def test_classes_invalidas(bundle_valido, chave):
    if chave == "xgb_denso":
        # XGBoost não permite sobrescrever classes_: treina um multiclasse com as mesmas 103 features.
        modelo = XGBClassifier(n_estimators=2, max_depth=1).fit(np.random.rand(9, 103), np.arange(9) % 3)
    else:
        modelo = copy.deepcopy(bundle_valido[chave])
        modelo.classes_ = np.array([0, 2])
    with pytest.raises(BundleIncompativelError, match=f"{chave} deve ter classes"):
        validar_bundle(_mutar(bundle_valido, **{chave: modelo}))


def test_modelos_tematicos_trocados(bundle_valido):
    """LDA8 no lugar de LDA30 muda a dimensionalidade (11 em vez de 33)."""
    with pytest.raises(BundleIncompativelError, match="lda_30 deve ter n_components=30"):
        validar_bundle(_mutar(bundle_valido, lda_30=bundle_valido["lda_8"]))


def test_vetorizador_trocado(bundle_valido):
    with pytest.raises(BundleIncompativelError, match="svm_caracteres espera"):
        validar_bundle(_mutar(bundle_valido, tfidf_char=bundle_valido["tfidf_word"]))


def test_meta_modelo_dimensao_errada(bundle_valido):
    from sklearn.linear_model import LogisticRegression

    meta = LogisticRegression().fit(np.random.rand(10, 2), np.arange(10) % 2)
    with pytest.raises(BundleIncompativelError, match="meta_modelo espera 2"):
        validar_bundle(_mutar(bundle_valido, meta_modelo=meta))


def test_todos_os_problemas_sao_reportados(bundle_valido):
    with pytest.raises(BundleIncompativelError) as exc:
        validar_bundle(_mutar(bundle_valido, feature_count=10, limiar=2.0, version=""))
    assert len(exc.value.problemas) == 3


def test_bundle_incompativel_impede_startup(apontar_modelo, bundle_valido, salvar):
    """Critério MI-17: com bundle incompatível, o lifespan falha e a API não fica pronta."""
    from api.main import app

    apontar_modelo(salvar(_mutar(bundle_valido, feature_count=102)))
    with pytest.raises(ModeloInvalidoError):
        with TestClient(app):
            pass


def test_modelo_ausente_impede_startup(apontar_modelo, tmp_path):
    from api.main import app

    apontar_modelo(str(tmp_path / "nao_existe.joblib"))
    with pytest.raises(ModeloAusenteError):
        with TestClient(app):
            pass


def test_bundle_valido_permite_startup(monkeypatch):
    """Com bundle válido, o startup conclui e /health expõe a versão do modelo."""
    from api import main

    monkeypatch.setattr(main.Base.metadata, "create_all", lambda **kw: None)
    resetar_cache_bundle()
    with TestClient(main.app) as c:
        data = c.get("/health").json()
    assert data["model_version"] == "sintetico-teste"
    assert data["feature_count"] == 103
