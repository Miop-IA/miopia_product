import sys
import os

# Garante a raiz do backend no sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.database import Base, get_db
from api.main import app

# Configuração de banco de dados SQLite em memória isolado para os testes
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Garante recriação limpa do schema antes de cada teste."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


TEXTO_VALIDO_LONGO = (
    "O Ministério da Fazenda divulgou hoje uma nota oficial com as diretrizes econômicas "
    "e parâmetros fiscais que serão apresentados aos governadores na reunião marcada "
    "para o início do próximo mês em Brasília, com o objetivo de equilibrar as contas públicas "
    "e impulsionar novos investimentos estruturais no país."
)

TEXTO_CURTO_INVALIDO = "Texto muito curto para análise prévia."


def test_health_check():
    """Verifica se a rota de monitoramento responde online."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Stacking" in data["model"]


def test_analisar_texto_curto_rejeicao():
    """Valida se textos com menos de 30 palavras recebem HTTP 400."""
    response = client.post(
        "/analisar",
        json={"texto": TEXTO_CURTO_INVALIDO, "url": "https://exemplo.com/curta"}
    )
    assert response.status_code == 400
    assert "30 palavras" in response.json()["detail"]


def test_analisar_noticia_sucesso_e_estrutura():
    """Testa a extração completa, inferência do Stacking e formato do contrato."""
    response = client.post(
        "/analisar",
        json={"texto": TEXTO_VALIDO_LONGO, "url": "https://portalexemplo.com/noticia"}
    )
    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert "hash_texto" in data
    assert len(data["hash_texto"]) == 64
    assert 0.0 <= data["prob_suspeita"] <= 1.0
    assert data["faixa"] in ["Confiavel", "Atencao", "Suspeita"]
    assert data["modelo_f1"] == 0.961

    metricas = data["metricas"]
    assert len(metricas) == 15
    assert 0.0 <= metricas["trunc_diversity"] <= 1.0
    assert "link_density" in metricas

    comunidade = data["avaliacoes_comunidade"]
    assert comunidade["total"] == 0


def test_cache_o1_hash_texto():
    """Garante que a segunda requisição com o mesmo conteúdo utiliza o cache."""
    res_1 = client.post("/analisar", json={"texto": TEXTO_VALIDO_LONGO})
    assert res_1.status_code == 200
    id_1 = res_1.json()["id"]

    res_2 = client.post("/analisar", json={"texto": TEXTO_VALIDO_LONGO})
    assert res_2.status_code == 200
    assert res_2.json()["id"] == id_1
    assert res_2.json()["hash_texto"] == res_1.json()["hash_texto"]


def test_avaliar_comunidade_ciclo_completo():
    """Testa submissão de voto e atualização idempotente do mesmo utilizador."""
    analise_res = client.post("/analisar", json={"texto": TEXTO_VALIDO_LONGO})
    noticia_id = analise_res.json()["id"]

    voto_res = client.post(
        "/avaliar",
        json={
            "noticia_id": noticia_id,
            "client_id": "cliente-sessao-abc-123",
            "avaliacao": 0
        }
    )
    assert voto_res.status_code == 200
    res_json = voto_res.json()
    assert res_json["sucesso"] is True
    assert res_json["avaliacoes_atualizadas"]["verdadeiro"] == 1
    assert res_json["avaliacoes_atualizadas"]["total"] == 1

    voto_att = client.post(
        "/avaliar",
        json={
            "noticia_id": noticia_id,
            "client_id": "cliente-sessao-abc-123",
            "avaliacao": 2
        }
    )
    assert voto_att.status_code == 200
    res_att_json = voto_att.json()
    assert res_att_json["avaliacoes_atualizadas"]["verdadeiro"] == 0
    assert res_att_json["avaliacoes_atualizadas"]["falso"] == 1
    assert res_att_json["avaliacoes_atualizadas"]["total"] == 1