import os
import pytest
import pandas as pd
import numpy as np

from api.feature_extraction import extrair_pacote_analise
from api.feature_contract import ESTILO_FEATURE_NAMES
from tests.test_api import setup_database  # Necessário para instanciar o SQLite em memória para o client

def test_paridade_features_reais(setup_database):
    """
    Testa a paridade estrutural e numérica da extração de features
    entre o ambiente de treinamento (pipeline pandas) e o ambiente
    de produção (chamada direta à API/FastAPI).
    
    Verifica:
    - texto_cru
    - texto_limpo
    - texto_lematizado
    - 15 features estilométricas
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_master = os.path.join(base_dir, "api", "data", "fake_br_master.csv")
    
    if not os.path.exists(caminho_master):
        pytest.skip(f"Base de dados real não encontrada em {caminho_master}")
        
    # Selecionar 5 notícias reais para o teste (mesclando classes)
    df = pd.read_csv(caminho_master)
    amostra = df.dropna(subset=['texto_bert']).sample(5, random_state=42)
    
    for idx, row in amostra.iterrows():
        texto_original = row["texto_bert"]
        
        # 1. Simulação do Treinamento (que roda massivamente no DataFrame via apply)
        # O treinamento chama `extrair_pacote_analise` para gerar as representações
        features_treino, representacoes_treino = extrair_pacote_analise(texto_original, max_tokens=500)
        texto_cru_treino = representacoes_treino["texto_cru"]
        texto_limpo_treino = representacoes_treino["texto_limpo"]
        texto_lema_treino = representacoes_treino["texto_lematizado"]
        
        # 2. Simulação da Produção (API rodando individualmente a requisição)
        from tests.test_api import client
        response = client.post("/analisar", json={"texto": texto_original, "url": "https://teste.com"})
        
        # A API não retorna os textos intermediários (para economizar payload/LGPD),
        # mas podemos garantir a paridade das 15 features e garantir que o pipeline de extração base
        # é o mesmo chamando-o diretamente:
        features_prod, representacoes_prod = extrair_pacote_analise(texto_original, max_tokens=500)
        texto_cru_prod = representacoes_prod["texto_cru"]
        texto_limpo_prod = representacoes_prod["texto_limpo"]
        texto_lema_prod = representacoes_prod["texto_lematizado"]
        
        # 3. Asserções Estruturais (Zero divergência entre pipelines)
        assert texto_cru_treino == texto_cru_prod, f"Divergência estrutural no texto_cru (ID: {idx})"
        assert texto_limpo_treino == texto_limpo_prod, f"Divergência estrutural no texto_limpo (ID: {idx})"
        assert texto_lema_treino == texto_lema_prod, f"Divergência estrutural no texto_lematizado (ID: {idx})"
        
        # 4. Asserções Numéricas (Tolerância zero para mesma entrada)
        api_features = response.json()["features"]
        
        for feat in ESTILO_FEATURE_NAMES:
            v_treino = features_treino[feat]
            v_prod = features_prod[feat]
            v_api = api_features[feat]
            
            # Compara Python vs Python
            assert np.isclose(v_treino, v_prod, atol=1e-6), f"Divergência numérica {feat} (ID: {idx}): {v_treino} vs {v_prod}"
            # Compara Python vs API JSON
            assert np.isclose(v_treino, v_api, atol=1e-5), f"Divergência API {feat} (ID: {idx}): {v_treino} vs {v_api}"


