import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.feature_extraction import extrair_pacote_analise
from api.inferencia import predizer_risco_stacking
import numpy as np

from tests.test_api import client, setup_database

# Gerador de 20 textos fixos variados para o teste de paridade
TEXTOS_FIXOS = [
    f"Noticia de teste {i} com um texto longo o suficiente para passar no filtro de 30 palavras. " * 3 + f" Diferencial: {i} " + " ".join([f"palavra{x}" for x in range(i)])
    for i in range(20)
]
# Vamos garantir que os textos tenham mais de 30 palavras
TEXTOS_FIXOS = [t + (" Apenas preenchendo o texto para garantir que ele passe pelo filtro de tamanho minimo com tranquilidade e possa gerar features validas." * 3) for t in TEXTOS_FIXOS]

def test_paridade_pipeline_vs_api():
    """
    Testa a paridade entre a extração direta do modelo (simulando o Notebook/Train)
    e a saída servida pela API HTTP de inferência.
    """
    for idx, texto in enumerate(TEXTOS_FIXOS):
        # 1. Pipeline Oficial (Python Direto)
        features_dict_py, textos_py = extrair_pacote_analise(texto)
        prob_py, faixa_py, _, _ = predizer_risco_stacking(features_dict_py, textos_py)
        
        # 2. Chamada na API
        response = client.post(
            "/analisar",
            json={"texto": texto, "url": "https://teste.com"}
        )
        assert response.status_code == 200, f"Falha na API para o texto {idx}"
        api_data = response.json()
        
        # 3. Verificações de Paridade
        
        # 3.1 Mesma representação textual truncada
        assert api_data["texto_truncado"] == textos_py["texto_cru"], f"Divergência de truncamento no texto {idx}"
        
        # 3.2 Mesmas 15 features dentro da tolerância
        api_features = api_data["features"]
        assert len(api_features) == 15, f"A API não retornou 15 features para o texto {idx}"
        assert len(features_dict_py) == 15, f"O pipeline direto não gerou 15 features para o texto {idx}"
        
        for feat_name, val_py in features_dict_py.items():
            val_api = api_features[feat_name]
            assert np.isclose(val_py, val_api, atol=1e-5), f"Divergência na feature {feat_name} no texto {idx}: Pipeline={val_py}, API={val_api}"
            
        # 3.3 Mesma probabilidade dentro da tolerância
        assert np.isclose(prob_py, api_data["prob_suspeita"], atol=1e-5), f"Divergência de probabilidade no texto {idx}: Pipeline={prob_py}, API={api_data['prob_suspeita']}"
        
        # 3.4 Mesma classificação (faixa)
        assert faixa_py == api_data["faixa"], f"Divergência de classificação no texto {idx}: Pipeline={faixa_py}, API={api_data['faixa']}"
