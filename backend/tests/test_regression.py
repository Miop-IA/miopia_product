import pytest
import os
import numpy as np

from api.feature_extraction import extrair_pacote_analise
from api.inferencia import predizer_risco_stacking, carregar_bundle_stacking
from tests.test_api import setup_database  # Instancia SQLite em memória se o router for acionado

def test_regression_golden_samples(monkeypatch):
    """
    Testes de regressão estritos com Golden Samples (Guardrails).
    Garante que alterações futuras no código, features ou heurísticas não quebrem
    a predição esperada de textos de baseline previamente aferidos.
    """
    from api.inferencia import resetar_cache_bundle
    from api.config import get_settings

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_model_path = os.path.join(base_dir, "models", "stacking_miopia_v4_candidato.joblib")
    if not os.path.exists(real_model_path):
        pytest.skip(f"Modelo real não encontrado em {real_model_path}")
        
    monkeypatch.setenv("MODEL_PATH", real_model_path)
    get_settings.cache_clear()
    resetar_cache_bundle()
    
    # Sample 1: Claramente Falso/Suspeito (Excesso de CAIXA ALTA, pronomes 1a pessoa, emotividade)
    texto_fake = (
        "URGENTE EU TE AVISO AGORA MESMO VEJA o que o governo MENTIU para a "
        "população. A VERDADE FOI REVELADA! Compartilhe imediatamente com sua "
        "família antes que APAGUEM tudo. A situação é terrível e chocante."
    )
    features_fake, textos_fake = extrair_pacote_analise(texto_fake, max_tokens=500)
    prob_fake, faixa_fake, orientacao_fake, f1_ref = predizer_risco_stacking(features_fake, textos_fake)
    
    assert faixa_fake in ["Atencao", "Suspeita"], f"Golden Sample Fake foi classificado erroneamente como {faixa_fake}"
    assert prob_fake > 0.40, f"Probabilidade do texto fake muito baixa: {prob_fake}"

    # Sample 2: Claramente Verdadeiro/Confiável (Jornalístico, estruturado, neutro)
    texto_real = (
        "O Banco Central anunciou nesta terça-feira a manutenção da taxa básica de juros "
        "em 10,50% ao ano. A decisão foi unânime entre os diretores do Comitê de Política "
        "Monetária (Copom). Segundo a nota oficial, a inflação tem se mantido dentro "
        "da meta estabelecida para o semestre."
    )
    features_real, textos_real = extrair_pacote_analise(texto_real, max_tokens=500)
    prob_real, faixa_real, orientacao_real, _ = predizer_risco_stacking(features_real, textos_real)
    
    assert faixa_real == "Confiavel", f"Golden Sample Real foi classificado erroneamente como {faixa_real}"
    assert prob_real < 0.40, f"Probabilidade do texto real muito alta: {prob_real}"

    # Sample 3: Limite de truncamento preserva comportamento
    texto_longo = texto_real + (" E mais palavras neutras para preencher." * 100)
    features_longo, textos_longo = extrair_pacote_analise(texto_longo, max_tokens=500)
    prob_longo, faixa_longo, _, _ = predizer_risco_stacking(features_longo, textos_longo)
    
    assert faixa_longo == "Confiavel", "Texto longo extrapolou e mudou de faixa"
    # Diferença pequena entre truncado e original (pois é essencialmente o mesmo texto)
    assert abs(prob_longo - prob_real) < 0.15, "Drift alto na predição pós-truncamento"

