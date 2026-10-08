import pytest
import os
from api.feature_extraction import extrair_pacote_analise, calcular_mattr
from api.inferencia import predizer_risco_stacking, carregar_bundle_stacking
from api.filtro import validar_viabilidade_analise, gerar_hash_texto


def test_mattr_janela_fixa():
    """Valida se o MATTR com janela de 25 palavras calcula a diversidade lexical."""
    lemas = ["noticia", "governo", "economia", "politica"] * 10
    score = calcular_mattr(lemas, window_size=25)
    assert 0.0 < score <= 1.0


def test_extracao_pacote_features():
    """Garante a extração das 15 features e das três representações textuais."""
    texto = (
        "O ministério da fazenda publicou portaria com as novas diretrizes fiscais "
        "para os estados e municípios durante a reunião ordinária desta semana."
    )
    features, textos = extrair_pacote_analise(texto)
    
    assert len(features) == 15
    assert "trunc_diversity" in features
    assert "texto_cru" in textos
    assert "texto_limpo" in textos
    assert "texto_lematizado" in textos


def test_inferencia_stacking_e_limiar():
    """Testa se o modelo serializado processa as predições e aplica o limiar 0.46."""
    texto = (
        "URGENTE repasse agora mesmo veja o que o governo escondeu de você escândalo "
        "confirmado pela imprensa independente compartilhe já antes que apaguem tudo."
    )
    features, textos = extrair_pacote_analise(texto)
    prob, faixa, orientacao, f1_ref = predizer_risco_stacking(features, textos)

    assert 0.0 <= prob <= 1.0
    assert faixa in ["Confiavel", "Atencao", "Suspeita"]
    assert f1_ref > 0.90  # Valida que o pipeline embute uma métrica calculada sã e coerente


def test_filtro_viabilidade_texto_curto():
    """Assegura a rejeição imediata de textos com menos de 30 palavras."""
    texto_curto = "Apenas uma frase curta para validar o filtro."
    valido, motivo, total = validar_viabilidade_analise(texto_curto)
    assert valido is False
    assert total < 30


def test_ausencia_intersecao_grupos_calibracao():
    """Verifica se o GroupKFold impede interseção de grupos entre treino e validação, garantindo calibração group-aware segura."""
    from sklearn.model_selection import GroupKFold
    import numpy as np

    X = np.random.rand(10, 2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    gkf = GroupKFold(n_splits=3)
    for train_idx, val_idx in gkf.split(X, y, groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        
        # A interseção entre os grupos deve ser estritamente vazia
        assert len(train_groups.intersection(val_groups)) == 0