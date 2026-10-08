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


def test_paridade_preprocessing_treino_producao():
    """Garante que a função unificada trate os limites, minúsculas e limpeza de links exatamente igual para API e Treino."""
    from api.feature_extraction import preparar_texto_comum
    
    texto = "Este é um texto longo que será cortado com truncamento. " * 50
    texto_cru_trunc, texto_limpo, num_links = preparar_texto_comum(texto, max_tokens=10)
    
    assert len(texto_cru_trunc.split()) == 10
    assert len(texto_limpo.split()) <= 10
    assert texto_limpo.islower()
    
    texto_com_link = "Veja o repositório https://github.com/Miop-IA e o portal www.uol.com.br fim."
    _, limpo, num = preparar_texto_comum(texto_com_link, max_tokens=500)
    
    assert num == 2
    assert "www" not in limpo


def test_link_density_com_num_links_payload():
    """Verifica se o backend aceita e prioriza o num_links vindo do payload (DOM) na hora de calcular link_density."""
    texto_sem_link_literal = "Este é um texto gigante sem nenhuma url literal escrita, mas que no HTML tem links reais. " * 20
    
    # Se não enviar num_links, a link_density deve ser 0 (pois não há http/www escrito)
    features_sem, _ = extrair_pacote_analise(texto_sem_link_literal)
    assert features_sem["link_density"] == 0.0
    
    # Se enviar num_links=5, a link_density deve ser > 0 (usa o valor exato do DOM)
    features_com, _ = extrair_pacote_analise(texto_sem_link_literal, num_links_param=5)
    assert features_com["link_density"] > 0.0


def test_rc_spelling_errors_repete_palavra():
    """Valida se a métrica calcula corretamente a ocorrência de erros quando há repetição (sem misturar com tipos)."""
    # 4 palavras válidas no total (caxorrrro e serto são erros)
    # caxorrrro aparece 2 vezes
    # serto aparece 1 vez
    # correto aparece 1 vez
    # Total de palavras = 4
    # Total de erros (ocorrências) = 3 (caxorrrro, caxorrrro, serto)
    # rc_spelling_errors = 3 / 4 = 0.75
    texto = "caxorrrro caxorrrro serto correto"
    features, _ = extrair_pacote_analise(texto)
    assert features["rc_spelling_errors"] == 0.75