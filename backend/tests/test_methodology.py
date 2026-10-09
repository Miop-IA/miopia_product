import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from train.train import gerar_dados_sinteticos_para_teste

@pytest.mark.fast
def test_esquema_dados_qualidade():
    """Valida que o gerador ou os dados mantêm o esquema esperado (target, texto_cru, grupo_identidade)"""
    df = gerar_dados_sinteticos_para_teste(n_samples=20)
    assert "target" in df.columns
    assert "texto_cru" in df.columns
    assert "grupo_identidade" in df.columns
    assert df["texto_cru"].notna().all(), "Não pode haver texto nulo"

@pytest.mark.fast
def test_cv_sem_intersecao_grupos():
    """Garante que a validação cruzada não divide o mesmo grupo_identidade entre treino e validação."""
    df = gerar_dados_sinteticos_para_teste(n_samples=50)
    
    # Adicionando alguns grupos repetidos artificialmente para garantir teste de leakage
    df.loc[0:5, "grupo_identidade"] = "grupo_teste_1"
    
    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in gkf.split(df, df["target"], groups=df["grupo_identidade"]):
        train_groups = set(df.iloc[train_idx]["grupo_identidade"])
        val_groups = set(df.iloc[val_idx]["grupo_identidade"])
        
        intersecao = train_groups.intersection(val_groups)
        assert len(intersecao) == 0, f"Vazamento de grupos detectado na Validação Cruzada: {intersecao}"

@pytest.mark.fast
def test_selecao_threshold_independente():
    """Garante que o teste do threshold pode ser instanciado e as probabilidades processadas."""
    # Simula as probabilidades do metamodelo
    y_true = np.array([0, 1, 0, 1, 1, 0])
    p_fake_oof = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4])
    
    from sklearn.metrics import f1_score
    melhor_limiar = 0.5
    melhor_f1_oof = 0.0
    
    for lim in np.arange(0.1, 0.9, 0.01):
        y_pred_cand = (p_fake_oof >= lim).astype(int)
        f1_cand = f1_score(y_true, y_pred_cand, pos_label=1)
        if f1_cand > melhor_f1_oof:
            melhor_f1_oof = f1_cand
            melhor_limiar = lim
            
    assert melhor_limiar > 0
    assert melhor_f1_oof > 0.8 # O limiar ideal deveria separar os dados dummy bem
