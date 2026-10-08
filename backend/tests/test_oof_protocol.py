import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.model_selection import GroupKFold

# Importa o módulo de treino para podermos testar o fluxo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train import treinar_stacking
from api.feature_contract import ESTILO_FEATURE_NAMES

def test_garantia_kfold_sem_vazamento():
    """
    Testa rigorosamente a teoria e prática do GroupKFold usado no OOF.
    Garante que:
    1. Nenhum grupo aparece simultaneamente no treino e validação do mesmo fold.
    2. Cada amostra aparece exatamente UMA VEZ como validação.
    """
    df_treino = pd.DataFrame({
        "id_noticia": [1, 1, 2, 2, 3, 4, 5, 5, 6, 7],
        "target": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    })
    groups_train = df_treino["id_noticia"].values
    y_train = df_treino["target"].values

    gkf = GroupKFold(n_splits=3)
    val_indices = []
    
    for train_idx, val_idx in gkf.split(df_treino, y_train, groups=groups_train):
        train_groups = set(groups_train[train_idx])
        val_groups = set(groups_train[val_idx])
        
        # 1. Nenhum grupo simultâneo
        assert len(train_groups.intersection(val_groups)) == 0, "Vazamento OOF: grupo(s) presentes no treino e validação simultaneamente."
        
        val_indices.extend(val_idx)

    # 2. Cada amostra aparece exatamente uma vez na validação
    assert sorted(val_indices) == list(range(len(df_treino))), "Protocolo OOF falhou: as predições de validação não cobrem todo o dataset exatamente uma vez."

def test_metamodelo_usa_oof(tmp_path):
    """
    Roda um mock do treinamento com um dataset minúsculo para provar que a
    Regressão Logística (Metamodelo) recebe como entrada de treino as
    matrizes OOF populadas (dimensão e integridade), e não as matrizes originais.
    """
    N = 30
    target = [i % 2 for i in range(N)]
    # Textos separáveis por classe: o treino exige F1 mínimo antes de empacotar o bundle.
    textos = {
        0: ("Ministério divulga relatório oficial.", "ministerio divulga relatorio oficial", "ministerio divulgar relatorio oficial"),
        1: ("URGENTE compartilhe antes que apaguem!", "urgente compartilhe antes apaguem", "urgente compartilhar antes apagar"),
    }
    df_dummy = pd.DataFrame({
        "id_noticia": list(range(1, N+1)),
        "target": target,
        "texto_cru": [textos[t][0] for t in target],
        "texto_limpo": [textos[t][1] for t in target],
        "texto_lematizado": [textos[t][2] for t in target],
    })

    df_val = df_dummy.copy()
    df_val["id_noticia"] = list(range(101, 101+N))
    for f in ESTILO_FEATURE_NAMES:
        df_dummy[f] = np.random.rand(N) + df_dummy["target"]
        df_val[f] = np.random.rand(N) + df_val["target"]
    
    from sklearn.linear_model import LogisticRegression
    real_fit = LogisticRegression.fit
    captured_shapes = []
    
    def fake_fit(self, X, y, *args, **kwargs):
        captured_shapes.append(X.shape)
        return real_fit(self, X, y, *args, **kwargs)

    with patch("train.train.LogisticRegression.fit", fake_fit):
        with patch("train.train.joblib.dump"): # Evita salvar no disco de verdade
            fake_path = os.path.join(tmp_path, "fake_path.joblib")
            treinar_stacking(df_dummy, df_val=df_val, output_path=fake_path)
    
    # Verifica se a regressão logística foi chamada
    assert len(captured_shapes) > 0, "O metamodelo (LogisticRegression) não foi treinado."
    
    X_meta_shape = captured_shapes[0]
    
    # 3. Meta-modelo usa OOF: a matriz X_meta_train deve ter exatas 3 colunas 
    # (probabilidade char, probabilidade word, probabilidade denso) para N amostras.
    assert X_meta_shape == (N, 3), f"Matriz OOF para o metamodelo tem shape incorreto: {X_meta_shape}. Esperado: {(N, 3)}"
