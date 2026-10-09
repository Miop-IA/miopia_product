import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.model_selection import StratifiedGroupKFold

# Importa o módulo de treino para podermos testar o fluxo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train import treinar_stacking
from api.feature_contract import ESTILO_FEATURE_NAMES

def test_garantia_kfold_sem_vazamento():
    """
    Testa rigorosamente a teoria e prática do StratifiedGroupKFold usado no OOF.
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

    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
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
    Regressão Logística (Metamodelo) usa cross_val_predict para achar o limiar OOF independentemente.
    """
    N = 30
    df_dummy = pd.DataFrame({
        "id_noticia": list(range(1, N+1)),
        "target": [i % 2 for i in range(N)],
        "texto_cru": ["Texto dummy cruel para teste."] * N,
        "texto_limpo": ["texto dummy limpo"] * N,
        "texto_lematizado": ["texto dummy lema"] * N,
        "grupo_identidade": [f"grupo_{i}" for i in range(N)]
    })

    df_val = df_dummy.copy()
    df_val["id_noticia"] = list(range(101, 101+N))
    df_val["grupo_identidade"] = [f"grupo_{i}" for i in range(101, 101+N)]
    for f in ESTILO_FEATURE_NAMES:
        df_dummy[f] = np.random.rand(N)
        df_val[f] = np.random.rand(N)
    
    captured_cvp_shapes = []
    
    def fake_cvp(estimator, X, y=None, **kwargs):
        captured_cvp_shapes.append(X.shape)
        # return dummy probabilities so pipeline doesn't crash
        import numpy as np
        return np.random.rand(X.shape[0], 2)

    def fake_dump(bundle, path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"dummy")

    with patch("sklearn.model_selection.cross_val_predict", side_effect=fake_cvp):
        with patch("train.train.joblib.dump", side_effect=fake_dump):
            fake_path = os.path.join(tmp_path, "fake_path.joblib")
            treinar_stacking(df_dummy, df_val=df_val, output_path=fake_path)
    
    assert len(captured_cvp_shapes) > 0, "O metamodelo não utilizou cross_val_predict."
    
    X_meta_shape = captured_cvp_shapes[0]
    
    # 3. Meta-modelo usa OOF: a matriz X_meta_train deve ter exatas 3 colunas 
    assert X_meta_shape == (N, 3), f"Matriz OOF para o metamodelo tem shape incorreto: {X_meta_shape}. Esperado: {(N, 3)}"
