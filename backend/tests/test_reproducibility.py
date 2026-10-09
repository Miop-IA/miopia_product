import pytest
import os
import numpy as np

from train.train import treinar_stacking, gerar_dados_sinteticos_para_teste, TrainingConfig

def test_training_reproducibility():
    """
    Treina dois bundles consecutivamente com a mesma configuração e mesmos dados,
    e valida que as saídas e pesos do modelo são idênticos dentro de tolerância numérica,
    provando o determinismo estrito do treinamento da Fase 8.
    """
    # 1. Dados e Configuração
    df_completo = gerar_dados_sinteticos_para_teste(n_samples=50)
    
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_completo, groups=df_completo["grupo_identidade"]))
    df_treino = df_completo.iloc[train_idx]
    df_teste = df_completo.iloc[test_idx]
    
    config = TrainingConfig(
        random_state=42,
        n_splits_oof=3,     # Menor para acelerar teste
        n_splits_calib=2,   # Menor para acelerar teste
        xgb_n_estimators=10, 
        xgb_max_depth=3,
        lda_max_iter=3,
        nmf_max_iter=3,
        svm_max_iter=100,
        is_official_run=False
    )
    
    # 2. Executar primeiro treinamento
    bundle_a = treinar_stacking(
        df_treino=df_treino, 
        df_val=df_teste, 
        output_path=None, 
        config=config
    )
    
    # 3. Executar segundo treinamento
    bundle_b = treinar_stacking(
        df_treino=df_treino, 
        df_val=df_teste, 
        output_path=None, 
        config=config
    )
    
    # 4. Comparações estritas de determinismo
    
    # 4.1 Composição do bundle
    assert bundle_a["feature_count"] == bundle_b["feature_count"]
    assert bundle_a["limiar"] == bundle_b["limiar"]
    assert np.isclose(bundle_a["f1_score"], bundle_b["f1_score"], atol=1e-5)
    
    # 4.2 Pesos do Metamodelo (Linear)
    coef_a = bundle_a["meta_modelo"].coef_
    coef_b = bundle_b["meta_modelo"].coef_
    np.testing.assert_allclose(coef_a, coef_b, rtol=1e-5, atol=1e-5, err_msg="Os pesos do metamodelo divergem!")
    
    # 4.3 Previsões do Ramo Denso sobre a mesma amostra fixa
    X_denso_dummy = np.random.RandomState(42).rand(5, bundle_a["feature_count"]).astype(np.float32)
    
    pred_a = bundle_a["xgb_denso"].predict_proba(X_denso_dummy)[:, 1]
    pred_b = bundle_b["xgb_denso"].predict_proba(X_denso_dummy)[:, 1]
    
    np.testing.assert_allclose(pred_a, pred_b, rtol=1e-5, atol=1e-5, err_msg="As previsões do ramo XGBoost divergem!")
