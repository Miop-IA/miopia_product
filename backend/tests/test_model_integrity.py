import os
import joblib
import json

def test_model_integrity():
    """
    Testa a integridade estrutural e contratual do modelo serializado (bundle)
    e do respectivo manifesto de versão (model_manifest.json).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "stacking_miopia_v2.joblib")
    manifest_path = os.path.join(base_dir, "models", "model_manifest.json")
    
    # 1. Bundle Existe
    assert os.path.exists(model_path), f"Arquivo de modelo não encontrado em: {model_path}"
    assert os.path.exists(manifest_path), f"Manifesto não encontrado em: {manifest_path}"
    
    # 2. Bundle Carrega
    bundle = joblib.load(model_path)
    assert isinstance(bundle, dict), "O bundle deve ser um dicionário"
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    # 3. Chaves Existem
    chaves_esperadas = [
        "tfidf_char", "svm_caracteres", 
        "tfidf_word", "svm_palavras", 
        "tfidf_lemmas", 
        "lda_8", "nmf_8", "lda_30", "nmf_30", 
        "scaler_estilo", "xgb_denso", "meta_modelo", 
        "limiar", "f1_score"
    ]
    for chave in chaves_esperadas:
        assert chave in bundle, f"Chave esperada '{chave}' está ausente no bundle!"
        
    # 4. 103 Features no modelo denso
    xgb_denso = bundle["xgb_denso"]
    assert xgb_denso.n_features_in_ == 103, f"O modelo denso deve ter exatamente 103 features de entrada, obteve: {xgb_denso.n_features_in_}"
    assert manifest["feature_count"] == 103, "O manifesto deve registrar 103 features"
    
    # 5. Meta-modelo possui 3 entradas
    meta_modelo = bundle["meta_modelo"]
    # meta_modelo.coef_ tem formato (n_classes, n_features) ou (1, n_features) para binária
    n_meta_entradas = meta_modelo.coef_.shape[1]
    assert n_meta_entradas == 3, f"O metamodelo deve combinar exatamente 3 modelos (char, word, denso), obteve {n_meta_entradas} entradas"
    
    # 6. Threshold Válido
    limiar = bundle["limiar"]
    assert isinstance(limiar, float), "O limiar de decisão deve ser float"
    assert 0.0 < limiar < 1.0, f"O limiar deve estar entre (0, 1), obteve {limiar}"
    assert manifest["threshold"] == limiar, "A divergência entre limiar do bundle e do manifesto"
    
    # 7. F1 Válido (deve ser alto o suficiente para justificar deploy)
    f1 = bundle["f1_score"]
    assert isinstance(f1, float), "O f1_score deve ser float"
    assert f1 > 0.85, f"O F1 score ({f1}) está abaixo do mínimo exigido de 0.85"
    assert manifest["F1"] == f1, "A divergência entre F1 do bundle e do manifesto"
