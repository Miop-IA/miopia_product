import os
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.stats
import joblib

logger = logging.getLogger(__name__)

from api.bundle_spec import (
    BLOCOS_TEMATICOS,
    ESTILO_FEATURE_NAMES,
    FEATURE_COUNT,
    N_FEATURES_TEMAS,
    BundleIncompativelError,
    ModeloAusenteError,
    ModeloCorrompidoError,
    ModeloInvalidoError,
    resumo_bundle,
    validar_bundle,
)

_stacking_bundle = None

CAMINHO_PADRAO_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stacking_miopia_0961.joblib"
)

_TEXTO_SMOKE = "verificacao de integridade do modelo de producao durante a inicializacao da api"


def caminho_modelo() -> str:
    """Caminho do bundle: variável MODEL_PATH (via settings) ou o artefato padrão em backend/models."""
    from api.config import get_settings

    return get_settings().model_path or CAMINHO_PADRAO_MODELO


def carregar_bundle_stacking(caminho: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega o pacote de artefatos do Stacking:
    - tfidf_char + svm_caracteres
    - tfidf_word + svm_palavras
    - lda_8, nmf_8, lda_30, nmf_30
    - xgb_denso (estilo + temas)
    - meta_modelo (Regressão Logística com corte dinâmico)
    """
    global _stacking_bundle
    if _stacking_bundle is not None and caminho is None:
        return _stacking_bundle

    model_path = caminho or caminho_modelo()

    if os.path.exists(model_path):
        try:
            _stacking_bundle = joblib.load(model_path)
            
            # Validação do Bundle Incompleto
            required_keys = [
                "tfidf_char", "svm_caracteres", 
                "tfidf_word", "svm_palavras", 
                "tfidf_lemmas", "lda_8", "nmf_8", "lda_30", "nmf_30", 
                "scaler_estilo", "xgb_denso", "meta_modelo", "limiar", "f1_score"
            ]
            for k in required_keys:
                if k not in _stacking_bundle:
                    raise RuntimeError(f"Bundle corrompido: chave obrigatória '{k}' ausente.")
                    
            # Validação Dimensional Completa
            if hasattr(_stacking_bundle["xgb_denso"], "n_features_in_") and _stacking_bundle["xgb_denso"].n_features_in_ != 103:
                raise RuntimeError("Dimensão errada: O modelo denso não possui 103 features.")
                
            meta_modelo = _stacking_bundle["meta_modelo"]
            n_meta = getattr(meta_modelo, "n_features_in_", meta_modelo.coef_.shape[1])
            if n_meta != 3:
                raise RuntimeError(f"Dimensão errada: O metamodelo deve ter 3 entradas, obteve {n_meta}.")
                
            for nome_modelo, n_topicos in [("lda_8", 8), ("nmf_8", 8), ("lda_30", 30), ("nmf_30", 30)]:
                if _stacking_bundle[nome_modelo].n_components != n_topicos:
                    raise RuntimeError(f"Dimensão errada: {nome_modelo} não possui {n_topicos} tópicos.")
                    
            vocab_size = len(_stacking_bundle["tfidf_lemmas"].vocabulary_)
            if hasattr(_stacking_bundle["lda_8"], "n_features_in_") and _stacking_bundle["lda_8"].n_features_in_ != vocab_size:
                raise RuntimeError("Incompatibilidade: tfidf_lemmas não alinhado com os modelos de tópicos.")
                
            # Validação do Manifesto
            manifest_path = os.path.join(base_dir, "models", "model_manifest.json")
            if not os.path.exists(manifest_path):
                raise RuntimeError("Manifesto ausente.")
                
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                
            # Manifesto Inconsistente
            if "feature_count" not in manifest or manifest["feature_count"] != 103:
                raise RuntimeError("Manifesto inconsistente: feature_count inválido.")
            if abs(manifest.get("threshold", 0.0) - _stacking_bundle["limiar"]) > 1e-4:
                raise RuntimeError("Manifesto inconsistente: divergência no threshold.")
            if abs(manifest.get("F1", 0.0) - _stacking_bundle["f1_score"]) > 1e-4:
                raise RuntimeError("Manifesto inconsistente: divergência no F1.")
                
            logger.info("Pipeline Stacking carregado e validado com sucesso.")
            return _stacking_bundle
        except Exception as e:
            _stacking_bundle = None
            logger.error(f"Erro ao deserializar ou validar o bundle do Stacking: {e}")
            raise RuntimeError(f"Erro ao carregar o modelo Stacking: {e}")

    error_msg = f"Artefato '{model_path}' não encontrado. O sistema não pode inicializar sem o modelo de inferência."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def extrair_features_topicos(texto_lematizado: str, bundle: Dict[str, Any]) -> np.ndarray:
    """Extrai features temáticas contínuas (k+3: distribuição soft, entropia, theta_max, ajuste)."""
    vec_lemmas = bundle["tfidf_lemmas"].transform([texto_lematizado])
    vecs = []
    for model in [bundle["lda_8"], bundle["nmf_8"], bundle["lda_30"], bundle["nmf_30"]]:
        theta = model.transform(vec_lemmas)[0]
        soma = float(np.sum(theta))
        theta_norm = theta / soma if soma > 0 else theta
        eps = 1e-9
        theta_safe = np.clip(theta_norm, eps, 1.0)
        entropia = float(scipy.stats.entropy(theta_safe))
        theta_max = float(np.max(theta_norm))
        ajuste = float(soma)
        vecs.extend(list(theta_norm) + [entropia, theta_max, ajuste])
    return np.array(vecs, dtype=np.float32)


def classificar_faixa_e_orientacao(prob_fake: float, limiar: float, f1_score: float = None) -> Tuple[str, str]:
    """Mapeia a probabilidade final para as faixas de confiança com corte centrado no limiar."""
    if prob_fake < (limiar - 0.15):
        faixa = "Confiavel"
        orientacao = (
            "Este conteúdo apresenta padrões textuais condizentes com o jornalismo profissional. "
            "Ainda assim, verifique a autoria e a data original da publicação."
        )
    elif prob_fake <= (limiar + 0.15):
        faixa = "Atencao"
        orientacao = (
            "O texto apresenta características híbridas ou traços de subjetividade elevada. "
            "Recomenda-se cautela antes de compartilhar e checagem dos fatos em outros veículos."
        )
    else:
        faixa = "Suspeita"
        f1_str = f" (F1={f1_score:.3f})" if f1_score else ""
        orientacao = (
            "Foram detectadas anomalias estilométricas e alta probabilidade de desinformação "
            f"pelo modelo Stacking{f1_str}. Consulte agências de checagem confiáveis."
        )

    return faixa, orientacao


def _inferir(bundle: Dict[str, Any], features_estilo: Dict[str, float], textos: Dict[str, str]) -> float:
    """Executa os três ramos e o metamodelo. Retorna a probabilidade final de 'fake'."""
    # 1. Ramo Caracteres
    X_char = bundle["tfidf_char"].transform([textos["texto_cru"]])
    p_fake_char = float(bundle["svm_caracteres"].predict_proba(X_char)[0][1])

    # 2. Ramo Palavras
    X_word = bundle["tfidf_word"].transform([textos["texto_limpo"]])
    p_fake_word = float(bundle["svm_palavras"].predict_proba(X_word)[0][1])

    # 3. Ramo XGBoost Denso: 15 estilo + 88 temas = 103
    faltando = [f for f in ESTILO_FEATURE_NAMES if f not in features_estilo]
    if faltando:
        raise ValueError(f"features estilométricas ausentes: {', '.join(faltando)}")
    vetor_estilo = np.array([features_estilo[f] for f in ESTILO_FEATURE_NAMES], dtype=np.float32)
    vetor_temas = extrair_features_topicos(textos["texto_lematizado"], bundle)
    X_denso = np.concatenate([vetor_estilo, vetor_temas]).reshape(1, -1)

    n_esperado = int(bundle["feature_count"])
    if X_denso.shape[1] != n_esperado or X_denso.shape[1] != FEATURE_COUNT:
        raise BundleIncompativelError(
            [f"vetor denso com {X_denso.shape[1]} features, bundle/contrato exigem {n_esperado}/{FEATURE_COUNT}"]
        )
    p_fake_denso = float(bundle["xgb_denso"].predict_proba(X_denso)[0][1])

    # 4. Metamodelo
    X_meta = np.array([[p_fake_char, p_fake_word, p_fake_denso]])
    return float(bundle["meta_modelo"].predict_proba(X_meta)[0][1])


def predizer_risco_stacking(
    features_estilo: Dict[str, float],
    textos: Dict[str, str]
) -> Tuple[float, str, str, float]:
    """
    Inferência completa do Stacking Ensemble:
    1. svm_caracteres: TF-IDF (char 3-5) no texto cru
    2. svm_palavras: TF-IDF (1-2 gramas) no texto limpo
    3. xgb_denso: Estilo (15 features) + Temas LDA/NMF (k=8 e k=30)
    4. Meta-modelo: Regressão Logística com limiar dinâmico do bundle
    """
    bundle = carregar_bundle_stacking()
    prob_fake_final = _inferir(bundle, features_estilo, textos)

    # 1. Ramo Caracteres
    vec_char = bundle["tfidf_char"]
    svm_char = bundle["svm_caracteres"]
    X_char = vec_char.transform([textos["texto_cru"]])
    p_fake_char = float(svm_char.predict_proba(X_char)[0][1])

    # 2. Ramo Palavras
    vec_word = bundle["tfidf_word"]
    svm_word = bundle["svm_palavras"]
    X_word = vec_word.transform([textos["texto_limpo"]])
    p_fake_word = float(svm_word.predict_proba(X_word)[0][1])

    # 3. Ramo XGBoost Denso
    vetor_estilo_raw = [features_estilo[f] for f in ESTILO_FEATURE_NAMES]
    scaler_estilo = bundle["scaler_estilo"]
    vetor_estilo = scaler_estilo.transform([vetor_estilo_raw])[0]
        
    vetor_temas = extrair_features_topicos(textos["texto_lematizado"], bundle)
    X_denso = np.concatenate([vetor_estilo, vetor_temas]).reshape(1, -1)

    xgb_model = bundle["xgb_denso"]
    p_fake_denso = float(xgb_model.predict_proba(X_denso)[0][1])

    # 4. Metamodelo
    meta_model = bundle["meta_modelo"]
    X_meta = np.array([[p_fake_char, p_fake_word, p_fake_denso]])
    prob_fake_final = float(meta_model.predict_proba(X_meta)[0][1])

    limiar = bundle.get("limiar", 0.46)
    f1_score_ref = bundle.get("f1_score", 0.0)

    faixa, orientacao = classificar_faixa_e_orientacao(prob_fake_final, limiar, f1_score_ref)

    return round(prob_fake_final, 3), faixa, orientacao, f1_score_ref
