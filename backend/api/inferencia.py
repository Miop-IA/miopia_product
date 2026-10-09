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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "stacking_miopia_v4_candidato.joblib"
)

_TEXTO_SMOKE = "verificacao de integridade do modelo de producao durante a inicializacao da api"


def caminho_modelo() -> str:
    """Caminho do bundle: variável MODEL_PATH (via settings) ou o artefato padrão em backend/models."""
    from api.config import get_settings

    return get_settings().model_path or CAMINHO_PADRAO_MODELO


def carregar_bundle_stacking(caminho: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega e valida o bundle do Stacking. Nunca cria modelo substituto:
    - arquivo ausente            -> ModeloAusenteError
    - arquivo corrompido         -> ModeloCorrompidoError
    - chave/dimensão/classe/meta -> BundleIncompativelError
    O bundle só é colocado em cache depois de passar por todas as validações.
    """
    global _stacking_bundle
    if _stacking_bundle is not None and caminho is None:
        return _stacking_bundle

    model_path = caminho or caminho_modelo()

    if not os.path.exists(model_path):
        raise ModeloAusenteError(f"Artefato do modelo não encontrado: {model_path}")

    try:
        bundle = joblib.load(model_path)
    except Exception as e:
        raise ModeloCorrompidoError(
            f"Falha ao desserializar o artefato do modelo ({model_path}): {type(e).__name__}: {e}"
        ) from e

    validar_bundle(bundle)
    _verificar_manifesto(bundle, model_path)
    _verificar_execucao(bundle)

    logger.info("Bundle Stacking validado: %s", resumo_bundle(bundle))
    if caminho is None:
        _stacking_bundle = bundle
    return bundle


def _verificar_manifesto(bundle: Dict[str, Any], model_path: str) -> None:
    """O model_manifest.json ao lado do .joblib precisa existir e bater com o bundle."""
    import json
    import hashlib

    manifest_path = os.path.join(os.path.dirname(model_path), "model_manifest.json")
    if not os.path.exists(manifest_path):
        raise BundleIncompativelError([f"manifesto ausente: {manifest_path}"])
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    problemas = []
    
    # 1. Validação do SHA-256 (Fase 7)
    manifest_hash = manifest.get("model_hash_sha256")
    if not manifest_hash:
        problemas.append("manifesto inconsistente: model_hash_sha256 ausente")
    else:
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        real_hash = sha256_hash.hexdigest()
        if real_hash != manifest_hash:
            problemas.append(f"manifesto inconsistente: hash real ({real_hash}) diverge do manifesto ({manifest_hash})")

    # 2. Compatibilidade Bundle vs Manifesto
    if manifest.get("model_version") != bundle.get("version"):
        problemas.append("manifesto inconsistente: model_version diverge do bundle")
    if manifest.get("pipeline_version") != bundle.get("pipeline_version"):
        problemas.append("manifesto inconsistente: pipeline_version diverge do bundle")
    
    if manifest.get("feature_count") != FEATURE_COUNT:
        problemas.append("manifesto inconsistente: feature_count inválido")
    if abs(float(manifest.get("threshold", -1.0)) - float(bundle["limiar"])) > 1e-4:
        problemas.append("manifesto inconsistente: divergência no threshold")
    if abs(float(manifest.get("F1", -1.0)) - float(bundle["f1_score"])) > 1e-4:
        problemas.append("manifesto inconsistente: divergência no F1")
        
    if problemas:
        raise BundleIncompativelError(problemas)


def _verificar_execucao(bundle: Dict[str, Any]) -> None:
    """Executa uma inferência sintética ponta a ponta para garantir que os artefatos funcionam juntos."""
    textos = {"texto_cru": _TEXTO_SMOKE, "texto_limpo": _TEXTO_SMOKE, "texto_lematizado": _TEXTO_SMOKE}
    features = {nome: 0.0 for nome in ESTILO_FEATURE_NAMES}
    try:
        prob = _inferir(bundle, features, textos)
    except ModeloInvalidoError:
        raise
    except Exception as e:
        raise BundleIncompativelError([f"inferência de verificação falhou: {type(e).__name__}: {e}"]) from e
    if not (0.0 <= prob <= 1.0) or not np.isfinite(prob):
        raise BundleIncompativelError([f"inferência de verificação retornou probabilidade inválida: {prob!r}"])


def resetar_cache_bundle() -> None:
    """Descarta o bundle em memória (usado em testes e recarga controlada)."""
    global _stacking_bundle
    _stacking_bundle = None


def extrair_features_topicos(texto_lematizado: str, bundle: Dict[str, Any]) -> np.ndarray:
    """
    Extrai as 88 features temáticas (k+3 por modelo: distribuição soft, entropia, theta_max, ajuste)
    na ordem LDA8, NMF8, LDA30, NMF30. Sem fallback: modelo ausente é erro.
    """
    faltando = [chave for chave, _, _ in BLOCOS_TEMATICOS if chave not in bundle]
    if "tfidf_lemmas" not in bundle:
        faltando.append("tfidf_lemmas")
    if faltando:
        raise BundleIncompativelError([f"modelos temáticos ausentes: {', '.join(faltando)}"])

    vec_lemmas = bundle["tfidf_lemmas"].transform([texto_lematizado])
    vecs = []
    for chave, _, k in BLOCOS_TEMATICOS:
        theta = bundle[chave].transform(vec_lemmas)[0]
        if theta.shape[0] != k:
            raise BundleIncompativelError([f"{chave} retornou {theta.shape[0]} temas, esperado {k}"])
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
        raise KeyError(f"features estilométricas ausentes: {', '.join(faltando)}")
    vetor_estilo_raw = np.array([[features_estilo[f] for f in ESTILO_FEATURE_NAMES]], dtype=np.float64)
    vetor_estilo = bundle["scaler_estilo"].transform(vetor_estilo_raw)[0].astype(np.float32)
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

    limiar = float(bundle["limiar"])
    f1_score_ref = float(bundle["f1_score"])

    faixa, orientacao = classificar_faixa_e_orientacao(prob_fake_final, limiar, f1_score_ref)

    return round(prob_fake_final, 3), faixa, orientacao, f1_score_ref
