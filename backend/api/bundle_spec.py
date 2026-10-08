"""
Contrato do bundle de produção do Stacking Miop.IA.

Fonte única da verdade para:
- chaves obrigatórias do bundle serializado;
- ordem e quantidade de features do ramo XGBoost denso (103);
- validação estrutural e dimensional executada no startup da API.

Usado pela API (validação no carregamento) e pelo script de treino
(geração dos metadados `feature_order` / `feature_count`).
"""
import math
from typing import Any, Dict, List

ESTILO_FEATURE_NAMES: List[str] = [
    "trunc_pausality",
    "trunc_emotiveness",
    "trunc_diversity",
    "trunc_upper_case_density",
    "trunc_verb_density",
    "trunc_noun_density",
    "trunc_adj_density",
    "trunc_adv_density",
    "trunc_pron_density",
    "link_density",
    "rc_spelling_errors",
    "rc_modal_verbs_density",
    "rc_subj_imp_verbs_density",
    "rc_pron_1_2_sing_density",
    "rc_pron_1_plur_density",
]

# (chave no bundle, prefixo da feature, k) — na ordem em que entram no vetor denso.
BLOCOS_TEMATICOS = [
    ("lda_8", "lda8", 8),
    ("nmf_8", "nmf8", 8),
    ("lda_30", "lda30", 30),
    ("nmf_30", "nmf30", 30),
]

# Cada bloco temático gera k + 3 features: distribuição soft, entropia, theta_max, ajuste.
FEATURES_EXTRAS_POR_BLOCO = ("entropia", "theta_max", "ajuste")


def construir_feature_order() -> List[str]:
    """Ordem canônica das features do XGBoost denso: 15 + 11 + 11 + 33 + 33 = 103."""
    ordem = list(ESTILO_FEATURE_NAMES)
    for _, prefixo, k in BLOCOS_TEMATICOS:
        ordem.extend(f"{prefixo}_tema_{i}" for i in range(k))
        ordem.extend(f"{prefixo}_{extra}" for extra in FEATURES_EXTRAS_POR_BLOCO)
    return ordem


FEATURE_ORDER: List[str] = construir_feature_order()
N_FEATURES_ESTILO = len(ESTILO_FEATURE_NAMES)  # 15
N_FEATURES_TEMAS = sum(k + len(FEATURES_EXTRAS_POR_BLOCO) for _, _, k in BLOCOS_TEMATICOS)  # 88
FEATURE_COUNT = N_FEATURES_ESTILO + N_FEATURES_TEMAS  # 103

assert FEATURE_COUNT == 103 == len(FEATURE_ORDER), "Contrato de features inconsistente"

N_FEATURES_META = 3  # p_char, p_word, p_denso
CLASSES_ESPERADAS = [0, 1]

CHAVES_OBRIGATORIAS = (
    "tfidf_char",
    "svm_caracteres",
    "tfidf_word",
    "svm_palavras",
    "tfidf_lemmas",
    "lda_8",
    "nmf_8",
    "lda_30",
    "nmf_30",
    "scaler_estilo",
    "xgb_denso",
    "meta_modelo",
    "limiar",
    "f1_score",
    "version",
    "feature_order",
    "feature_count",
    "feature_names_estilo",
)


class ModeloInvalidoError(RuntimeError):
    """Base: o modelo de produção não pode ser usado. A API não deve servir classificações."""


class ModeloAusenteError(ModeloInvalidoError):
    """O arquivo do bundle não existe."""


class ModeloCorrompidoError(ModeloInvalidoError):
    """O arquivo existe mas não pôde ser desserializado."""


class BundleIncompativelError(ModeloInvalidoError):
    """O bundle foi carregado mas viola o contrato (chaves, dimensões, classes, metadados)."""

    def __init__(self, problemas: List[str]):
        self.problemas = problemas
        super().__init__(
            "Bundle do modelo incompatível com o contrato de produção:\n- " + "\n- ".join(problemas)
        )


def _numero_real(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(float(valor))


def _classes(modelo: Any) -> Any:
    classes = getattr(modelo, "classes_", None)
    return None if classes is None else [int(c) for c in list(classes)]


def _tamanho_vocabulario(vetorizador: Any) -> Any:
    vocab = getattr(vetorizador, "vocabulary_", None)
    return None if vocab is None else len(vocab)


def validar_bundle(bundle: Any) -> None:
    """
    Valida estrutura e dimensionalidade do bundle. Levanta BundleIncompativelError
    listando todos os problemas encontrados. Não corrige nem preenche nada.
    """
    if not isinstance(bundle, dict):
        raise BundleIncompativelError([f"bundle deve ser dict, recebido {type(bundle).__name__}"])

    faltando = [c for c in CHAVES_OBRIGATORIAS if c not in bundle]
    if faltando:
        raise BundleIncompativelError([f"chaves obrigatórias ausentes: {', '.join(faltando)}"])

    problemas: List[str] = []

    # --- Metadados ---------------------------------------------------------
    versao = bundle["version"]
    if not isinstance(versao, str) or not versao.strip():
        problemas.append(f"version deve ser string não vazia, recebido {versao!r}")

    limiar = bundle["limiar"]
    if not _numero_real(limiar) or not (0.0 < float(limiar) < 1.0):
        problemas.append(f"limiar (threshold) deve ser número em (0, 1), recebido {limiar!r}")

    f1 = bundle["f1_score"]
    if not _numero_real(f1) or not (0.0 < float(f1) <= 1.0):
        problemas.append(f"f1_score deve ser número em (0, 1], recebido {f1!r}")

    if list(bundle["feature_names_estilo"]) != ESTILO_FEATURE_NAMES:
        problemas.append("feature_names_estilo diverge das 15 features estilométricas da API")

    # --- Ordem e quantidade de features -----------------------------------
    feature_count = bundle["feature_count"]
    if not isinstance(feature_count, int) or isinstance(feature_count, bool) or feature_count != FEATURE_COUNT:
        problemas.append(f"feature_count deve ser {FEATURE_COUNT}, recebido {feature_count!r}")

    feature_order = bundle["feature_order"]
    if not isinstance(feature_order, (list, tuple)):
        problemas.append("feature_order deve ser lista de nomes de features")
    else:
        if len(feature_order) != FEATURE_COUNT:
            problemas.append(f"feature_order tem {len(feature_order)} itens, esperado {FEATURE_COUNT}")
        elif list(feature_order) != FEATURE_ORDER:
            divergente = next(i for i, (a, b) in enumerate(zip(feature_order, FEATURE_ORDER)) if a != b)
            problemas.append(
                f"feature_order diverge na posição {divergente}: "
                f"bundle={feature_order[divergente]!r}, esperado={FEATURE_ORDER[divergente]!r}"
            )

    # --- Ramos esparsos (SVM) ---------------------------------------------
    for chave_vec, chave_clf in (("tfidf_char", "svm_caracteres"), ("tfidf_word", "svm_palavras")):
        n_vocab = _tamanho_vocabulario(bundle[chave_vec])
        if n_vocab is None:
            problemas.append(f"{chave_vec} não está ajustado (sem vocabulary_)")
            continue
        n_in = getattr(bundle[chave_clf], "n_features_in_", None)
        if n_in != n_vocab:
            problemas.append(f"{chave_clf} espera {n_in} features, mas {chave_vec} produz {n_vocab}")

    # --- Modelos temáticos -------------------------------------------------
    n_lemas = _tamanho_vocabulario(bundle["tfidf_lemmas"])
    if n_lemas is None:
        problemas.append("tfidf_lemmas não está ajustado (sem vocabulary_)")
    for chave, _, k in BLOCOS_TEMATICOS:
        modelo = bundle[chave]
        n_comp = getattr(modelo, "n_components", None)
        if n_comp != k:
            problemas.append(f"{chave} deve ter n_components={k}, recebido {n_comp!r}")
        n_in = getattr(modelo, "n_features_in_", None)
        if n_lemas is not None and n_in != n_lemas:
            problemas.append(f"{chave} espera {n_in} features, mas tfidf_lemmas produz {n_lemas}")

    # --- Normalizador das features estilométricas -------------------------
    n_scaler = getattr(bundle["scaler_estilo"], "n_features_in_", None)
    if n_scaler != len(ESTILO_FEATURE_NAMES):
        problemas.append(
            f"scaler_estilo espera {n_scaler} features, esperado {len(ESTILO_FEATURE_NAMES)}"
        )

    # --- XGBoost denso -----------------------------------------------------
    n_xgb = getattr(bundle["xgb_denso"], "n_features_in_", None)
    if n_xgb != FEATURE_COUNT:
        problemas.append(
            f"xgb_denso espera {n_xgb} features, contrato exige {FEATURE_COUNT} "
            f"(15 estilo + 11 LDA8 + 11 NMF8 + 33 LDA30 + 33 NMF30)"
        )

    # --- Meta-modelo -------------------------------------------------------
    n_meta = getattr(bundle["meta_modelo"], "n_features_in_", None)
    if n_meta != N_FEATURES_META:
        problemas.append(f"meta_modelo espera {n_meta} features, esperado {N_FEATURES_META}")

    # --- Classes -----------------------------------------------------------
    for chave in ("svm_caracteres", "svm_palavras", "xgb_denso", "meta_modelo"):
        classes = _classes(bundle[chave])
        if classes != CLASSES_ESPERADAS:
            problemas.append(f"{chave} deve ter classes {CLASSES_ESPERADAS}, recebido {classes!r}")

    if problemas:
        raise BundleIncompativelError(problemas)


def resumo_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Metadados seguros para log e /health."""
    return {
        "version": bundle["version"],
        "f1_score": float(bundle["f1_score"]),
        "limiar": float(bundle["limiar"]),
        "feature_count": int(bundle["feature_count"]),
    }
