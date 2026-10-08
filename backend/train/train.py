import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import scipy.stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stacking")

# Contrato compartilhado com a API (15 estilo + 88 temas = 103 features)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from api.bundle_spec import ESTILO_FEATURE_NAMES, FEATURE_COUNT, FEATURE_ORDER, validar_bundle  # noqa: E402


def extrair_vetor_k_mais_3(model, X_text_transformed: np.ndarray) -> np.ndarray:
    """
    Calcula as k+3 características temáticas:
    [tema_0 ... tema_{k-1}, entropia, theta_max, ajuste]
    """
    thetas = model.transform(X_text_transformed)
    
    # Normalização L1 para distribuição de probabilidade contínua (soft)
    somas = thetas.sum(axis=1, keepdims=True)
    somas[somas == 0] = 1.0
    thetas_norm = thetas / somas

    # Entropia de Shannon
    eps = 1e-9
    thetas_safe = np.clip(thetas_norm, eps, 1.0)
    entropia = scipy.stats.entropy(thetas_safe, axis=1, base=np.e)

    # Theta_max (peso do tema predominante)
    theta_max = np.max(thetas_norm, axis=1)

    # Ajuste (massa total original de ativação do modelo no texto)
    ajuste = thetas.sum(axis=1)

    return np.hstack([thetas_norm, entropia.reshape(-1, 1), theta_max.reshape(-1, 1), ajuste.reshape(-1, 1)])


def treinar_stacking(
    df_treino: pd.DataFrame,
    df_val: pd.DataFrame = None,
    output_path: str = None,
    version: str = None,
) -> Dict:
    """
    Treina os 3 ramos do Stacking Ensemble e o metamodelo de Regressão Logística.
    """
    logger.info("A iniciar treino do pipeline Stacking Parte C...")

    y_train = df_treino["target"].values

    # -------------------------------------------------------------
    # 1. Ramo SVM Caracteres (TF-IDF de n-gramas 3 a 5 no texto cru)
    # -------------------------------------------------------------
    logger.info("A treinar TF-IDF de Caracteres (3 a 5) + LinearSVC...")
    tfidf_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=50000,
        sublinear_tf=True
    )
    X_char_train = tfidf_char.fit_transform(df_treino["texto_cru"])
    
    base_svm_char = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm_char = CalibratedClassifierCV(estimator=base_svm_char, cv=3)
    svm_char.fit(X_char_train, y_train)

    # -------------------------------------------------------------
    # 2. Ramo SVM Palavras (TF-IDF de 1 e 2 gramas no texto limpo)
    # -------------------------------------------------------------
    logger.info("A treinar TF-IDF de Palavras (1 e 2 gramas) + LinearSVC...")
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
        sublinear_tf=True
    )
    X_word_train = tfidf_word.fit_transform(df_treino["texto_limpo"])

    base_svm_word = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm_word = CalibratedClassifierCV(estimator=base_svm_word, cv=3)
    svm_word.fit(X_word_train, y_train)

    # -------------------------------------------------------------
    # 3. Modelos Temáticos: LDA e NMF (k=8 e k=30) no texto lematizado
    # -------------------------------------------------------------
    logger.info("A ajustar modelos temáticos não supervisionados (LDA e NMF com k=8 e k=30)...")
    tfidf_lemmas = TfidfVectorizer(max_features=10000, min_df=3)
    X_lemmas_train = tfidf_lemmas.fit_transform(df_treino["texto_lematizado"])

    # Modelos com k=8
    lda_8 = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=15, n_jobs=-1)
    lda_8.fit(X_lemmas_train)

    nmf_8 = NMF(n_components=8, random_state=42, max_iter=200)
    nmf_8.fit(X_lemmas_train)

    # Modelos com k=30
    lda_30 = LatentDirichletAllocation(n_components=30, random_state=42, max_iter=15, n_jobs=-1)
    lda_30.fit(X_lemmas_train)

    nmf_30 = NMF(n_components=30, random_state=42, max_iter=200)
    nmf_30.fit(X_lemmas_train)

    v_lda8 = extrair_vetor_k_mais_3(lda_8, X_lemmas_train)
    v_nmf8 = extrair_vetor_k_mais_3(nmf_8, X_lemmas_train)
    v_lda30 = extrair_vetor_k_mais_3(lda_30, X_lemmas_train)
    v_nmf30 = extrair_vetor_k_mais_3(nmf_30, X_lemmas_train)

    # -------------------------------------------------------------
    # 4. Ramo XGBoost Denso: Estilo (15 features) + Temas (k=8 e k=30)
    # -------------------------------------------------------------
    logger.info("A treinar XGBoost Denso (Estilo + Temas)...")
    X_estilo = df_treino[ESTILO_FEATURE_NAMES].values
    X_denso_train = np.hstack([X_estilo, v_lda8, v_nmf8, v_lda30, v_nmf30])
    if X_denso_train.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Vetor denso com {X_denso_train.shape[1]} features, contrato exige {FEATURE_COUNT}")

    xgb_denso = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_denso.fit(X_denso_train, y_train)

    # -------------------------------------------------------------
    # 5. Meta-Modelo: Regressão Logística
    # -------------------------------------------------------------
    logger.info("A ajustar Meta-Modelo de Regressão Logística...")
    p_char = svm_char.predict_proba(X_char_train)[:, 1].reshape(-1, 1)
    p_word = svm_word.predict_proba(X_word_train)[:, 1].reshape(-1, 1)
    p_denso = xgb_denso.predict_proba(X_denso_train)[:, 1].reshape(-1, 1)

    X_meta_train = np.hstack([p_char, p_word, p_denso])

    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    meta_lr.fit(X_meta_train, y_train)

    logger.info(f"Pesos do Meta-Modelo (Char, Word, Denso): {meta_lr.coef_[0]}")

    # -------------------------------------------------------------
    # 6. Avaliação e Verificação do Limiar 0.46
    # -------------------------------------------------------------
    df_eval = df_val if df_val is not None else df_treino
    y_eval = df_eval["target"].values

    X_char_eval = tfidf_char.transform(df_eval["texto_cru"])
    X_word_eval = tfidf_word.transform(df_eval["texto_limpo"])
    X_lem_eval = tfidf_lemmas.transform(df_eval["texto_lematizado"])

    v_lda8_eval = extrair_vetor_k_mais_3(lda_8, X_lem_eval)
    v_nmf8_eval = extrair_vetor_k_mais_3(nmf_8, X_lem_eval)
    v_lda30_eval = extrair_vetor_k_mais_3(lda_30, X_lem_eval)
    v_nmf30_eval = extrair_vetor_k_mais_3(nmf_30, X_lem_eval)

    X_denso_eval = np.hstack([
        df_eval[ESTILO_FEATURE_NAMES].values,
        v_lda8_eval, v_nmf8_eval, v_lda30_eval, v_nmf30_eval
    ])

    p_char_eval = svm_char.predict_proba(X_char_eval)[:, 1].reshape(-1, 1)
    p_word_eval = svm_word.predict_proba(X_word_eval)[:, 1].reshape(-1, 1)
    p_denso_eval = xgb_denso.predict_proba(X_denso_eval)[:, 1].reshape(-1, 1)

    X_meta_eval = np.hstack([p_char_eval, p_word_eval, p_denso_eval])
    p_fake_final = meta_lr.predict_proba(X_meta_eval)[:, 1]

    limiar_decisao = 0.46
    y_pred = (p_fake_final >= limiar_decisao).astype(int)

    f1_obtido = f1_score(y_eval, y_pred, pos_label=1)
    logger.info(f"F1-Score obtido (com limiar {limiar_decisao}): {f1_obtido:.4f}")
    logger.info("\n" + classification_report(y_eval, y_pred, target_names=["Verdadeiro", "Falso"]))

    # -------------------------------------------------------------
    # 7. Empacotamento do Bundle de Produção
    # -------------------------------------------------------------
    bundle = {
        "tfidf_char": tfidf_char,
        "svm_caracteres": svm_char,
        "tfidf_word": tfidf_word,
        "svm_palavras": svm_word,
        "tfidf_lemmas": tfidf_lemmas,
        "lda_8": lda_8,
        "nmf_8": nmf_8,
        "lda_30": lda_30,
        "nmf_30": nmf_30,
        "xgb_denso": xgb_denso,
        "meta_modelo": meta_lr,
        "limiar": limiar_decisao,
        "f1_score": float(f1_obtido),
        "f1_avaliado_em": "validacao" if df_val is not None else "treino",
        "feature_names_estilo": list(ESTILO_FEATURE_NAMES),
        "feature_order": list(FEATURE_ORDER),
        "feature_count": FEATURE_COUNT,
        "version": version or datetime.now(timezone.utc).strftime("stacking-%Y%m%dT%H%M%SZ"),
    }

    # O mesmo contrato verificado no startup da API: nunca serializar bundle inválido.
    validar_bundle(bundle)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(bundle, output_path, compress=3)
        logger.info(f"Bundle serializado com sucesso em: {output_path}")

    return bundle


def carregar_dados_reais(caminho_dataset_11: str, caminho_master: str) -> pd.DataFrame:
    """
    Carrega e une os datasets conforme a arquitetura da Parte A/B:
    Junção por (id_noticia, target) + alinhamento de colunas.
    """
    logger.info("A carregar ficheiros de dados...")
    df_11 = pd.read_csv(caminho_dataset_11)
    df_master = pd.read_csv(caminho_master)

    df_unificado = pd.merge(
        df_11,
        df_master,
        on=["id_noticia", "target"],
        suffixes=("_11", "_master")
    )
    return df_unificado


def gerar_dados_sinteticos_para_teste(n_samples: int = 80) -> pd.DataFrame:
    """Gera um DataFrame mock estruturado para validar a execução técnica."""
    dados = []
    textos_true = [
        "O ministério da fazenda publicou portaria com as novas regras fiscais para os estados.",
        "Pesquisa científica da universidade mapeia os efeitos do clima na agricultura regional.",
        "Dados divulgados pelo instituto apontam redução do índice de desemprego no trimestre."
    ]
    textos_fake = [
        "URGENTE repasse agora mesmo veja o que o governo escondeu de você escândalo confirmado",
        "Bomba caiu na rede o plano secreto que a mídia não divulga compartilhe antes que apaguem",
        "Atenção segredo revelado por fonte anônima tudo vai mudar amanhã repasse já"
    ]

    for i in range(n_samples):
        is_fake = i % 2 == 1
        t_cru = textos_fake[i % len(textos_fake)] if is_fake else textos_true[i % len(textos_true)]
        t_limpo = t_cru.lower()
        t_lem = " ".join([w for w in t_limpo.split() if len(w) > 3])

        row = {
            "texto_cru": t_cru,
            "texto_limpo": t_limpo,
            "texto_lematizado": t_lem,
            "target": 1 if is_fake else 0,
            "trunc_pausality": np.random.uniform(0.1, 0.4),
            "trunc_emotiveness": np.random.uniform(0.3, 0.8) if is_fake else np.random.uniform(0.1, 0.4),
            "trunc_diversity": np.random.uniform(0.6, 0.8),
            "trunc_upper_case_density": np.random.uniform(0.05, 0.2) if is_fake else np.random.uniform(0.0, 0.05),
            "trunc_verb_density": np.random.uniform(0.1, 0.3),
            "trunc_noun_density": np.random.uniform(0.2, 0.5),
            "trunc_adj_density": np.random.uniform(0.05, 0.2),
            "trunc_adv_density": np.random.uniform(0.02, 0.1),
            "trunc_pron_density": np.random.uniform(0.05, 0.15),
            "link_density": 0.0,
            "rc_spelling_errors": np.random.uniform(0.02, 0.1) if is_fake else 0.0,
            "rc_modal_verbs_density": np.random.uniform(0.0, 0.2),
            "rc_subj_imp_verbs_density": np.random.uniform(0.0, 0.2),
            "rc_pron_1_2_sing_density": np.random.uniform(0.02, 0.1) if is_fake else 0.0,
            "rc_pron_1_plur_density": np.random.uniform(0.0, 0.05),
        }
        dados.append(row)

    return pd.DataFrame(dados)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treina o bundle Stacking Miop.IA.")
    parser.add_argument("--dados-dir", default=os.path.join(_BACKEND_DIR, "api", "data"),
                        help="Pasta com dataset_11.csv e fake_br_master.csv")
    parser.add_argument("--saida", default=os.path.join(_BACKEND_DIR, "models", "stacking_miopia_0961.joblib"))
    parser.add_argument("--versao", default=None, help="Identificador de versão gravado no bundle")
    parser.add_argument("--sintetico", action="store_true",
                        help="Treina com dados sintéticos (apenas para testes; exige --saida diferente da produção)")
    args = parser.parse_args()

    saida_producao = os.path.abspath(os.path.join(_BACKEND_DIR, "models", "stacking_miopia_0961.joblib"))

    if args.sintetico:
        if os.path.abspath(args.saida) == saida_producao:
            sys.exit("Recusado: bundle sintético não pode sobrescrever o artefato de produção. Use --saida.")
        logger.warning("Treinando com dados SINTÉTICOS (somente teste).")
        df_mock = gerar_dados_sinteticos_para_teste(n_samples=80)
        treinar_stacking(df_mock, output_path=args.saida, version=args.versao or "sintetico-teste")
    else:
        caminho_11 = os.path.join(args.dados_dir, "dataset_11.csv")
        caminho_master = os.path.join(args.dados_dir, "fake_br_master.csv")
        faltando = [c for c in (caminho_11, caminho_master) if not os.path.exists(c)]
        if faltando:
            sys.exit(f"Bases reais não encontradas: {faltando}. Nenhum modelo foi gerado.")
        df_completo = carregar_dados_reais(caminho_11, caminho_master)
        treinar_stacking(df_completo, output_path=args.saida, version=args.versao)

    logger.info("Pipeline concluído.")
