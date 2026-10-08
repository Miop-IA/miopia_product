import os
import sys
import logging
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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stacking")

# 15 características estilométricas definidas via Contrato Oficial Unificado
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.feature_contract import ESTILO_FEATURE_NAMES


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
    output_path: str = None
) -> Dict:
    """
    Treina os 3 ramos do Stacking Ensemble e o metamodelo de Regressão Logística
    usando previsões Out-of-Fold (OOF) baseadas no GroupKFold por id_noticia.
    """
    logger.info("A iniciar treino do pipeline Stacking Parte C...")
    y_train = df_treino["target"].values
    groups_train = df_treino["id_noticia"].values

    # -------------------------------------------------------------
    # 0. Geração de Predições Out-of-Fold (OOF)
    # -------------------------------------------------------------
    logger.info("Gerando predições OOF com GroupKFold (5 splits)...")
    gkf = GroupKFold(n_splits=5)
    
    p_char_oof = np.zeros(len(df_treino))
    p_word_oof = np.zeros(len(df_treino))
    p_denso_oof = np.zeros(len(df_treino))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_treino, y_train, groups=groups_train)):
        logger.info(f"Processando fold {fold + 1}/5...")
        df_fold_train = df_treino.iloc[train_idx]
        df_fold_val = df_treino.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        
        # Char
        tfidf_char_fold = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=50000, sublinear_tf=True)
        X_char_fold_train = tfidf_char_fold.fit_transform(df_fold_train["texto_cru"])
        X_char_fold_val = tfidf_char_fold.transform(df_fold_val["texto_cru"])
        # Usa GroupKFold com 3 splits para calibração interna respeitando id_noticia
        cv_char_fold = list(GroupKFold(n_splits=3).split(X_char_fold_train, y_fold_train, groups=df_fold_train["id_noticia"]))
        svm_char_fold = CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=cv_char_fold)
        svm_char_fold.fit(X_char_fold_train, y_fold_train)
        p_char_oof[val_idx] = svm_char_fold.predict_proba(X_char_fold_val)[:, 1]
        
        # Word
        tfidf_word_fold = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
        X_word_fold_train = tfidf_word_fold.fit_transform(df_fold_train["texto_limpo"])
        X_word_fold_val = tfidf_word_fold.transform(df_fold_val["texto_limpo"])
        # Usa GroupKFold com 3 splits para calibração interna respeitando id_noticia
        cv_word_fold = list(GroupKFold(n_splits=3).split(X_word_fold_train, y_fold_train, groups=df_fold_train["id_noticia"]))
        svm_word_fold = CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=cv_word_fold)
        svm_word_fold.fit(X_word_fold_train, y_fold_train)
        p_word_oof[val_idx] = svm_word_fold.predict_proba(X_word_fold_val)[:, 1]
        
        # Denso
        tfidf_lemmas_fold = TfidfVectorizer(max_features=10000, min_df=3)
        X_lemmas_fold_train = tfidf_lemmas_fold.fit_transform(df_fold_train["texto_lematizado"])
        X_lemmas_fold_val = tfidf_lemmas_fold.transform(df_fold_val["texto_lematizado"])
        
        lda_8_fold = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=15, n_jobs=-1).fit(X_lemmas_fold_train)
        nmf_8_fold = NMF(n_components=8, random_state=42, max_iter=200).fit(X_lemmas_fold_train)
        lda_30_fold = LatentDirichletAllocation(n_components=30, random_state=42, max_iter=15, n_jobs=-1).fit(X_lemmas_fold_train)
        nmf_30_fold = NMF(n_components=30, random_state=42, max_iter=200).fit(X_lemmas_fold_train)
        
        scaler_estilo_fold = StandardScaler()
        X_estilo_fold_train = scaler_estilo_fold.fit_transform(df_fold_train[ESTILO_FEATURE_NAMES].values)
        X_denso_fold_train = np.hstack([
            X_estilo_fold_train,
            extrair_vetor_k_mais_3(lda_8_fold, X_lemmas_fold_train),
            extrair_vetor_k_mais_3(nmf_8_fold, X_lemmas_fold_train),
            extrair_vetor_k_mais_3(lda_30_fold, X_lemmas_fold_train),
            extrair_vetor_k_mais_3(nmf_30_fold, X_lemmas_fold_train)
        ])
        
        xgb_fold = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
        xgb_fold.fit(X_denso_fold_train, y_fold_train)
        
        X_estilo_fold_val = scaler_estilo_fold.transform(df_fold_val[ESTILO_FEATURE_NAMES].values)
        X_denso_fold_val = np.hstack([
            X_estilo_fold_val,
            extrair_vetor_k_mais_3(lda_8_fold, X_lemmas_fold_val),
            extrair_vetor_k_mais_3(nmf_8_fold, X_lemmas_fold_val),
            extrair_vetor_k_mais_3(lda_30_fold, X_lemmas_fold_val),
            extrair_vetor_k_mais_3(nmf_30_fold, X_lemmas_fold_val)
        ])
        p_denso_oof[val_idx] = xgb_fold.predict_proba(X_denso_fold_val)[:, 1]

    # -------------------------------------------------------------
    # 1. Treinamento Final dos Modelos-Base em Todo o Conjunto
    # -------------------------------------------------------------
    logger.info("Treinando modelos-base finais em todo o conjunto de desenvolvimento...")
    tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=50000, sublinear_tf=True)
    X_char_train = tfidf_char.fit_transform(df_treino["texto_cru"])
    base_svm_char = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    # Calibração explícita group-aware para o modelo final
    cv_char_final = list(GroupKFold(n_splits=3).split(X_char_train, y_train, groups=df_treino["id_noticia"]))
    svm_char = CalibratedClassifierCV(estimator=base_svm_char, cv=cv_char_final)
    svm_char.fit(X_char_train, y_train)

    tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    X_word_train = tfidf_word.fit_transform(df_treino["texto_limpo"])
    base_svm_word = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    # Calibração explícita group-aware para o modelo final
    cv_word_final = list(GroupKFold(n_splits=3).split(X_word_train, y_train, groups=df_treino["id_noticia"]))
    svm_word = CalibratedClassifierCV(estimator=base_svm_word, cv=cv_word_final)
    svm_word.fit(X_word_train, y_train)

    tfidf_lemmas = TfidfVectorizer(max_features=10000, min_df=3)
    X_lemmas_train = tfidf_lemmas.fit_transform(df_treino["texto_lematizado"])
    
    lda_8 = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=15, n_jobs=-1).fit(X_lemmas_train)
    nmf_8 = NMF(n_components=8, random_state=42, max_iter=200).fit(X_lemmas_train)
    lda_30 = LatentDirichletAllocation(n_components=30, random_state=42, max_iter=15, n_jobs=-1).fit(X_lemmas_train)
    nmf_30 = NMF(n_components=30, random_state=42, max_iter=200).fit(X_lemmas_train)

    scaler_estilo = StandardScaler()
    X_estilo_train = scaler_estilo.fit_transform(df_treino[ESTILO_FEATURE_NAMES].values)
    X_denso_train = np.hstack([
        X_estilo_train,
        extrair_vetor_k_mais_3(lda_8, X_lemmas_train),
        extrair_vetor_k_mais_3(nmf_8, X_lemmas_train),
        extrair_vetor_k_mais_3(lda_30, X_lemmas_train),
        extrair_vetor_k_mais_3(nmf_30, X_lemmas_train)
    ])
    
    xgb_denso = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_denso.fit(X_denso_train, y_train)

    # -------------------------------------------------------------
    # 2. Meta-Modelo (Treinado Estritamente com OOF)
    # -------------------------------------------------------------
    logger.info("A ajustar Meta-Modelo de Regressão Logística sobre previsões OOF...")
    X_meta_train = np.hstack([p_char_oof.reshape(-1, 1), p_word_oof.reshape(-1, 1), p_denso_oof.reshape(-1, 1)])
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    meta_lr.fit(X_meta_train, y_train)
    logger.info(f"Pesos do Meta-Modelo (Char, Word, Denso): {meta_lr.coef_[0]}")

    # -------------------------------------------------------------
    # 6. Avaliação e Verificação do Limiar 0.46
    # -------------------------------------------------------------
    limiar_decisao = 0.46
    
    # 6.1 Relatório de Desenvolvimento (OOF)
    logger.info("--- RELATÓRIO DO CONJUNTO DE DESENVOLVIMENTO (OOF) ---")
    p_fake_oof = meta_lr.predict_proba(X_meta_train)[:, 1]
    y_pred_oof = (p_fake_oof >= limiar_decisao).astype(int)
    f1_oof = f1_score(y_train, y_pred_oof, pos_label=1)
    logger.info(f"F1-Score OOF: {f1_oof:.4f}")
    logger.info("\n" + classification_report(y_train, y_pred_oof, target_names=["Verdadeiro", "Falso"]))
    
    # 6.2 Relatório de Teste
    logger.info("--- RELATÓRIO DO CONJUNTO DE TESTE ---")
    df_eval = df_val if df_val is not None else df_treino
    y_eval = df_eval["target"].values

    X_char_eval = tfidf_char.transform(df_eval["texto_cru"])
    X_word_eval = tfidf_word.transform(df_eval["texto_limpo"])
    X_lem_eval = tfidf_lemmas.transform(df_eval["texto_lematizado"])

    v_lda8_eval = extrair_vetor_k_mais_3(lda_8, X_lem_eval)
    v_nmf8_eval = extrair_vetor_k_mais_3(nmf_8, X_lem_eval)
    v_lda30_eval = extrair_vetor_k_mais_3(lda_30, X_lem_eval)
    v_nmf30_eval = extrair_vetor_k_mais_3(nmf_30, X_lem_eval)

    X_estilo_eval = scaler_estilo.transform(df_eval[ESTILO_FEATURE_NAMES].values)
    X_denso_eval = np.hstack([
        X_estilo_eval,
        v_lda8_eval, v_nmf8_eval, v_lda30_eval, v_nmf30_eval
    ])

    p_char_eval = svm_char.predict_proba(X_char_eval)[:, 1].reshape(-1, 1)
    p_word_eval = svm_word.predict_proba(X_word_eval)[:, 1].reshape(-1, 1)
    p_denso_eval = xgb_denso.predict_proba(X_denso_eval)[:, 1].reshape(-1, 1)

    X_meta_eval = np.hstack([p_char_eval, p_word_eval, p_denso_eval])
    p_fake_final = meta_lr.predict_proba(X_meta_eval)[:, 1]

    y_pred = (p_fake_final >= limiar_decisao).astype(int)

    f1_obtido = f1_score(y_eval, y_pred, pos_label=1)
    logger.info(f"F1-Score Teste (limiar {limiar_decisao}): {f1_obtido:.4f}")
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
        "scaler_estilo": scaler_estilo,
        "xgb_denso": xgb_denso,
        "meta_modelo": meta_lr,
        "limiar": limiar_decisao,
        "f1_score": float(f1_obtido),
        "n_exemplos_treino": len(df_treino),
        "n_exemplos_teste": len(df_eval) if df_val is not None else 0,
        "dataset_version": "fake_br_master + dataset_11",
        "feature_names_estilo": ESTILO_FEATURE_NAMES,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(bundle, output_path, compress=3)
        logger.info(f"Bundle serializado com sucesso em: {output_path}")
        
        # Gerar o manifest JSON
        manifest_path = os.path.join(os.path.dirname(output_path), "model_manifest.json")
        try:
            import json, datetime, subprocess, sklearn, xgboost, spacy, sys
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            
            manifest = {
                "model_version": "v1",
                "pipeline_version": "1.0",
                "dataset_version": bundle["dataset_version"],
                "training_commit": commit_hash,
                "python_version": sys.version.split()[0],
                "scikit-learn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "spacy_version": spacy.__version__,
                "spacy_model": "pt_core_news_lg",
                "threshold": bundle["limiar"],
                "F1": bundle["f1_score"],
                "feature_count": int(bundle["xgb_denso"].n_features_in_),
                "training_date": datetime.datetime.now().isoformat()
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4)
            logger.info(f"Manifest serializado com sucesso em: {manifest_path}")
        except Exception as e:
            logger.error(f"Erro ao gerar model_manifest.json: {e}")

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
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.feature_extraction import extrair_pacote_analise
    from api.feature_contract import ESTILO_FEATURE_NAMES

    logger.info("Recalculando TODAS as features estilométricas usando as regras unificadas de produção...")
    def _aplicar_tudo(row):
        texto_original = row.get("texto_bert")
        if pd.isna(texto_original) or not texto_original:
            texto_original = row.get("texto_truncado", "")
        
        features_dict, representacoes = extrair_pacote_analise(str(texto_original), max_tokens=500)
        
        out = {
            "texto_cru": representacoes["texto_cru"],
            "texto_limpo": representacoes["texto_limpo"],
            "texto_lematizado": representacoes["texto_lematizado"]
        }
        for feat in ESTILO_FEATURE_NAMES:
            out[feat] = features_dict[feat]
            
        return pd.Series(out)

    cols = ["texto_cru", "texto_limpo", "texto_lematizado"] + ESTILO_FEATURE_NAMES
    df_unificado[cols] = df_unificado.apply(_aplicar_tudo, axis=1)

    return df_unificado



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_joblib = os.path.join(base_dir, "models", "stacking_miopia_v1.joblib")

    caminho_11 = os.path.join(base_dir, "api", "data", "dataset_11.csv")
    caminho_master = os.path.join(base_dir, "api", "data", "fake_br_master.csv")

    if os.path.exists(caminho_11) and os.path.exists(caminho_master):
        logger.info(f"A carregar bases reais: {caminho_11} e {caminho_master}")
        df_completo = carregar_dados_reais(caminho_11, caminho_master)
        
        # Divisão com GroupShuffleSplit para que pares da mesma id_noticia não se separem
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df_completo, groups=df_completo["id_noticia"]))
        
        df_treino = df_completo.iloc[train_idx]
        df_teste = df_completo.iloc[test_idx]
        
        logger.info(f"Grupos no Treino: {df_treino['id_noticia'].nunique()} | Tamanho: {len(df_treino)}")
        logger.info(f"Grupos no Teste Reservado: {df_teste['id_noticia'].nunique()} | Tamanho: {len(df_teste)}")
        
        treinar_stacking(df_treino=df_treino, df_val=df_teste, output_path=target_joblib)
    else:
        erro_msg = f"Bases reais não encontradas: {caminho_11} ou {caminho_master}"
        logger.error(erro_msg)
        raise FileNotFoundError(erro_msg)
    
    logger.info("Pipeline concluído.")