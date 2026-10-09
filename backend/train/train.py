import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import scipy.stats

import random
import numpy as np

# Fixando sementes globais para reprodutibilidade absoluta
random.seed(42)
np.random.seed(42)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_stacking")

# F1 mínimo no conjunto de teste para o bundle ser empacotado
F1_MINIMO = 0.85

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
    df_val: pd.DataFrame,
    output_path: str = None,
    version: str = None,
) -> Dict:
    """
    Treina os 3 ramos do Stacking Ensemble e o metamodelo de Regressão Logística
    usando previsões Out-of-Fold (OOF) baseadas no StratifiedGroupKFold por grupo_identidade.
    """
    if df_val is None:
        raise ValueError("O conjunto de teste/validação (df_val) é obrigatório para evitar avaliação viciada no treino.")
        
    logger.info("A iniciar treino do pipeline Stacking Parte C...")
    
    if "grupo_identidade" in df_treino.columns and "grupo_identidade" in df_val.columns:
        train_groups = set(df_treino["grupo_identidade"])
        test_groups = set(df_val["grupo_identidade"])
        if len(train_groups.intersection(test_groups)) > 0:
            raise ValueError("Sobreposição detectada (Data Leakage)! Existem grupos compartilhados entre treino e teste.")

    y_train = df_treino["target"].values
    groups_train = df_treino["grupo_identidade"].values

    # -------------------------------------------------------------
    # 0. Geração de Predições Out-of-Fold (OOF)
    # -------------------------------------------------------------
    logger.info("Gerando predições OOF com StratifiedGroupKFold (5 splits)...")
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
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
        cv_char_fold = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_char_fold_train, y_fold_train, groups=df_fold_train["grupo_identidade"]))
        svm_char_fold = CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=cv_char_fold)
        svm_char_fold.fit(X_char_fold_train, y_fold_train)
        p_char_oof[val_idx] = svm_char_fold.predict_proba(X_char_fold_val)[:, 1]
        
        # Word
        tfidf_word_fold = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
        X_word_fold_train = tfidf_word_fold.fit_transform(df_fold_train["texto_limpo"])
        X_word_fold_val = tfidf_word_fold.transform(df_fold_val["texto_limpo"])
        # Usa GroupKFold com 3 splits para calibração interna respeitando id_noticia
        cv_word_fold = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_word_fold_train, y_fold_train, groups=df_fold_train["grupo_identidade"]))
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
    cv_char_final = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_char_train, y_train, groups=df_treino["grupo_identidade"]))
    svm_char = CalibratedClassifierCV(estimator=base_svm_char, cv=cv_char_final)
    svm_char.fit(X_char_train, y_train)

    tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    X_word_train = tfidf_word.fit_transform(df_treino["texto_limpo"])
    base_svm_word = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    # Calibração explícita group-aware para o modelo final
    cv_word_final = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_word_train, y_train, groups=df_treino["grupo_identidade"]))
    svm_word = CalibratedClassifierCV(estimator=base_svm_word, cv=cv_word_final)
    svm_word.fit(X_word_train, y_train)

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
    scaler_estilo = StandardScaler()
    X_estilo = scaler_estilo.fit_transform(df_treino[ESTILO_FEATURE_NAMES].values)
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
    # 2. Meta-Modelo e Seleção de Limiar via OOF
    # -------------------------------------------------------------

    logger.info("A calcular limiar ótimo via predições cruzadas OOF do próprio Meta-Modelo...")
    X_meta_train = np.hstack([p_char_oof.reshape(-1, 1), p_word_oof.reshape(-1, 1), p_denso_oof.reshape(-1, 1)])
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    
    cv_meta = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    import sklearn.model_selection
    p_fake_oof = sklearn.model_selection.cross_val_predict(
        meta_lr, X_meta_train, y_train, 
        cv=cv_meta, groups=groups_train, method="predict_proba"
    )[:, 1]
    
    melhor_limiar = 0.5
    melhor_f1_oof = 0.0
    
    for lim in np.arange(0.1, 0.9, 0.01):
        y_pred_cand = (p_fake_oof >= lim).astype(int)
        f1_cand = f1_score(y_train, y_pred_cand, pos_label=1)
        if f1_cand > melhor_f1_oof:
            melhor_f1_oof = f1_cand
            melhor_limiar = lim

    limiar_decisao = round(float(melhor_limiar), 2)
    logger.info(f"Limiar ótimo selecionado rigorosamente sem vazamento (Meta-OOF): {limiar_decisao} com F1={melhor_f1_oof:.4f}")
    
    logger.info("A ajustar Meta-Modelo final em toda a massa de validação cruzada...")
    meta_lr.fit(X_meta_train, y_train)
    logger.info(f"Pesos do Meta-Modelo Final (Char, Word, Denso): {meta_lr.coef_[0]}")
    
    # 6.1 Relatório de Desenvolvimento (OOF)
    logger.info("--- RELATÓRIO DO CONJUNTO DE DESENVOLVIMENTO (OOF) ---")
    y_pred_oof = (p_fake_oof >= limiar_decisao).astype(int)
    f1_oof = f1_score(y_train, y_pred_oof, pos_label=1)
    logger.info(f"F1-Score OOF (limiar {limiar_decisao}): {f1_oof:.4f}")
    logger.info("\n" + classification_report(y_train, y_pred_oof, target_names=["Verdadeiro", "Falso"]))
    
    # 6.2 Relatório de Teste
    logger.info("--- RELATÓRIO DO CONJUNTO DE TESTE ---")
    df_eval = df_val
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
    prec_obtido = precision_score(y_eval, y_pred, pos_label=1)
    rec_obtido = recall_score(y_eval, y_pred, pos_label=1)
    acc_obtido = accuracy_score(y_eval, y_pred)
    cm_obtido = confusion_matrix(y_eval, y_pred).tolist()
    
    logger.info(f"F1-Score Teste (limiar {limiar_decisao}): {f1_obtido:.4f}")
    report_teste = classification_report(y_eval, y_pred, target_names=["Verdadeiro", "Falso"])
    logger.info("\n" + report_teste)
    
    # Removido o bloqueio F1_MINIMO baseado no conjunto de teste para evitar viés de seleção.

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
        "precision": float(prec_obtido),
        "recall": float(rec_obtido),
        "accuracy": float(acc_obtido),
        "confusion_matrix": cm_obtido,
        "classification_report_teste": report_teste,
        "n_exemplos_treino": len(df_treino),
        "n_exemplos_teste": len(df_eval),
        "n_grupos_treino": df_treino['grupo_identidade'].nunique() if 'grupo_identidade' in df_treino.columns else 0,
        "n_grupos_teste": df_eval['grupo_identidade'].nunique() if 'grupo_identidade' in df_eval.columns else 0,
        "dataset_version": "fake_br_master + dataset_11",
        "feature_names_estilo": list(ESTILO_FEATURE_NAMES),
        "feature_order": list(FEATURE_ORDER),
        "feature_count": FEATURE_COUNT,
        "version": version or datetime.now(timezone.utc).strftime("stacking-%Y%m%dT%H%M%SZ"),
    }

    # O mesmo contrato verificado no startup da API: nunca serializar bundle inválido.
    validar_bundle(bundle)

    if output_path:
        import hashlib
        import tempfile
        import shutil

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=os.path.dirname(output_path))
        
        temp_model_path = os.path.join(temp_dir, os.path.basename(output_path))
        temp_manifest_path = os.path.join(temp_dir, "model_manifest.json")
        
        try:
            # Salvar modelo temporário
            joblib.dump(bundle, temp_model_path, compress=3)
            
            # Calcular SHA256 do arquivo joblib
            sha256_hash = hashlib.sha256()
            with open(temp_model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            model_hash = sha256_hash.hexdigest()

            # Gerar manifesto
            import json, subprocess, sklearn, xgboost, spacy
            try:
                commit_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode("utf-8").strip()
            except Exception:
                commit_hash = "desconhecido"
                
            manifest = {
                "model_version": bundle["version"],
                "pipeline_version": "1.0",
                "dataset_version": bundle["dataset_version"],
                "training_commit": commit_hash,
                "python_version": sys.version.split()[0],
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "spacy": spacy.__version__,
                "spacy_model": "pt_core_news_lg",
                "threshold": bundle["limiar"],
                "F1": bundle["f1_score"],
                "precision": bundle["precision"],
                "recall": bundle["recall"],
                "accuracy": bundle["accuracy"],
                "confusion_matrix": bundle["confusion_matrix"],
                "n_train": bundle["n_exemplos_treino"],
                "n_test": bundle["n_exemplos_teste"],
                "n_grupos_treino": bundle["n_grupos_treino"],
                "n_grupos_teste": bundle["n_grupos_teste"],
                "feature_count": int(bundle["xgb_denso"].n_features_in_),
                "training_date": datetime.now(timezone.utc).isoformat(),
                "model_hash_sha256": model_hash
            }
            
            with open(temp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4)
                
            # Mover atomicamente (os.replace garante overwrite)
            manifest_path = os.path.join(os.path.dirname(output_path), "model_manifest.json")
            os.replace(temp_manifest_path, manifest_path)
            os.replace(temp_model_path, output_path)

            # Gravar report em markdown
            md_path = os.path.join(os.path.dirname(output_path), "metrics_report.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Relatório de Validação Stacking - {bundle['version']}\n\n")
                f.write(f"**Data:** {manifest['training_date']}\n")
                f.write(f"**Hash SHA256:** `{model_hash}`\n")
                f.write(f"**Limiar OOF:** {bundle['limiar']}\n\n")
                f.write("## Performance (Teste)\n")
                f.write(f"- **F1-Score**: {bundle['f1_score']:.4f}\n")
                f.write(f"- **Accuracy**: {bundle['accuracy']:.4f}\n\n")
                f.write("### Classification Report\n```text\n")
                f.write(bundle['classification_report_teste'])
                f.write("\n```\n\n### Matriz de Confusão\n```text\n")
                f.write(str(bundle['confusion_matrix']))
                f.write("\n```\n")

            logger.info(f"Bundle, manifest e metrics_report.md serializados atomicamente com sucesso em: {output_path}")
            
        except Exception as e:
            logger.error(f"Erro durante o empacotamento atômico: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return bundle


def carregar_dados_reais(caminho_dataset_11: str, caminho_master: str) -> pd.DataFrame:
    """
    Carrega e une os datasets conforme a arquitetura da Parte A/B:
    Valida esquema, resolve duplicidades textuais e cria a chave de agrupamento real.
    """
    logger.info("A carregar ficheiros de dados...")
    df_11 = pd.read_csv(caminho_dataset_11)
    df_master = pd.read_csv(caminho_master)

    logger.info(f"n_rows_dataset_11: {len(df_11)}")
    logger.info(f"n_rows_master: {len(df_master)}")

    # 1.1 Validar esquema
    req_11 = {"id_noticia", "target", "texto_truncado"}
    req_master = {"id_noticia", "target", "texto_bert"}
    
    if not req_11.issubset(df_11.columns):
        raise ValueError(f"dataset_11 faltam colunas: {req_11 - set(df_11.columns)}")
    if not req_master.issubset(df_master.columns):
        raise ValueError(f"fake_br_master faltam colunas: {req_master - set(df_master.columns)}")
        
    for col in ["id_noticia", "target"]:
        if df_11[col].isnull().any() or df_master[col].isnull().any():
            raise ValueError(f"Valores nulos proibidos na coluna {col}")

    valid_targets = {0, 1}
    if not set(df_11["target"].unique()).issubset(valid_targets):
        raise ValueError("Targets inválidos encontrados em dataset_11")

    # 1.2 Junção explícita
    try:
        df_unificado = pd.merge(
            df_11,
            df_master,
            on=["id_noticia", "target"],
            suffixes=("_11", "_master"),
            validate="1:1"
        )
    except Exception as e:
        logger.error(f"Erro na junção dos dados: {e}")
        raise ValueError(f"Junção falhou: {e}")

    logger.info(f"n_rows_apos_merge: {len(df_unificado)}")
    logger.info(f"n_grupos_id_noticia_originais: {df_unificado['id_noticia'].nunique()}")

    import sys
    import hashlib
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.feature_extraction import extrair_pacote_analise
    from api.feature_contract import ESTILO_FEATURE_NAMES

    logger.info("Filtrando textos e recalculando features estilométricas...")
    
    registros_processados = []
    rejeitados = 0
    origem_bert = 0
    origem_truncado = 0

    for idx, row in df_unificado.iterrows():
        texto = row.get("texto_bert")
        if pd.isna(texto) or not str(texto).strip():
            texto = row.get("texto_truncado", "")
            origem_truncado += 1
        else:
            origem_bert += 1
            
        texto = str(texto).strip()
        
        # 1.4 Validar tamanho mínimo
        if len(texto.split()) < 30:
            rejeitados += 1
            continue
            
        features_dict, representacoes = extrair_pacote_analise(texto, max_tokens=500)
        
        # 1.3 Identidade de grupos
        hash_texto = hashlib.sha256(representacoes["texto_limpo"].encode("utf-8")).hexdigest()
        
        out = row.to_dict()
        out["texto_cru"] = representacoes["texto_cru"]
        out["texto_limpo"] = representacoes["texto_limpo"]
        out["texto_lematizado"] = representacoes["texto_lematizado"]
        out["grupo_identidade"] = hash_texto
        
        for feat in ESTILO_FEATURE_NAMES:
            out[feat] = features_dict[feat]
            
        registros_processados.append(out)

    df_limpo = pd.DataFrame(registros_processados)
    
    logger.info(f"Registros mantidos: {len(df_limpo)}. Rejeitados por texto insuficiente/ausente: {rejeitados}.")
    logger.info(f"Origem texto_bert: {origem_bert}, Origem texto_truncado: {origem_truncado}.")
    
    if len(df_limpo) == 0:
        raise ValueError("Dataset ficou vazio após a limpeza de texto.")

    targets_por_grupo = df_limpo.groupby("grupo_identidade")["target"].nunique()
    if (targets_por_grupo > 1).any():
        raise ValueError("Existem grupos textuais (textos idênticos) com targets conflitantes.")
        
    n_grupos_finais = df_limpo["grupo_identidade"].nunique()
    logger.info(f"Número de grupos textuais únicos: {n_grupos_finais}")
    
    if n_grupos_finais < 10:
        raise ValueError("Quantidade insuficiente de grupos para validação cruzada.")

    return df_limpo



def gerar_dados_sinteticos_para_teste(n_samples: int = 80) -> pd.DataFrame:
    """Gera um DataFrame mock estruturado para validar a execução técnica (somente testes)."""
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

    import hashlib
    for i in range(n_samples):
        grupo_id = i // 4
        is_fake = grupo_id % 2 == 1
        t_cru = textos_fake[i % len(textos_fake)] if is_fake else textos_true[i % len(textos_true)]
        t_cru += f" Variacao {i}"
        
        t_limpo = t_cru.lower()
        t_lem = " ".join([w for w in t_limpo.split() if len(w) > 3])
        
        grupo_identidade = hashlib.sha256(f"grupo_{grupo_id}".encode("utf-8")).hexdigest()

        row = {
            "id_noticia": grupo_id,
            "grupo_identidade": grupo_identidade,
            "texto_cru": t_cru,
            "texto_limpo": t_limpo,
            "texto_lematizado": t_lem,
            "target": 1 if is_fake else 0,
            "trunc_pausality": 0.2,
            "trunc_emotiveness": 0.5,
            "trunc_diversity": 0.7,
            "trunc_upper_case_density": 0.1,
            "trunc_verb_density": 0.2,
            "trunc_noun_density": 0.3,
            "trunc_adj_density": 0.1,
            "trunc_adv_density": 0.05,
            "trunc_pron_density": 0.1,
            "link_density": 0.0,
            "rc_spelling_errors": 0.05,
            "rc_modal_verbs_density": 0.1,
            "rc_subj_imp_verbs_density": 0.1,
            "rc_pron_1_2_sing_density": 0.05,
            "rc_pron_1_plur_density": 0.02,
        }
        dados.append(row)

    return pd.DataFrame(dados)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treina o bundle Stacking Miop.IA.")
    parser.add_argument("--dados-dir", default=os.path.join(_BACKEND_DIR, "api", "data"),
                        help="Pasta com dataset_11.csv e fake_br_master.csv")
    parser.add_argument("--saida", default=os.path.join(_BACKEND_DIR, "models", "stacking_miopia_v3.joblib"))
    parser.add_argument("--versao", default=None, help="Identificador de versão gravado no bundle")
    args = parser.parse_args()

    caminho_11 = os.path.join(args.dados_dir, "dataset_11.csv")
    caminho_master = os.path.join(args.dados_dir, "fake_br_master.csv")
    faltando = [c for c in (caminho_11, caminho_master) if not os.path.exists(c)]
    if faltando:
        sys.exit(f"Bases reais não encontradas: {faltando}. Nenhum modelo foi gerado.")

    df_completo = carregar_dados_reais(caminho_11, caminho_master)

    # GroupShuffleSplit: pares da mesma id_noticia nunca ficam separados entre treino e teste
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_completo, groups=df_completo["grupo_identidade"]))
    df_treino = df_completo.iloc[train_idx]
    df_teste = df_completo.iloc[test_idx]
    logger.info(f"n_treino: {len(df_treino)} | n_teste: {len(df_teste)}")

    treinar_stacking(df_treino=df_treino, df_val=df_teste, output_path=args.saida, version=args.versao)

    logger.info("Pipeline concluído.")
