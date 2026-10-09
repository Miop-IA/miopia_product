import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Ajuste de PYTHONPATH para imports
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from api.bundle_spec import validar_bundle

def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 150
    })

def plot_class_distribution(y_true, output_path, meta_info):
    """
    1. Distribuição das classes
    """
    plt.figure()
    counts = pd.Series(y_true).value_counts().sort_index()
    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(f"Distribuição de Classes no Conjunto de Avaliação\n{meta_info}")
    plt.xlabel("Classe Real (0 = Falso, 1 = Verdadeiro)")
    plt.ylabel("Contagem")
    for i, v in enumerate(counts.values):
        ax.text(i, v + len(y_true)*0.01, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path, meta_info):
    """
    2. Matriz de Confusão do Modelo Avaliado
    """
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predito Falso (0)", "Predito Verdadeiro (1)"],
                yticklabels=["Real Falso (0)", "Real Verdadeiro (1)"])
    plt.title(f"Matriz de Confusão\n{meta_info}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pr_roc_curves(y_true, y_prob, output_path, meta_info):
    """
    3. Curva Precision-Recall e Curva ROC
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.suptitle(f"Avaliação do Limiar e Discriminação\n{meta_info}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_calibration_curve(y_true, y_prob, output_path, meta_info):
    """
    4. Curva de Calibração (Reliability Diagram)
    """
    plt.figure()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Modelo (Brier={brier:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfeitamente Calibrado')
    plt.xlabel('Probabilidade Predita Média')
    plt.ylabel('Fração de Positivos Reais')
    plt.title(f"Curva de Calibração (Reliability Diagram)\n{meta_info}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Gerador de visualizações de avaliação do modelo (Fase 9).")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o arquivo .joblib")
    parser.add_argument("--dataset_path", type=str, required=True, help="Caminho para o CSV com dados de validação reais")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório onde as imagens serão salvas")
    args = parser.parse_args()

    # Validação de dependências de entrada
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {args.model_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado: {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Carregando dataset de avaliação...")
    df = pd.read_csv(args.dataset_path)
    if "target" not in df.columns or "texto_cru" not in df.columns:
        raise ValueError("O dataset deve conter as colunas 'target' e 'texto_cru'.")

    # Utilizando inferencia.py para garantir o pipeline idêntico de prod
    print("Inicializando modelo...")
    bundle = joblib.load(args.model_path)
    validar_bundle(bundle) # Assegurar integridade estrita
    
    # Extrair metadata
    version = bundle.get("version", "unknown")
    pipeline_version = bundle.get("pipeline_version", "unknown")
    threshold = bundle.get("limiar", 0.5)
    meta_info = f"Model: {version} | Pipeline: v{pipeline_version}"
    
    print("Gerando predições (isso pode levar algum tempo)...")
    y_true = df["target"].values
    
    # Avaliando todos os textos usando as peças do bundle isoladamente para performance no batch
    # (Ou poderíamos usar a API analisar_texto iterativamente, mas batch é mais rápido)
    
    # Reconstruindo pipeline em batch para eficiência do teste:
    tfidf_char = bundle["tfidf_char"]
    svm_char = bundle["svm_caracteres"]
    tfidf_word = bundle["tfidf_word"]
    svm_palavras = bundle["svm_palavras"]
    tfidf_lemmas = bundle["tfidf_lemmas"]
    lda_8, nmf_8 = bundle["lda_8"], bundle["nmf_8"]
    lda_30, nmf_30 = bundle["lda_30"], bundle["nmf_30"]
    scaler_estilo = bundle["scaler_estilo"]
    xgb_denso = bundle["xgb_denso"]
    meta_modelo = bundle["meta_modelo"]
    
    from api.bundle_spec import ESTILO_FEATURE_NAMES
    from train.train import extrair_vetor_k_mais_3
    
    # Prepara features para predição OOM - assumimos que o CSV tem texto_limpo e texto_lematizado
    # Se não tiver, falhamos por exigência de dataset válido pré-processado
    if "texto_limpo" not in df.columns or "texto_lematizado" not in df.columns:
        raise ValueError("O dataset de avaliação precisa estar previamente limpo e lematizado para processamento em batch.")
        
    X_char = tfidf_char.transform(df["texto_cru"])
    X_word = tfidf_word.transform(df["texto_limpo"])
    X_lem = tfidf_lemmas.transform(df["texto_lematizado"])
    
    p_char = svm_char.predict_proba(X_char)[:, 1].reshape(-1, 1)
    p_word = svm_palavras.predict_proba(X_word)[:, 1].reshape(-1, 1)
    
    v_lda8 = extrair_vetor_k_mais_3(lda_8, X_lem)
    v_nmf8 = extrair_vetor_k_mais_3(nmf_8, X_lem)
    v_lda30 = extrair_vetor_k_mais_3(lda_30, X_lem)
    v_nmf30 = extrair_vetor_k_mais_3(nmf_30, X_lem)
    
    X_estilo = scaler_estilo.transform(df[ESTILO_FEATURE_NAMES].values)
    X_denso = np.hstack([X_estilo, v_lda8, v_nmf8, v_lda30, v_nmf30])
    
    p_denso = xgb_denso.predict_proba(X_denso)[:, 1].reshape(-1, 1)
    X_meta = np.hstack([p_char, p_word, p_denso])
    
    y_prob = meta_modelo.predict_proba(X_meta)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    configure_plot_style()

    print("Gerando gráficos...")
    # Gráficos exigidos pela Fase 9
    plot_class_distribution(y_true, os.path.join(args.output_dir, "01_class_distribution.png"), meta_info)
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "02_confusion_matrix.png"), meta_info)
    plot_pr_roc_curves(y_true, y_prob, os.path.join(args.output_dir, "03_roc_pr_curves.png"), meta_info)
    plot_calibration_curve(y_true, y_prob, os.path.join(args.output_dir, "04_calibration_curve.png"), meta_info)

    print(f"Avaliação concluída. Imagens salvas em {args.output_dir}.")

if __name__ == "__main__":
    main()
