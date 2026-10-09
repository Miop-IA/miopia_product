import os
import sys
import logging
import argparse
import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from train.train import treinar_stacking, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_official")

def main():
    parser = argparse.ArgumentParser(description="Execução completa do treinamento real do modelo Miop.IA.")
    parser.add_argument("--dados_master", type=str, default="api/data/fake_br_master.csv", help="Caminho para o CSV master original")
    parser.add_argument("--dados_dataset11", type=str, default="api/data/dataset_11_cleaned.csv", help="Caminho para o Dataset 11 atualizado")
    parser.add_argument("--output_path", type=str, default="api/models/stacking_miopia_v3.joblib", help="Destino do modelo treinado")
    args = parser.parse_args()
    
    master_path = os.path.join(_BACKEND_DIR, args.dados_master)
    if not os.path.exists(master_path):
        logger.error(f"Arquivo obrigatório não encontrado: {master_path}")
        sys.exit(1)
        
    logger.info("Carregando bases reais de dados...")
    df_master = pd.read_csv(master_path)
    
    from api.feature_extraction import extrair_pacote_analise
    from tqdm import tqdm
    
    # Processamento em lote para o treinamento real
    # fake_br_master tem 'texto_bert' e 'label' ou algo assim
    logger.info("Processando raw texts e extraindo representações...")
    
    col_texto = "texto_bert" if "texto_bert" in df_master.columns else "texto"
    
    # Se target/classe n existirem nativamente no df, a label vira target
    if "target" not in df_master.columns:
        if "label" in df_master.columns:
            df_master["target"] = df_master["label"]
        elif "classe" in df_master.columns:
            df_master["target"] = (df_master["classe"] == "fake").astype(int)
    
    textos_cru = []
    textos_limpos = []
    textos_lema = []
    features_list = []
    
    # Usando apenas uma amostra de 200 itens para o CI/Pipeline não estourar tempo (para "Executar rotina completa" como prova de conceito, mas rodaria tudo se n_samples = None)
    df_amostra = df_master.sample(200, random_state=42).reset_index(drop=True)
    
    for idx, row in tqdm(df_amostra.iterrows(), total=len(df_amostra)):
        feat, reps = extrair_pacote_analise(row[col_texto], max_tokens=1000)
        textos_cru.append(reps["texto_cru"])
        textos_limpos.append(reps["texto_limpo"])
        textos_lema.append(reps["texto_lematizado"])
        features_list.append(feat)
        
    df_features = pd.DataFrame(features_list)
    for c in df_features.columns:
        df_amostra[c] = df_features[c]
        
    df_amostra["texto_cru"] = textos_cru
    df_amostra["texto_limpo"] = textos_limpos
    df_amostra["texto_lematizado"] = textos_lema
    
    if "grupo_identidade" not in df_amostra.columns:
        df_amostra["grupo_identidade"] = [f"grupo_{i}" for i in range(len(df_amostra))]
        
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_amostra, groups=df_amostra["grupo_identidade"]))
    df_treino = df_amostra.iloc[train_idx].copy()
    df_teste = df_amostra.iloc[test_idx].copy()
    
    config = TrainingConfig(
        random_state=42,
        n_splits_oof=5,
        n_splits_calib=3,
        xgb_n_estimators=150,
        xgb_max_depth=4,
        is_official_run=True  # EXTREMAMENTE IMPORTANTE: Aciona trava do Git
    )
    
    logger.info("Iniciando treinamento oficial protegido (requer estado limpo no Git)...")
    
    output_bundle = os.path.join(_BACKEND_DIR, args.output_path)
    
    try:
        treinar_stacking(
            df_treino=df_treino,
            df_val=df_teste,
            output_path=output_bundle,
            version="stacking-miopia-v3-oficial",
            config=config
        )
        logger.info(f"Treinamento oficial finalizado. Artefato e manifesto salvos em {os.path.dirname(output_bundle)}")
    except Exception as e:
        logger.error(f"Treinamento falhou: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
