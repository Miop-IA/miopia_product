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
    
    # Aqui a equipe adicionaria lógica de concatenação com dataset_11 e feature extractions
    # Assumimos que o master e o dataset_11 já passaram pela limpeza e extração de features textuais
    # Exemplo: pipeline completo exigiria Apply de `extrair_pacote_analise` (que é lento)
    # Por isso o CI roda com mock sintético, e esta rotina é rodada à mão em batch.
    
    # Exemplo simples de split se o CSV já estivesse pronto:
    from sklearn.model_selection import GroupShuffleSplit
    
    # Simulação da checagem de colunas para o treinamento estrito:
    colunas_obrigatorias = ["target", "texto_cru", "texto_limpo", "texto_lematizado", "grupo_identidade"]
    faltando = [c for c in colunas_obrigatorias if c not in df_master.columns]
    if faltando:
        logger.error(f"O CSV fornecido não possui todas as representações extraídas. Faltam: {faltando}")
        logger.error("Você deve rodar uma rotina de pré-processamento batch antes de chamar o treinamento final.")
        sys.exit(1)
        
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_master, groups=df_master["grupo_identidade"]))
    df_treino = df_master.iloc[train_idx]
    df_teste = df_master.iloc[test_idx]
    
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
