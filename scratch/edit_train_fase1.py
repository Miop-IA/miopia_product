import os
import re

path = r"c:\Users\Usuario\RES_ELD\miopia_product\backend\train\train.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Edit carregar_dados_reais
new_carregar = '''def carregar_dados_reais(caminho_dataset_11: str, caminho_master: str) -> pd.DataFrame:
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

    return df_limpo'''

content = re.sub(r'def carregar_dados_reais.*?return df_unificado', new_carregar, content, flags=re.DOTALL)

# Edit gerar_dados_sinteticos_para_teste
new_gerar = '''def gerar_dados_sinteticos_para_teste(n_samples: int = 80) -> pd.DataFrame:
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

    return pd.DataFrame(dados)'''

content = re.sub(r'def gerar_dados_sinteticos_para_teste.*?return pd\.DataFrame\(dados\)', new_gerar, content, flags=re.DOTALL)

# Replace id_noticia usages EXCEPT the ones strictly inside carregar_dados_reais loop or sinteticos
# We can do target replacements
content = content.replace('groups=df_treino["id_noticia"]', 'groups=df_treino["grupo_identidade"]')
content = content.replace('groups=df_fold_train["id_noticia"]', 'groups=df_fold_train["grupo_identidade"]')
content = content.replace('df_treino["id_noticia"].values', 'df_treino["grupo_identidade"].values')
content = content.replace('groups=df_completo["id_noticia"]', 'groups=df_completo["grupo_identidade"]')

# in len(train_groups.intersection(test_groups)):
content = content.replace('set(df_treino["id_noticia"])', 'set(df_treino["grupo_identidade"])')
content = content.replace('set(df_val["id_noticia"])', 'set(df_val["grupo_identidade"])')
content = content.replace('"id_noticia" in df_treino.columns and "id_noticia" in df_val.columns', '"grupo_identidade" in df_treino.columns and "grupo_identidade" in df_val.columns')
content = content.replace('compartilhados entre treino e teste.', 'compartilhados entre treino e teste.') # Keep same error msg maybe? 
# The prompt says: "Existem grupos (id_noticia) compartilhados entre treino e teste." -> Let's change it
content = content.replace('Existem grupos (id_noticia) compartilhados', 'Existem grupos compartilhados')

content = content.replace("df_treino['id_noticia'].nunique() if 'id_noticia' in df_treino.columns else 0", "df_treino['grupo_identidade'].nunique() if 'grupo_identidade' in df_treino.columns else 0")
content = content.replace("df_eval['id_noticia'].nunique() if 'id_noticia' in df_eval.columns else 0", "df_eval['grupo_identidade'].nunique() if 'grupo_identidade' in df_eval.columns else 0")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
