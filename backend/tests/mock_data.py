import numpy as np
import pandas as pd

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
