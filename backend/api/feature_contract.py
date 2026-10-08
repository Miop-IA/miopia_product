"""
Contrato oficial das features estilométricas utilizadas no pipeline MiopIA.
Centraliza a definição, ordem e documentação das features para garantir paridade entre Treino e Produção.
"""

ESTILO_FEATURE_CONTRACT = [
    {
        "name": "trunc_pausality",
        "formula": "pausas_count / num_sentences",
        "denominador": "num_sentences (número de sentenças detectadas pelo SpaCy)",
        "tokens_considerados": "PONTUACOES_PAUSALIDADE (',', ';', ':', '-', '—')",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 0
    },
    {
        "name": "trunc_emotiveness",
        "formula": "(pos_counts['ADJ'] + pos_counts['ADV']) / max(pos_counts['NOUN'] + pos_counts['VERB'], 1)",
        "denominador": "Substantivos + Verbos",
        "tokens_considerados": "ADJ, ADV (numerador) / NOUN, PROPN, VERB, AUX (denominador)",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 1
    },
    {
        "name": "trunc_diversity",
        "formula": "MATTR (Moving-Average Type-Token Ratio) com janela de 25 palavras",
        "denominador": "25 (tamanho da janela) e total de janelas",
        "tokens_considerados": "Lemas de tokens alfabéticos",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 2
    },
    {
        "name": "trunc_upper_case_density",
        "formula": "caixa_alta_count / num_words",
        "denominador": "num_words (total de palavras alfabéticas)",
        "tokens_considerados": "Tokens onde t.text.isupper() e len(t.text) > 1",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 3
    },
    {
        "name": "trunc_verb_density",
        "formula": "pos_counts['VERB'] / num_words",
        "denominador": "num_words",
        "tokens_considerados": "VERB, AUX",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 4
    },
    {
        "name": "trunc_noun_density",
        "formula": "pos_counts['NOUN'] / num_words",
        "denominador": "num_words",
        "tokens_considerados": "NOUN, PROPN",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 5
    },
    {
        "name": "trunc_adj_density",
        "formula": "pos_counts['ADJ'] / num_words",
        "denominador": "num_words",
        "tokens_considerados": "ADJ",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 6
    },
    {
        "name": "trunc_adv_density",
        "formula": "pos_counts['ADV'] / num_words",
        "denominador": "num_words",
        "tokens_considerados": "ADV",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 7
    },
    {
        "name": "trunc_pron_density",
        "formula": "pos_counts['PRON'] / num_words",
        "denominador": "num_words",
        "tokens_considerados": "PRON, DET",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 8
    },
    {
        "name": "link_density",
        "formula": "num_links / num_words",
        "denominador": "num_words",
        "tokens_considerados": "URLs via Regex (https?:// ou www.)",
        "entrada_utilizada": "texto_cru_trunc",
        "ordem_no_vetor": 9
    },
    {
        "name": "rc_spelling_errors",
        "formula": "spell_errors_count / num_words",
        "denominador": "num_words",
        "tokens_considerados": "Palavras não reconhecidas pelo pyspellchecker",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 10
    },
    {
        "name": "rc_modal_verbs_density",
        "formula": "modal_verbs_count / num_words",
        "denominador": "num_words",
        "tokens_considerados": "Verbos (VERB, AUX) com lema em MODAL_LEMMAS ('poder', 'dever', 'precisar', 'querer', 'costumar', 'saber') + locução 'ter que/de'",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 11
    },
    {
        "name": "rc_subj_imp_verbs_density",
        "formula": "subj_imp_count / num_words",
        "denominador": "num_words",
        "tokens_considerados": "Verbos (VERB, AUX) com Morph Mood 'Sub' ou 'Imp'",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 12
    },
    {
        "name": "rc_pron_1_2_sing_density",
        "formula": "pron_1_2_sing_count / num_words",
        "denominador": "num_words",
        "tokens_considerados": "Pronomes/Determinantes (PRON, DET) de 1ª ou 2ª pessoa do singular (via spaCy) ou em lista explícita",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 13
    },
    {
        "name": "rc_pron_1_plur_density",
        "formula": "pron_1_plur_count / num_words",
        "denominador": "num_words",
        "tokens_considerados": "Pronomes/Determinantes (PRON, DET) de 1ª pessoa do plural (via spaCy) ou em lista explícita",
        "entrada_utilizada": "texto_limpo",
        "ordem_no_vetor": 14
    }
]

ESTILO_FEATURE_NAMES = [f["name"] for f in ESTILO_FEATURE_CONTRACT]
