import logging
import re
from typing import Dict, Any, List, Tuple
import spacy
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)

def _load_spacy_model():
    models_to_try = ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"]
    for model_name in models_to_try:
        try:
            loaded_nlp = spacy.load(model_name, disable=["ner"])
            logger.info(f"Modelo SpaCy carregado: {model_name}")
            return loaded_nlp
        except OSError:
            continue
    raise RuntimeError("Instale o modelo SpaCy: python -m spacy download pt_core_news_sm")

nlp = _load_spacy_model()
spell = SpellChecker(language="pt")

MODAL_LEMMAS = frozenset(["poder", "dever", "precisar", "querer", "costumar", "saber"])
PRON_1_2_SING = frozenset(["eu", "me", "mim", "comigo", "você", "tu", "te", "ti", "contigo", "teu", "tua", "teus", "tuas"])
PRON_1_PLUR = frozenset(["nós", "nos", "conosco", "nosso", "nossa", "nossos", "nossas"])
PONTUACOES_PAUSALIDADE = frozenset([",", ";", ":", "-", "—"])

TAIL_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)continua\s+após\s+a\s+publicidade.*", re.DOTALL),
    re.compile(r"(?i)opinião\s*texto\s+em\s+que\s+o\s+autor\s+apresenta.*", re.DOTALL),
    re.compile(r"(?i)este\s+texto\s+não\s+reflete.*", re.DOTALL),
    re.compile(r"(?i)comunicar\s+erro.*", re.DOTALL),
    re.compile(r"(?i)deixe\s+seu\s+comentário.*", re.DOTALL),
    re.compile(r"(?i)o\s+autor\s+da\s+mensagem.*", re.DOTALL),
    re.compile(r"(?i)leia\s+as\s+regras\s+de\s+uso.*", re.DOTALL),
    re.compile(r"(?i)adicione\s+como\s+fonte\s+preferencial.*", re.DOTALL),
    re.compile(r"(?i)produzido\s+pela\s+ri7a.*", re.DOTALL),
    re.compile(r"(?i)gerando\s+resumo.*", re.DOTALL),
    re.compile(r"(?i)siga\s+(o|a)?\s*(nosso|nossa)?\s*canal.*", re.DOTALL),
    re.compile(r"(?i)conteúdo\s+criado\s+em\s+parceria.*", re.DOTALL),
]


def sanitizar_texto_noticia(texto: str) -> str:
    """Higieniza ruídos estruturais comuns em portais de notícia."""
    if not texto or not isinstance(texto, str):
        return ""
    limpo = texto
    for pattern in TAIL_BOILERPLATE_PATTERNS:
        limpo = pattern.sub("", limpo)
    limpo = re.sub(r"^.*?\d{1,2}:\d{2}\s*/\s*\d{1,2}:\d{2}x?\s*", "", limpo, flags=re.DOTALL)
    limpo = re.sub(r"(?i)imagem:\s*arte/uol", "", limpo)
    limpo = re.sub(r"(?i)veja\s+horários\s+das\s+lives[^\n.]*", "", limpo)
    return limpo


def calcular_mattr(lemas: List[str], window_size: int = 25) -> float:
    """
    Moving-Average Type-Token Ratio (MATTR) com janela de 25 palavras.
    Remove o viés de tamanho do texto presente no TTR convencional.
    """
    n_tokens = len(lemas)
    if n_tokens == 0:
        return 0.0
    if n_tokens < window_size:
        return len(set(lemas)) / n_tokens

    ttrs = []
    for i in range(n_tokens - window_size + 1):
        janela = lemas[i : i + window_size]
        ttrs.append(len(set(janela)) / window_size)

    return float(sum(ttrs) / len(ttrs))


def preparar_texto_comum(texto_bruto: str, max_tokens: int = 500) -> Tuple[str, str, int]:
    """
    Função unificada de preprocessing e truncamento para treino e inferência.
    Garante que o texto seja limpo e truncado da mesma forma nos dois pipelines.
    Retorna (texto_cru_trunc, texto_limpo, num_links).
    """
    if not isinstance(texto_bruto, str) or pd.isna(texto_bruto) if 'pd' in globals() else not texto_bruto:
        return "", "", 0

    texto_sanitizado = sanitizar_texto_noticia(texto_bruto)

    palavras_cruas = texto_sanitizado.split()
    if len(palavras_cruas) > max_tokens:
        texto_cru_trunc = " ".join(palavras_cruas[:max_tokens])
    else:
        texto_cru_trunc = texto_sanitizado

    links = re.findall(r"(?:https?://|www\.)[^\s]+", texto_cru_trunc)
    num_links = len(links)

    texto_limpo = re.sub(r"(?:https?://|www\.)[^\s]+", "", texto_cru_trunc)
    # Importante: texto_limpo é convertido para lower case para paridade total com o TF-IDF
    texto_limpo = re.sub(r"[\s\xa0\u200b]+", " ", texto_limpo).strip().lower()

    return texto_cru_trunc, texto_limpo, num_links


def extrair_pacote_analise(texto_bruto: str, max_tokens: int = 500) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Processa o texto sob o limite de truncamento e retorna:
    1. Dicionário das 15 features estilométricas (com MATTR em trunc_diversity).
    2. Dicionário com as 3 representações de texto: texto_cru, texto_limpo, texto_lematizado.
    """
    import pandas as pd
    texto_cru_trunc, texto_limpo, num_links = preparar_texto_comum(texto_bruto, max_tokens)

    doc = nlp(texto_limpo)
    sents = list(doc.sents)
    num_sentences = max(len(sents), 1)

    tokens_validos = [t for t in doc if not t.is_space]
    tokens_palavras = [t for t in tokens_validos if t.is_alpha]
    num_words = max(len(tokens_palavras), 1)

    pausas_count = sum(1 for t in tokens_validos if t.text in PONTUACOES_PAUSALIDADE)
    trunc_pausality = pausas_count / num_sentences

    lemas_palavras = [t.lemma_.lower() for t in tokens_palavras]
    mattr_25 = calcular_mattr(lemas_palavras, window_size=25)

    texto_lematizado = " ".join([t.lemma_.lower() for t in tokens_validos if not t.is_punct and not t.is_stop])

    caixa_alta_count = sum(1 for t in tokens_palavras if t.text.isupper() and len(t.text) > 1)
    trunc_upper_case_density = caixa_alta_count / num_words

    pos_counts = {"VERB": 0, "NOUN": 0, "ADJ": 0, "ADV": 0, "PRON": 0}
    modal_verbs_count = 0
    subj_imp_count = 0
    pron_1_2_sing_count = 0
    pron_1_plur_count = 0
    spell_candidates: List[str] = []

    doc_length = len(doc)
    for i, t in enumerate(doc):
        if not t.is_alpha:
            continue
        pos = t.pos_
        t_lower = t.lower_

        if pos in ("VERB", "AUX"):
            pos_counts["VERB"] += 1
        elif pos in ("NOUN", "PROPN"):
            pos_counts["NOUN"] += 1
        elif pos == "ADJ":
            pos_counts["ADJ"] += 1
        elif pos == "ADV":
            pos_counts["ADV"] += 1
        elif pos == "PRON":
            pos_counts["PRON"] += 1

        if t.lemma_ in MODAL_LEMMAS:
            modal_verbs_count += 1
        elif t.lemma_ == "ter" and i < doc_length - 1:
            if doc[i + 1].lower_ in ("que", "de"):
                modal_verbs_count += 1

        moods = t.morph.get("Mood")
        if moods and ("Sub" in moods or "Imp" in moods):
            subj_imp_count += 1

        person = t.morph.get("Person")
        number = t.morph.get("Number")
        is_1_2_sing = (
            pos == "PRON" and person and ("1" in person or "2" in person) and number and "Sing" in number
        ) or (t_lower in PRON_1_2_SING)
        if is_1_2_sing:
            pron_1_2_sing_count += 1

        is_1_plur = (
            pos == "PRON" and person and "1" in person and number and "Plur" in number
        ) or (t_lower in PRON_1_PLUR)
        if is_1_plur:
            pron_1_plur_count += 1

        if t.is_lower and len(t.text) > 2 and pos != "PROPN":
            spell_candidates.append(t.text)

    rc_spelling_errors = 0.0
    if spell_candidates:
        erros_unicos = spell.unknown(list(set(spell_candidates)))
        qtd_erros = sum(1 for word in spell_candidates if word in erros_unicos)
        rc_spelling_errors = qtd_erros / len(spell_candidates)

    emotiveness_den = pos_counts["NOUN"] + pos_counts["VERB"]
    trunc_emotiveness = (pos_counts["ADJ"] + pos_counts["ADV"]) / emotiveness_den if emotiveness_den > 0 else 0.0
    total_verbos = max(pos_counts["VERB"], 1)

    features = {
        "trunc_pausality": round(float(trunc_pausality), 4),
        "trunc_emotiveness": round(float(trunc_emotiveness), 4),
        "trunc_diversity": round(float(mattr_25), 4),
        "trunc_upper_case_density": round(float(trunc_upper_case_density), 4),
        "trunc_verb_density": round(float(pos_counts["VERB"] / num_words), 4),
        "trunc_noun_density": round(float(pos_counts["NOUN"] / num_words), 4),
        "trunc_adj_density": round(float(pos_counts["ADJ"] / num_words), 4),
        "trunc_adv_density": round(float(pos_counts["ADV"] / num_words), 4),
        "trunc_pron_density": round(float(pos_counts["PRON"] / num_words), 4),
        "link_density": round(float(num_links / num_words), 4),
        "rc_spelling_errors": round(float(rc_spelling_errors), 4),
        "rc_modal_verbs_density": round(float(modal_verbs_count / total_verbos), 4),
        "rc_subj_imp_verbs_density": round(float(subj_imp_count / total_verbos), 4),
        "rc_pron_1_2_sing_density": round(float(pron_1_2_sing_count / num_words), 4),
        "rc_pron_1_plur_density": round(float(pron_1_plur_count / num_words), 4),
    }

    textos_processados = {
        "texto_cru": texto_cru_trunc,
        "texto_limpo": texto_limpo,
        "texto_lematizado": texto_lematizado,
    }

    return features, textos_processados