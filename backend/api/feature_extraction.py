import spacy
import re
import logging
from spellchecker import SpellChecker
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Carregamento resiliente do modelo SpaCy
def _load_spacy_model():
    models_to_try = ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"]
    for model_name in models_to_try:
        try:
            loaded_nlp = spacy.load(model_name, disable=["ner"])
            logger.info(f"Modelo SpaCy carregado com sucesso: {model_name}")
            return loaded_nlp
        except OSError:
            continue
    raise RuntimeError(
        "Nenhum modelo SpaCy em português encontrado. "
        "Instale executando: pip install pt_core_news_sm ou python -m spacy download pt_core_news_sm"
    )

nlp = _load_spacy_model()
spell = SpellChecker(language='pt')

# Conjuntos imutáveis para buscas O(1)
MODAL_LEMMAS = frozenset(['poder', 'dever', 'precisar'])
PRON_1_2_SING = frozenset(['você', 'tu', 'te', 'ti', 'contigo', 'teu', 'tua', 'teus', 'tuas'])
PRON_1_PLUR = frozenset(['nós', 'nos', 'conosco', 'nosso', 'nossa', 'nossos', 'nossas'])

# Padrões de cauda (disclaimers, comentários, avisos institucionais)
TAIL_BOILERPLATE_PATTERNS = [
    re.compile(r'(?i)continua\s+após\s+a\s+publicidade.*', re.DOTALL),
    re.compile(r'(?i)opinião\s*texto\s+em\s+que\s+o\s+autor\s+apresenta.*', re.DOTALL),
    re.compile(r'(?i)este\s+texto\s+não\s+reflete.*', re.DOTALL),
    re.compile(r'(?i)comunicar\s+erro.*', re.DOTALL),
    re.compile(r'(?i)deixe\s+seu\s+comentário.*', re.DOTALL),
    re.compile(r'(?i)o\s+autor\s+da\s+mensagem.*', re.DOTALL),
    re.compile(r'(?i)leia\s+as\s+regras\s+de\s+uso.*', re.DOTALL),
]

def sanitizar_texto_noticia(texto: str) -> str:
    """
    Remove ruídos comuns de portais de notícias:
    - Metadados de player de áudio (ex: '0:00 / 0:00x')
    - Disclaimers e seções de comentários no final da matéria
    - Legendas de imagens residuais
    """
    if not texto or not isinstance(texto, str):
        return ""

    limpo = texto

    # 1. Remove disclaimers de rodapé comuns
    for pattern in TAIL_BOILERPLATE_PATTERNS:
        limpo = pattern.sub('', limpo)

    # 2. Remove player de áudio e timestamp no início do texto (ex: "0:00 / 0:00x")
    # Caso haja cabeçalho de autoria seguido por tempo de áudio antes do corpo
    limpo = re.sub(r'^.*?\d{1,2}:\d{2}\s*/\s*\d{1,2}:\d{2}x?\s*', '', limpo, flags=re.DOTALL)


    # 3. Remove legendas e referências a imagens comuns
    limpo = re.sub(r'(?i)imagem:\s*arte/uol', '', limpo)
    limpo = re.sub(r'(?i)veja\s+horários\s+das\s+lives[^\n.]*', '', limpo)

    return limpo

def pre_processar_e_truncar(texto_bruto: str, max_tokens: int = 500) -> str:
    """Valida, sanitiza ruídos, normaliza espaços e aplica truncamento seguro."""
    if not texto_bruto or not isinstance(texto_bruto, str):
        return ""
    
    # 1. Sanitização defensiva de ruídos de portais
    texto_sanitizado = sanitizar_texto_noticia(texto_bruto)
    
    # 2. Normaliza quebras de linha e caracteres especiais de espaço (\xa0, \u200b)
    texto_norm = re.sub(r'[\s\xa0\u200b]+', ' ', texto_sanitizado).strip()
    
    # 3. Truncamento seguro de até 500 tokens
    tokens = texto_norm.split()
    if len(tokens) > max_tokens:
        texto_norm = " ".join(tokens[:max_tokens])
        
    return texto_norm

def get_default_features(url: str = "") -> Dict[str, Any]:
    """Fallback: Retorna atributos zerados caso o texto seja inválido ou o processo falhe."""
    return {
        "url": url,
        "texto_normalizado": "",
        "trunc_pausality": 0.0,
        "trunc_emotiveness": 0.0,
        "trunc_upper_case_density": 0.0,
        "trunc_verb_density": 0.0,
        "trunc_noun_density": 0.0,
        "trunc_adj_density": 0.0,
        "trunc_adv_density": 0.0,
        "trunc_pron_density": 0.0,
        "link_density": 0.0,
        "rc_spelling_errors": 0.0,
        "rc_modal_verbs_density": 0.0,
        "rc_subj_imp_verbs_density": 0.0,
        "rc_pron_1_2_sing_density": 0.0,
        "rc_pron_1_plur_density": 0.0
    }

def extract_features(url: str, texto_bruto: str) -> Dict[str, Any]:
    """
    Extrai 14 features estilométricas e linguísticas para detecção de credibilidade/viés.
    """
    try:
        # 1. Validação, sanitização e truncamento
        texto_norm = pre_processar_e_truncar(texto_bruto)
        if not texto_norm:
            return get_default_features(url)
        
        # 2. Extração e sanitização de links
        links = re.findall(r'(?:https?://|www\.)[^\s]+', texto_norm)
        num_links = len(links)
        
        # Remove URLs para não interferir na análise sintática e ortográfica
        texto_limpo = re.sub(r'(?:https?://|www\.)[^\s]+', '', texto_norm).strip()
        if not texto_limpo:
            return get_default_features(url)
        
        # 3. Processamento com SpaCy
        doc = nlp(texto_limpo)
        
        sents = list(doc.sents)
        num_sentences = max(len(sents), 1)
        
        tokens_palavras = [t for t in doc if t.is_alpha]
        num_words = max(len(tokens_palavras), 1)
        
        # 4. Inicialização de Contadores
        pontuacao_count = sum(1 for t in doc if t.is_punct)
        caixa_alta_count = sum(1 for t in tokens_palavras if t.text.isupper() and len(t.text) > 1)
        
        pos_counts = {'VERB': 0, 'NOUN': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0}
        modal_verbs_count = 0
        subj_imp_count = 0
        pron_1_2_sing_count = 0
        pron_1_plur_count = 0
        spell_candidates: List[str] = []
        
        # 5. Iteração Otimizada
        doc_length = len(doc)
        for i, t in enumerate(doc):
            if not t.is_alpha:
                continue
                
            pos = t.pos_
            t_lower = t.lower_
            
            # Contagem morfológica (POS Tags)
            if pos in ('VERB', 'AUX'):
                pos_counts['VERB'] += 1
            elif pos in ('NOUN', 'PROPN'):
                pos_counts['NOUN'] += 1
            elif pos == 'ADJ':
                pos_counts['ADJ'] += 1
            elif pos == 'ADV':
                pos_counts['ADV'] += 1
            elif pos == 'PRON':
                pos_counts['PRON'] += 1
            
            # Verbos modais: lema ou locução 'ter que/de'
            if t.lemma_ in MODAL_LEMMAS:
                modal_verbs_count += 1
            elif t.lemma_ == 'ter' and i < doc_length - 1:
                if doc[i + 1].lower_ in ('que', 'de'):
                    modal_verbs_count += 1
                
            # Modo verbal: Subjuntivo ou Imperativo
            moods = t.morph.get('Mood')
            if moods and ('Sub' in moods or 'Imp' in moods):
                subj_imp_count += 1
                
            # Pronomes de 1ª/2ª pessoa (singular e plural)
            person = t.morph.get('Person')
            number = t.morph.get('Number')
            
            is_1_2_sing = (
                pos == 'PRON' and person and ('1' in person or '2' in person) and number and 'Sing' in number
            ) or (t_lower in PRON_1_2_SING)
            if is_1_2_sing:
                pron_1_2_sing_count += 1
                
            is_1_plur = (
                pos == 'PRON' and person and '1' in person and number and 'Plur' in number
            ) or (t_lower in PRON_1_PLUR)
            if is_1_plur:
                pron_1_plur_count += 1
                
            # Candidatos para verificação ortográfica (palavras minúsculas comuns > 2 letras)
            if t.is_lower and len(t.text) > 2 and pos != 'PROPN':
                spell_candidates.append(t.text)
                
        # 6. Cálculos Seguros de Densidade e Ortografia
        rc_spelling_errors = 0.0
        if spell_candidates:
            unique_candidates = list(set(spell_candidates))
            erros_unicos = spell.unknown(unique_candidates)
            qtd_erros = sum(1 for word in spell_candidates if word in erros_unicos)
            rc_spelling_errors = qtd_erros / len(spell_candidates)
        
        # Emotividade = (Adjetivos + Advérbios) / (Substantivos + Verbos)
        emotiveness_den = pos_counts['NOUN'] + pos_counts['VERB']
        trunc_emotiveness = (pos_counts['ADJ'] + pos_counts['ADV']) / emotiveness_den if emotiveness_den > 0 else 0.0
        
        # 7. Retorno do Dicionário
        return {
            "url": url,
            "texto_normalizado": texto_norm,
            "trunc_pausality": round(pontuacao_count / num_sentences, 4),
            "trunc_emotiveness": round(trunc_emotiveness, 4),
            "trunc_upper_case_density": round(caixa_alta_count / num_words, 4),
            "trunc_verb_density": round(pos_counts['VERB'] / num_words, 4),
            "trunc_noun_density": round(pos_counts['NOUN'] / num_words, 4),
            "trunc_adj_density": round(pos_counts['ADJ'] / num_words, 4),
            "trunc_adv_density": round(pos_counts['ADV'] / num_words, 4),
            "trunc_pron_density": round(pos_counts['PRON'] / num_words, 4),
            "link_density": round(num_links / num_words, 4),
            "rc_spelling_errors": round(rc_spelling_errors, 4),
            "rc_modal_verbs_density": round(modal_verbs_count / num_words, 4),
            "rc_subj_imp_verbs_density": round(subj_imp_count / num_words, 4),
            "rc_pron_1_2_sing_density": round(pron_1_2_sing_count / num_words, 4),
            "rc_pron_1_plur_density": round(pron_1_plur_count / num_words, 4)
        }
        
    except Exception as e:
        logger.exception(f"Exceção ao extrair features da URL '{url}': {str(e)}")
        return get_default_features(url)