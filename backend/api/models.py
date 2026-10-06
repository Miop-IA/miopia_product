from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class NewsRequest(BaseModel):
    url: str = Field(default="", description="URL de origem da notícia")
    text: str = Field(..., min_length=1, max_length=500_000, description="Texto da notícia a ser analisada")

class StylometricFeatures(BaseModel):
    url: str
    texto_normalizado: str
    trunc_pausality: float = Field(..., description="Pausalidade: pontuação por sentença")
    trunc_emotiveness: float = Field(..., description="Índice de emotividade (adj + adv) / (noun + verb)")
    trunc_upper_case_density: float = Field(..., description="Densidade de palavras em caixa alta")
    trunc_verb_density: float = Field(..., description="Densidade de verbos")
    trunc_noun_density: float = Field(..., description="Densidade de substantivos")
    trunc_adj_density: float = Field(..., description="Densidade de adjetivos")
    trunc_adv_density: float = Field(..., description="Densidade de advérbios")
    trunc_pron_density: float = Field(..., description="Densidade de pronomes")
    link_density: float = Field(..., description="Densidade de links")
    rc_spelling_errors: float = Field(..., description="Taxa de possíveis erros ortográficos")
    rc_modal_verbs_density: float = Field(..., description="Densidade de verbos modais")
    rc_subj_imp_verbs_density: float = Field(..., description="Densidade de verbos no subjuntivo/imperativo")
    rc_pron_1_2_sing_density: float = Field(..., description="Densidade de pronomes de 1ª/2ª pessoa do singular")
    rc_pron_1_plur_density: float = Field(..., description="Densidade de pronomes de 1ª pessoa do plural")

class NewsResponse(BaseModel):
    success: bool
    texto_original: str = Field(..., description="Texto completo recebido da página")
    texto_truncado: str = Field(..., description="Texto normalizado e truncado (até 500 tokens)")
    total_caracteres_original: int
    total_palavras_original: int
    total_palavras_truncado: int
    features: StylometricFeatures
    prediction: Optional[Dict[str, Any]] = Field(default=None, description="Resultado do modelo classificador")
    message: Optional[str] = None