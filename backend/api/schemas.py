from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class MetricasEstilometricas(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    trunc_pausality: float
    trunc_emotiveness: float
    trunc_diversity: float
    trunc_upper_case_density: float
    trunc_verb_density: float
    trunc_noun_density: float
    trunc_adj_density: float
    trunc_adv_density: float
    trunc_pron_density: float
    link_density: float
    rc_spelling_errors: float
    rc_modal_verbs_density: float
    rc_subj_imp_verbs_density: float
    rc_pron_1_2_sing_density: float
    rc_pron_1_plur_density: float


class ContagemAvaliacoes(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    verdadeiro: int = 0
    duvidoso: int = 0
    falso: int = 0
    total: int = 0


class AnaliseRequest(BaseModel):
    # Removido min_length rígido do Pydantic para deixar a regra de negócio (>30 palavras)
    # em filtro.py emitir o HTTP 400 Bad Request esperado pela API
    texto: str = Field(default="", description="Texto da notícia para análise")
    text: Optional[str] = Field(None, description="Alias em inglês para compatibilidade")
    url: Optional[str] = Field(None, description="URL de origem opcional")

    @model_validator(mode="before")
    @classmethod
    def compatibilidade_texto(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("texto") and values.get("text"):
                values["texto"] = values["text"]
        return values


class AnaliseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    hash_texto: str
    prob_suspeita: float
    faixa: str
    modelo_f1: float
    orientacao: str
    metricas: MetricasEstilometricas
    avaliacoes_comunidade: ContagemAvaliacoes
    features: Optional[Dict[str, float]] = None
    texto_truncado: Optional[str] = None
    total_palavras_truncado: Optional[int] = None


class AvaliacaoRequest(BaseModel):
    noticia_id: int
    client_id: str
    # 0 = Verdadeiro, 1 = Duvidoso, 2 = Falso
    avaliacao: int = Field(..., ge=0, le=2, description="Voto da comunidade: 0=Verdadeiro, 1=Duvidoso, 2=Falso")


class AvaliacaoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sucesso: bool
    mensagem: str
    avaliacoes_atualizadas: ContagemAvaliacoes