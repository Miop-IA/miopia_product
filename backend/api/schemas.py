from enum import IntEnum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, HttpUrl, field_validator
import uuid


class FaixaRisco(str):
    CONFIÁVEL = "Confiavel"
    ATENÇÃO = "Atencao"
    SUSPEITA = "Suspeita"


class TipoAvaliacao(IntEnum):
    VERDADEIRO = 0
    DUVIDOSO = 1
    FALSO = 2


# ---------------------------------------------------------
# Schemas de Entrada (Requests)
# ---------------------------------------------------------

class AnaliseRequest(BaseModel):
    """
    Payload de entrada para o endpoint de inferência /analisar.
    """
    texto: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Corpo textual da matéria a ser analisado"
    )
    url: Optional[str] = Field(
        None,
        max_length=2048,
        description="URL da notícia (opcional para rastreabilidade)"
    )

    @field_validator("texto")
    @classmethod
    def normalizar_espacos(cls, v: str) -> str:
        v_stripped = v.strip()
        if len(v_stripped) < 50:
            raise ValueError("O texto deve conter pelo menos 50 caracteres úteis.")
        return v_stripped


class AvaliacaoRequest(BaseModel):
    """
    Payload de entrada para o endpoint de feedback /avaliar.
    """
    noticia_id: int = Field(
        ...,
        gt=0,
        description="ID numérico da notícia retornado na análise prévia"
    )
    client_id: uuid.UUID = Field(
        ...,
        description="UUID persistente da extensão para controle de unicidade"
    )
    avaliacao: TipoAvaliacao = Field(
        ...,
        description="Voto do usuário: 0=Verdadeiro, 1=Duvidoso, 2=Falso"
    )


# ---------------------------------------------------------
# Schemas de Saída (Responses)
# ---------------------------------------------------------

class MetricasEstilometricas(BaseModel):
    """
    Indicadores estilométricos truncados e recalculados calculados pelo pipeline.
    """
    trunc_pausality: Optional[float] = None
    trunc_emotiveness: Optional[float] = None
    trunc_diversity: Optional[float] = None
    trunc_upper_case_density: Optional[float] = None
    trunc_verb_density: Optional[float] = None
    trunc_noun_density: Optional[float] = None
    trunc_adj_density: Optional[float] = None
    trunc_adv_density: Optional[float] = None
    trunc_pron_density: Optional[float] = None
    link_density: Optional[float] = None
    rc_spelling_errors: Optional[float] = None
    rc_modal_verbs_density: Optional[float] = None
    rc_subj_imp_verbs_density: Optional[float] = None
    rc_pron_1_2_sing_density: Optional[float] = None
    rc_pron_1_plur_density: Optional[float] = None


class ContagemAvaliacoes(BaseModel):
    """
    Consolidação agregada dos palpites da comunidade.
    """
    verdadeiro: int = Field(0, ge=0)
    duvidoso: int = Field(0, ge=0)
    falso: int = Field(0, ge=0)
    total: int = Field(0, ge=0)


class AnaliseResponse(BaseModel):
    """
    Contrato final entregue à extensão Chrome contendo o veredito e métricas.
    """
    id: int = Field(..., description="ID da notícia gravada para submissão de feedback")
    hash_texto: str = Field(..., description="Hash SHA-256 do conteúdo normalizado")
    prob_suspeita: float = Field(..., description="Probabilidade calibrada de desinformação [0.0 a 1.0]")
    faixa: str = Field(..., description="Faixa qualitativa: Confiavel, Atencao ou Suspeita")
    modelo_f1: float = Field(..., description="F1-Score macro do modelo treinado")
    orientacao: str = Field(..., description="Mensagem contextual de apoio ao leitor")
    metricas: MetricasEstilometricas
    avaliacoes_comunidade: ContagemAvaliacoes

    class Config:
        from_attributes = True


class AvaliacaoResponse(BaseModel):
    """
    Retorno da confirmação de voto computado.
    """
    sucesso: bool
    mensagem: str
    avaliacoes_atualizadas: ContagemAvaliacoes