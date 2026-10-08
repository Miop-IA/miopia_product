from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from api.database import Base



class Noticia(Base):
    __tablename__ = "noticias"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hash_texto = Column(String(64), index=True, nullable=False)
    url = Column(String(500), nullable=True)
    texto = Column(Text, nullable=False)
    texto_truncado = Column(Text, nullable=True)
    
    model_version = Column(String(50), nullable=False, default="unknown")
    pipeline_version = Column(String(50), nullable=False, default="unknown")
    
    # Métricas preditivas
    prob_suspeita = Column(Float, nullable=False)
    faixa = Column(String(30), nullable=False)
    modelo_f1 = Column(Float, default=0.961)

    # 15 Características Estilométricas (com MATTR-25 em trunc_diversity)
    trunc_pausality = Column(Float, nullable=False)
    trunc_emotiveness = Column(Float, nullable=False)
    trunc_diversity = Column(Float, nullable=False)
    trunc_upper_case_density = Column(Float, nullable=False)
    trunc_verb_density = Column(Float, nullable=False)
    trunc_noun_density = Column(Float, nullable=False)
    trunc_adj_density = Column(Float, nullable=False)
    trunc_adv_density = Column(Float, nullable=False)
    trunc_pron_density = Column(Float, nullable=False)
    link_density = Column(Float, nullable=False)
    rc_spelling_errors = Column(Float, nullable=False)
    rc_modal_verbs_density = Column(Float, nullable=False)
    rc_subj_imp_verbs_density = Column(Float, nullable=False)
    rc_pron_1_2_sing_density = Column(Float, nullable=False)
    rc_pron_1_plur_density = Column(Float, nullable=False)

    criado_em = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('hash_texto', 'model_version', 'pipeline_version', name='_hash_model_pipeline_uc'),
    )

    # Relacionamento com as avaliações da comunidade
    avaliacoes = relationship("Avaliacao", back_populates="noticia", cascade="all, delete-orphan")


class Avaliacao(Base):
    __tablename__ = "avaliacoes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    noticia_id = Column(Integer, ForeignKey("noticias.id", ondelete="CASCADE"), nullable=False)
    client_id = Column(String(100), nullable=False, index=True)
    
    # 0 = Verdadeiro, 1 = Duvidoso, 2 = Falso
    avaliacao = Column(Integer, nullable=False)
    criado_em = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    noticia = relationship("Noticia", back_populates="avaliacoes")

    __table_args__ = (
        UniqueConstraint("noticia_id", "client_id", name="uq_noticia_cliente"),
    )