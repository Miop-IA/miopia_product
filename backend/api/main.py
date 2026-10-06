import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.config import get_settings
from api.database import engine, Base, get_db
from api.models import Noticia, Avaliacao
from api.schemas import (
    AnaliseRequest,
    AnaliseResponse,
    AvaliacaoRequest,
    AvaliacaoResponse,
    ContagemAvaliacoes,
    MetricasEstilometricas,
)
from api.filtro import normalizar_texto, gerar_hash_texto, validar_viabilidade_analise
from api.feature_extraction import extrair_pacote_analise
from api.inferencia import predizer_risco_stacking, classificar_faixa_e_orientacao

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miopia_api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tabelas inicializadas com sucesso.")
    except Exception as e:
        logger.warning(f"Aviso de banco: {e}")
    yield


app = FastAPI(
    title="Miop.IA - API de Análise e Credibilidade de Notícias",
    description="Backend com Stacking Multivisão (SVM Char + SVM Word + XGBoost Estilo/Temas + Meta Regressão Logística).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if isinstance(settings.cors_origins, list) else [settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def obter_contagem_avaliacoes(db: Session, noticia_id: int) -> ContagemAvaliacoes:
    votos = (
        db.query(Avaliacao.avaliacao, func.count(Avaliacao.id))
        .filter(Avaliacao.noticia_id == noticia_id)
        .group_by(Avaliacao.avaliacao)
        .all()
    )
    mapa = {v: c for v, c in votos}
    return ContagemAvaliacoes(
        verdadeiro=mapa.get(0, 0),
        duvidoso=mapa.get(1, 0),
        falso=mapa.get(2, 0),
        total=sum(mapa.values()),
    )


@app.get("/health", tags=["Monitoramento"])
def health_check():
    return {"status": "online", "environment": settings.environment, "model": "Stacking Parte C (F1=0.961)"}


@app.post("/analisar", response_model=AnaliseResponse, status_code=status.HTTP_200_OK, tags=["Análise"])
def analisar_noticia(payload: AnaliseRequest, db: Session = Depends(get_db)):
    texto_puro = payload.texto

    is_valido, motivo, _ = validar_viabilidade_analise(texto_puro)
    if not is_valido:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=motivo)

    hash_txt = gerar_hash_texto(texto_puro)

    # 1. Consulta em Cache O(1)
    noticia_existente = db.query(Noticia).filter(Noticia.hash_texto == hash_txt).first()

    if noticia_existente:
        logger.info(f"Cache hit para a hash: {hash_txt}")
        avaliacoes = obter_contagem_avaliacoes(db, noticia_existente.id)
        _, orientacao = classificar_faixa_e_orientacao(noticia_existente.prob_suspeita)

        metricas_dto = MetricasEstilometricas(
            trunc_pausality=noticia_existente.trunc_pausality,
            trunc_emotiveness=noticia_existente.trunc_emotiveness,
            trunc_diversity=noticia_existente.trunc_diversity,
            trunc_upper_case_density=noticia_existente.trunc_upper_case_density,
            trunc_verb_density=noticia_existente.trunc_verb_density,
            trunc_noun_density=noticia_existente.trunc_noun_density,
            trunc_adj_density=noticia_existente.trunc_adj_density,
            trunc_adv_density=noticia_existente.trunc_adv_density,
            trunc_pron_density=noticia_existente.trunc_pron_density,
            link_density=noticia_existente.link_density,
            rc_spelling_errors=noticia_existente.rc_spelling_errors,
            rc_modal_verbs_density=noticia_existente.rc_modal_verbs_density,
            rc_subj_imp_verbs_density=noticia_existente.rc_subj_imp_verbs_density,
            rc_pron_1_2_sing_density=noticia_existente.rc_pron_1_2_sing_density,
            rc_pron_1_plur_density=noticia_existente.rc_pron_1_plur_density,
        )

        return AnaliseResponse(
            id=noticia_existente.id,
            hash_texto=noticia_existente.hash_texto,
            prob_suspeita=noticia_existente.prob_suspeita,
            faixa=noticia_existente.faixa,
            modelo_f1=noticia_existente.modelo_f1,
            orientacao=orientacao,
            metricas=metricas_dto,
            avaliacoes_comunidade=avaliacoes,
        )

    # 2. Processamento de texto inédito
    features, textos = extrair_pacote_analise(texto_puro)
    prob_suspeita, faixa, orientacao, f1_score = predizer_risco_stacking(features, textos)

    nova_noticia = Noticia(
        url=payload.url,
        hash_texto=hash_txt,
        texto=normalizar_texto(texto_puro),
        prob_suspeita=prob_suspeita,
        faixa=faixa,
        modelo_f1=f1_score,
        **features,
    )
    db.add(nova_noticia)
    db.commit()
    db.refresh(nova_noticia)

    metricas_dto = MetricasEstilometricas(**features)
    avaliacoes = ContagemAvaliacoes(verdadeiro=0, duvidoso=0, falso=0, total=0)

    return AnaliseResponse(
        id=nova_noticia.id,
        hash_texto=nova_noticia.hash_texto,
        prob_suspeita=nova_noticia.prob_suspeita,
        faixa=nova_noticia.faixa,
        modelo_f1=nova_noticia.modelo_f1,
        orientacao=orientacao,
        metricas=metricas_dto,
        avaliacoes_comunidade=avaliacoes,
    )


@app.post("/avaliar", response_model=AvaliacaoResponse, status_code=status.HTTP_200_OK, tags=["Feedback"])
def registrar_avaliacao(payload: AvaliacaoRequest, db: Session = Depends(get_db)):
    noticia = db.query(Noticia).filter(Noticia.id == payload.noticia_id).first()
    if not noticia:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notícia não encontrada para registro de voto.",
        )

    avaliacao_existente = (
        db.query(Avaliacao)
        .filter(
            Avaliacao.noticia_id == payload.noticia_id,
            Avaliacao.client_id == payload.client_id,
        )
        .first()
    )

    if avaliacao_existente:
        avaliacao_existente.avaliacao = payload.avaliacao
        db.commit()
        msg = "Voto atualizado com sucesso."
    else:
        novo_voto = Avaliacao(
            noticia_id=payload.noticia_id,
            client_id=payload.client_id,
            avaliacao=payload.avaliacao,
        )
        db.add(novo_voto)
        db.commit()
        msg = "Voto registrado com sucesso."

    contagem = obter_contagem_avaliacoes(db, payload.noticia_id)
    return AvaliacaoResponse(sucesso=True, mensagem=msg, avaliacoes_atualizadas=contagem)