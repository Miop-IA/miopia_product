import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

import json

from api.config import get_settings
from api.database import engine, Base, get_db
from api.models import Noticia, Avaliacao, Modelo
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
from api.inferencia import (
    carregar_bundle_stacking,
    classificar_faixa_e_orientacao,
    predizer_risco_stacking,
)
from api.bundle_spec import ModeloInvalidoError, resumo_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miopia_api")
settings = get_settings()

def get_current_model_info() -> dict:
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(base_dir, "models", "model_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "model_version": data.get("model_version", "unknown"),
                    "pipeline_version": data.get("pipeline_version", "unknown"),
                    "dataset_version": data.get("dataset_version", "unknown"),
                    "f1": float(data.get("F1", 0.0)),
                    "threshold": float(data.get("threshold", 0.0))
                }
        except Exception as e:
            logger.error(f"Erro ao ler model_manifest.json: {e}")
    return {
        "model_version": "unknown", "pipeline_version": "unknown",
        "dataset_version": "unknown", "f1": 0.0, "threshold": 0.0
    }

def get_or_create_modelo(db: Session, model_info: dict) -> Modelo:
    modelo = db.query(Modelo).filter(
        Modelo.model_version == model_info["model_version"],
        Modelo.pipeline_version == model_info["pipeline_version"]
    ).first()
    
    if not modelo:
        modelo = Modelo(
            model_version=model_info["model_version"],
            pipeline_version=model_info["pipeline_version"],
            dataset_version=model_info["dataset_version"],
            f1=model_info["f1"],
            threshold=model_info["threshold"]
        )
        db.add(modelo)
        db.commit()
        db.refresh(modelo)
    
    return modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bundle ausente, corrompido ou incompatível impede a API de ficar pronta.
    # As tabelas são gerenciadas pelo Alembic ('alembic upgrade head').
    try:
        bundle = carregar_bundle_stacking()
    except ModeloInvalidoError as e:
        logger.critical(f"Modelo de produção inválido; abortando startup: {e}")
        raise
    logger.info(f"Modelo de produção pronto: {resumo_bundle(bundle)}")
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


def obter_bundle_ou_503():
    try:
        return carregar_bundle_stacking()
    except ModeloInvalidoError as e:
        logger.error(f"Modelo de produção indisponível: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de classificação indisponível.",
        )


@app.get("/health", tags=["Monitoramento"])
def health_check():
    """Retorna sucesso caso a API esteja operando."""
    meta = resumo_bundle(obter_bundle_ou_503())
    return {
        "status": "online",
        "environment": settings.environment,
        "model": f"Stacking {meta['version']} (F1={meta['f1_score']:.3f})",
        "model_version": meta["version"],
        "feature_count": meta["feature_count"],
    }


@app.get("/ready", tags=["Monitoramento"])
def readiness_check(db: Session = Depends(get_db)):
    """
    Retorna 200 OK se e somente se:
    1. O modelo está carregado em memória.
    2. O bundle do modelo possui as chaves obrigatórias.
    3. O banco de dados está acessível.
    """
    reasons = []

    # 1 e 2. Validação do Bundle e Modelo Carregado
    try:
        from api.inferencia import carregar_bundle_stacking
        bundle = carregar_bundle_stacking()
        
        required_keys = [
            "tfidf_char", "svm_caracteres", "tfidf_word", "svm_palavras",
            "xgb_denso", "meta_modelo", "f1_score", "limiar"
        ]
        missing = [k for k in required_keys if k not in bundle]
        if missing:
            reasons.append(f"Bundle incompleto. Chaves ausentes: {missing}")
    except Exception as e:
        reasons.append(f"Erro ao carregar modelo: {str(e)}")

    # 3. Validação do Banco de Dados
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception as e:
        reasons.append(f"Banco de dados inacessível: {str(e)}")

    if reasons:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "reasons": reasons}
        )

    return {"status": "ready"}


@app.post("/analisar", response_model=AnaliseResponse, status_code=status.HTTP_200_OK, tags=["Análise"])
def analisar_noticia(payload: AnaliseRequest, db: Session = Depends(get_db)):
    texto_puro = payload.texto
    bundle = obter_bundle_ou_503()

    is_valido, motivo, _ = validar_viabilidade_analise(texto_puro)
    if not is_valido:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=motivo)

    hash_txt = gerar_hash_texto(texto_puro)
    model_info = get_current_model_info()
    modelo_atual = get_or_create_modelo(db, model_info)

    # 1. Consulta em Cache O(1)
    noticia_existente = db.query(Noticia).filter(
        Noticia.hash_texto == hash_txt,
        Noticia.modelo_id == modelo_atual.id
    ).first()

    if noticia_existente:
        logger.info(f"Cache hit para a hash: {hash_txt}")
        avaliacoes = obter_contagem_avaliacoes(db, noticia_existente.id)
        _, orientacao = classificar_faixa_e_orientacao(noticia_existente.prob_suspeita, limiar=modelo_atual.threshold)

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
            modelo_f1=modelo_atual.f1,
            model_version=modelo_atual.model_version,
            pipeline_version=modelo_atual.pipeline_version,
            orientacao=orientacao,
            metricas=metricas_dto,
            avaliacoes_comunidade=avaliacoes,
            features=metricas_dto.model_dump(),
            texto_truncado=noticia_existente.texto_truncado,
            total_palavras_truncado=len(noticia_existente.texto_truncado.split()) if noticia_existente.texto_truncado else 0,
        )

    # 2. Processamento de texto inédito
    features, textos = extrair_pacote_analise(texto_puro, num_links_param=payload.num_links)
    
    try:
        prob_suspeita, faixa, orientacao, f1_score = predizer_risco_stacking(features, textos)
    except ModeloInvalidoError as e:
        logger.error(f"Inferência recusada, modelo inválido: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de classificação indisponível."
        )

    texto_trunc = textos.get("texto_cru", texto_puro)

    nova_noticia = Noticia(
        url=payload.url,
        hash_texto=hash_txt,
        texto_truncado=texto_trunc,
        prob_suspeita=prob_suspeita,
        faixa=faixa,
        modelo_id=modelo_atual.id,
        **features,
    )
    db.add(nova_noticia)
    db.commit()
    db.refresh(nova_noticia)

    metricas_dto = MetricasEstilometricas(**features)
    avaliacoes = ContagemAvaliacoes(verdadeiro=0, duvidoso=0, falso=0, total=0)
    texto_trunc = textos.get("texto_cru", texto_puro)
    total_palavras_trunc = len(texto_trunc.split()) if texto_trunc else 0

    return AnaliseResponse(
        id=nova_noticia.id,
        hash_texto=nova_noticia.hash_texto,
        prob_suspeita=nova_noticia.prob_suspeita,
        faixa=nova_noticia.faixa,
        modelo_f1=modelo_atual.f1,
        model_version=modelo_atual.model_version,
        pipeline_version=modelo_atual.pipeline_version,
        orientacao=orientacao,
        metricas=metricas_dto,
        avaliacoes_comunidade=avaliacoes,
        features=features,
        texto_truncado=texto_trunc,
        total_palavras_truncado=total_palavras_trunc,
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