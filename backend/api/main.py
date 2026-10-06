import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib

try:
    from api.config import get_settings
    from api.filtro import validar_viabilidade_analise
    from api.models import NewsRequest, NewsResponse, StylometricFeatures
    from api.feature_extraction import extract_features
except ImportError:
    from .config import get_settings
    from .filtro import validar_viabilidade_analise
    from .models import NewsRequest, NewsResponse, StylometricFeatures
    from .feature_extraction import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("miopia.api")

settings = get_settings()

CAMINHO_MODELO = Path(__file__).parent.parent / "modelos" / "classificador.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if CAMINHO_MODELO.exists():
        try:
            app.state.modelo = joblib.load(CAMINHO_MODELO)
            logger.info(f"Modelo carregado com sucesso de {CAMINHO_MODELO}")
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo de {CAMINHO_MODELO}: {e}")
            app.state.modelo = None
    else:
        logger.info(f"Arquivo de modelo não encontrado em {CAMINHO_MODELO}. Iniciando com app.state.modelo = None")
        app.state.modelo = None
    yield

app = FastAPI(
    title="MiopIA API",
    description="API para análise estilométrica e detecção de credibilidade de notícias.",
    version="1.0.0",
    lifespan=lifespan
)

origins = settings.cors_origins if isinstance(settings.cors_origins, list) else [settings.cors_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def health_check():
    return {
        "status": "healthy",
        "app": "MiopIA API",
        "version": "1.0.0"
    }

@app.post(
    "/analisar",
    response_model=NewsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Análise"]
)
async def analisar_noticia(request: NewsRequest):
    """
    Valida volume textual (filtro), extrai features estilométricas,
    trunca o texto (até 500 tokens) e retorna as características para predição.
    """
    # 1. Validação de tamanho da notícia utilizando o filtro da dev
    is_valido, motivo, _ = validar_viabilidade_analise(request.text)
    if not is_valido:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=motivo
        )

    palavras_original = request.text.split()
    total_palavras_original = len(palavras_original)
    total_caracteres_original = len(request.text)

    logger.info(
        f"Analisando notícia. URL: {request.url or 'N/A'} | "
        f"Chars: {total_caracteres_original} | Palavras: {total_palavras_original}"
    )

    try:
        raw_features = extract_features(url=request.url, texto_bruto=request.text)
        features = StylometricFeatures(**raw_features)
        texto_truncado = features.texto_normalizado
        total_palavras_truncado = len(texto_truncado.split()) if texto_truncado else 0

        # Predição opcional se o modelo já estiver carregado no lifespan
        prediction = None
        if hasattr(app.state, "modelo") and app.state.modelo is not None:
            try:
                # Placeholder para predição futura via scikit-learn/joblib
                pass
            except Exception as e:
                logger.warning(f"Erro ao inferir com o modelo classificador: {e}")

        return NewsResponse(
            success=True,
            texto_original=request.text,
            texto_truncado=texto_truncado,
            total_caracteres_original=total_caracteres_original,
            total_palavras_original=total_palavras_original,
            total_palavras_truncado=total_palavras_truncado,
            features=features,
            prediction=prediction,
            message="Texto processado e features extraídas com sucesso."
        )
    except Exception as e:
        logger.exception(f"Erro ao processar notícia: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno de processamento: {str(e)}"
        )
