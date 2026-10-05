import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from .models import NewsRequest, NewsResponse, StylometricFeatures
from .feature_extraction import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("miopia.api")

app = FastAPI(
    title="MiopIA API",
    description="API para análise estilométrica e detecção de credibilidade de notícias.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    Recebe a notícia bruta, extrai as features e retorna o texto completo,
    o texto truncado (500 tokens) e o vetor de características estilométricas.
    """
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

        return NewsResponse(
            success=True,
            texto_original=request.text,
            texto_truncado=texto_truncado,
            total_caracteres_original=total_caracteres_original,
            total_palavras_original=total_palavras_original,
            total_palavras_truncado=total_palavras_truncado,
            features=features,
            prediction=None,
            message="Texto processado e features extraídas com sucesso."
        )
    except Exception as e:
        logger.exception(f"Erro ao processar notícia: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno de processamento: {str(e)}"
        )