from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, Request

#from api.config
#from api.database
#from api.models

CAMINHO_MODELO = Path(__file__).parent.parent / "modelos" / "classificador.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = joblib.load(CAMINHO_MODELO)   # roda ao iniciar
    yield

app = FastAPI(lifespan=lifespan)