import sys
import os
import logging

# Adiciona a raiz do backend ao sys.path para importações absolutas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import engine, Base
from api.models import Noticia, Avaliacao

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("init_db")


def inicializar_banco():
    """
    Cria as tabelas declaradas nos modelos SQLAlchemy caso ainda não existam.
    """
    try:
        url_segura = engine.url.render_as_string(hide_password=True)
        logger.info(f"A ligar à base de dados configurada: {url_segura}")
        Base.metadata.create_all(bind=engine)
        logger.info("Tabelas 'noticias' e 'avaliacoes' criadas/verificadas com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao inicializar a base de dados: {e}")
        sys.exit(1)


if __name__ == "__main__":
    inicializar_banco()