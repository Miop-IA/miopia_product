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
    Roda as migrações do Alembic para inicializar ou atualizar a base de dados.
    """
    from alembic.config import Config
    from alembic import command
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
        alembic_cfg.set_main_option("script_location", os.path.join(base_dir, "alembic"))
        logger.info("Executando migrações do Alembic...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Banco de dados inicializado com sucesso via Alembic.")
    except Exception as e:
        logger.error(f"Falha ao inicializar a base de dados: {e}")
        sys.exit(1)


if __name__ == "__main__":
    inicializar_banco()