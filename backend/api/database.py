from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from api.config import get_settings

settings = get_settings()

db_url = settings.sqlalchemy_url

if db_url.startswith("sqlite"):
    # Configuração local para SQLite (desenvolvimento sem bloqueio de portas)
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    # Configuração de produção para PostgreSQL (Neon / Render)
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        echo=False,
        connect_args={"connect_timeout": 10}
    )

# Fábrica de sessões do SQLAlchemy
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base declarativa para os modelos ORM
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependência FastAPI que injeta a sessão do banco por requisição
    e garante o encerramento correto da conexão.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()