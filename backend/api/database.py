from typing import Generator
from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from api.config import get_settings

settings = get_settings()

db_url = settings.sqlalchemy_url

# Correção automática de prefixo legado do Neon / Render
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

if db_url.startswith("sqlite"):
    # Configuração local para SQLite (desenvolvimento)
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
else:
    # Configuração de produção para PostgreSQL (Neon / Render)
    # SSL obrigatório só para hosts remotos (Neon/Render); Postgres local e do CI não tem SSL.
    connect_args = {"connect_timeout": 10}
    host_local = make_url(db_url).host in (None, "localhost", "127.0.0.1", "::1")
    if "sslmode" not in db_url and not host_local:
        connect_args["sslmode"] = "require"

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        echo=False,
        connect_args=connect_args
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