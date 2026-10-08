from functools import lru_cache
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações da aplicação Miop.IA backend.
    Lê valores do ambiente ou do ficheiro .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/miopia_db",
        description="URL de conexão com o banco de dados PostgreSQL"
    )
    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:3000"],
        description="Lista de origens permitidas para requisições CORS. Em produção, adicione o ID da extensão."
    )
    limite_diario_por_cliente: int = Field(
        default=50,
        ge=1,
        description="Limite máximo de requisições de análise diárias por client_id"
    )
    environment: str = Field(
        default="development",
        description="Ambiente de execução (development, production, testing)"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return v

    @property
    def sqlalchemy_url(self) -> str:
        """
        Garante explicitamente o uso do driver psycopg2 no SQLAlchemy 2.
        Substitui prefixos legados como 'postgres://' ou 'postgresql://'
        por 'postgresql+psycopg2://' para evitar falhas com psycopg v3.
        """
        url = self.database_url
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg2://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """
    Retorna uma instância única em cache das configurações.
    """
    return Settings()