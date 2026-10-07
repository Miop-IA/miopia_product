import hashlib
import re
from typing import Tuple


def normalizar_texto(texto: str) -> str:
    """
    Higieniza o texto cru removendo quebras de linha repetidas,
    espaços duplicados e caracteres de controlo invisíveis.
    """
    # Substitui sequências de espaços em branco e quebras de linha por espaço simples
    texto_limpo = re.sub(r"\s+", " ", texto)
    return texto_limpo.strip()


def gerar_hash_texto(texto: str) -> str:
    """
    Gera o identificador único SHA-256 (64 caracteres hexadecimais)
    a partir do texto devidamente normalizado para busca de cache e deduplicação.
    """
    texto_base = normalizar_texto(texto)
    return hashlib.sha256(texto_base.encode("utf-8")).hexdigest()


def validar_viabilidade_analise(texto: str) -> Tuple[bool, str, int]:
    """
    Verifica se o conteúdo possui volume textual suficiente para análise estilométrica.
    Retorna (is_valido, mensagem_motivo, total_palavras).
    """
    texto_normalizado = normalizar_texto(texto)
    # Extrai tokens alfanuméricos simples para contabilização rápida de palavras
    palavras = re.findall(r"\b\w+\b", texto_normalizado)
    total_palavras = len(palavras)

    if total_palavras < 30:
        return (
            False,
            f"Texto com volume insuficiente ({total_palavras} palavras). O pipeline requer ao menos 30 palavras para calcular indicadores de estilo com estabilidade estatística.",
            total_palavras,
        )

    if len(texto_normalizado) > 60000:
        return (
            False,
            "Texto excede o limite máximo permitido de 60.000 caracteres.",
            total_palavras,
        )

    return True, "", total_palavras