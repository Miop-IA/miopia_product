# Benchmark de Extração de Notícias - Miop.IA

Este diretório contém uma suíte de testes de extração (benchmark) real para avaliar a assertividade e eficiência do `content.js` nos principais portais de notícias brasileiros (G1, UOL, Folha, R7, Metrópoles, CNN Brasil, Estadão, BBC).

## Objetivo
O benchmark avalia automaticamente a qualidade de:
1. **Seletores específicos** (Camada 1 do content script).
2. **Mozilla Readability** (Camada 2 de fallback).
3. **Fallback Semântico** (Camada 3).
4. **Filtros de Ruído e Densidade de Links**.

## Requisitos
- Node.js v18+ instalado.
- Pacotes instalados (rode `npm install` na pasta `benchmark`).

## Como rodar
Dentro da pasta `miopia/benchmark`, execute:

```bash
npm run benchmark
```

## Como adicionar um novo portal (Fixture)
1. Salve o código-fonte (HTML) de uma reportagem na pasta `fixtures/` com o nome do portal (ex: `uol_noticia.html`).
2. Crie um arquivo JSON com o mesmo nome (`uol_noticia.json`).
3. O JSON deve possuir o seguinte formato de contrato:

```json
{
    "url": "https://url-original-da-noticia-para-base-do-jsdom",
    "expected_method": "specific_selector",
    "words": 450,
    "tolerance": 10,
    "must_include": [
        "Trecho exato de um parágrafo que não pode ter sido excluído (Testa Falso Negativo)"
    ],
    "must_not_include": [
        "Leia também: Assine nosso jornal",
        "Publicidade"
    ]
}
```

O script testará cada página HTML isolando-a no `jsdom`, executando os scripts originais da extensão (`readability.js` e `content.js`) e aferindo os falsos positivos (ruídos) e falsos negativos (cortes).
