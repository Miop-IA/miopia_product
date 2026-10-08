# Miop.IA — Auditoria e Documentação Técnica do Projeto

O **Miop.IA** é uma solução para análise de credibilidade e detecção de padrões de desinformação em notícias escritas em Língua Portuguesa. O sistema é composto por:
1. **Extensão para Google Chrome (Manifest V3):** Interface client-side para extração contextual do corpo da matéria e exibição do diagnóstico.
2. **API Backend em Python (FastAPI):** Servidor ASGI assíncrono para validação textual, extração de features, inferência em cache $O(1)$ e registro de feedback comunitário.
3. **Pipeline de Machine Learning (Stacking Ensemble Multivisão):** Classificador ensemble híbrido combinando estilometria avançada (MATTR, POS tags, ortografia), modelagem temática (LDA e NMF), $n$-gramas em nível de caractere/palavra e metamodelo supervisionado (F1 aferido no manifesto).

---

## 📐 Arquitetura da Solução

```
[ Usuário no Navegador (G1, Folha, Estadão, CNN, etc.) ]
                          │
                          ▼
            [ Extensão Chrome (Manifest V3) ]
         (Content Script: sanitização e extração de texto)
                          │  HTTP POST /analisar
                          ▼
             [ Backend FastAPI (api/main.py) ]
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
[ Cache / Banco de Dados ]           [ Pipeline de ML (Stacking) ]
  • SQLite local (dev)                 1. Ramo Char: TF-IDF (3-5) + LinearSVC
  • PostgreSQL/Neon (prod)             2. Ramo Word: TF-IDF (1-2) + LinearSVC
  • Busca por Hash SHA-256             3. Ramo Estilo + Temas: 15 métricas + LDA/NMF -> XGBoost
  • Votos da Comunidade                4. Meta-Modelo: Regressão Logística (corte 0.46)
```

---

## 📂 Estrutura de Diretórios e Componentes

| Caminho | Descrição Técnica |
|---|---|
| [`miopia/`](file:///c:/Users/25894064/Documents/miopia_product/miopia) | **Extensão Google Chrome (Manifest V3):** Scripts de conteúdo (`scripts/content.js`), popup de interface (`popup.html`, `popup.js`, `popup.css`), manifesto e service worker. |
| [`backend/api/`](file:///c:/Users/25894064/Documents/miopia_product/backend/api) | **Núcleo da API FastAPI:** Endpoints REST (`main.py`), modelos de dados Pydantic (`schemas.py`), ORM SQLAlchemy (`models.py`, `database.py`), configurações (`config.py`), sanitização/validação (`filtro.py`), extrator estilométrico (`feature_extraction.py`) e orquestrador de inferência (`inferencia.py`). |
| [`backend/models/`](file:///c:/Users/25894064/Documents/miopia_product/backend/models) | **Artefatos Treinados de IA:** Pacote serializado com o modelo Stacking (`stacking_miopia_v1.joblib`). |
| [`backend/train/`](file:///c:/Users/25894064/Documents/miopia_product/backend/train) | **Módulo de Treinamento e Calibração:** Script de treinamento (`train.py`) com validação cruzada, ajuste de hiperparâmetros e exportação de artefatos. |
| [`backend/tests/`](file:///c:/Users/25894064/Documents/miopia_product/backend/tests) | **Suíte de Testes Automatizados:** Testes de ponta a ponta da API (`test_api.py`), testes unitários do pipeline e limiares (`test_pipeline.py`) e script de teste direto de inferência (`teste_inferencia.py`). |
| [`backend/scripts/`](file:///c:/Users/25894064/Documents/miopia_product/backend/scripts) | **Scripts Utilitários:** Inicialização e criação do schema de banco de dados (`init_db.py`). |
| [`backend/schema.sql`](file:///c:/Users/25894064/Documents/miopia_product/backend/schema.sql) | Definição DDL em SQL das tabelas de `noticias` e `avaliacoes`. |

---

## 🤖 Engenharia de Machine Learning e Estilometria

O classificador não se apoia exclusivamente em palavras-chave contextuais, utilizando uma abordagem **multivisão** resistente a *concept drift* e manipulações intencionais de vocabulário:

1. **Ramo de Caracteres (`svm_caracteres`):**
   * TF-IDF em $n$-gramas de caracteres (3 a 5 gramas). Captura prefixos, sufixos, pontuação e microestruturas morfológicas.
2. **Ramo de Palavras (`svm_palavras`):**
   * TF-IDF de 1 a 2 gramas de palavras limpas, com remoção de stopwords padrão.
3. **Ramo Estilométrico e Temático (`xgb_denso`):**
   * **15 Features Estilométricas:**
     * `trunc_diversity`: **MATTR** (*Moving-Average Type-Token Ratio*) com janela deslizante de 25 palavras, eliminando viés pelo tamanho do texto.
     * `trunc_pausality`: Densidade de quebras estruturais (vírgulas, ponto e vírgula, dois-pontos, travessões).
     * `trunc_emotiveness`: Razão entre classes emotivas (adjetivos + advérbios) e substantivas (substantivos + verbos).
     * `trunc_upper_case_density`: Proporção de caracteres em caixa alta (indicativo de apelo sensacionalista).
     * `trunc_verb_density`, `trunc_noun_density`, `trunc_adj_density`, `trunc_adv_density`, `trunc_pron_density`: Densidade morfossintática via POS-tagging com spaCy (`pt_core_news_lg`).
     * `rc_spelling_errors`: Proporção de palavras fora do léxico em português via `pyspellchecker`.
     * `rc_modal_verbs_density`: Verbos modais de certeza/probabilidade (*poder, dever, precisar, etc.*).
     * `rc_subj_imp_verbs_density`: Verbos no subjuntivo e imperativo.
     * `rc_pron_1_2_sing_density` e `rc_pron_1_plur_density`: Densidade de pronomes pessoais e possessivos de 1ª/2ª pessoa do singular e plural.
   * **Modelagem Temática ($k=8$ e $k=30$):** Distribuição suave de tópicos por LDA (*Latent Dirichlet Allocation*) e NMF (*Non-Negative Matrix Factorization*), entropia e valores máximos de ativação.
4. **Metamodelo e Faixas de Decisão:**
   * Regressão Logística com limiar e score F1 calibrados dinamicamente via manifesto.
   * **Confiável:** Probabilidade $< 0{,}31$ (padrões textuais compatíveis com jornalismo profissional).
   * **Atenção:** $0{,}31 \le \text{Prob} \le 0{,}61$ (traços híbridos ou subjetividade elevada).
   * **Suspeita:** Probabilidade $> 0{,}61$ (anomalias estilométricas e alta probabilidade de desinformação).

---

## 🚀 Guia de Reprodução e Execução Local

### Pré-requisitos
* **Python 3.13.3 (64-bit)** (Recomendado para compatibilidade exata com o ambiente de treinamento e dependências de NLP/ML).
* **Navegador Google Chrome** (ou navegadores baseados em Chromium com suporte a Manifest V3).

### 1. Configuração do Backend

No terminal (PowerShell no Windows ou Bash no Linux/macOS):

```bash
cd backend

# Criação do ambiente virtual
python -m venv .venv

# Ativação do ambiente virtual
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux / macOS:
source .venv/bin/activate

# Instalação das dependências de produção e testes
pip install --upgrade pip
pip install -r requirements-dev.txt

# Inicialização do banco de dados local (SQLite)
python scripts/init_db.py
```

### 2. Execução da API

```bash
# Iniciar o servidor ASGI com recarregamento em tempo real
uvicorn api.main:app --reload --port 8000
```

* **Healthcheck:** [`http://localhost:8000/health`](http://localhost:8000/health)
* **Documentação Interativa (Swagger UI):** [`http://localhost:8000/docs`](http://localhost:8000/docs)
* **Especificação OpenAPI:** [`http://localhost:8000/openapi.json`](http://localhost:8000/openapi.json)

### 3. Instalação da Extensão Chrome

1. Abra o navegador e acesse: `chrome://extensions/`.
2. Ative o alternador **Modo do desenvolvedor** (*Developer mode*) no canto superior direito.
3. Clique no botão **Carregar sem compactação** (*Load unpacked*).
4. Selecione a pasta [`miopia/`](file:///c:/Users/25894064/Documents/miopia_product/miopia) da raiz do projeto.
5. Abra uma página de notícia (ex: *G1, Folha, Estadão, CNN*) e clique no ícone da extensão para testar.

---

## 🧪 Auditoria de Qualidade e Testes Automatizados

O repositório conta com testes unitários e de integração utilizando `pytest` cobrindo regras de negócio, endpoints e sanidade do modelo de ML:

```bash
cd backend
.\.venv\Scripts\pytest -v
```

### Resultados da Suíte de Testes (100% de Aprovação):
* `tests/test_api.py::test_health_check` — Validação do status e metadados de versão da API.
* `tests/test_api.py::test_analisar_noticia_sucesso_e_estrutura` — Validação de schema completo, persistência e métricas.
* `tests/test_api.py::test_analisar_noticia_volume_insuficiente` — Garantia de rejeição (HTTP 400) para textos com menos de 30 palavras.
* `tests/test_api.py::test_avaliar_noticia_sucesso` — Registro de votos da comunidade (0=Verdadeiro, 1=Duvidoso, 2=Falso).
* `tests/test_api.py::test_avaliar_noticia_inexistente` — Retorno HTTP 404 para identificadores de notícia inexistentes.
* `tests/test_pipeline.py::test_extracao_features_e_chaves` — Conferência das 15 chaves estilométricas geradas pelo spaCy.
* `tests/test_pipeline.py::test_calcular_mattr_janela_deslizante` — Exatidão matemática do algoritmo MATTR (janela 25).
* `tests/test_pipeline.py::test_inferencia_stacking_e_limiar` — Carregamento do artefato e cálculo de probabilidades dentro do intervalo $[0, 1]$.
* `tests/test_pipeline.py::test_validacao_viabilidade_texto` — Regras de proteção contra textos curtos ou excessivamente longos.

### Teste de Inferência Rápido (Script Standalone):
```bash
python tests/teste_inferencia.py
```

---

## 🔒 Conformidade, Privacidade e LGPD

O projeto foi desenhado sob as diretrizes de **Privacy by Design** e a Lei Geral de Proteção de Dados (Lei nº 13.709/2018):

1. **Minimização de Dados (Art. 6º, III):**
   * A extensão não realiza coleta passiva ou em segundo plano. O envio ocorre exclusivamente sob comando explícito do usuário (*"Ler e Analisar Notícia"*).
   * O banco de dados em produção utiliza hashing criptográfico **SHA-256** para identificar artigos em cache, evitando a retenção permanente do texto bruto. O sistema armazena unicamente uma versão sanitizada e truncada (limite máximo de 500 palavras), estritamente necessária para reprodução iterativa das métricas e auditoria técnica do classificador.
2. **Anonimização de Feedback (Art. 12):**
   * Os votos da comunidade utilizam um identificador aleatório de cliente (`X-Client-Id`), sem coleta de nomes, e-mails, endereços IP ou credenciais do usuário.
3. **Limitação de Taxa (*Rate Limiting*):**
   * Proteção contra abusos via limite diário de requisições por cliente configurável no backend (`LIMITE_DIARIO_POR_CLIENTE`).