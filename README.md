# Miop.IA — Auditoria e Documentação Técnica do Projeto

O **Miop.IA** é uma solução para análise de credibilidade e detecção de padrões de desinformação em notícias escritas em Língua Portuguesa. O sistema é composto por:
1. **Extensão para Google Chrome (Manifest V3):** Interface client-side para extração contextual do corpo da matéria e exibição do diagnóstico.
2. **API Backend em Python (FastAPI):** Servidor ASGI assíncrono para validação textual, extração de features, inferência estruturada e registro de feedback comunitário.
3. **Pipeline de Machine Learning (Stacking Ensemble Multivisão):** Classificador ensemble híbrido combinando estilometria avançada (MATTR, POS tags, ortografia), modelagem temática (LDA e NMF), $n$-gramas em nível de caractere/palavra e metamodelo supervisionado (F1 atual $\approx 0.67$ aferido no manifesto).

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
[ Banco de Dados ]                   [ Pipeline de ML (Stacking) ]
  • PostgreSQL/Neon (prod)             1. Ramo Char: TF-IDF (3-5) + LinearSVC
  • Armazenamento Completo do Texto    2. Ramo Word: TF-IDF (1-2) + LinearSVC
  • Votos da Comunidade                3. Ramo Estilo + Temas: 15 métricas + LDA/NMF -> XGBoost
                                       4. Meta-Modelo: Regressão Logística (corte 0.46)
```

---

## 📂 Estrutura de Diretórios e Componentes

| Caminho | Descrição Técnica |
|---|---|
| [`miopia/`](file:///c:/Users/25894064/Documents/miopia_product/miopia) | **Extensão Google Chrome (Manifest V3):** Scripts de conteúdo (`scripts/content.js`), popup de interface (`popup.html`, `popup.js`, `popup.css`), manifesto e service worker. |
| [`backend/api/`](file:///c:/Users/25894064/Documents/miopia_product/backend/api) | **Núcleo da API FastAPI:** Endpoints REST (`main.py`), modelos de dados Pydantic (`schemas.py`), ORM SQLAlchemy (`models.py`, `database.py`), configurações (`config.py`), sanitização/validação (`filtro.py`), extrator estilométrico (`feature_extraction.py`) e orquestrador de inferência (`inferencia.py`). |
| [`backend/models/`](file:///c:/Users/25894064/Documents/miopia_product/backend/models) | **Artefatos Treinados de IA:** Pacote serializado (`stacking_miopia_v1.joblib`) e respectivo manifesto em JSON (`model_manifest.json`) com registro de `python_version`, `scikit-learn`, `xgboost`, `spacy`, `spacy_model`, `F1`, `threshold`, `feature_count`, `n_train`, `n_test`, `training_date`, etc. |
| [`backend/train/`](file:///c:/Users/25894064/Documents/miopia_product/backend/train) | **Módulo de Treinamento e Calibração:** Script de treinamento (`train.py`) com validação cruzada, ajuste de hiperparâmetros e exportação de artefatos. |
| [`backend/tests/`](file:///c:/Users/25894064/Documents/miopia_product/backend/tests) | **Suíte de Testes Automatizados:** Testes de ponta a ponta da API (`test_api.py`), falhas estruturais (`test_model_failures.py`), integridade de manifesto (`test_model_integrity.py`), paridade de extração (`test_parity_real.py`) e testes de OOF (`test_oof_protocol.py`). |
| [`backend/scripts/`](file:///c:/Users/25894064/Documents/miopia_product/backend/scripts) | **Scripts Utilitários:** Inicialização do banco. |
| [`backend/alembic/`](file:///c:/Users/25894064/Documents/miopia_product/backend/alembic) | **Migrations Alembic:** Versões de schema e atualização do banco. |

---

## 🤖 Engenharia de Machine Learning e Estilometria

O classificador não se apoia exclusivamente em palavras-chave contextuais, utilizando uma abordagem **multivisão** resistente a *concept drift* e manipulações intencionais de vocabulário:

1. **Ramo de Caracteres (`svm_caracteres`):**
   * TF-IDF em $n$-gramas de caracteres (3 a 5 gramas). Captura prefixos, sufixos, pontuação e microestruturas morfológicas.
2. **Ramo de Palavras (`svm_palavras`):**
   * TF-IDF de 1 a 2 gramas de palavras limpas (minúsculas, espaços normalizados), **sem** remoção de stopwords (para preservar coesão linguística e evitar viés estrutural).
3. **Ramo Estilométrico e Temático (`xgb_denso`):**
   * **15 Features Estilométricas:**
     * `trunc_diversity`: **MATTR** (*Moving-Average Type-Token Ratio*) com janela deslizante de 25 palavras, eliminando viés pelo tamanho do texto.
     * `trunc_pausality`: Densidade de quebras estruturais (vírgulas, ponto e vírgula, dois-pontos, travessões).
     * `trunc_emotiveness`: Razão entre classes emotivas (adjetivos + advérbios) e substantivas (substantivos + verbos).
     * `trunc_upper_case_density`: Proporção de caracteres em caixa alta (indicativo de apelo sensacionalista).
     * `trunc_verb_density`, `trunc_noun_density`, `trunc_adj_density`, `trunc_adv_density`, `trunc_pron_density`: Densidade morfossintática via POS-tagging com spaCy 3.8.16 (`pt_core_news_lg`).
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
* **Python 3.13.11 (64-bit)** (Recomendado para compatibilidade exata com o ambiente de treinamento e dependências de NLP/ML).
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

### Resultados da Suíte de Testes e CI (100% de Aprovação):
* **CI GitHub Actions:** Garantia de integração contínua (alembic, testes estruturais e de integridade).
* **`test_oof_protocol.py`:** Proteção absoluta contra vazamento de dados (*data leakage*) e garantia matemática de predições únicas no metamodelo Out-of-Fold (OOF).
* **`test_model_failures.py`:** A API adota postura "falha fechada" (HTTP 503) em caso de adulteração de hiperparâmetros, corrupção do bundle, manifesto incompleto ou dimensões adulteradas de features.
* **`test_model_integrity.py`:** Valida rigidamente o contrato do manifesto, assegurando F1-score estrito e estrutura idêntica para os vetores de treinamento.
* **`test_parity_real.py`:** Validação que o modelo de treinamento e de produção operam *exatamente* sob o mesmo pipeline de extração de features, garantindo divergência numérica perto de zero (`atol=1e-5`).
* **`test_api.py`:** Validação de comportamento da API REST (feedback da comunidade, proteção de volume insuficiente, retornos e rate limiting).

---

## 🔒 Conformidade, Privacidade e LGPD

O projeto foi desenhado sob as diretrizes de **Privacy by Design** e a Lei Geral de Proteção de Dados (Lei nº 13.709/2018):

1. **Retenção de Dados:**
   * A extensão não realiza coleta passiva ou em segundo plano. O envio ocorre exclusivamente sob comando explícito do usuário (*"Ler e Analisar Notícia"*).
   * O sistema armazena o texto completo em banco para viabilizar auditorias analíticas posteriores, re-treinamento e registro dos votos de validação da comunidade, mas abstém-se de manter PIIs ou perfis rastreáveis.
2. **Anonimização de Feedback (Art. 12):**
   * Os votos da comunidade utilizam um identificador aleatório de cliente (`X-Client-Id`), sem coleta de nomes, e-mails, endereços IP ou credenciais do usuário.
3. **Limitação de Taxa (*Rate Limiting*):**
   * Proteção contra abusos via limite diário de requisições por cliente configurável no backend (`LIMITE_DIARIO_POR_CLIENTE`).