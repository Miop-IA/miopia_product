# PRD — Miop.IA

> Documento de contexto de produto. Leia isto primeiro, antes de qualquer tarefa de
> front-end ou back-end. Este documento não substitui a leitura do código já
> existente no repositório — ele explica o *porquê* das decisões, pra que qualquer
> tarefa nova seja coerente com o que já foi construído e com os princípios do
> produto, não só com a implementação técnica.

## 1. Do que se trata o projeto

Miop.IA é uma extensão de navegador que lê a notícia que o usuário está lendo no
momento e devolve sinais e contexto sobre possíveis características de
desinformação no texto — desenvolvida como projeto de residência em Inteligência
Artificial (Instituto de Pesquisas Eldorado / PUC-Campinas).

O nome é um jogo com "miopia": a extensão existe pra corrigir uma visão turva sobre
a notícia, não pra enxergar por quem lê.

## 2. O que ele faz

1. Lê o texto da notícia aberta na página atual (via content script).
2. Analisa o texto usando métricas estilométricas e um léxico de palavras-gatilho,
   previamente validados num dataset de notícias verdadeiras/falsas.
3. Abre uma modal mostrando: quais trechos do texto acenderam algum sinal de alerta,
   o que cada sinal costuma indicar, e lembretes gerais sobre como avaliar uma
   notícia antes de compartilhar.
4. Coleta a avaliação do próprio usuário sobre a notícia (verdadeira / duvidosa /
   falsa), armazenando isso para gerar evidência de impacto do produto.

## 3. Qual a pretensão com o projeto (visão e princípio central)

**O produto NÃO é um verificador de fatos, e NUNCA deve se comportar como um.**

A decisão de produto mais importante, tomada desde o início do projeto, é que a
extensão não dá veredito ("isso é falso"). Ela sinaliza *características do texto*
associadas a desinformação (ex.: linguagem de urgência, números sem fonte,
emotividade alta) e devolve isso ao usuário como insumo — a decisão final de
acreditar ou não continua sendo do usuário. O objetivo é estimular pensamento
crítico, não substituí-lo.

Isso tem implicações diretas de implementação que qualquer agente de
desenvolvimento precisa respeitar:
- Nunca exibir um número de confiança/probabilidade como se fosse uma certeza
  absoluta (ex.: "98% falsa"). Se um score for exibido, ele precisa vir qualificado
  ("sinais fortes de", "possível", não "é").
- Nunca apresentar a detecção de palavras-gatilho ou métricas estilométricas como
  verificação factual. Exemplo real usado no design: a frase "muitos jovens irão
  morrer" tem um número sem fonte — isso é um *sinal estilístico* (ausência de
  fonte, generalização numérica), não uma verificação de que o fato é falso.
  Fact-checking de conteúdo está **fora do escopo** deste produto.

## 4. Comunidade real / público-alvo

Usuários de navegador que consomem notícias online e querem apoio pra avaliar
criticamente o que leem, sem depender de um veredito externo. O produto foi
validado com usuários reais ao longo do projeto (Design Sprint, Mágico de Oz —
ver histórico do Slot D da metodologia).

## 5. Definições de negócio

| Termo | Definição no contexto do produto |
|---|---|
| Sinal / destaque | Um trecho do texto ou métrica que acendeu um alerta (ex.: palavra-gatilho encontrada, emotividade acima do normal) |
| Avaliação do usuário | A opinião do próprio usuário sobre a notícia lida: Verdadeira / Duvidosa / Falsa |
| "Pense" | Bloco de conteúdo majoritariamente estático com padrões comuns de desinformação, não gerado dinamicamente por notícia |
| Evidência de impacto | Dado coletado (avaliação do usuário, uso recorrente) usado para demonstrar que o produto cumpre seu objetivo de promover pensamento crítico |

## 6. Fundação técnica (por que as decisões de produto são como são)

O time testou e comparou vários modelos de classificação (Regressão Logística
como baseline, SVM, XGBoost, Naive Bayes, KNN, Random Forest) sobre o Fake.br
Corpus (7.198 notícias em português, balanceadas verdadeira/falsa), usando tanto
TF-IDF quanto features estilométricas (pausalidade, emotividade, diversidade
lexical, densidade de classes gramaticais, tamanho de sentença/palavra).

**Achado metodológico crítico, que qualquer implementação de IA do produto precisa
respeitar:** usar metadados (autor, domínio, data) ou o texto sem truncamento
infla a acurácia artificialmente (vazamento de dado) — o time chegou a
resultados "perfeitos" (100%) que eram, na prática, o modelo aprendendo o
*tamanho* do texto ou a *fonte*, não o conteúdo. O resultado honesto, validado em
cenário sem vazamento e sem colinearidade, fica na faixa de 74–98% de F1
dependendo do modelo e das features usadas — SVM/XGBoost combinando TF-IDF com
estilometria performam melhor que modelos usando só estilometria tabular
(KNN/Random Forest).

**Consequência prática:** o backend deve expor os números reais (do cenário
validado, não do cenário com vazamento), e o front-end nunca deve arredondar isso
pra uma mensagem de certeza absoluta.

Além da classificação, o time também usou LDA e NMF (modelagem de tópicos) pra
identificar o tema da notícia (política, saúde etc.) — usado para a parte de
contexto, não para a classificação em si.

## 7. Requisitos funcionais

### RF1 — Leitura da página
A extensão captura o texto da notícia aberta na aba ativa (content script, DOM).

### RF2 — Análise do texto
O texto capturado é enviado para análise (backend já implementado pelos colegas
— ver nota na Seção 9) e retorna: sinais encontrados, classificação/confiança
(qualificada, nunca absoluta) e, quando aplicável, o tema identificado (LDA/NMF).

### RF3 — Detecção de palavras-gatilho (client-side, independente do modelo)
Lista de termos associados a alarmismo/urgência (ex.: "urgente", "chocante",
"compartilhe antes que apaguem") checada diretamente no texto, sem depender do
backend — serve como sinal rápido e também como fallback se a API estiver
indisponível.

### RF4 — Exibição da modal de análise
Ver `FRONTEND_SPEC_MiopIA.md` para o detalhamento completo de blocos, estados e
interações.

### RF5 — Coleta de avaliação do usuário
Usuário escolhe Verdadeira/Duvidosa/Falsa e confirma envio; resposta é persistida
no backend (ver proposta de contrato de dados no spec de front-end).

### RF6 — Armazenamento para evidência de impacto
Toda avaliação enviada, associada à análise que a originou, deve poder ser
consultada depois para demonstrar engajamento e evolução do uso do produto
(constraint de "evidências de impacto" do projeto).

## 8. Requisitos não funcionais

- **Nunca incluir o dataset bruto (texto das notícias de treino) no repositório** —
  é conteúdo de terceiros; o que viaja com o produto é o modelo treinado (pesos),
  não os dados de treino.
- **Privacidade (LGPD):** nenhum dado pessoal identificável do usuário deve ser
  coletado junto da avaliação, além do necessário para o produto funcionar
  (ex.: não armazenar nome/e-mail do usuário atrelado à avaliação sem necessidade
  clara).
- **Transparência:** qualquer número de confiança exibido precisa ser honesto em
  relação ao desempenho real validado (Seção 6), não ao cenário com vazamento de
  dado.
- **Leveza:** a extensão deve permanecer responsiva; evitar bloquear a UI
  esperando resposta do backend sem indicação de carregamento.

## 9. Arquitetura (alto nível) e o que já existe

```
Extensão (JS, content script) → captura texto da página
        ↓
API backend (Python) → JÁ IMPLEMENTADA PARCIALMENTE pelos colegas de equipe.
        │                Qualquer agente de desenvolvimento DEVE inspecionar o
        │                código já existente no repositório antes de propor
        │                endpoints novos ou alterar o contrato de dados.
        ↓
Modelo treinado + lógica de extração de feature (estilometria) → classificação
        ↓
Resposta → Modal no front-end (ver FRONTEND_SPEC_MiopIA.md)
        ↓
Avaliação do usuário → API → banco de dados (schema proposto no spec de front-end,
                                a validar contra o schema real já implementado)
```

**Decisão de arquitetura já tomada:** API própria (não Pyodide/PyScript, não
conversão do modelo para JS) — os colegas já começaram essa implementação, então
esse é o caminho confirmado, não mais uma decisão em aberto.

## 10. Fora de escopo (explicitamente)

- Fact-checking / verificação factual de afirmações específicas da notícia.
- Veredito definitivo de verdadeiro/falso apresentado como fato.
- Qualquer funcionalidade que substitua o julgamento do usuário em vez de
  informá-lo.

## 11. Assets de referência

- `frontend/src/assets/design.png` — design da modal gerado no Figma.
- `frontend/src/assets/Miop.IA logo.svg` (ou nome correspondente no repositório) —
  logo do produto, posicionado no canto superior esquerdo da modal.
